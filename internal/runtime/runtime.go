package runtime

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Handler is the minimal "reserve a GPU, run a container, release it"
// surface. Two implementations: Docker (real) and Mock (control-plane only).
type Handler interface {
	Reserve(gpuIndex int) (claimID string, err error)
	Run(claimID, image string, ports []int, env map[string]string, keepAliveMinutes int) (serviceURL string, err error)
	Release(claimID string) error
}

// NewHandler returns either a Docker or Mock implementation based on the
// requested runtime mode.
func NewHandler(mode string, cacheDir string) Handler {
	if mode == "mock" {
		return &mockHandler{}
	}
	return &dockerHandler{cacheDir: cacheDir, claims: make(map[string]*claim)}
}

// -------------------- Mock implementation --------------------

type mockHandler struct{ mu sync.Mutex }

func (m *mockHandler) Reserve(gpuIndex int) (string, error) {
	id := fmt.Sprintf("mock-claim-%d-%d", gpuIndex, time.Now().UnixNano())
	log.Printf("[runtime/mock] reserve gpu=%d claim=%s", gpuIndex, id)
	return id, nil
}

func (m *mockHandler) Run(claimID, image string, ports []int, env map[string]string, keep int) (string, error) {
	url := fmt.Sprintf("https://i-%s.i.cloudgpu.app", claimID)
	log.Printf("[runtime/mock] run claim=%s image=%s ports=%v env=%d keys keep=%dm -> %s",
		claimID, image, ports, len(env), keep, url)
	return url, nil
}

func (m *mockHandler) Release(claimID string) error {
	log.Printf("[runtime/mock] release claim=%s", claimID)
	return nil
}

// -------------------- Docker implementation --------------------

type claim struct {
	gpuIndex    int
	containerID string
	frpcCmd     *exec.Cmd // nil if no tunnel set up
	frpcCfgPath string    // path to the per-claim frpc ini, for cleanup on Release
	lockFile    *os.File  // held for the lifetime of the claim so a concurrent Reserve can't take this GPU
}

type dockerHandler struct {
	cacheDir string
	mu       sync.Mutex
	claims   map[string]*claim
}

// frps config from env. If FRPS_SERVER is unset the agent skips frpc
// spawn and returns a nil service URL — the control plane fills in a
// fallback URL (which will 502 until the tunnel is live).
var (
	frpsServer     = os.Getenv("FRPS_SERVER")      // e.g. "frps.cloudgpu.app:7000"
	frpsToken      = os.Getenv("FRPS_TOKEN")       // shared secret
	frpsSubdomain  = os.Getenv("FRPS_SUBDOMAIN_HOST") // e.g. "i.cloudgpu.app"
	frpcBinaryPath = firstExisting("/usr/local/bin/frpc", "/opt/frp/frpc", "/usr/bin/frpc")
)

func firstExisting(paths ...string) string {
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func (d *dockerHandler) Reserve(gpuIndex int) (string, error) {
	// Three-step coexistence guarantee:
	//   1. Take an exclusive flock on /var/lock/cloudgpu-gpu-<i>.lock so our
	//      own concurrent Reserves race safely.
	//   2. Re-scan nvidia-smi — if another platform's container (Vast,
	//      RunPod) is using this card right now, bail without launching.
	//      Race window from the hub-side busy check to this point is
	//      ~100-500ms; re-checking closes it.
	//   3. Set nvidia-smi compute mode EXCLUSIVE_PROCESS so that from this
	//      point on, any OTHER CUDA init on this device fails. Our own
	//      container starts first, so it owns the exclusive handle.
	id := fmt.Sprintf("claim-%d-%d", gpuIndex, time.Now().UnixNano())

	// Step 1: flock
	lockPath := fmt.Sprintf("/var/lock/cloudgpu-gpu-%d.lock", gpuIndex)
	lockFile, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return "", fmt.Errorf("open flock %s: %w", lockPath, err)
	}
	if err := syscall.Flock(int(lockFile.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		lockFile.Close()
		return "", fmt.Errorf("gpu %d is locked by a concurrent reserve", gpuIndex)
	}

	// Step 2: last-moment busy recheck. If nvidia-smi shows any compute
	// processes on this specific index, somebody got in first.
	busyOut, _ := exec.Command("nvidia-smi",
		"-i", fmt.Sprintf("%d", gpuIndex),
		"--query-compute-apps=pid",
		"--format=csv,noheader,nounits",
	).Output()
	if len(strings.TrimSpace(string(busyOut))) > 0 {
		syscall.Flock(int(lockFile.Fd()), syscall.LOCK_UN)
		lockFile.Close()
		return "", fmt.Errorf("gpu %d became busy between scan and reserve — another platform claimed it", gpuIndex)
	}

	// Step 3: lock out other CUDA consumers. Best-effort: on some unusual
	// driver configs (headless servers missing persistence daemon) this
	// call can fail non-fatally. We log + continue; the VRAM-hog inside
	// our container still provides partial protection.
	if out, cerr := exec.Command("nvidia-smi",
		"-i", fmt.Sprintf("%d", gpuIndex),
		"-c", "EXCLUSIVE_PROCESS",
	).CombinedOutput(); cerr != nil {
		log.Printf("[runtime/docker] WARN: could not set EXCLUSIVE_PROCESS on gpu %d: %v / %s",
			gpuIndex, cerr, strings.TrimSpace(string(out)))
	} else {
		log.Printf("[runtime/docker] gpu %d compute-mode = EXCLUSIVE_PROCESS", gpuIndex)
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	d.claims[id] = &claim{gpuIndex: gpuIndex, lockFile: lockFile}
	log.Printf("[runtime/docker] reserve gpu=%d claim=%s", gpuIndex, id)
	return id, nil
}

func (d *dockerHandler) Run(claimID, image string, ports []int, env map[string]string, keepAliveMinutes int) (string, error) {
	d.mu.Lock()
	cl, ok := d.claims[claimID]
	d.mu.Unlock()
	if !ok {
		return "", fmt.Errorf("unknown claim %s", claimID)
	}
	args := []string{
		"run", "-d",
		"--name", "cloudgpu-" + claimID,
		"--gpus", fmt.Sprintf("device=%d", cl.gpuIndex),
	}
	// Publish each container port to a random host port (Docker picks).
	// We'll read the mapping back via `docker port` after the container
	// starts, then point frpc at the host-side port.
	for _, p := range ports {
		args = append(args, "-p", fmt.Sprintf("%d", p))
	}
	for k, v := range env {
		args = append(args, "-e", k+"="+v)
	}
	args = append(args, image)
	log.Printf("[runtime/docker] %s", strings.Join(append([]string{"docker"}, args...), " "))
	cmd := exec.Command("docker", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("docker run: %w: %s", err, strings.TrimSpace(string(out)))
	}
	containerID := strings.TrimSpace(string(out))
	d.mu.Lock()
	cl.containerID = containerID
	d.mu.Unlock()

	// Resolve host-side port. The FIRST declared port is what we tunnel;
	// multi-port containers will grow a json-per-port later.
	hostPort := 0
	if len(ports) > 0 {
		hostPort = inspectHostPort(containerID, ports[0])
	}

	// Set up frpc tunnel if configured AND we captured a host port.
	if frpsServer != "" && frpcBinaryPath != "" && hostPort > 0 {
		if url, ferr := d.startFrpcTunnel(cl, claimID, hostPort); ferr != nil {
			log.Printf("[runtime/docker] frpc setup failed: %v — service will be unreachable until tunnel restarts", ferr)
		} else {
			return url, nil
		}
	}

	// No tunnel — return empty so the control plane knows to mint a
	// fallback URL (which will 502 until the tunnel grows up).
	return "", nil
}

// inspectHostPort runs `docker port <id> <container>/tcp` and parses the
// output like "0.0.0.0:49153". Returns 0 on any failure so the caller
// degrades gracefully.
func inspectHostPort(containerID string, containerPort int) int {
	out, err := exec.Command("docker", "port", containerID, fmt.Sprintf("%d/tcp", containerPort)).Output()
	if err != nil {
		return 0
	}
	// output: "0.0.0.0:49153\n:::49153\n" (IPv4 + IPv6 rows)
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		idx := strings.LastIndex(line, ":")
		if idx < 0 {
			continue
		}
		portStr := line[idx+1:]
		var p int
		if _, err := fmt.Sscanf(portStr, "%d", &p); err == nil && p > 0 {
			return p
		}
	}
	return 0
}

// startFrpcTunnel writes a per-claim frpc ini, spawns frpc, and returns
// the public URL the tunnel resolves to. Cleanup is done in Release via
// the stored *exec.Cmd. The subdomain is the claim id (URL-safe).
func (d *dockerHandler) startFrpcTunnel(cl *claim, claimID string, hostPort int) (string, error) {
	host := frpsSubdomain
	if host == "" {
		host = "i.cloudgpu.app"
	}

	cfgDir := "/var/run/cloudgpu-agent"
	_ = os.MkdirAll(cfgDir, 0755)
	cfgPath := filepath.Join(cfgDir, "frpc-"+claimID+".ini")

	// frp 0.x / 1.x still accept INI. The HTTP vhost on frps routes
	// subdomain.<host> -> this tunnel -> local_port.
	cfg := fmt.Sprintf(`[common]
server_addr = %s
token = %s
log_level = warn

[cloudgpu-%s]
type = http
local_ip = 127.0.0.1
local_port = %d
subdomain = %s
`, strings.TrimSuffix(frpsServer, ":7000")+":7000",
		frpsToken, claimID, hostPort, claimID)

	if err := os.WriteFile(cfgPath, []byte(cfg), 0600); err != nil {
		return "", fmt.Errorf("write frpc config: %w", err)
	}

	cmd := exec.Command(frpcBinaryPath, "-c", cfgPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("spawn frpc: %w", err)
	}
	d.mu.Lock()
	cl.frpcCmd = cmd
	cl.frpcCfgPath = cfgPath
	d.mu.Unlock()

	url := fmt.Sprintf("https://%s.%s/", claimID, host)
	log.Printf("[runtime/docker] frpc tunnel up: %s -> localhost:%d", url, hostPort)
	return url, nil
}

// unused helper to keep encoding/json imported in case we expand run_ack
// later with a multi-port mapping payload.
var _ = json.Marshal

func (d *dockerHandler) Release(claimID string) error {
	d.mu.Lock()
	cl, ok := d.claims[claimID]
	delete(d.claims, claimID)
	d.mu.Unlock()
	if !ok {
		return nil
	}
	// Kill the tunnel first so the customer's connections close cleanly
	// before their container disappears. Best-effort — if frpc was already
	// dead the Process.Kill returns an error we don't care about.
	if cl.frpcCmd != nil && cl.frpcCmd.Process != nil {
		_ = cl.frpcCmd.Process.Kill()
		_, _ = cl.frpcCmd.Process.Wait()
	}
	if cl.frpcCfgPath != "" {
		_ = os.Remove(cl.frpcCfgPath)
	}
	if cl.containerID != "" {
		_ = exec.Command("docker", "stop", "-t", "5", cl.containerID).Run()
		_ = exec.Command("docker", "rm", "-f", cl.containerID).Run()
	}
	// Restore compute mode DEFAULT so the supplier's other rental platforms
	// (Vast, RunPod) can again launch containers on this GPU. Best-effort —
	// failure only means the GPU stays in EXCLUSIVE mode until next reboot
	// or agent restart, which is a degraded but not broken state.
	if out, cerr := exec.Command("nvidia-smi",
		"-i", fmt.Sprintf("%d", cl.gpuIndex),
		"-c", "DEFAULT",
	).CombinedOutput(); cerr != nil {
		log.Printf("[runtime/docker] WARN: could not restore DEFAULT compute-mode on gpu %d: %v / %s",
			cl.gpuIndex, cerr, strings.TrimSpace(string(out)))
	}
	// Release the flock last so a new Reserve can't race in until the
	// compute mode is back to DEFAULT.
	if cl.lockFile != nil {
		_ = syscall.Flock(int(cl.lockFile.Fd()), syscall.LOCK_UN)
		_ = cl.lockFile.Close()
	}
	log.Printf("[runtime/docker] released claim=%s, gpu %d back to DEFAULT", claimID, cl.gpuIndex)
	return nil
}

// keepAlive timer hooks: future work. The agent currently relies on the
// hub calling Release explicitly. If we want keep_alive_minutes to
// auto-release we'd fire a time.AfterFunc here and record the *time.Timer.
var _ = time.Second
