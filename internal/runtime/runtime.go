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
	// Real claim uses flock on /var/lock/cloudgpu-gpu-<i>.lock AND re-checks
	// nvidia-smi --query-compute-apps to confirm nothing else is on the device.
	// See docs/P2P_SUPPLIER_DESIGN.md §coexistence. Stub for now — returns
	// success + tracks the intent so Run can attach to the right device.
	id := fmt.Sprintf("claim-%d-%d", gpuIndex, time.Now().UnixNano())
	d.mu.Lock()
	defer d.mu.Unlock()
	d.claims[id] = &claim{gpuIndex: gpuIndex}
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
	return nil
}

// keepAlive timer hooks: future work. The agent currently relies on the
// hub calling Release explicitly. If we want keep_alive_minutes to
// auto-release we'd fire a time.AfterFunc here and record the *time.Timer.
var _ = time.Second
