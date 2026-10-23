package runtime

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/mrhy7176/cloudgpu-agent/internal/gpu"
)

// Handler is the minimal "reserve a GPU, run a container, release it"
// surface. Two implementations: Docker (real) and Mock (control-plane only).
type Handler interface {
	Reserve(gpuIndex int) (claimID string, err error)
	Run(spec RunSpec) (serviceURL string, err error)
	Release(claimID string) error
	// CleanupOrphans 在 agent 启动时清掉上一个进程留下的容器：claim 只记在内存里，
	// 进程一重启它们就没人管了，而 hub 那边早已按掉线把租约关掉。
	CleanupOrphans()
	// Prefetch 后台拉镜像，不阻塞调用方。第一单来时镜像已在本地，开机才赶得上超时。
	Prefetch(images []string)
}

// RunSpec 是 hub 的 docker_run 消息。
type RunSpec struct {
	ClaimID string
	Image   string
	// Entrypoint 覆盖镜像的 ENTRYPOINT；空 = 用镜像自带的。
	Entrypoint string
	// User 以哪个用户起容器（docker run --user）；空 = 镜像默认。
	User string
	// Command / Args 覆盖镜像默认 CMD；为空则用镜像自带的。
	Command          string
	Args             []string
	Ports            []int
	Env              map[string]string
	KeepAliveMinutes int
	// Tunnel 非空时用 frpc 把 Tunnel.LocalPort 挂到 https://<CustomDomain>。
	Tunnel *Tunnel
}

// Tunnel 由 hub 下发：每条租约一个登记过的 id，frps 插件只放行登记过的 id。
// 以前 frps 地址和 token 要供给方自己写进环境变量，安装脚本一个都没写，结果没有一台机器起过隧道。
type Tunnel struct {
	ProxyName    string   `json:"proxy_name"`
	CustomDomain string   `json:"custom_domain"`
	ServerAddr   string   `json:"server_addr"`
	ServerPort   int      `json:"server_port"`
	Token        string   `json:"token"`
	LocalPort    int      `json:"local_port"`
	Bandwidth    string   `json:"bandwidth"`
	FrpcURLs     []string `json:"frpc_urls"`
	FrpcSHA256   string   `json:"frpc_sha256"`
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

// mock 模式不起容器也不起隧道，回一个隧道地址的样子，给控制面做联调。
func (m *mockHandler) Run(spec RunSpec) (string, error) {
	url := fmt.Sprintf("https://i-%s.i.cloudgpu.app", spec.ClaimID)
	if spec.Tunnel != nil && spec.Tunnel.CustomDomain != "" {
		url = "https://" + spec.Tunnel.CustomDomain
	}
	log.Printf("[runtime/mock] run claim=%s image=%s ports=%v env=%d keys -> %s",
		spec.ClaimID, spec.Image, spec.Ports, len(spec.Env), url)
	return url, nil
}

func (m *mockHandler) CleanupOrphans() {}

func (m *mockHandler) Prefetch(images []string) {
	log.Printf("[runtime/mock] prefetch %v (skipped)", images)
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
	busyOut, _ := exec.Command(gpuSMI(),
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
	if out, cerr := exec.Command(gpuSMI(),
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

func (d *dockerHandler) Run(spec RunSpec) (string, error) {
	d.mu.Lock()
	cl, ok := d.claims[spec.ClaimID]
	d.mu.Unlock()
	if !ok {
		return "", fmt.Errorf("unknown claim %s", spec.ClaimID)
	}
	if spec.Image == "" {
		return "", fmt.Errorf("docker_run without an image")
	}
	args := []string{
		"run", "-d",
		"--name", "cloudgpu-" + spec.ClaimID,
		"--label", orphanLabel + "=" + spec.ClaimID,
		"--gpus", fmt.Sprintf("device=%d", cl.gpuIndex),
		// PyTorch 的 DataLoader 多进程要共享内存，docker 默认 64M 一跑就崩。
		"--shm-size", "8g",
	}
	if spec.Entrypoint != "" {
		args = append(args, "--entrypoint", spec.Entrypoint)
	}
	if spec.User != "" {
		args = append(args, "--user", spec.User)
	}
	// 端口只发布到本机回环：外面的人只能经隧道进来。原来发布在 0.0.0.0，
	// 供给方机器有公网 IP 的话，客户的 Jupyter / Ollama 直接裸奔在公网上。
	for _, p := range spec.Ports {
		args = append(args, "-p", fmt.Sprintf("127.0.0.1::%d", p))
	}
	for k, v := range spec.Env {
		args = append(args, "-e", k+"="+v)
	}
	args = append(args, spec.Image)
	// 镜像名之后紧跟 command + args，即 `docker run ... <image> <command> <args...>`，
	// 覆盖镜像默认 CMD。command 为空时不追加，让镜像自己的 ENTRYPOINT/CMD 生效。
	if spec.Command != "" {
		args = append(args, spec.Command)
	}
	args = append(args, spec.Args...)
	log.Printf("[runtime/docker] docker run %s (env %d keys)", spec.Image, len(spec.Env))
	cmd := exec.Command("docker", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("docker run: %w: %s", err, strings.TrimSpace(string(out)))
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	containerID := strings.TrimSpace(lines[len(lines)-1])
	d.mu.Lock()
	cl.containerID = containerID
	d.mu.Unlock()

	if spec.Tunnel == nil {
		// 没有隧道就没有人能连进来。hub 自 0.2.0 起总是下发隧道；没下发说明控制面版本不对，别假装成功。
		return "", fmt.Errorf("docker_run without tunnel — nothing could reach this container")
	}
	hostPort := inspectHostPort(containerID, spec.Tunnel.LocalPort)
	if hostPort == 0 {
		return "", fmt.Errorf("container port %d is not published", spec.Tunnel.LocalPort)
	}
	url, err := d.startFrpcTunnel(cl, spec.Tunnel, hostPort)
	if err != nil {
		return "", fmt.Errorf("tunnel: %w", err)
	}
	return url, nil
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

// startFrpcTunnel writes a per-claim frpc.toml, spawns frpc, and waits until
// frps has accepted the proxy. Cleanup is done in Release via the stored
// *exec.Cmd. Format matches the backend's ApiTunnelInstaller (TOML, TLS,
// customDomains = <id>-i.cloudgpu.app).
func (d *dockerHandler) startFrpcTunnel(cl *claim, t *Tunnel, hostPort int) (string, error) {
	if t.ProxyName == "" || t.CustomDomain == "" || t.ServerAddr == "" {
		return "", fmt.Errorf("incomplete tunnel spec")
	}
	bin, err := ensureFrpc(t.FrpcURLs, t.FrpcSHA256)
	if err != nil {
		return "", err
	}
	cfgDir := "/var/run/cloudgpu-agent"
	_ = os.MkdirAll(cfgDir, 0700)
	cfgPath := filepath.Join(cfgDir, "frpc-"+t.ProxyName+".toml")
	logPath := filepath.Join(cfgDir, "frpc-"+t.ProxyName+".log")
	port := t.ServerPort
	if port == 0 {
		port = 7000
	}
	bw := ""
	if t.Bandwidth != "" {
		bw = fmt.Sprintf("transport.bandwidthLimit = %q\ntransport.bandwidthLimitMode = \"server\"\n", t.Bandwidth)
	}
	cfg := fmt.Sprintf(`serverAddr = %q
serverPort = %d
auth.method = "token"
auth.token = %q
transport.tls.enable = true
loginFailExit = false
log.to = %q
log.level = "info"

[[proxies]]
name = %q
type = "http"
localIP = "127.0.0.1"
localPort = %d
customDomains = [%q]
%s`, t.ServerAddr, port, t.Token, logPath, t.ProxyName, hostPort, t.CustomDomain, bw)
	if err := os.WriteFile(cfgPath, []byte(cfg), 0600); err != nil {
		return "", fmt.Errorf("write frpc config: %w", err)
	}
	_ = os.Remove(logPath)
	cmd := exec.Command(bin, "-c", cfgPath)
	// 供给方机器上常带着 http_proxy（国内机房尤其）：frpc 走代理连我们 IP 会断，必须直连。
	cmd.Env = withoutProxy(os.Environ())
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("spawn frpc: %w", err)
	}
	d.mu.Lock()
	cl.frpcCmd = cmd
	cl.frpcCfgPath = cfgPath
	d.mu.Unlock()

	// 等 frps 放行（日志里出现 start proxy success）。不等的话 hub 会把一个还没通的地址交给客户。
	deadline := time.Now().Add(45 * time.Second)
	for time.Now().Before(deadline) {
		time.Sleep(1 * time.Second)
		b, _ := os.ReadFile(logPath)
		text := string(b)
		if strings.Contains(text, "start proxy success") {
			url := "https://" + t.CustomDomain
			log.Printf("[runtime/docker] tunnel up: %s -> localhost:%d", url, hostPort)
			return url, nil
		}
		if strings.Contains(text, "start error") || strings.Contains(text, "reject") {
			return "", fmt.Errorf("frps refused the tunnel: %s", lastLine(text))
		}
	}
	b, _ := os.ReadFile(logPath)
	return "", fmt.Errorf("frpc did not connect within 45s: %s", lastLine(string(b)))
}

func lastLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.LastIndex(s, "\n"); i >= 0 {
		s = s[i+1:]
	}
	if len(s) > 300 {
		s = s[len(s)-300:]
	}
	return s
}

func withoutProxy(env []string) []string {
	out := env[:0:0]
	for _, kv := range env {
		k := strings.ToLower(strings.SplitN(kv, "=", 2)[0])
		if k == "http_proxy" || k == "https_proxy" || k == "all_proxy" {
			continue
		}
		out = append(out, kv)
	}
	return out
}

// frpc 放在 agent 自己的目录里，不碰供给方系统里可能已有的别的 frpc。
const frpcPath = "/usr/local/lib/cloudgpu-agent/frpc"

var frpcMu sync.Mutex

// ensureFrpc 返回可用的 frpc 路径：已有且校验通过就直接用，否则按顺序下载（.gz 自动解压）并校验 sha256。
func ensureFrpc(urls []string, sha string) (string, error) {
	frpcMu.Lock()
	defer frpcMu.Unlock()
	if ok, _ := fileSHA256Matches(frpcPath, sha); ok {
		return frpcPath, nil
	}
	if len(urls) == 0 {
		return "", fmt.Errorf("frpc missing and no download url")
	}
	_ = os.MkdirAll(filepath.Dir(frpcPath), 0755)
	var lastErr error
	for _, u := range urls {
		tmp := frpcPath + ".tmp"
		if err := download(u, tmp); err != nil {
			lastErr = err
			continue
		}
		if ok, got := fileSHA256Matches(tmp, sha); !ok {
			lastErr = fmt.Errorf("sha256 mismatch from %s: %s", u, got)
			_ = os.Remove(tmp)
			continue
		}
		_ = os.Chmod(tmp, 0755)
		if err := os.Rename(tmp, frpcPath); err != nil {
			lastErr = err
			continue
		}
		log.Printf("[runtime/docker] frpc installed from %s", u)
		return frpcPath, nil
	}
	return "", fmt.Errorf("could not install frpc: %v", lastErr)
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
	if out, cerr := exec.Command(gpuSMI(),
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

const orphanLabel = "cloudgpu.claim"

func gpuSMI() string { return gpu.SMIPath() }

func download(url, dst string) error {
	client := &http.Client{
		Timeout: 4 * time.Minute,
		// 直连：供给方机器上的 http_proxy 到我们 IP 常常是断的。
		Transport: &http.Transport{Proxy: nil},
	}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	var r io.Reader = resp.Body
	if strings.HasSuffix(url, ".gz") {
		gz, err := gzip.NewReader(resp.Body)
		if err != nil {
			return err
		}
		defer gz.Close()
		r = gz
	}
	f, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0755)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, r); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// fileSHA256Matches：want 为空时只要文件存在就算过（hub 没下发校验值）。
func fileSHA256Matches(path, want string) (bool, string) {
	f, err := os.Open(path)
	if err != nil {
		return false, ""
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false, ""
	}
	got := hex.EncodeToString(h.Sum(nil))
	return want == "" || strings.EqualFold(got, want), got
}

// CleanupOrphans 删掉带我们标签的容器，并把所有卡的计算模式恢复成 DEFAULT。
// 只在启动时调用：此刻内存里没有任何 claim，凡是带标签的容器都是上一个进程留下的。
func (d *dockerHandler) CleanupOrphans() {
	out, err := exec.Command("docker", "ps", "-aq", "--filter", "label="+orphanLabel).Output()
	if err != nil {
		log.Printf("[runtime/docker] WARN: list orphan containers: %v", err)
		return
	}
	ids := strings.Fields(string(out))
	for _, id := range ids {
		_ = exec.Command("docker", "rm", "-f", id).Run()
	}
	if len(ids) > 0 {
		log.Printf("[runtime/docker] removed %d orphan container(s) from a previous run", len(ids))
		_ = exec.Command(gpuSMI(), "-c", "DEFAULT").Run()
	}
}

var prefetchMu sync.Mutex

func (d *dockerHandler) Prefetch(images []string) {
	go func() {
		// 一次只拉一轮：agent 断线重连会再收到一次 prefetch，别叠着拉。
		if !prefetchMu.TryLock() {
			return
		}
		defer prefetchMu.Unlock()
		for _, img := range images {
			if img == "" {
				continue
			}
			if exec.Command("docker", "image", "inspect", img).Run() == nil {
				continue
			}
			log.Printf("[runtime/docker] prefetch: pulling %s", img)
			start := time.Now()
			out, err := exec.Command("docker", "pull", img).CombinedOutput()
			if err != nil {
				log.Printf("[runtime/docker] prefetch %s failed: %v: %s", img, err, lastLine(string(out)))
				continue
			}
			log.Printf("[runtime/docker] prefetch %s done in %s", img, time.Since(start).Round(time.Second))
		}
	}()
}

// keepAlive timer hooks: future work. The agent currently relies on the
// hub calling Release explicitly. If we want keep_alive_minutes to
// auto-release we'd fire a time.AfterFunc here and record the *time.Timer.
var _ = time.Second
