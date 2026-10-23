package gpu

import (
	"bufio"
	"bytes"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
)

// SMIPath 找 nvidia-smi。systemd 起的进程 PATH 很短，而 WSL 把它放在 /usr/lib/wsl/lib ——
// 2026-10-22 一位 Windows 供给方的 agent 连上来四次、一张卡都没报，就是在这里静默返回了空。
func SMIPath() string {
	smiOnce.Do(func() {
		if p, err := exec.LookPath("nvidia-smi"); err == nil {
			smiPath = p
			return
		}
		for _, p := range []string{"/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi"} {
			if st, err := os.Stat(p); err == nil && !st.IsDir() {
				smiPath = p
				return
			}
		}
		smiPath = "nvidia-smi"
	})
	return smiPath
}

var (
	smiOnce sync.Once
	smiPath string
)

// Device is one physical GPU as seen on this host.
type Device struct {
	Index     int    `json:"index"`
	Model     string `json:"model"`
	VRAMGB    int    `json:"vram_gb"`
	UUID      string `json:"uuid"`
	FreeVRAM  int    `json:"free_vram_gb"`
	Util      int    `json:"utilization_pct"`
	Processes int    `json:"compute_processes"`
	// ProcNames 是 GPU 上计算进程的名字。
	//
	// 服务端 P2pWatchService 的违规判定认的是名字（xmrig / ethminer / hashcat …），
	// 只有 pid 判不了任何事。采名字不采内容 —— 这条界线和条款一致：
	// 在租来的机器上管行为，不管客户生成了什么。
	ProcNames []string `json:"compute_process_names,omitempty"`
	// Busy is our own composite verdict: utilization < 5% AND no processes
	// AND no "other platform" containers using this device.
	Busy bool `json:"busy"`
}

// Scan runs `nvidia-smi` and returns the current state of every GPU on the
// host. Returns an empty slice (not an error) if nvidia-smi is missing or
// has no devices — that way the agent can still connect, heartbeat with
// zero GPUs, and show up in the supplier dashboard as "idle, no cards".
func Scan() ([]Device, error) {
	out, err := exec.Command(SMIPath(),
		"--query-gpu=index,name,memory.total,memory.free,utilization.gpu,uuid",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		if _, ok := err.(*exec.Error); ok {
			// nvidia-smi not found — treat as zero GPUs, but say so: a silent
			// empty list looks exactly like a host with no cards.
			logOnce("nvidia-smi not found on PATH or in /usr/lib/wsl/lib — reporting 0 GPUs")
			return []Device{}, nil
		}
		return nil, fmt.Errorf("nvidia-smi query-gpu: %w", err)
	}

	devices := []Device{}
	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		fields := strings.Split(scanner.Text(), ",")
		if len(fields) < 6 {
			continue
		}
		for i := range fields {
			fields[i] = strings.TrimSpace(fields[i])
		}
		idx, _ := strconv.Atoi(fields[0])
		totalMB, _ := strconv.Atoi(fields[2])
		freeMB, _ := strconv.Atoi(fields[3])
		util, _ := strconv.Atoi(fields[4])
		devices = append(devices, Device{
			Index:    idx,
			Model:    fields[1],
			VRAMGB:   totalMB / 1024,
			UUID:     fields[5],
			FreeVRAM: freeMB / 1024,
			Util:     util,
		})
	}

	// Now count compute processes per GPU via a second nvidia-smi call.
	// This is the "is anyone else using this" signal that makes coexistence
	// with other rental platforms possible.
	// 连名字一起取。原来只取了 gpu_uuid,pid —— 而服务端的违规判定
	// 认的是名字，有 pid 没名字等于采了个用不上的东西。
	procOut, err := exec.Command(SMIPath(),
		"--query-compute-apps=gpu_uuid,pid,process_name",
		"--format=csv,noheader,nounits",
	).Output()
	if err == nil {
		procByUUID := map[string]int{}
		namesByUUID := map[string][]string{}
		ps := bufio.NewScanner(bytes.NewReader(procOut))
		for ps.Scan() {
			f := strings.Split(ps.Text(), ",")
			if len(f) < 1 {
				continue
			}
			uuid := strings.TrimSpace(f[0])
			procByUUID[uuid]++
			if len(f) >= 3 {
				// process_name 本身可能含逗号，把第三列之后的重新拼回去
				name := strings.TrimSpace(strings.Join(f[2:], ","))
				if name != "" && len(namesByUUID[uuid]) < 16 {
					namesByUUID[uuid] = append(namesByUUID[uuid], name)
				}
			}
		}
		for i := range devices {
			devices[i].Processes = procByUUID[devices[i].UUID]
			devices[i].ProcNames = namesByUUID[devices[i].UUID]
		}
	}

	// Composite busy verdict. 显存只作兜底：桌面机接着显示器，桌面本身就占几百 MB 到 1 GB 多，
	// 原来「用了 1 GB 以上就算忙」让这类卡永远上不了架。计算进程才是「别人正在用」的信号；
	// 显存门槛留给 WSL 这类看不到进程列表的环境。
	for i := range devices {
		devices[i].Busy = devices[i].Util >= 5 ||
			devices[i].Processes > 0 ||
			(devices[i].VRAMGB-devices[i].FreeVRAM) > 3
	}

	return devices, nil
}

// MockScan returns a single fake RTX 4090 that is always idle. Used when
// the agent is started with --runtime=mock so the control plane can be
// tested without real hardware.
func MockScan() []Device {
	return []Device{{
		Index:     0,
		Model:     "RTX 4090",
		VRAMGB:    24,
		UUID:      "GPU-mock-0",
		FreeVRAM:  24,
		Util:      0,
		Processes: 0,
		Busy:      false,
	}}
}

var loggedOnce sync.Map

func logOnce(msg string) {
	if _, dup := loggedOnce.LoadOrStore(msg, true); !dup {
		log.Printf("[gpu] %s", msg)
	}
}

// MemoryGB 读 /proc/meminfo 的 MemTotal。读不到返回 0。
func MemoryGB() int {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "MemTotal:") {
			f := strings.Fields(line)
			if len(f) >= 2 {
				kb, _ := strconv.Atoi(f[1])
				return (kb + 512*1024) / (1024 * 1024)
			}
		}
	}
	return 0
}
