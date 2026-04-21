package gpu

import (
	"bufio"
	"bytes"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
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
	// Busy is our own composite verdict: utilization < 5% AND no processes
	// AND no "other platform" containers using this device.
	Busy bool `json:"busy"`
}

// Scan runs `nvidia-smi` and returns the current state of every GPU on the
// host. Returns an empty slice (not an error) if nvidia-smi is missing or
// has no devices — that way the agent can still connect, heartbeat with
// zero GPUs, and show up in the supplier dashboard as "idle, no cards".
func Scan() ([]Device, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,name,memory.total,memory.free,utilization.gpu,uuid",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		if _, ok := err.(*exec.Error); ok {
			// nvidia-smi not found — treat as zero GPUs
			return nil, nil
		}
		return nil, fmt.Errorf("nvidia-smi query-gpu: %w", err)
	}

	var devices []Device
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
	procOut, err := exec.Command("nvidia-smi",
		"--query-compute-apps=gpu_uuid,pid",
		"--format=csv,noheader,nounits",
	).Output()
	if err == nil {
		procByUUID := map[string]int{}
		ps := bufio.NewScanner(bytes.NewReader(procOut))
		for ps.Scan() {
			f := strings.Split(ps.Text(), ",")
			if len(f) >= 1 {
				procByUUID[strings.TrimSpace(f[0])]++
			}
		}
		for i := range devices {
			devices[i].Processes = procByUUID[devices[i].UUID]
		}
	}

	// Composite busy verdict.
	for i := range devices {
		devices[i].Busy = devices[i].Util >= 5 ||
			devices[i].Processes > 0 ||
			(devices[i].VRAMGB-devices[i].FreeVRAM) > 1
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
