package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/mrhy7176/cloudgpu-agent/internal/config"
	"github.com/mrhy7176/cloudgpu-agent/internal/hub"
	"github.com/mrhy7176/cloudgpu-agent/internal/runtime"
)

// agentHandler is the glue between the hub client and the runtime handler.
// Inbound hub messages (reserve / docker_run / release) call these methods
// which in turn delegate to the Docker or Mock runtime.
type agentHandler struct {
	rt runtime.Handler
}

func (a *agentHandler) OnReserve(reqID string, gpuIndex int) (string, error) {
	return a.rt.Reserve(gpuIndex)
}

func (a *agentHandler) OnDockerRun(reqID string, spec runtime.RunSpec) (string, error) {
	return a.rt.Run(spec)
}

func (a *agentHandler) OnRelease(claimID string) error {
	return a.rt.Release(claimID)
}

func (a *agentHandler) OnPrefetch(images []string) {
	a.rt.Prefetch(images)
}

func main() {
	cfgPath := flag.String("config", "/etc/cloudgpu-agent/config.yaml", "path to config.yaml")
	runtimeOverride := flag.String("runtime", "", "override config runtime: docker or mock")
	flag.Parse()

	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.Printf("cloudgpu-agent %s starting, config=%s", hub.AgentVersion, *cfgPath)

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		log.Fatalf("config load failed: %v", err)
	}
	// agent_id 为空时只生成一次并写回配置文件。以前 Save() 从没被调用过，
	// 每次重启都换一个新 id：hub 侧多一行 p2p_agent，lease 表还指着旧的死 id。
	// 写回失败只告警不退出 —— 本次进程内的 id 仍然一致，下次重启再试。
	// 必须在应用 --runtime 覆盖之前写，否则命令行临时指定的 mock 会被固化进文件。
	if cfg.AgentID == "" {
		host, _ := os.Hostname()
		cfg.AgentID = fmt.Sprintf("agent-%s-%d", host, time.Now().UnixNano())
		if err := config.Save(*cfgPath, cfg); err != nil {
			log.Printf("WARN: could not persist generated agent_id to %s: %v (id will change on next restart)", *cfgPath, err)
		} else {
			log.Printf("generated agent_id=%s and saved to %s", cfg.AgentID, *cfgPath)
		}
	}
	if *runtimeOverride != "" {
		cfg.Runtime = *runtimeOverride
	}
	log.Printf("runtime=%s hub=%s heartbeat=%ds cache=%s(%dGB)",
		cfg.Runtime, cfg.HubURL, cfg.HeartbeatSeconds, cfg.CacheDir, cfg.CacheMaxGB)

	rt := runtime.NewHandler(cfg.Runtime, cfg.CacheDir)
	rt.CleanupOrphans()
	handler := &agentHandler{rt: rt}
	client := hub.New(cfg, handler)

	ctx, cancel := context.WithCancel(context.Background())
	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		s := <-sigc
		log.Printf("signal %s — shutting down", s)
		cancel()
	}()

	if err := client.Run(ctx); err != nil {
		log.Printf("hub client exited: %v", err)
	}
	log.Printf("bye")
}
