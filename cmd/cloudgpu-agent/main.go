package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

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

func (a *agentHandler) OnDockerRun(reqID, claimID, image string, ports []int, env map[string]string, keepAlive int) (string, error) {
	return a.rt.Run(claimID, image, ports, env, keepAlive)
}

func (a *agentHandler) OnRelease(claimID string) error {
	return a.rt.Release(claimID)
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
	if *runtimeOverride != "" {
		cfg.Runtime = *runtimeOverride
	}
	log.Printf("runtime=%s hub=%s heartbeat=%ds cache=%s(%dGB)",
		cfg.Runtime, cfg.HubURL, cfg.HeartbeatSeconds, cfg.CacheDir, cfg.CacheMaxGB)

	rt := runtime.NewHandler(cfg.Runtime, cfg.CacheDir)
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
