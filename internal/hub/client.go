package hub

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mrhy7176/cloudgpu-agent/internal/config"
	"github.com/mrhy7176/cloudgpu-agent/internal/gpu"
)

// Handler is what the main loop passes in to receive commands from the hub.
// Returning an error lets the client log and, optionally, send a nack.
type Handler interface {
	OnReserve(reqID string, gpuIndex int) (claimID string, err error)
	OnDockerRun(reqID, claimID, image string, ports []int, env map[string]string, keepAlive int) (serviceURL string, err error)
	OnRelease(claimID string) error
}

// Client is a long-lived connection to the CloudGPU hub. It owns the socket,
// does register + heartbeat + reconnect, and dispatches inbound commands
// to the Handler.
type Client struct {
	cfg       *config.Config
	handler   Handler
	agentID   string
	hostname  string
	conn      *websocket.Conn
	writeMu   sync.Mutex
	reconnect chan struct{}
}

// New constructs a Client. agentID is read from cfg; if empty a new one
// will be generated on first connect.
func New(cfg *config.Config, h Handler) *Client {
	host, _ := os.Hostname()
	return &Client{
		cfg:      cfg,
		handler:  h,
		agentID:  cfg.AgentID,
		hostname: host,
	}
}

// Run blocks until ctx is cancelled. Handles reconnect with exponential
// backoff on disconnects.
func (c *Client) Run(ctx context.Context) error {
	backoff := 5 * time.Second
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}
		if err := c.connectAndServe(ctx); err != nil {
			log.Printf("[hub] disconnected: %v — reconnecting in %s", err, backoff)
			select {
			case <-ctx.Done():
				return nil
			case <-time.After(backoff):
			}
			if backoff < 60*time.Second {
				backoff *= 2
			}
			continue
		}
		backoff = 5 * time.Second
	}
}

// connectAndServe opens the websocket, sends register, and runs the read +
// heartbeat loops until the socket closes or ctx is cancelled.
func (c *Client) connectAndServe(ctx context.Context) error {
	target, err := url.Parse(c.cfg.HubURL)
	if err != nil {
		return fmt.Errorf("parse hub_url: %w", err)
	}
	// Append the token as the final path segment.
	target.Path = target.Path + "/" + c.cfg.SupplierToken
	log.Printf("[hub] dialing %s://%s%s", target.Scheme, target.Host, "/api/p2p/ws/<redacted>")

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}
	conn, _, err := dialer.DialContext(ctx, target.String(), http.Header{})
	if err != nil {
		return fmt.Errorf("dial: %w", err)
	}
	defer conn.Close()
	c.conn = conn

	// Send register immediately.
	if err := c.sendRegister(); err != nil {
		return fmt.Errorf("register: %w", err)
	}
	log.Printf("[hub] connected and registered (agent_id=%s)", c.agentID)

	// Launch heartbeat + read loops. Whichever ends first kills the other.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	errc := make(chan error, 2)
	go func() { errc <- c.heartbeatLoop(ctx) }()
	go func() { errc <- c.readLoop(ctx) }()
	return <-errc
}

func (c *Client) sendRegister() error {
	if c.agentID == "" {
		c.agentID = fmt.Sprintf("agent-%s-%d", c.hostname, time.Now().UnixNano())
	}
	var gpus []gpu.Device
	if c.cfg.Runtime == "mock" {
		gpus = gpu.MockScan()
	} else {
		scanned, err := gpu.Scan()
		if err != nil {
			log.Printf("[hub] gpu scan failed (continuing with empty list): %v", err)
		}
		gpus = scanned
	}
	payload := map[string]interface{}{
		"type":      "register",
		"agent_id":  c.agentID,
		"hostname":  c.hostname,
		"os":        runtime.GOOS + "-" + runtime.GOARCH,
		"cpu_cores": runtime.NumCPU(),
		"memory_gb": 0, // TODO: read from /proc/meminfo
		"version":   AgentVersion,
		"gpus":      gpus,
	}
	return c.sendJSON(payload)
}

func (c *Client) heartbeatLoop(ctx context.Context) error {
	t := time.NewTicker(time.Duration(c.cfg.HeartbeatSeconds) * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
			var gpus []gpu.Device
			if c.cfg.Runtime == "mock" {
				gpus = gpu.MockScan()
			} else {
				gpus, _ = gpu.Scan()
			}
			if err := c.sendJSON(map[string]interface{}{
				"type":     "heartbeat",
				"agent_id": c.agentID,
				"gpus":     gpus,
			}); err != nil {
				return fmt.Errorf("heartbeat send: %w", err)
			}
		}
	}
}

func (c *Client) readLoop(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		_, msg, err := c.conn.ReadMessage()
		if err != nil {
			if errors.Is(err, websocket.ErrCloseSent) {
				return err
			}
			return fmt.Errorf("read: %w", err)
		}
		c.dispatch(msg)
	}
}

// dispatch parses an inbound message and routes it to the handler. The
// handler's return value is packaged into an ack message and sent back on
// the same socket.
func (c *Client) dispatch(raw []byte) {
	var generic map[string]interface{}
	if err := json.Unmarshal(raw, &generic); err != nil {
		log.Printf("[hub] bad message: %v", err)
		return
	}
	t, _ := generic["type"].(string)
	reqID, _ := generic["req_id"].(string)
	switch t {
	case "reserve":
		idx, _ := generic["gpu_index"].(float64)
		claimID, err := c.handler.OnReserve(reqID, int(idx))
		ack := map[string]interface{}{
			"type":     "reserve_ack",
			"req_id":   reqID,
			"agent_id": c.agentID,
			"ok":       err == nil,
		}
		if err != nil {
			ack["reason"] = err.Error()
		} else {
			ack["claim_id"] = claimID
		}
		_ = c.sendJSON(ack)

	case "docker_run":
		claimID, _ := generic["claim_id"].(string)
		image, _ := generic["image"].(string)
		ports := toIntList(generic["ports"])
		env := toStringMap(generic["env"])
		keepAlive := 60
		if v, ok := generic["keep_alive_minutes"].(float64); ok {
			keepAlive = int(v)
		}
		url, err := c.handler.OnDockerRun(reqID, claimID, image, ports, env, keepAlive)
		ack := map[string]interface{}{
			"type":     "run_ack",
			"req_id":   reqID,
			"agent_id": c.agentID,
		}
		if err != nil {
			ack["status"] = "failed"
			ack["reason"] = err.Error()
		} else {
			ack["status"] = "running"
			ack["service_url"] = url
		}
		_ = c.sendJSON(ack)

	case "release":
		claimID, _ := generic["claim_id"].(string)
		err := c.handler.OnRelease(claimID)
		_ = c.sendJSON(map[string]interface{}{
			"type":     "release_ack",
			"claim_id": claimID,
			"agent_id": c.agentID,
			"ok":       err == nil,
		})

	case "ping":
		_ = c.sendJSON(map[string]interface{}{"type": "pong", "agent_id": c.agentID})

	default:
		log.Printf("[hub] ignoring unknown message type %q", t)
	}
}

// sendJSON serialises v and writes it to the socket. Synchronised because
// gorilla/websocket forbids concurrent writers on the same connection.
func (c *Client) sendJSON(v interface{}) error {
	if c.conn == nil {
		return errors.New("not connected")
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	return c.conn.WriteJSON(v)
}

// AgentID returns the id we registered with (generated on first connect).
func (c *Client) AgentID() string { return c.agentID }

func toIntList(v interface{}) []int {
	if v == nil {
		return nil
	}
	list, ok := v.([]interface{})
	if !ok {
		return nil
	}
	out := make([]int, 0, len(list))
	for _, x := range list {
		if f, ok := x.(float64); ok {
			out = append(out, int(f))
		}
	}
	return out
}

func toStringMap(v interface{}) map[string]string {
	if v == nil {
		return nil
	}
	m, ok := v.(map[string]interface{})
	if !ok {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, val := range m {
		if s, ok := val.(string); ok {
			out[k] = s
		}
	}
	return out
}

// AgentVersion is injected at build time via -ldflags -X. The default
// placeholder value is used in development builds.
var AgentVersion = "0.1.0-dev"
