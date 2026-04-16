package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config is loaded from /etc/cloudgpu-agent/config.yaml (or wherever --config
// points). Only a handful of fields are mandatory; everything else has a
// sensible default so the installer-generated file stays short.
type Config struct {
	// URL the agent dials. Default wss://cloudgpu.app/api/p2p/ws — the token
	// is appended at runtime: /api/p2p/ws/<token>.
	HubURL string `yaml:"hub_url"`

	// Plaintext supplier token from /suppliers/signup. SHA-256 hash on the
	// server side matches against p2p_supplier.token_hash.
	SupplierToken string `yaml:"supplier_token"`

	// Persistent cache dir for model weights. Per-template subdirs get
	// mounted into containers as the template requests.
	CacheDir string `yaml:"cache_dir"`

	// Soft cap on disk used by the cache. LRU eviction kicks in above this.
	CacheMaxGB int `yaml:"cache_max_gb"`

	// Heartbeat cadence (seconds). Server marks us offline after >60s gap.
	HeartbeatSeconds int `yaml:"heartbeat_seconds"`

	// Idle detection window in seconds — a GPU must be continuously idle for
	// this long before we report it as available.
	IdleWindowSeconds int `yaml:"idle_window_seconds"`

	// Runtime mode: "docker" (real) or "mock" (dry-run, for control-plane tests).
	Runtime string `yaml:"runtime"`

	// Optional persistent agent id. If blank, a new uuid is generated on
	// first run and written back to this file so subsequent starts keep the
	// same p2p_agent row.
	AgentID string `yaml:"agent_id"`
}

// Defaults used when the config file omits a field.
func Defaults() Config {
	return Config{
		HubURL:            "wss://cloudgpu.app/api/p2p/ws",
		CacheDir:          "/var/cache/cloudgpu-data",
		CacheMaxGB:        200,
		HeartbeatSeconds:  15,
		IdleWindowSeconds: 60,
		Runtime:           "docker",
	}
}

// Load reads the yaml file at path, filling defaults for missing fields.
// Returns an error if the file is missing, unreadable, or lacks a token.
func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	cfg := Defaults()
	if err := yaml.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if cfg.SupplierToken == "" {
		return nil, fmt.Errorf("supplier_token is required in %s", path)
	}
	return &cfg, nil
}

// Save writes the config back to disk. Used only to persist a newly
// generated agent_id so restarts keep the same identity.
func Save(path string, cfg *Config) error {
	out, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(path, out, 0o600)
}
