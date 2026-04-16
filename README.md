# cloudgpu-agent

Supplier-side agent that connects a GPU host to the CloudGPU control plane.

## What it does

- Detects local NVIDIA GPUs via `nvidia-smi`.
- Opens an outbound WebSocket to `wss://cloudgpu.app/api/p2p/ws/<token>`.
- Reports availability every 15 seconds (heartbeat).
- Accepts reserve/docker_run/release commands from the hub.
- Atomically claims GPUs via `flock` so a race with other rental platforms
  sharing the same host is safe.
- Runs containers with `docker run --gpus device=<i>` + persistent cache
  volumes per template.
- Opens a reverse tunnel via `frpc` so the container's ports are reachable
  at `https://i-<claimId>.i.cloudgpu.app`.

## Install (supplier side)

```sh
curl -fsSL https://cloudgpu.app/install.sh | sudo bash -s <YOUR_SUPPLIER_TOKEN>
```

Get your token at https://cloudgpu.app/suppliers/signup.

## Build from source

```sh
cd cloudgpu-agent
go build -o cloudgpu-agent ./cmd/cloudgpu-agent
```

## Run modes

The agent supports two runtime modes selected via `--runtime=` or
`runtime:` in `config.yaml`:

- **`docker`** (default) — real mode. Requires Docker + nvidia-container-toolkit
  on the host. Runs actual containers.
- **`mock`** — control-plane dry-run. Does not touch Docker or nvidia-smi
  for side effects. Reports 1× fake RTX 4090, always idle; reserve/run
  commands return fake success with a fake service URL. Used for end-to-end
  testing of the hub and scheduler without needing a real GPU.

## Status

Phase 2 scaffolding — register + heartbeat + message loop work end-to-end
against the CloudGPU hub. Reserve/run/release command handlers are stubbed
to return success so the control plane can be demonstrated. Real Docker
execution + frpc tunnel spawning are in-progress.
