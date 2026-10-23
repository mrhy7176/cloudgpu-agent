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
- Opens a reverse tunnel via `frpc` so the container's port is reachable
  at `https://<tunnelId>-i.cloudgpu.app` (the hub issues and registers the id).
- Pulls the container images in the background as soon as it connects, so the
  first rental does not wait on a multi-GB download.

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

## Tunnel (docker mode)

Nothing to configure on the supplier side. For every rental the hub sends a
`tunnel` object inside `docker_run` (server address, a per-lease proxy name,
`customDomains = <id>-i.cloudgpu.app`, frpc download URLs + sha256). The agent
downloads frpc to `/usr/local/lib/cloudgpu-agent/frpc` if it is missing,
verifies the sha256, writes `/var/run/cloudgpu-agent/frpc-<id>.toml` and waits
until frps accepts the proxy before acking. frps' auth plugin only accepts ids
the hub registered, so the shared frps token cannot publish anything else.

Container ports are published on `127.0.0.1` only; the tunnel is the only way in.

## Status

Register + heartbeat + message loop work end-to-end against the CloudGPU hub,
and the docker runtime is real (not stubbed):

- **reserve** — takes an exclusive `flock` on `/var/lock/cloudgpu-gpu-<i>.lock`,
  re-checks `nvidia-smi --query-compute-apps` so a card another platform just
  grabbed is refused, then sets the GPU compute mode to `EXCLUSIVE_PROCESS` so
  no other CUDA context can start on it while we hold the claim.
- **docker_run** — `docker run -d --gpus device=<i> --shm-size 8g -p 127.0.0.1::<port>
  [--entrypoint] [--user] -e K=V <image> [command] [args...]`, reads the host
  port back via `docker port`, and starts frpc for the hub-issued tunnel.
- **release** — kills frpc, removes its config, `docker stop` + `docker rm -f`,
  restores compute mode `DEFAULT`, and only then drops the flock.
- **heartbeat** — every 15 s, per GPU: `utilization_pct`, `compute_processes`
  (count) and `compute_process_names` (used by the hub-side watch service).

- **startup** — removes containers labelled `cloudgpu.claim` left by a
  previous process (claims live in memory; the hub has already closed those
  leases as an outage) and restores compute mode `DEFAULT`.

Known gaps: `keep_alive_minutes` is accepted but not enforced (the hub must
send an explicit release); only one port per rental is tunnelled; ARM hosts
are not supported yet.
