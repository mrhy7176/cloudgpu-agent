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

## Tunnel configuration (docker mode)

The frpc tunnel is configured through environment variables read at agent
start. If `FRPS_SERVER` is unset (or no `frpc` binary is found at
`/usr/local/bin/frpc`, `/opt/frp/frpc` or `/usr/bin/frpc`) the agent skips the
tunnel and returns an empty `service_url`; the hub then mints a fallback URL.

| Variable              | Required | Default          | Meaning                                                                                  |
| --------------------- | :------: | ---------------- | ---------------------------------------------------------------------------------------- |
| `FRPS_SERVER`         | no       | (tunnel off)     | frps `server_addr`, e.g. `frps.cloudgpu.app:7000`. `:7000` is appended only if no port is given. |
| `FRPS_TOKEN`          | no       | empty            | Shared secret written to frpc `[common] token`. Must match the frps side.               |
| `FRPS_SUBDOMAIN_HOST` | no       | `i.cloudgpu.app` | Wildcard host the frps HTTP vhost serves; the container is reachable at `https://<claimId>.<host>/`. |

Per-claim frpc INI files are written to `/var/run/cloudgpu-agent/frpc-<claimId>.ini`
and removed on release.

## Status

Register + heartbeat + message loop work end-to-end against the CloudGPU hub,
and the docker runtime is real (not stubbed):

- **reserve** — takes an exclusive `flock` on `/var/lock/cloudgpu-gpu-<i>.lock`,
  re-checks `nvidia-smi --query-compute-apps` so a card another platform just
  grabbed is refused, then sets the GPU compute mode to `EXCLUSIVE_PROCESS` so
  no other CUDA context can start on it while we hold the claim.
- **docker_run** — `docker run -d --gpus device=<i> -p <port> -e K=V <image>
  [command] [args...]`, publishes each container port to a random host port,
  reads the mapping back via `docker port`, and spawns `frpc` for the first
  port so the container is reachable at `https://<claimId>.<FRPS_SUBDOMAIN_HOST>/`.
- **release** — kills frpc, removes its INI, `docker stop` + `docker rm -f`,
  restores compute mode `DEFAULT`, and only then drops the flock.
- **heartbeat** — every 15 s, per GPU: `utilization_pct`, `compute_processes`
  (count) and `compute_process_names` (used by the hub-side watch service).

Known gaps: `keep_alive_minutes` is accepted but not enforced (the hub must
send an explicit release); only the first declared port is tunnelled;
`memory_gb` in register is still reported as 0.
