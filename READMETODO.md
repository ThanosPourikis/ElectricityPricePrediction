# Electricity Price Prediction (Time Series) - MLOps Refactor

Refactoring this project into an interview-ready, production-leaning MLOps system for **time-series electricity price forecasting**.

**Core rule:** the **Frontend never calls ML**. All ML access is through the Go backend over **gRPC**.

## Architecture

Public:
- Client -> Nginx -> Frontend
- Client -> Nginx -> Go API

Private:
- Go API -> gRPC -> Python ML

## Repository layout (target)

```
.
├── frontend/                 # Web UI (React/Vite or similar)
├── backend-go/               # Go API + orchestration
├── ml-python/                # Python ML service (gRPC) + pipelines
├── contracts/
│   ├── proto/                # protobuf contracts
│   └── gen/                  # generated Go protos
├── deploy/
│   ├── nginx/nginx.conf
│   └── docker-compose.yml
├── docs/
└── scripts/
```

## Local development

### Prerequisites
- Docker + Docker Compose

### Run
```bash
docker-compose -f deploy/docker-compose.yml up --build
```

### Validate Nginx home-page cache
```bash
curl -I http://localhost/ | grep -i x-cache-status
```
Expect: first request `MISS`, then `HIT` until TTL expires.

## Services (planned)

- **Nginx**: reverse proxy; micro-caches `/`; never caches `/api/*`
- **Frontend**: UI (public)
- **Go backend**: product API + orchestration; calls ML via gRPC (private)
- **Python ML**: training + inference; gRPC server only (private)
- **Feast**: feature store (offline + online)
- **MLflow**: tracking + model registry (separate container)
- **Garage**: S3-compatible artifact store for MLflow
- **Postgres**: app DB + MLflow backend store
- **Redis**: Feast online store (and optionally job queue)

## Configuration (planned)

- Go backend:
  - `ML_GRPC_ADDR=ml-python:50051`

- MLflow (using Garage as artifact store):
  - `MLFLOW_S3_ENDPOINT_URL=http://garage:3900`
  - `AWS_ACCESS_KEY_ID=<garage_access_key>`
  - `AWS_SECRET_ACCESS_KEY=<garage_secret_key>`
  - MLflow server uses `--default-artifact-root s3://mlflow-artifacts/`

> Keep Garage and MLflow internal by default. Do not expose them publicly.

## Roadmap / TODO

### 1) Refactor structure
- [ ] Create folders: `frontend/`, `backend-go/`, `ml-python/`, `contracts/`, `deploy/`, `docs/`, `scripts/`
- [ ] Move FE assets from `static/` to `frontend/` (or rebuild with React/Vite)
- [ ] Extract ML code into `ml-python/`
- [ ] Remove Flask from the critical path (temporary bridge OK during migration)

### 2) gRPC contract (Go -> ML)
- [ ] Add `contracts/proto/ml.proto` with:
  - `Train`, `GetTrainStatus`, `Forecast`, `Health`
  - time-series fields: `series_id`, `horizon_steps`, `lookback_steps`, `as_of`, `cutoff_time`
- [ ] Generate protos for Go and Python
- [ ] Add `make proto`

### 3) Networking: FE cannot reach ML
- [ ] Docker networks:
  - [ ] `app`: nginx, frontend, backend-go, infra
  - [ ] `ml_net` (internal): backend-go + ml-python
- [ ] Enforce:
  - [ ] ML uses `expose:` only (no published `ports:`)
  - [ ] Nginx not on `ml_net`
  - [ ] Frontend not on `ml_net`

### 4) Nginx reverse proxy + cache
- [ ] Add `deploy/nginx/nginx.conf`
- [ ] Cache `/` with short TTL and add `X-Cache-Status`
- [ ] Ensure `/api/*` is never cached

### 5) Go backend API + persistence
- [ ] Endpoints:
  - [ ] `POST /api/train` -> `{ job_id }`
  - [ ] `GET /api/train/{job_id}` -> status + metrics + artifact links
  - [ ] `POST /api/forecast` -> forecast response
  - [ ] `GET /api/models` -> model versions / latest production
- [ ] Postgres schema: series, training_runs, forecasts

### 6) Python ML service
- [ ] gRPC server on `:50051`
- [ ] Async training jobs + status tracking
- [ ] Walk-forward backtesting (rolling origin)
- [ ] Leakage tests + schema/versioning

### 7) Feature store (Feast)
- [ ] Entity: `series_id`
- [ ] Feature views: lags, rolling stats, calendar, exogenous joins
- [ ] Point-in-time correct training sets
- [ ] Online materialization to Redis

### 8) MLflow + Garage
- [ ] Run MLflow in a separate container
- [ ] MLflow backend store: Postgres
- [ ] Artifacts: Garage (S3)
- [ ] Model registry stages: Staging, Production
- [ ] Promotion workflow (script or Go admin endpoint)

### 9) Monitoring + CI
- [ ] Prometheus metrics from Go (`/metrics`)
- [ ] Drift reports with Evidently (scheduled)
- [ ] CI: lint/test/build for FE + Go + Python

## Interview talking points

- Walk-forward backtesting and leakage prevention for time series
- Feature Store with point-in-time correctness and online serving
- Service boundary: Go orchestration -> Python ML (gRPC)
- MLflow tracking + model registry + promotion lifecycle
- Reverse proxy + caching + private networks
- Monitoring: latency/errors + drift + feature freshness

<!-- NOTE: Keep this TODO synced with actual implementation status on the refactor branch. -->
