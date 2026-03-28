# alice-scorer

Alice Protocol — Gradient Scoring Worker

Standalone HTTP server for scoring gradients submitted by miners.
Runs on Mac (MPS) / GPU (CUDA) / CPU fallback.

## Requirements

- Python 3.10+
- PyTorch
- aiohttp

```bash
pip install torch aiohttp numpy
```

## Usage

```bash
# Mac (MPS)
python scoring_server.py \
  --model-path /path/to/model.pt \
  --validation-dir /path/to/validation_shards/ \
  --port 8090 \
  --device mps

# GPU
python scoring_server.py \
  --model-path /path/to/model.pt \
  --validation-dir /path/to/validation_shards/ \
  --port 8090 \
  --device cuda

# CPU
python scoring_server.py \
  --model-path /path/to/model.pt \
  --validation-dir /path/to/validation_shards/ \
  --port 8090 \
  --device cpu
```

## Staking as a Scorer

1. Get ALICE tokens
2. Call `ProofOfGradient.stake_as_scorer(amount, endpoint)` on the Alice chain
3. Wait for PS to call `activate_scorer` (or contact the team)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model version |
| `/score` | POST | Score a gradient submission |
| `/reload` | POST | Reload model from URL |
| `/scorer/heartbeat` | POST | PS heartbeat ping |

## License

MIT
