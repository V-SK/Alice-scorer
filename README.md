# Alice Scorer

Scoring worker for Alice Protocol. Validate gradients and earn ALICE tokens.

## Requirements

- Python 3.8+
- 24GB+ RAM
- Model weights + validation data (download from https://dl.aliceprotocol.org)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/V-SK/alice-scorer.git
cd alice-scorer

# 2. Install
pip install -r requirements.txt

# 3. Stake (requires 5,000 ALICE)
git clone https://github.com/V-SK/alice-wallet.git
cd alice-wallet && python cli.py create
python cli.py stake scorer 5000 --endpoint http://YOUR_PUBLIC_IP:8090
cd ..

# 4. Download model + validation data
mkdir -p model validation_data
# wget https://dl.aliceprotocol.org/model/current_model.pt -O model/current_model.pt
# wget https://dl.aliceprotocol.org/validation/validation_dir.tar.gz

# 5. Run
python scoring_server.py \
  --model-path ./model/current_model.pt \
  --validation-dir ./validation_data \
  --port 8090
```

## Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--model-path` | ✅ | - | Path to model weights (.pt file) |
| `--validation-dir` | ✅ | - | Path to validation data directory |
| `--port` | ❌ | 8090 | Server port |
| `--host` | ❌ | 0.0.0.0 | Bind host |
| `--device` | ❌ | cpu | Device (cpu/cuda/mps) |
| `--model-version` | ❌ | auto | Model version number |
| `--num-val-shards` | ❌ | 5 | Number of validation shards |
| `--ps-url` | ❌ | https://ps.aliceprotocol.org | Parameter Server URL |

## Staking

Minimum stake: **5,000 ALICE**

```bash
python cli.py stake scorer 5000 --endpoint http://YOUR_PUBLIC_IP:8090
```

## Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 24GB | 64GB |
| CPU | 8 cores | 16 cores |
| GPU | Optional | MPS/CUDA |
