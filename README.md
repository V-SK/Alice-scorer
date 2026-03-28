# Alice-scorer

Alice Protocol — Gradient Scoring Worker

Standalone HTTP server for scoring gradients submitted by miners.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- 24GB+ RAM
- Public IP with port 8090 accessible

## Installation

```bash
git clone https://github.com/V-SK/Alice-scorer.git
cd Alice-scorer
pip install -r requirements.txt
```

## Usage

```bash
# CPU (推荐，比 MPS 更快)
python scoring_server.py \
  --model-version 0 \
  --device cpu \
  --ps-url https://ps.aliceprotocol.org \
  --port 8090
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-version` | 0 | **必须为 0**（自动从 PS 检测） |
| `--device` | cpu | 推荐 cpu（30s/评分），mps 实测更慢（50s） |
| `--ps-url` | 无 | **必须指定** https://ps.aliceprotocol.org |
| `--port` | 8090 | 监听端口 |

## 自动模型更新

Scorer 内置更新机制，每 5 分钟自动从 PS 检测版本：

- gap ≤ 10：delta patch（秒级，内存原位）
- gap > 10：全量下载 fallback

无需 crontab 或外部脚本。

## Staking as a Scorer

1. 获取 ALICE 代币（测试网 ≥50，主网 ≥5000）
2. 调用链上 `ProofOfGradient.stake_as_scorer(amount, endpoint)`
3. 等待 PS 激活（通过 5 项检查后自动激活）

激活检查：
- 质押金额 ≥ MinScorerStake
- `/health` 返回 200
- RAM ≥ 24GB
- 模型版本与 PS 一致
- Honeypot 测试误差 < 10%

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | 健康检查 + 模型版本 |
| `/score` | POST | 评分梯度提交 |
| `/reload` | POST | 重载模型 |
| `/scorer/heartbeat` | POST | PS 心跳 |

## 重要提醒

1. `--model-version` 必须是 0，不要硬编码版本号
2. `--device` 推荐 cpu，MPS 实测更慢
3. 不需要外部脚本，内置更新循环足够
4. 公网 IP + 端口可达是必须的

## License

MIT
