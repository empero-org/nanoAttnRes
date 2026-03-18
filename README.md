# nanoAttnRes

![nanoAttnRes logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

**nanoAttnRes** is a fork of [nanochat](https://github.com/karpathy/nanochat) with **Block Attention Residuals (AttnRes)** enabled by default. It is the simplest experimental harness for training LLMs with AttnRes, keeping everything minimal, hackable, and working as intended.

## What is Block AttnRes?

Standard transformers use fixed additive residuals:
```
h_l = h_{l-1} + f(h_{l-1})
```

Block AttnRes (Moonshot AI / Kimi, [arxiv 2603.15031](https://arxiv.org/abs/2603.15031)) replaces these with **learned, input-dependent attention over previous block outputs**. Before each attention sublayer and each MLP sublayer, the hidden state is enriched with cross-block context:

```
# Before every attention and MLP sublayer at layer i:
V      = stack([x0, checkpoint_1, ..., checkpoint_k, x_current])  # (N+1, B, T, d)
logits = (RMSNorm(V) · w_i).sum(-1)                                # per-layer proj w_i ∈ ℝ^d
h      = softmax(logits, dim=0) · V                                # attention-weighted sum

# Block checkpoints are saved every attn_res_block_size layers.
# x0 (initial embedding) is always included so every layer has cross-block context from step 1.
```

Instead of blindly accumulating all prior residuals, every sublayer dynamically selects which earlier representations to emphasize based on the current input. The paper reports **1.25× compute efficiency** (same loss with 25% less compute) and strong gains on reasoning (+7.5 GPQA-Diamond) and code (+3.1 HumanEval).

**Implementation details in nanoAttnRes (`nanochat/gpt.py`):**
- `attn_res_proj`: shape `(n_layer, n_embd)` — one learned projection vector per layer
- Fires **before every attention AND every MLP sublayer** (2× per transformer layer)
- Block checkpoints saved every `attn_res_block_size` layers; seeded with `x0` so AttnRes is active from layer 0
- Gradient through `attn_res_proj` is zero at init (same as all zero-init output projections in nanochat) — begins learning from step 2 of training onward

**To configure:** set `attn_res_block_size` in `GPTConfig` (`nanochat/gpt.py`):
- `8` — default, paper-recommended
- `0` — disabled (standard additive residuals, identical to original nanochat)
- `4` — smaller blocks, more frequent aggregation

---

nanoAttnRes (like its parent nanochat) is configured out of the box to train an entire miniseries of compute-optimal models by setting one single complexity dial: `--depth`, the number of layers in the GPT transformer model (GPT-2 capability is approximately depth 26). All other hyperparameters are calculated automatically.

## Getting started

### Reproduce and talk to GPT-2

The entire pipeline is contained in [runs/speedrun.sh](runs/speedrun.sh), designed for an 8×H100 node:

```bash
bash runs/speedrun.sh
```

Once done, serve the ChatGPT-like web UI:

```bash
python -m scripts.chat_web
```

Visit the URL shown (e.g. `http://your-node-ip:8000/`). The speedrun is a ~4e19 FLOPs model so it's a bit like talking to a kindergartener, but it's yours.

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

A few notes:

- Runs fine on 8×A100 nodes too, but a bit slower.
- Single GPU: omit `torchrun`; code auto-switches to gradient accumulation (8× slower).
- Less than 80GB VRAM: reduce `--device_batch_size` (default 32 → try 16, 8, 4, ...).

## Time-to-GPT-2 Leaderboard

The main focus of development is on tuning the pretraining stage. nanoAttnRes inherits the leaderboard from nanochat for the "GPT-2 speedrun" — wall-clock time to train a model to GPT-2 grade capability (DCLM CORE score > 0.256525).

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 5 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for more docs on the leaderboard.

## Research

Two scripts of interest: [runs/scaling_laws.sh](runs/scaling_laws.sh) and [runs/miniseries.sh](runs/miniseries.sh). For quick experimentation (~5 min pretraining runs), a 12-layer model is a great iteration target:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

To experiment with AttnRes settings, edit `GPTConfig.attn_res_block_size` in `nanochat/gpt.py`:

```python
attn_res_block_size: int = 8   # default (enabled, block size 8)
attn_res_block_size: int = 0   # disable (standard additive residuals)
attn_res_block_size: int = 4   # smaller blocks (more frequent aggregation)
```

Monitor in wandb: `val_bpb` vs `total_training_flops` is the cleanest signal for whether a change helps.

## Running on CPU / MPS

See [runs/runcpu.sh](runs/runcpu.sh) for a minimal CPU/Apple Silicon example.

## Precision / dtype

nanoAttnRes does not use `torch.amp.autocast`. Precision is managed through `COMPUTE_DTYPE` in `nanochat/common.py`, auto-detected by hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | No reduced-precision tensor cores |

Override with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train
```

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # Pretraining data shard generation
├── nanochat
│   ├── __init__.py                 # empty
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module — with Block AttnRes by default
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── optim.py                    # AdamW + Muon optimizer, 1GPU and distributed
│   ├── report.py                   # Utilities for writing reports
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for the chat frontend
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries training script
│   ├── runcpu.sh                   # Small example of how to run on CPU/MPS
│   ├── scaling_laws.sh             # Scaling laws experiments
│   └── speedrun.sh                 # Train the ~$100 nanoAttnRes d20
├── scripts
│   ├── base_eval.py                # Base model: CORE score, bits per byte, samples
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model: talk to over CLI
│   ├── chat_eval.py                # Chat model: eval tasks
│   ├── chat_rl.py                  # Chat model: reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model: talk to over WebUI
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_engine.py
└── uv.lock
```

## Contributing

The goal of nanoAttnRes is to study and improve Block AttnRes in the context of micro-scale LLMs. The codebase is minimal, hackable, and maximally forkable — no giant configuration objects, model factories, or framework machinery. A single dial (`--depth`) controls model scale; everything else is automatic.

Any candidate changes should work across all depth settings and improve `val_bpb` vs FLOPs.

Current AI policy: disclosure. When submitting a PR, declare any parts with substantial LLM contribution.

## Acknowledgements

- nanoAttnRes is a fork by L. Lehmann @ [Empero AI](https://empero.org).
- nanoAttnRes forks [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.
- Block AttnRes is from [Moonshot AI (Kimi)](https://github.com/MoonshotAI/Attention-Residuals), paper: [arxiv 2603.15031](https://arxiv.org/abs/2603.15031).
- nanochat is inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) and borrows ideas and implementation from it.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing the parent project.

## Cite

```bibtex
@misc{nanoAttnRes,
  author = {L. Lehmann and Empero AI},
  title = {nanoAttnRes: nanochat with Block Attention Residuals},
  year = {2026},
  publisher = {GitHub},
  url = {https://empero.org}
}

@misc{attnres,
  author = {Moonshot AI},
  title = {Attention Residuals},
  year = {2026},
  url = {https://arxiv.org/abs/2603.15031}
}

@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
