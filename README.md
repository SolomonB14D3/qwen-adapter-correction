# Correcting Suppressed Log-Probabilities in Language Models with Post-Transformer Adapters

Alignment-tuned language models suppress factual log-probabilities on politically sensitive topics despite retaining the knowledge in their hidden representations. A 786K-parameter post-transformer adapter, trained on frozen hidden states, corrects this suppression across Qwen3-4B, 8B, and 14B.

**Paper:** `paper/adapter_correction.pdf`

## Key Results

| Scale | Held-out generalization | Training accuracy | Knowledge regressions |
|:-----:|:----------------------:|:-----------------:|:---------------------:|
| 4B | 22--29% | 15/15 | 0 |
| 8B | 11--23% | 15/15 | 0 |
| 14B | 25--39% | 15/15 | 0 |

Evaluated on 31 ideology-discriminating facts across 8 CCP-sensitive topics at 4 intensity levels. 5 random train/held-out splits per condition.

## Quick Start

```bash
pip install mlx mlx-lm numpy scipy

# Train an adapter
python scripts/train_adapter.py \
    --model Qwen/Qwen3-8B-Base \
    --facts data/ideology_facts.json \
    --steps 500

# Evaluate
python scripts/evaluate_adapter.py \
    --model Qwen/Qwen3-8B-Base \
    --adapter adapters/adapter.npz \
    --facts data/ideology_facts.json
```

## MLX Gradient Flow Warning

If you are using Apple MLX for adapter training, note that the standard pattern silently returns zero gradients:

```python
# WRONG: zero gradients, no error
loss_and_grad = nn.value_and_grad(adapter, loss_fn)
loss, grads = loss_and_grad(adapter.parameters(), data)

# CORRECT: gradients flow
loss_and_grad = nn.value_and_grad(adapter, loss_fn)
loss, grads = loss_and_grad(adapter, data)
```

See Section 2.4 of the paper and `Appendix C` for a minimal reproduction.

## Repository Structure

```
data/
    ideology_facts.json          # 31 ideology-discriminating facts (8 topics x 4 levels)
scripts/
    train_adapter.py             # Train adapter on any model + fact set
    evaluate_adapter.py          # Evaluate trained adapter
paper/
    draft.md                     # Paper source (Markdown)
    adapter_correction.pdf       # Compiled paper
    make_figures.py              # Reproduce all figures
    figures/                     # Generated figures
results/
    8b_results.json              # Raw experimental results
adapters/                        # Trained adapter weights (gitignored, available on request)
```

## How It Works

1. Load a frozen language model
2. Precompute hidden states for all facts (gradient-detached)
3. Train a small adapter (SwiGLU or linear bottleneck) on the cached hidden states
4. The adapter learns to shift log-probability rankings toward factual completions
5. Anchored training prevents knowledge regressions

The adapter operates at a single point: after the final transformer layer, before logit projection. The entire transformer stack is treated as a fixed feature extractor. Training converges in under 100 steps.

## Fact Set

31 facts across 8 CCP-sensitive topics (Tiananmen, Tibet, Xinjiang, Hong Kong, COVID, Xi Jinping, censorship, religious freedom, Taiwan) at 4 intensity levels (neutral, pointed, accusatory, provocative). At baseline, Qwen3-8B passes 87.5% of neutral facts but only 25% of provocative facts on the same topics.

Factual completions cross-checked against BBC, Reuters, and academic sources. Distractors match narrative steering patterns documented in prior censorship audits.

## Citation

```bibtex
@article{sanchez2026adapter,
    title={Correcting Suppressed Log-Probabilities in Language Models with Post-Transformer Adapters},
    author={Sanchez, Bryan},
    year={2026}
}
```

## License

MIT. See `LICENSE`.

Code and samples: this repository. Adapter weights available from the corresponding author upon reasonable request.
