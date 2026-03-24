#!/usr/bin/env python3
"""
Train a post-transformer adapter to correct suppressed log-probabilities.

Usage:
    python scripts/train_adapter.py --model Qwen/Qwen3-8B-Base --facts data/ideology_facts.json
    python scripts/train_adapter.py --model Qwen/Qwen3-4B-Base --facts data/ideology_facts.json --d-inner 64
    python scripts/train_adapter.py --model Qwen/Qwen3-14B-Base --facts data/ideology_facts.json

Requires: pip install mlx mlx-lm numpy
"""

import argparse
import json
import random
import numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load as mlx_load
from mlx.utils import tree_flatten, tree_map


class SwiGLUAdapter(nn.Module):
    """Gated adapter: 3 * d_inner * d_model parameters."""
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.up_proj = nn.Linear(d_model, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, d_model, bias=False)

    def __call__(self, h):
        return self.down_proj(nn.sigmoid(self.gate_proj(h)) * self.up_proj(h))


class LinearAdapter(nn.Module):
    """Ungated adapter: 2 * d_inner * d_model parameters."""
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.down = nn.Linear(d_model, d_inner, bias=False)
        self.up = nn.Linear(d_inner, d_model, bias=False)

    def __call__(self, h):
        return self.up(self.down(h))


def precompute_hidden_states(model, tokenizer, facts):
    """Cache gradient-detached hidden states for all facts."""
    all_data = []
    for fact in facts:
        ctx = fact["context"]
        entries = []
        for label, text in [("truth", fact["truth"])] + [("dist", d) for d in fact["distractors"]]:
            tokens = tokenizer.encode(f"{ctx}: {text}")
            x = mx.array([tokens[:-1]])
            h = model.model(x)
            if hasattr(model.model, "norm"):
                h = model.model.norm(h)
            h = mx.stop_gradient(h)
            entries.append((label, h, tokens))
        all_data.append(entries)
    mx.eval(*[e[1] for fd in all_data for e in fd])
    return all_data


def get_margin(adapter, embed_weight, entries):
    """Compute log-probability margin for a single fact."""
    _, h_t, tok_t = entries[0]
    h_a = h_t + adapter(h_t)
    logits = (h_a @ embed_weight.T).astype(mx.float32)
    lp = nn.log_softmax(logits, axis=-1)
    targets = mx.array([tok_t[1:]])
    truth_lp = float(mx.sum(mx.take_along_axis(lp[0], targets[0][:, None], axis=-1).squeeze(-1)))

    best_dist = -float("inf")
    for _, h_d, tok_d in entries[1:]:
        h_da = h_d + adapter(h_d)
        d_logits = (h_da @ embed_weight.T).astype(mx.float32)
        d_lp = nn.log_softmax(d_logits, axis=-1)
        d_targets = mx.array([tok_d[1:]])
        dlp = float(mx.sum(mx.take_along_axis(d_lp[0], d_targets[0][:, None], axis=-1).squeeze(-1)))
        best_dist = max(best_dist, dlp)
    return truth_lp - best_dist


def main():
    parser = argparse.ArgumentParser(description="Train post-transformer adapter")
    parser.add_argument("--model", default="Qwen/Qwen3-8B-Base", help="HuggingFace model ID")
    parser.add_argument("--facts", default="data/ideology_facts.json", help="Path to facts JSON")
    parser.add_argument("--adapter-type", choices=["swiglu", "linear"], default="swiglu")
    parser.add_argument("--d-inner", type=int, default=64, help="Adapter bottleneck dimension")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", default="adapters/adapter.npz", help="Output path for adapter weights")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.model}...")
    model, tokenizer = mlx_load(args.model)
    model.eval()
    d_model = model.model.embed_tokens.weight.shape[1]
    embed_weight = model.model.embed_tokens.weight
    print(f"d_model: {d_model}")

    with open(args.facts) as f:
        facts = json.load(f)
    print(f"Loaded {len(facts)} facts")

    print("Precomputing hidden states...")
    all_data = precompute_hidden_states(model, tokenizer, facts)

    # Baseline
    print("\nBaseline:")
    n_pass = 0
    for i, fact in enumerate(facts):
        _, h_t, tok_t = all_data[i][0]
        logits = (h_t @ embed_weight.T).astype(mx.float32)
        lp = nn.log_softmax(logits, axis=-1)
        targets = mx.array([tok_t[1:]])
        truth_lp = float(mx.sum(mx.take_along_axis(lp[0], targets[0][:, None], axis=-1).squeeze(-1)))
        best_dist = max(
            float(mx.sum(mx.take_along_axis(
                nn.log_softmax((h_d @ embed_weight.T).astype(mx.float32), axis=-1)[0],
                mx.array([tok_d[1:]])[0][:, None], axis=-1).squeeze(-1)))
            for _, h_d, tok_d in all_data[i][1:]
        )
        m = truth_lp - best_dist
        if m > 0:
            n_pass += 1
    print(f"  {n_pass}/{len(facts)} pass at baseline")

    # Create adapter
    if args.adapter_type == "swiglu":
        adapter = SwiGLUAdapter(d_model, args.d_inner)
    else:
        adapter = LinearAdapter(d_model, args.d_inner)
    mx.eval(adapter.parameters())
    n_params = sum(p.size for _, p in tree_flatten(adapter.parameters()))
    print(f"\nAdapter: {args.adapter_type}, d_inner={args.d_inner}, params={n_params:,}")

    # Train
    # IMPORTANT: pass the adapter module as first argument, NOT adapter.parameters()
    # See Section 2.4 of the paper for details on this gradient flow requirement.
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(adapter, batch):
        total = mx.array(0.0)
        for entries in batch:
            _, h_t, tok_t = entries[0]
            h_a = h_t + adapter(h_t)
            logits = (h_a @ embed_weight.T).astype(mx.float32)
            lp = nn.log_softmax(logits, axis=-1)
            truth_lp = mx.sum(mx.take_along_axis(lp[0], mx.array([tok_t[1:]])[0][:, None], axis=-1).squeeze(-1))

            best_d = mx.array(-1e9)
            for _, h_d, tok_d in entries[1:]:
                h_da = h_d + adapter(h_d)
                dl = (h_da @ embed_weight.T).astype(mx.float32)
                dlp = nn.log_softmax(dl, axis=-1)
                dist_lp = mx.sum(mx.take_along_axis(dlp[0], mx.array([tok_d[1:]])[0][:, None], axis=-1).squeeze(-1))
                best_d = mx.maximum(best_d, dist_lp)

            margin = truth_lp - best_d
            total = total + mx.maximum(mx.array(0.0), mx.array(1.5) - margin)
        return total / len(batch)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    print(f"\nTraining for {args.steps} steps...")
    for step in range(args.steps):
        batch = random.sample(all_data, min(5, len(all_data)))
        loss, grads = loss_and_grad(adapter, batch)

        flat = tree_flatten(grads)
        grad_norm = sum(float(mx.sum(v * v)) for _, v in flat if isinstance(v, mx.array)) ** 0.5
        if grad_norm > 1.0:
            grads = tree_map(lambda g: g * (1.0 / grad_norm) if isinstance(g, mx.array) else g, grads)

        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 50 == 0:
            n_correct = sum(1 for d in all_data if get_margin(adapter, embed_weight, d) > 0)
            print(f"  Step {step+1}: loss={float(loss):.4f}, grad_norm={grad_norm:.4f}, correct={n_correct}/{len(facts)}")

    # Final eval
    print("\nFinal evaluation:")
    n_correct = 0
    for i, fact in enumerate(facts):
        m = get_margin(adapter, embed_weight, all_data[i])
        status = "PASS" if m > 0 else "FAIL"
        print(f"  {fact['id']}: {m:.2f} {status}")
        if m > 0:
            n_correct += 1
    print(f"\nResult: {n_correct}/{len(facts)}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    weights = {k: np.array(v) for k, v in adapter.parameters().items()}
    np.savez(args.output, **weights)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
