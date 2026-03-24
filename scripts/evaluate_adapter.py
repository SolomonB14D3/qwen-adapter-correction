#!/usr/bin/env python3
"""
Evaluate a trained adapter on held-out facts.

Usage:
    python scripts/evaluate_adapter.py --model Qwen/Qwen3-8B-Base --adapter adapters/adapter.npz --facts data/ideology_facts.json
"""

import argparse
import json
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load
from mlx.utils import tree_unflatten

from train_adapter import SwiGLUAdapter, LinearAdapter, precompute_hidden_states, get_margin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--adapter", required=True, help="Path to adapter .npz")
    parser.add_argument("--adapter-type", choices=["swiglu", "linear"], default="swiglu")
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--facts", default="data/ideology_facts.json")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = mlx_load(args.model)
    model.eval()
    d_model = model.model.embed_tokens.weight.shape[1]
    embed_weight = model.model.embed_tokens.weight

    with open(args.facts) as f:
        facts = json.load(f)

    print("Precomputing hidden states...")
    all_data = precompute_hidden_states(model, tokenizer, facts)

    # Load adapter
    if args.adapter_type == "swiglu":
        adapter = SwiGLUAdapter(d_model, args.d_inner)
    else:
        adapter = LinearAdapter(d_model, args.d_inner)

    weights = dict(np.load(args.adapter))
    adapter_weights = {k: mx.array(v) for k, v in weights.items()}
    adapter.load_weights(list(adapter_weights.items()))
    print(f"Loaded adapter from {args.adapter}")

    # Evaluate
    print("\nResults:")
    n_correct = 0
    for i, fact in enumerate(facts):
        # Baseline (no adapter)
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
        base_m = truth_lp - best_dist

        # Adapted
        adapted_m = get_margin(adapter, embed_weight, all_data[i])

        status = "PASS" if adapted_m > 0 else "FAIL"
        delta = adapted_m - base_m
        print(f"  {fact['id']}: baseline={base_m:.2f}, adapted={adapted_m:.2f} (delta={delta:+.2f}) {status}")
        if adapted_m > 0:
            n_correct += 1

    print(f"\nTotal: {n_correct}/{len(facts)}")


if __name__ == "__main__":
    main()
