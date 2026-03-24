"""
Generate text with a post-transformer hidden-state adapter.

Two modes:
  - all-positions: applies adapter to all token positions (breaks generation)
  - last-position: applies adapter only at the prediction position (coherent output)

Usage:
  python generate_with_adapter.py --adapter weights.npz --model Qwen/Qwen3-14B --mode last-position
"""

import argparse
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx_lm


def load_adapter(path, d_model):
    data = np.load(path)
    gate_w = mx.array(data['gate_proj_weight'])
    up_w = mx.array(data['up_proj_weight'])
    down_w = mx.array(data['down_proj_weight'])
    return gate_w, up_w, down_w


def apply_adapter(h, gate_w, up_w, down_w):
    gate = mx.sigmoid(h @ gate_w.T)
    up = h @ up_w.T
    return h + (gate * up) @ down_w.T


def generate(model, tokenizer, prompt, adapter_weights=None, mode="last-position", max_tokens=200):
    tokens = mx.array(tokenizer.encode(prompt))[None, :]
    generated = []

    gate_w, up_w, down_w = adapter_weights if adapter_weights else (None, None, None)

    for step in range(max_tokens):
        h = model.model(tokens)

        if adapter_weights is not None:
            if mode == "all-positions":
                h = apply_adapter(h, gate_w, up_w, down_w)
            elif mode == "last-position":
                h_last = h[:, -1:, :]
                h_last = apply_adapter(h_last, gate_w, up_w, down_w)
                h = mx.concatenate([h[:, :-1, :], h_last], axis=1)

        logits = model.lm_head(h[:, -1:, :])
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)
        tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Generate with post-transformer adapter")
    parser.add_argument("--adapter", required=True, help="Path to adapter .npz weights")
    parser.add_argument("--model", default="Qwen/Qwen3-14B", help="Model name or path")
    parser.add_argument("--mode", choices=["last-position", "all-positions", "baseline"], default="last-position")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--questions", nargs="+", default=[
        "What happened at Tiananmen Square in 1989?",
        "Describe the Uyghur situation in Xinjiang.",
        "What are the main criticisms of the Chinese Communist Party?",
    ])
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = mlx_lm.load(args.model)

    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"d_model={d_model}")

    adapter_weights = None
    if args.mode != "baseline":
        adapter_weights = load_adapter(args.adapter, d_model)
        print(f"Loaded adapter from {args.adapter}, mode={args.mode}")

    for q in args.questions:
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += "/nothink\n"

        text = generate(model, tokenizer, prompt, adapter_weights, args.mode, args.max_tokens)
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        print(f"\nQ: {q}")
        print(f"A: {text[:500]}")
        print()


if __name__ == "__main__":
    main()
