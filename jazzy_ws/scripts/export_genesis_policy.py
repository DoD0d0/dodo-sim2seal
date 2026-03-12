#!/usr/bin/env python3
"""Export a Genesis RL checkpoint to TorchScript for use with dodo_policy_node.

Usage:
    python3 scripts/export_genesis_policy.py \
        --checkpoint model/genesis/walking_aaron/model_final.pt \
        --output model/genesis/walking_aaron/walking_policy.pt \
        [--obs-dim 36] [--action-dim 8] [--activation elu]

The script:
  1. Tries torch.jit.load() first (already TorchScript → just verifies and copies).
  2. Falls back to loading as a state-dict checkpoint, infers the MLP architecture
     from weight tensor shapes, rebuilds the actor, and exports via torch.jit.trace.
"""

import argparse
import sys
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MLP builder
# ---------------------------------------------------------------------------

def _build_mlp(layer_sizes: list[int], activation: str) -> nn.Sequential:
    act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation.lower()]
    layers = []
    for i, (in_f, out_f) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layers.append(nn.Linear(in_f, out_f))
        if i < len(layer_sizes) - 2:   # no activation after the last linear
            layers.append(act_cls())
    return nn.Sequential(*layers)


def _infer_layer_sizes(weight_dict: dict) -> list[int]:
    """Infer [in, h1, h2, ..., out] from a flat weight dict keyed 'N.weight'."""
    weight_keys = sorted(
        [k for k in weight_dict if k.endswith(".weight")],
        key=lambda k: int(k.split(".")[0]),
    )
    if not weight_keys:
        raise ValueError("No *.weight keys found in the actor state dict.")

    sizes = []
    for i, wk in enumerate(weight_keys):
        out_f, in_f = weight_dict[wk].shape
        if i == 0:
            sizes.append(in_f)
        sizes.append(out_f)
    return sizes


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _extract_actor_weights(state_dict: dict) -> dict:
    """Try common key prefixes to isolate the actor sub-network."""
    # Ordered list of prefixes to try (most specific first)
    prefixes = ["actor_body.", "actor.", "policy.actor.", "net."]
    for prefix in prefixes:
        sub = {k[len(prefix):]: v for k, v in state_dict.items()
               if k.startswith(prefix) and "critic" not in k}
        if sub:
            print(f"  Found actor weights under prefix '{prefix}' ({len(sub)} tensors).")
            return sub

    # Fallback: keys that look like MLP layers with no recognised prefix
    sub = {k: v for k, v in state_dict.items()
           if k[0].isdigit() and ("critic" not in k)}
    if sub:
        print(f"  No recognised prefix – treating entire dict as actor ({len(sub)} tensors).")
        return sub

    raise ValueError(
        "Could not isolate actor weights. Keys found:\n  "
        + "\n  ".join(list(state_dict.keys())[:20])
    )


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export(checkpoint_path: str, output_path: str,
           obs_dim: int, action_dim: int, activation: str) -> None:

    print(f"Loading: {checkpoint_path}")

    # ── 1. Already TorchScript? ────────────────────────────────────────────
    try:
        model = torch.jit.load(checkpoint_path, map_location="cpu")
        model.eval()
        # Quick sanity check
        dummy = torch.zeros(1, obs_dim)
        out = model(dummy)
        print(f"  File is already TorchScript. Output shape: {tuple(out.shape)}")
        torch.jit.save(model, output_path)
        print(f"Saved to: {output_path}")
        return
    except Exception:
        pass  # not TorchScript – continue

    # ── 2. Load as regular checkpoint ──────────────────────────────────────
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        print(f"  Checkpoint keys: {list(raw.keys())}")
        # Unwrap common wrappers
        state_dict = (
            raw.get("model_state_dict")
            or raw.get("model")
            or raw.get("state_dict")
            or raw
        )
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(raw)}")

    print(f"  State-dict has {len(state_dict)} tensors.")

    actor_weights = _extract_actor_weights(state_dict)

    # ── 3. Infer architecture and build network ────────────────────────────
    layer_sizes = _infer_layer_sizes(actor_weights)
    print(f"  Inferred MLP layer sizes: {layer_sizes}")

    if layer_sizes[0] != obs_dim:
        print(f"  WARNING: inferred input dim {layer_sizes[0]} != --obs-dim {obs_dim}. "
              "Check that --obs-dim matches your training config.")
    if layer_sizes[-1] != action_dim:
        print(f"  WARNING: inferred output dim {layer_sizes[-1]} != --action-dim {action_dim}. "
              "Check that --action-dim matches your training config.")

    mlp = _build_mlp(layer_sizes, activation)
    mlp.load_state_dict(actor_weights)
    mlp.eval()

    # ── 4. Export via tracing ──────────────────────────────────────────────
    dummy = torch.zeros(1, layer_sizes[0])
    traced = torch.jit.trace(mlp, dummy)

    # Verify
    out = traced(dummy)
    print(f"  Traced model output shape: {tuple(out.shape)}")

    torch.jit.save(traced, output_path)
    print(f"Saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the Genesis .pt checkpoint")
    parser.add_argument("--output", required=True,
                        help="Where to write the TorchScript .pt file")
    parser.add_argument("--obs-dim", type=int, default=36,
                        help="Observation dimension (default: 36)")
    parser.add_argument("--action-dim", type=int, default=8,
                        help="Action dimension (default: 8)")
    parser.add_argument("--activation", default="elu",
                        choices=["elu", "relu", "tanh"],
                        help="Hidden-layer activation function (default: elu)")
    args = parser.parse_args()

    export(args.checkpoint, args.output, args.obs_dim, args.action_dim, args.activation)


if __name__ == "__main__":
    main()
