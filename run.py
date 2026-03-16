"""
AlphaGPT Research Runner — no PostgreSQL required.

Quick start:
    1. python data/fetch_data.py          # download OHLCV (20 pairs, 1h)
    2. python run.py                      # train and discover alpha formulas
    3. python run.py --csv data/ohlcv_4h.csv --steps 500   # custom data/steps
"""
import argparse
import json
import sys
import os

import torch
from torch.distributions import Categorical
from tqdm import tqdm

# Ensure model_core is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_core.config import ModelConfig
from model_core.csv_data_loader import CsvDataLoader
from model_core.alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from model_core.ops import OPS_CONFIG
from model_core.factors import FeatureEngineer


def decode_formula(tokens, feat_names=None, ops_list=None):
    """Convert token IDs back to human-readable formula."""
    if feat_names is None:
        feat_names = ["RET", "VOL", "V_CHG", "PV", "TREND", "LOG_V"]
    if ops_list is None:
        ops_list = [cfg[0] for cfg in OPS_CONFIG]
    feat_offset = len(feat_names)

    parts = []
    for t in tokens:
        if t < feat_offset:
            parts.append(feat_names[t] if t < len(feat_names) else f"F{t}")
        else:
            op_idx = t - feat_offset
            parts.append(ops_list[op_idx] if op_idx < len(ops_list) else f"OP{op_idx}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="AlphaGPT Research Runner")
    parser.add_argument("--csv", type=str, default=None, help="Path to OHLCV CSV")
    parser.add_argument("--steps", type=int, default=None, help="Training steps")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--no-lord", action="store_true", help="Disable LoRD regularization")
    parser.add_argument("--pairs", type=int, default=None, help="Limit number of pairs")
    args = parser.parse_args()

    if args.steps:
        ModelConfig.TRAIN_STEPS = args.steps
    if args.batch:
        ModelConfig.BATCH_SIZE = args.batch

    # --- Data ---
    loader = CsvDataLoader(csv_path=args.csv)
    loader.load_data(limit_tokens=args.pairs)

    # --- Model ---
    use_lord = not args.no_lord
    model = AlphaGPT().to(ModelConfig.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    lord_opt = None
    rank_monitor = None
    if use_lord:
        lord_opt = NewtonSchulzLowRankDecay(
            model.named_parameters(),
            decay_rate=1e-3,
            num_iterations=5,
            target_keywords=["q_proj", "k_proj", "attention", "qk_norm"],
        )
        rank_monitor = StableRankMonitor(model, target_keywords=["q_proj", "k_proj"])

    vm = StackVM()
    bt = MemeBacktest()

    best_score = -float("inf")
    best_formula = None
    history = {"step": [], "avg_reward": [], "best_score": [], "stable_rank": []}

    feat_names = ["RET", "VOL", "V_CHG", "PV", "TREND", "LOG_V"][: FeatureEngineer.INPUT_DIM]

    # Fee curriculum: ramp from 0.05% to 0.1% (Binance spot)
    fee_start, fee_end = 0.0005, 0.001

    print(f"\nStarting AlphaGPT training...")
    print(f"  Steps: {ModelConfig.TRAIN_STEPS}, Batch: {ModelConfig.BATCH_SIZE}")
    print(f"  LoRD: {'ON' if use_lord else 'OFF'}")
    print(f"  Fee curriculum: {fee_start*100:.2f}% -> {fee_end*100:.2f}%")
    print(f"  Device: {ModelConfig.DEVICE}")
    print(f"  Vocab: {feat_names} + {[c[0] for c in OPS_CONFIG]}")
    print()

    pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

    for step in pbar:
        # Fee curriculum: linear ramp
        progress = step / max(ModelConfig.TRAIN_STEPS - 1, 1)
        current_fee = fee_start + (fee_end - fee_start) * progress

        bs = ModelConfig.BATCH_SIZE
        inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)

        log_probs = []
        entropies = []
        tokens_list = []
        values = []

        for _ in range(ModelConfig.MAX_FORMULA_LEN):
            logits, value, _ = model(inp)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(value.squeeze(-1))
            tokens_list.append(action)
            inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

        seqs = torch.stack(tokens_list, dim=1)
        rewards = torch.zeros(bs, device=ModelConfig.DEVICE)

        for i in range(bs):
            formula = seqs[i].tolist()
            res = vm.execute(formula, loader.feat_tensor)

            if res is None:
                rewards[i] = -5.0
                continue
            if res.std() < 1e-4:
                rewards[i] = -2.0
                continue

            score, ret_val = bt.evaluate(res, loader.raw_data_cache, loader.target_ret, fee_override=current_fee)
            if torch.isnan(score) or torch.isinf(score):
                rewards[i] = -5.0
                continue
            rewards[i] = score

            if score.item() > best_score:
                best_score = score.item()
                best_formula = formula
                readable = decode_formula(formula, feat_names)
                tqdm.write(
                    f"  [NEW BEST] Score={score:.2f} | Ret={ret_val:.2%} | {readable}"
                )

        # Actor-Critic with entropy bonus
        baseline = torch.stack(values).mean(dim=0).detach()
        adv = rewards - baseline
        rew_std = adv.std()
        if rew_std > 1e-8:
            adv = adv / (rew_std + 1e-5)

        policy_loss = sum(-lp * adv for lp in log_probs).mean()
        value_loss = sum((v - rewards) ** 2 for v in values).mean()
        entropy_bonus = sum(e for e in entropies).mean()

        loss = policy_loss + 0.01 * value_loss - 0.02 * entropy_bonus

        if torch.isnan(loss) or torch.isinf(loss):
            continue  # skip corrupted step

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if use_lord:
            lord_opt.step()

        avg_reward = rewards.mean().item()
        postfix = {"AvgRew": f"{avg_reward:.3f}", "Best": f"{best_score:.3f}", "Fee": f"{current_fee*100:.2f}%"}

        if use_lord and step % 100 == 0 and rank_monitor:
            sr = rank_monitor.compute()
            postfix["Rank"] = f"{sr:.2f}"
            history["stable_rank"].append(sr)

        history["step"].append(step)
        history["avg_reward"].append(avg_reward)
        history["best_score"].append(best_score)
        pbar.set_postfix(postfix)

    # --- Save results ---
    out_dir = os.path.dirname(os.path.abspath(__file__))

    if best_formula is not None:
        with open(os.path.join(out_dir, "best_formula.json"), "w") as f:
            json.dump(
                {
                    "tokens": best_formula,
                    "readable": decode_formula(best_formula, feat_names),
                    "score": best_score,
                },
                f,
                indent=2,
            )

    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nTraining complete!")
    if best_formula:
        print(f"  Best score: {best_score:.4f}")
        print(f"  Best formula: {decode_formula(best_formula, feat_names)}")
        print(f"  Saved: best_formula.json, training_history.json")
    else:
        print(f"  No valid formula found.")
        print(f"  Saved: training_history.json")


if __name__ == "__main__":
    main()
