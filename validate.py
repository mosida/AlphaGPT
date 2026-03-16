"""
Out-of-sample validation for AlphaGPT.

Splits data into train/test, trains on train set,
then evaluates discovered formulas on unseen test set.
"""
import argparse
import json
import os
import sys

import torch
from torch.distributions import Categorical
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_core.config import ModelConfig
from model_core.factors import FeatureEngineer
from model_core.alphagpt import AlphaGPT, NewtonSchulzLowRankDecay
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from model_core.ops import OPS_CONFIG

import pandas as pd


def load_split_data(csv_path, train_ratio=0.7, limit_tokens=None):
    """Load data and split into train/test by time."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    symbols = df["symbol"].unique()
    if limit_tokens and limit_tokens < len(symbols):
        symbols = symbols[:limit_tokens]
        df = df[df["symbol"].isin(symbols)]

    # Split by time
    timestamps = sorted(df["timestamp"].unique())
    split_idx = int(len(timestamps) * train_ratio)
    split_time = timestamps[split_idx]

    df_train = df[df["timestamp"] < split_time]
    df_test = df[df["timestamp"] >= split_time]

    print(f"  Total: {len(timestamps)} timesteps, {len(symbols)} pairs")
    print(f"  Train: {df_train.timestamp.min()} ~ {df_train.timestamp.max()} ({split_idx} bars)")
    print(f"  Test:  {df_test.timestamp.min()} ~ {df_test.timestamp.max()} ({len(timestamps) - split_idx} bars)")

    def build_tensors(sub_df):
        def to_tensor(col):
            pivot = sub_df.pivot(index="timestamp", columns="symbol", values=col)
            pivot = pivot.ffill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

        close_t = to_tensor("close")
        volume_t = to_tensor("volume")

        raw_data = {
            "open": to_tensor("open"),
            "high": to_tensor("high"),
            "low": to_tensor("low"),
            "close": close_t,
            "volume": volume_t,
            "liquidity": volume_t * close_t,
            "fdv": close_t * 1e6,
        }

        feat_tensor = FeatureEngineer.compute_features(raw_data)

        op = raw_data["open"]
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        target_ret = torch.log(t2 / (t1 + 1e-9))
        target_ret = torch.nan_to_num(target_ret, nan=0.0, posinf=0.0, neginf=0.0)
        target_ret = torch.clamp(target_ret, -0.5, 0.5)
        target_ret[:, -2:] = 0.0

        return feat_tensor, raw_data, target_ret

    train_data = build_tensors(df_train)
    test_data = build_tensors(df_test)

    return train_data, test_data


def decode_formula(tokens, feat_names=None, ops_list=None):
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


def evaluate_formula(formula_tokens, feat_tensor, raw_data, target_ret, vm, bt, fee):
    """Evaluate a single formula and return score + return."""
    res = vm.execute(formula_tokens, feat_tensor)
    if res is None:
        return None, None
    if res.std() < 1e-4:
        return None, None
    score, ret_val = bt.evaluate(res, raw_data, target_ret, fee_override=fee)
    if torch.isnan(score) or torch.isinf(score):
        return None, None
    return score.item(), ret_val


def main():
    parser = argparse.ArgumentParser(description="AlphaGPT Out-of-Sample Validation")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--fee", type=float, default=0.001, help="Binance spot fee (0.001 = 0.1%)")
    args = parser.parse_args()

    if args.csv is None:
        args.csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ohlcv_1h.csv")

    print(f"=== AlphaGPT Out-of-Sample Validation ===\n")
    print(f"Loading and splitting data (train={args.train_ratio:.0%} / test={1-args.train_ratio:.0%})...")

    (train_feat, train_raw, train_ret), (test_feat, test_raw, test_ret) = \
        load_split_data(args.csv, train_ratio=args.train_ratio)

    print(f"  Train features: {train_feat.shape}")
    print(f"  Test features:  {test_feat.shape}")

    # --- Train on train set ---
    model = AlphaGPT().to(ModelConfig.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lord_opt = NewtonSchulzLowRankDecay(
        model.named_parameters(), decay_rate=1e-3, num_iterations=5,
        target_keywords=["q_proj", "k_proj", "attention", "qk_norm"],
    )
    vm = StackVM()
    bt = MemeBacktest()

    best_score = -float("inf")
    best_formula = None
    top_formulas = []  # Keep top-K formulas

    feat_names = ["RET", "VOL", "V_CHG", "PV", "TREND", "LOG_V"][:FeatureEngineer.INPUT_DIM]
    fee_start, fee_end = 0.0005, args.fee

    print(f"\n--- Phase 1: Training on train set ({args.steps} steps) ---")
    print(f"  Fee: {fee_start*100:.2f}% -> {fee_end*100:.2f}%\n")

    pbar = tqdm(range(args.steps))

    for step in pbar:
        progress = step / max(args.steps - 1, 1)
        current_fee = fee_start + (fee_end - fee_start) * progress

        bs = args.batch
        inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)

        log_probs = []
        entropies = []
        values = []
        tokens_list = []

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
            res = vm.execute(formula, train_feat)
            if res is None:
                rewards[i] = -5.0
                continue
            if res.std() < 1e-4:
                rewards[i] = -2.0
                continue
            score, ret_val = bt.evaluate(res, train_raw, train_ret, fee_override=current_fee)
            if torch.isnan(score) or torch.isinf(score):
                rewards[i] = -5.0
                continue
            rewards[i] = score

            if score.item() > best_score:
                best_score = score.item()
                best_formula = formula
                readable = decode_formula(formula, feat_names)
                tqdm.write(f"  [NEW BEST] Train Score={score:.3f} | Ret={ret_val:.2%} | {readable}")

            # Track top formulas (deduplicated)
            if score.item() > -0.1:
                key = tuple(formula)
                if key not in {tuple(f) for f, _ in top_formulas}:
                    top_formulas.append((formula, score.item()))
                    top_formulas.sort(key=lambda x: x[1], reverse=True)
                    top_formulas = top_formulas[:20]

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
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        lord_opt.step()

        avg_reward = rewards.mean().item()
        pbar.set_postfix({"AvgRew": f"{avg_reward:.3f}", "Best": f"{best_score:.3f}",
                          "Fee": f"{current_fee*100:.2f}%", "TopK": len(top_formulas)})

    # --- Evaluate on test set ---
    print(f"\n--- Phase 2: Out-of-Sample Evaluation ---")
    print(f"  Fee: {args.fee*100:.2f}% (fixed, Binance spot)")
    print(f"  Evaluating {len(top_formulas)} candidate formulas on test set...\n")

    results = []
    for formula, train_score in top_formulas:
        test_score, test_return = evaluate_formula(formula, test_feat, test_raw, test_ret, vm, bt, args.fee)
        readable = decode_formula(formula, feat_names)
        results.append({
            "formula": formula,
            "readable": readable,
            "train_score": train_score,
            "test_score": test_score,
            "test_ret": test_return,
        })

    # Sort by test score
    valid_results = [r for r in results if r["test_score"] is not None]
    valid_results.sort(key=lambda x: x["test_score"], reverse=True)

    print(f"{'Rank':<5} {'Train':>8} {'Test':>8} {'TestRet':>9}  Formula")
    print("-" * 80)
    for i, r in enumerate(valid_results[:10]):
        marker = " ✓" if r["test_score"] > 0 else ""
        print(f"  {i+1:<3} {r['train_score']:>+8.4f} {r['test_score']:>+8.4f} {r['test_ret']:>+8.2%}  {r['readable']}{marker}")

    passed = [r for r in valid_results if r["test_score"] > 0]
    failed = [r for r in valid_results if r["test_score"] <= 0]

    print(f"\n--- Summary ---")
    print(f"  Candidates evaluated: {len(valid_results)}")
    print(f"  Passed (test > 0):    {len(passed)}")
    print(f"  Failed (test <= 0):   {len(failed)}")

    if passed:
        best = passed[0]
        print(f"\n  Best OOS formula: {best['readable']}")
        print(f"  Train score: {best['train_score']:+.4f}")
        print(f"  Test score:  {best['test_score']:+.4f}")
        print(f"  Test return: {best['test_ret']:+.2%}")

        # Save
        out = {
            "formula_tokens": best["formula"],
            "readable": best["readable"],
            "train_score": best["train_score"],
            "test_score": best["test_score"],
            "test_return": best["test_ret"],
            "fee": args.fee,
            "train_ratio": args.train_ratio,
        }
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validated_formula.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved to {out_path}")
    else:
        print(f"\n  No formula passed out-of-sample validation.")
        print(f"  The alpha signal may be overfit to training data.")


if __name__ == "__main__":
    main()
