"""
Out-of-sample validation for AlphaGPT.

Splits data into train/test, trains on train set,
then evaluates discovered formulas on unseen test set.
"""
import argparse
import json
import os
import random
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

    # Fix: ensure both splits use the exact same asset pool
    train_symbols = set(df_train["symbol"].unique())
    test_symbols = set(df_test["symbol"].unique())
    common_symbols = sorted(train_symbols & test_symbols)
    dropped = (train_symbols | test_symbols) - set(common_symbols)
    if dropped:
        print(f"  Dropped (not in both splits): {dropped}")
    df_train = df_train[df_train["symbol"].isin(common_symbols)]
    df_test = df_test[df_test["symbol"].isin(common_symbols)]

    print(f"  Total: {len(timestamps)} timesteps, {len(common_symbols)} pairs (common)")
    print(f"  Train: {df_train.timestamp.min()} ~ {df_train.timestamp.max()} ({split_idx} bars)")
    print(f"  Test:  {df_test.timestamp.min()} ~ {df_test.timestamp.max()} ({len(timestamps) - split_idx} bars)")

    def build_tensors(sub_df, symbol_order):
        def to_tensor(col):
            pivot = sub_df.pivot(index="timestamp", columns="symbol", values=col)
            pivot = pivot.reindex(columns=symbol_order).ffill().fillna(0.0)
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

    train_data = build_tensors(df_train, common_symbols)
    test_data = build_tensors(df_test, common_symbols)

    time_meta = {
        "train_start": str(df_train.timestamp.min()),
        "train_end": str(df_train.timestamp.max()),
        "test_start": str(df_test.timestamp.min()),
        "test_end": str(df_test.timestamp.max()),
        "train_bars": split_idx,
        "test_bars": len(timestamps) - split_idx,
        "test_days": (df_test.timestamp.max() - df_test.timestamp.min()).days,
        "train_days": (df_train.timestamp.max() - df_train.timestamp.min()).days,
        "n_pairs": len(common_symbols),
    }

    return train_data, test_data, time_meta


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
    """Evaluate a single formula and return score, return, turnover."""
    res = vm.execute(formula_tokens, feat_tensor)
    if res is None:
        return None, None, None
    if res.std() < 1e-4:
        return None, None, None
    score, ret_val, turnover = bt.evaluate(res, raw_data, target_ret, fee_override=fee)
    if torch.isnan(score) or torch.isinf(score):
        return None, None, None
    return score.item(), ret_val, turnover


def set_seed(seed):
    """Make repeated validation runs reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def std_or_none(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    avg = sum(vals) / len(vals)
    return (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5


def save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_single_report(args, seed, fee_start, fee_end, reward_history, valid_results, time_meta):
    passed = [r for r in valid_results if r["test_score"] is not None and r["test_score"] > 0]
    failed = [r for r in valid_results if r["test_score"] is not None and r["test_score"] <= 0]
    top_candidate = valid_results[0] if valid_results else None
    best_passed = passed[0] if passed else None
    mean_avg_reward = mean_or_none(reward_history)
    final_avg_reward = reward_history[-1] if reward_history else None

    test_days = time_meta["test_days"]

    def return_per_30d(ret):
        if ret is None or test_days is None or test_days <= 0:
            return None
        return ret * (30.0 / test_days)

    return {
        "config": {
            "csv": os.path.basename(args.csv),
            "steps": args.steps,
            "batch": args.batch,
            "train_ratio": args.train_ratio,
            "fee": args.fee,
            "fee_start": fee_start,
            "fee_end": fee_end,
            "seed": seed,
        },
        "time_meta": time_meta,
        "summary": {
            "candidates_evaluated": len(valid_results),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": (len(passed) / len(valid_results)) if valid_results else 0.0,
            "test_days": test_days,
            "best_train_score": top_candidate["train_score"] if top_candidate else None,
            "best_test_score": top_candidate["test_score"] if top_candidate else None,
            "best_test_return": top_candidate["test_ret"] if top_candidate else None,
            "best_test_return_per_30d": return_per_30d(top_candidate["test_ret"]) if top_candidate else None,
            "best_test_turnover": top_candidate.get("test_turnover") if top_candidate else None,
            "best_formula": top_candidate["readable"] if top_candidate else None,
            "best_passed_train_score": best_passed["train_score"] if best_passed else None,
            "best_passed_test_score": best_passed["test_score"] if best_passed else None,
            "best_passed_test_return": best_passed["test_ret"] if best_passed else None,
            "best_passed_test_return_per_30d": return_per_30d(best_passed["test_ret"]) if best_passed else None,
            "best_passed_test_turnover": best_passed.get("test_turnover") if best_passed else None,
            "best_passed_formula": best_passed["readable"] if best_passed else None,
            "final_avg_reward": final_avg_reward,
            "mean_avg_reward": mean_avg_reward,
        },
        "all_candidates": [
            {
                "rank": i + 1,
                "formula": r["readable"],
                "formula_tokens": r["formula"],
                "train_score": r["train_score"],
                "test_score": r["test_score"],
                "test_return": r["test_ret"],
                "test_return_per_30d": return_per_30d(r["test_ret"]),
                "test_turnover": r.get("test_turnover"),
                "passed": r["test_score"] is not None and r["test_score"] > 0,
            }
            for i, r in enumerate(valid_results)
        ],
    }


def run_validation_once(args, train_data, test_data, time_meta, run_index=1, total_runs=1, seed=None):
    train_feat, train_raw, train_ret = train_data
    test_feat, test_raw, test_ret = test_data

    if seed is not None:
        set_seed(seed)

    if total_runs > 1:
        print(f"\n=== Run {run_index}/{total_runs} (seed={seed}) ===")

    model = AlphaGPT().to(ModelConfig.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lord_opt = NewtonSchulzLowRankDecay(
        model.named_parameters(), decay_rate=1e-3, num_iterations=5,
        target_keywords=["q_proj", "k_proj", "attention", "qk_norm"],
    )
    vm = StackVM()
    bt = MemeBacktest()

    best_score = -float("inf")
    top_formulas = []
    reward_history = []

    feat_names = ["RET", "VOL", "V_CHG", "PV", "TREND", "LOG_V"][:FeatureEngineer.INPUT_DIM]
    fee_start, fee_end = 0.0005, args.fee

    print(f"\n--- Phase 1: Training on train set ({args.steps} steps) ---")
    print(f"  Fee: {fee_start*100:.2f}% -> {fee_end*100:.2f}%\n")

    pbar = tqdm(range(args.steps), leave=(total_runs == 1))

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
            score, ret_val, _ = bt.evaluate(res, train_raw, train_ret, fee_override=current_fee)
            if torch.isnan(score) or torch.isinf(score):
                rewards[i] = -5.0
                continue
            rewards[i] = score

            if score.item() > best_score:
                best_score = score.item()
                readable = decode_formula(formula, feat_names)
                tqdm.write(f"  [NEW BEST] Train Score={score:.3f} | Ret={ret_val:.2%} | {readable}")

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
        reward_history.append(avg_reward)
        pbar.set_postfix({"AvgRew": f"{avg_reward:.3f}", "Best": f"{best_score:.3f}",
                          "Fee": f"{current_fee*100:.2f}%", "TopK": len(top_formulas)})

    print(f"\n--- Phase 2: Out-of-Sample Evaluation ---")
    print(f"  Fee: {args.fee*100:.2f}% (fixed, Binance spot)")
    print(f"  Evaluating {len(top_formulas)} candidate formulas on test set...\n")

    results = []
    for formula, train_score in top_formulas:
        test_score, test_return, test_turnover = evaluate_formula(formula, test_feat, test_raw, test_ret, vm, bt, args.fee)
        readable = decode_formula(formula, feat_names)
        results.append({
            "formula": formula,
            "readable": readable,
            "train_score": train_score,
            "test_score": test_score,
            "test_ret": test_return,
            "test_turnover": test_turnover,
        })

    valid_results = [r for r in results if r["test_score"] is not None]
    valid_results.sort(key=lambda x: x["test_score"], reverse=True)
    report = build_single_report(args, seed, fee_start, fee_end, reward_history, valid_results, time_meta)

    test_days = time_meta["test_days"]
    print(f"{'Rank':<5} {'Train':>8} {'Test':>8} {'TestRet':>9} {'Ret/30d':>8} {'Turn':>6}  Formula")
    print("-" * 95)
    for i, r in enumerate(valid_results[:10]):
        marker = " ✓" if r["test_score"] > 0 else ""
        ret_30d = r["test_ret"] * (30.0 / test_days) if r["test_ret"] is not None and test_days > 0 else 0.0
        turnover_str = f"{r['test_turnover']:.1f}" if r.get("test_turnover") is not None else "N/A"
        print(f"  {i+1:<3} {r['train_score']:>+8.4f} {r['test_score']:>+8.4f} {r['test_ret']:>+8.2%} {ret_30d:>+7.2%} {turnover_str:>6}  {r['readable']}{marker}")

    summary = report["summary"]
    print(f"\n--- Summary ---")
    print(f"  Test window:          {test_days} days")
    print(f"  Candidates evaluated: {summary['candidates_evaluated']}")
    print(f"  Passed (test > 0):    {summary['passed']}")
    print(f"  Failed (test <= 0):   {summary['failed']}")
    print(f"  Final AvgRew:         {summary['final_avg_reward']:+.4f}" if summary["final_avg_reward"] is not None else "  Final AvgRew:         N/A")
    print(f"  Mean AvgRew:          {summary['mean_avg_reward']:+.4f}" if summary["mean_avg_reward"] is not None else "  Mean AvgRew:          N/A")

    if summary["best_passed_formula"] is not None:
        print(f"\n  Best OOS formula: {summary['best_passed_formula']}")
        print(f"  Train score: {summary['best_passed_train_score']:+.4f}")
        print(f"  Test score:  {summary['best_passed_test_score']:+.4f}")
        print(f"  Test return: {summary['best_passed_test_return']:+.2%}")
    elif summary["best_formula"] is not None:
        print(f"\n  Best candidate did not pass OOS.")
        print(f"  Best candidate: {summary['best_formula']}")
        print(f"  Train score: {summary['best_train_score']:+.4f}")
        print(f"  Test score:  {summary['best_test_score']:+.4f}")
        print(f"  Test return: {summary['best_test_return']:+.2%}")
    else:
        print(f"\n  No formula produced a valid out-of-sample score.")

    return report


def build_multi_run_report(args, reports):
    summaries = [r["summary"] for r in reports]
    best_run = None
    scored_runs = [r for r in reports if r["summary"]["best_test_score"] is not None]
    if scored_runs:
        best_run = max(scored_runs, key=lambda r: r["summary"]["best_test_score"])

    return {
        "config": {
            "csv": os.path.basename(args.csv),
            "steps": args.steps,
            "batch": args.batch,
            "train_ratio": args.train_ratio,
            "fee": args.fee,
            "runs": args.runs,
            "seed": args.seed,
        },
        "aggregate": {
            "runs_completed": len(reports),
            "avg_candidates_evaluated": mean_or_none([s["candidates_evaluated"] for s in summaries]),
            "avg_passed": mean_or_none([s["passed"] for s in summaries]),
            "avg_failed": mean_or_none([s["failed"] for s in summaries]),
            "avg_pass_rate": mean_or_none([s["pass_rate"] for s in summaries]),
            "pass_rate_std": std_or_none([s["pass_rate"] for s in summaries]),
            "avg_best_train_score": mean_or_none([s["best_train_score"] for s in summaries]),
            "avg_best_test_score": mean_or_none([s["best_test_score"] for s in summaries]),
            "avg_best_test_return": mean_or_none([s["best_test_return"] for s in summaries]),
            "avg_best_test_return_per_30d": mean_or_none([s["best_test_return_per_30d"] for s in summaries]),
            "avg_best_test_turnover": mean_or_none([s["best_test_turnover"] for s in summaries]),
            "test_days": summaries[0]["test_days"] if summaries else None,
            "avg_final_avg_reward": mean_or_none([s["final_avg_reward"] for s in summaries]),
            "avg_mean_avg_reward": mean_or_none([s["mean_avg_reward"] for s in summaries]),
            "runs_with_positive_candidate": sum(
                1 for s in summaries
                if s["best_test_score"] is not None and s["best_test_score"] > 0
            ),
            "best_run_index": best_run["config"]["run_index"] if best_run else None,
            "best_run_seed": best_run["config"]["seed"] if best_run else None,
            "best_run_formula": best_run["summary"]["best_formula"] if best_run else None,
            "best_run_test_score": best_run["summary"]["best_test_score"] if best_run else None,
            "best_run_test_return": best_run["summary"]["best_test_return"] if best_run else None,
        },
        "runs": [
            {
                "run_index": report["config"]["run_index"],
                "seed": report["config"]["seed"],
                "summary": report["summary"],
                "report_file": report["config"].get("report_file"),
            }
            for report in reports
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="AlphaGPT Out-of-Sample Validation")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--fee", type=float, default=0.001, help="Binance spot fee (0.001 = 0.1%)")
    parser.add_argument("--runs", type=int, default=1, help="Repeat validation N times with different seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducible runs")
    args = parser.parse_args()

    if args.csv is None:
        args.csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ohlcv_1h.csv")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    print(f"=== AlphaGPT Out-of-Sample Validation ===\n")
    print(f"Loading and splitting data (train={args.train_ratio:.0%} / test={1-args.train_ratio:.0%})...")

    train_data, test_data, time_meta = \
        load_split_data(args.csv, train_ratio=args.train_ratio)

    train_feat, _, _ = train_data
    test_feat, _, _ = test_data
    print(f"  Train features: {train_feat.shape}")
    print(f"  Test features:  {test_feat.shape}")
    csv_basename = os.path.basename(args.csv)
    tf_tag = os.path.splitext(csv_basename)[0]
    out_dir = os.path.dirname(os.path.abspath(__file__))
    reports = []

    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        report = run_validation_once(
            args,
            train_data,
            test_data,
            time_meta,
            run_index=run_idx + 1,
            total_runs=args.runs,
            seed=seed,
        )
        report["config"]["run_index"] = run_idx + 1

        if args.runs == 1:
            report_path = os.path.join(out_dir, f"validation_report_{tf_tag}.json")
        else:
            report_path = os.path.join(out_dir, f"validation_report_{tf_tag}_run{run_idx + 1:02d}.json")

        save_json(report_path, report)
        report["config"]["report_file"] = os.path.basename(report_path)
        print(f"\n  Full report saved to {report_path}")

        if args.runs == 1 and report["summary"]["best_passed_formula"] is not None:
            best_idx = next(
                i for i, item in enumerate(report["all_candidates"])
                if item["passed"]
            )
            best_out = {
                "formula_tokens": report["all_candidates"][best_idx]["formula_tokens"],
                "readable": report["summary"]["best_passed_formula"],
                "train_score": report["summary"]["best_passed_train_score"],
                "test_score": report["summary"]["best_passed_test_score"],
                "test_return": report["summary"]["best_passed_test_return"],
                "fee": args.fee,
                "train_ratio": args.train_ratio,
                "seed": seed,
            }
            best_path = os.path.join(out_dir, "validated_formula.json")
            save_json(best_path, best_out)
            print(f"  Best formula saved to {best_path}")

        reports.append(report)

    if args.runs > 1:
        aggregate = build_multi_run_report(args, reports)
        aggregate_path = os.path.join(out_dir, f"validation_report_{tf_tag}_runs{args.runs}.json")
        save_json(aggregate_path, aggregate)

        agg = aggregate["aggregate"]
        print(f"\n=== Aggregate Summary ({args.runs} runs) ===")
        print(f"  Test window:         {agg['test_days']} days" if agg["test_days"] is not None else "  Test window:         N/A")
        print(f"  Avg pass rate:       {agg['avg_pass_rate']:.2%}" if agg["avg_pass_rate"] is not None else "  Avg pass rate:       N/A")
        print(f"  Pass rate std:       {agg['pass_rate_std']:.2%}" if agg["pass_rate_std"] is not None else "  Pass rate std:       N/A")
        print(f"  Avg best test score: {agg['avg_best_test_score']:+.4f}" if agg["avg_best_test_score"] is not None else "  Avg best test score: N/A")
        print(f"  Avg best return:     {agg['avg_best_test_return']:+.2%}" if agg["avg_best_test_return"] is not None else "  Avg best return:     N/A")
        print(f"  Avg return/30d:      {agg['avg_best_test_return_per_30d']:+.2%}" if agg["avg_best_test_return_per_30d"] is not None else "  Avg return/30d:      N/A")
        print(f"  Avg turnover:        {agg['avg_best_test_turnover']:.1f}" if agg["avg_best_test_turnover"] is not None else "  Avg turnover:        N/A")
        print(f"  Best run:            #{agg['best_run_index']} (seed={agg['best_run_seed']})" if agg["best_run_index"] is not None else "  Best run:            N/A")
        print(f"  Aggregate report:    {aggregate_path}")


if __name__ == "__main__":
    main()
