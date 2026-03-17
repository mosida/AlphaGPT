"""
Walk-Forward Multi-Factor Portfolio Validation for AlphaGPT.

Trains once on the training set to discover formulas, clusters them by
signal correlation, builds a multi-factor composite portfolio, then
evaluates on rolling walk-forward windows in the test set.

Normalization uses training-set parameters (no look-ahead bias).
Performance metrics are computed from bar-level equity curves.

Usage:
    python walk_forward.py --csv data/ohlcv_4h.csv
    python walk_forward.py --csv data/ohlcv_4h.csv --steps 500 --max-corr 0.4
    python walk_forward.py --csv data/ohlcv_4h.csv --wf-days 30
"""
import argparse
import json
import math
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
from model_core.portfolio import (
    compute_signals, greedy_cluster, build_composite, compute_norm_stats,
)

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


# ---------------------------------------------------------------------------
# Data loading (reuses validate.py logic)
# ---------------------------------------------------------------------------

def load_split_data(csv_path, train_ratio=0.7, limit_tokens=None):
    """Load data and split into train/test by time. Returns tensors + metadata."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    symbols = df["symbol"].unique()
    if limit_tokens and limit_tokens < len(symbols):
        symbols = symbols[:limit_tokens]
        df = df[df["symbol"].isin(symbols)]

    timestamps = sorted(df["timestamp"].unique())
    split_idx = int(len(timestamps) * train_ratio)
    split_time = timestamps[split_idx]

    df_train = df[df["timestamp"] < split_time]
    df_test = df[df["timestamp"] >= split_time]

    train_symbols = set(df_train["symbol"].unique())
    test_symbols = set(df_test["symbol"].unique())
    common_symbols = sorted(train_symbols & test_symbols)
    dropped = (train_symbols | test_symbols) - set(common_symbols)
    if dropped:
        print(f"  Dropped (not in both splits): {dropped}")

    df_train = df_train[df_train["symbol"].isin(common_symbols)]
    df_test = df_test[df_test["symbol"].isin(common_symbols)]

    test_timestamps = sorted(df_test["timestamp"].unique())

    print(f"  Total: {len(timestamps)} timesteps, {len(common_symbols)} pairs")
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
        "test_timestamps": [str(t) for t in test_timestamps],
    }
    return train_data, test_data, time_meta


# ---------------------------------------------------------------------------
# Training loop (discovers formulas)
# ---------------------------------------------------------------------------

def discover_formulas(args, train_data, feat_names, seed=42):
    """Train AlphaGPT and collect top candidate formulas.

    Returns:
        formulas: list of token lists
        scores: list of corresponding train scores
    """
    set_seed(seed)
    train_feat, train_raw, train_ret = train_data

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
    fee_start, fee_end = 0.0005, args.fee

    print(f"\n--- Phase 1: Training ({args.steps} steps, seed={seed}) ---")
    print(f"  Fee: {fee_start*100:.2f}% -> {fee_end*100:.2f}%\n")

    pbar = tqdm(range(args.steps))
    for step in pbar:
        progress = step / max(args.steps - 1, 1)
        current_fee = fee_start + (fee_end - fee_start) * progress

        bs = args.batch
        inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
        log_probs, entropies, values, tokens_list = [], [], [], []

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
                tqdm.write(f"  [NEW BEST] Train={score:.3f} | Ret={ret_val:.2%} | {readable}")

            if score.item() > -0.1:
                key = tuple(formula)
                if key not in {tuple(f) for f, _ in top_formulas}:
                    top_formulas.append((formula, score.item()))
                    top_formulas.sort(key=lambda x: x[1], reverse=True)
                    top_formulas = top_formulas[:40]  # keep more for clustering

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

    formulas = [f for f, _ in top_formulas]
    scores = [s for _, s in top_formulas]
    print(f"\n  Discovered {len(formulas)} candidate formulas (best train={best_score:.4f})")
    return formulas, scores


# ---------------------------------------------------------------------------
# Bar-level equity curve utilities
# ---------------------------------------------------------------------------

def compute_bar_equity(net_pnl_2d):
    """Compute bar-level equity curve from per-asset net PnL.

    Args:
        net_pnl_2d: (n_pairs, n_bars) per-bar net PnL per asset
    Returns:
        bar_pnl: (n_bars,) equal-weight cross-sectional mean per bar
        cum_pnl: (n_bars,) cumulative PnL curve
    """
    bar_pnl = net_pnl_2d.mean(dim=0)   # equal-weight across assets
    cum_pnl = bar_pnl.cumsum(dim=0)
    return bar_pnl, cum_pnl


def bar_level_max_dd(cum_pnl):
    """Max drawdown from a cumulative PnL curve (additive, not multiplicative).

    Returns scalar max drawdown (positive number = loss from peak).
    """
    running_max = cum_pnl.cummax(dim=0)[0]
    drawdown = running_max - cum_pnl
    return drawdown.max().item()


def bar_level_sharpe(bar_pnl, bars_per_year):
    """Annualized Sharpe ratio from bar-level returns.

    Args:
        bar_pnl: (n_bars,) per-bar PnL
        bars_per_year: e.g. 2190 for 4h bars
    Returns:
        annualized Sharpe or None if std ≈ 0
    """
    mean = bar_pnl.mean().item()
    std = bar_pnl.std().item()
    if std < 1e-10:
        return None
    return (mean / std) * math.sqrt(bars_per_year)


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def walk_forward_evaluate(formulas, selected_indices, train_data, test_data,
                          norm_stats, time_meta, args, feat_names):
    """Walk-forward evaluate portfolio on test set using bar-level equity curve.

    Composite signal is normalized using training-set parameters (norm_stats)
    to avoid look-ahead bias.  All performance metrics are computed from the
    bar-level PnL series, not from window-aggregated numbers.

    Args:
        train_data: (feat, raw, ret) training tensors (for individual formula eval context)
        test_data: (feat, raw, ret) test tensors
        norm_stats: dict {idx: (mean, std)} from compute_norm_stats on training set
    """
    test_feat, test_raw, test_ret = test_data
    vm = StackVM()
    bt = MemeBacktest()
    fee = args.fee
    n_bars = test_feat.shape[2]
    test_days = time_meta["test_days"]
    bars_per_day = n_bars / max(test_days, 1)
    bars_per_year = bars_per_day * 365

    # --- Evaluate individual formulas on full test set ---
    formula_results = []
    for idx in selected_indices:
        res = vm.execute(formulas[idx], test_feat)
        if res is None or res.std() < 1e-4:
            formula_results.append(None)
            continue
        detail = bt.evaluate_detailed(res, test_raw, test_ret, fee_override=fee)
        score, ret_val, turnover = bt.evaluate(res, test_raw, test_ret, fee_override=fee)
        bar_pnl, cum_pnl = compute_bar_equity(detail["net_pnl"])
        formula_results.append({
            "index": idx,
            "readable": decode_formula(formulas[idx], feat_names),
            "score": score.item() if not torch.isnan(score) else None,
            "return": ret_val,
            "turnover": turnover,
            "bar_pnl": bar_pnl,
            "cum_pnl": cum_pnl,
            "net_pnl_2d": detail["net_pnl"],
            "turnover_2d": detail["turnover"],
        })

    # --- Portfolio composite on full test set (train-set normalization) ---
    composite = build_composite(
        formulas, selected_indices, test_feat, vm, norm_stats=norm_stats
    )

    pf_bar_pnl = None
    pf_cum_pnl = None
    portfolio_score = None
    portfolio_return = None
    portfolio_turnover = None
    pf_detail = None
    if composite is not None:
        pf_detail = bt.evaluate_detailed(composite, test_raw, test_ret, fee_override=fee)
        portfolio_score, portfolio_return, portfolio_turnover = bt.evaluate(
            composite, test_raw, test_ret, fee_override=fee
        )
        if torch.isnan(portfolio_score):
            portfolio_score = None
        else:
            portfolio_score = portfolio_score.item()
        pf_bar_pnl, pf_cum_pnl = compute_bar_equity(pf_detail["net_pnl"])

    # --- Bar-level aggregate metrics ---
    pf_max_dd = None
    pf_sharpe = None
    pf_cum_ret = None
    pf_ann_ret = None
    if pf_bar_pnl is not None:
        pf_max_dd = bar_level_max_dd(pf_cum_pnl)
        pf_sharpe = bar_level_sharpe(pf_bar_pnl, bars_per_year)
        pf_cum_ret = pf_cum_pnl[-1].item()
        pf_ann_ret = pf_cum_ret * (365.0 / max(test_days, 1)) if test_days > 0 else None

    # --- Walk-forward windows (slicing bar-level PnL) ---
    test_timestamps = time_meta.get("test_timestamps", [])
    wf_bars = max(int(args.wf_days * bars_per_day), 10)

    windows = []
    t_start = 0
    while t_start + wf_bars <= n_bars:
        t_end = min(t_start + wf_bars, n_bars)
        windows.append((t_start, t_end))
        t_start = t_end
    if t_start < n_bars and (n_bars - t_start) >= wf_bars // 2:
        windows.append((t_start, n_bars))

    wf_windows = []
    for w_idx, (t_start, t_end) in enumerate(windows):
        window_days = (t_end - t_start) / bars_per_day
        ts_start = test_timestamps[t_start] if t_start < len(test_timestamps) else "?"
        ts_end = test_timestamps[min(t_end - 1, len(test_timestamps) - 1)] if test_timestamps else "?"

        w_ret = None
        w_max_dd = None
        w_sharpe = None
        w_turnover = None
        if pf_bar_pnl is not None:
            w_slice = pf_bar_pnl[t_start:t_end]
            w_ret = w_slice.sum().item()
            w_cum = w_slice.cumsum(dim=0)
            w_max_dd = bar_level_max_dd(w_cum)
            w_sharpe = bar_level_sharpe(w_slice, bars_per_year)
        if pf_detail is not None:
            w_turnover = pf_detail["turnover"][:, t_start:t_end].sum(dim=1).mean().item()

        ret_30d = w_ret * (30.0 / window_days) if w_ret is not None and window_days > 0 else None

        wf_windows.append({
            "window": w_idx + 1,
            "bars": f"{t_start}-{t_end}",
            "period": f"{ts_start} ~ {ts_end}",
            "days": round(window_days, 1),
            "portfolio_return": w_ret,
            "portfolio_return_per_30d": ret_30d,
            "portfolio_max_dd": w_max_dd,
            "portfolio_sharpe_ann": w_sharpe,
            "portfolio_turnover": w_turnover,
        })

    # --- Window-level summary ---
    positive_windows = sum(1 for w in wf_windows if w["portfolio_return"] is not None and w["portfolio_return"] > 0)
    total_windows = len(wf_windows)

    return {
        "portfolio": {
            "n_formulas": len(selected_indices),
            "formula_indices": selected_indices,
            "normalization": "train-set mean/std (causal)",
            "full_test_score": portfolio_score,
            "full_test_return": portfolio_return,
            "full_test_turnover": portfolio_turnover,
        },
        "individual_formulas": [
            {
                "index": fr["index"],
                "readable": fr["readable"],
                "score": fr["score"],
                "return": fr["return"],
                "turnover": fr["turnover"],
            }
            for fr in formula_results if fr is not None
        ],
        "equity_curve": {
            "cumulative_return": pf_cum_ret,
            "annualized_return": pf_ann_ret,
            "max_drawdown_bar": pf_max_dd,
            "sharpe_annualized": pf_sharpe,
            "bars_per_day": round(bars_per_day, 2),
            "bars_per_year": round(bars_per_year, 1),
            "total_bars": n_bars,
        },
        "walk_forward": {
            "wf_days_setting": args.wf_days,
            "wf_bars": wf_bars,
            "n_windows": total_windows,
            "positive_windows": positive_windows,
            "win_rate": positive_windows / total_windows if total_windows > 0 else None,
            "windows": wf_windows,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Multi-Factor Portfolio Validation")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--fee", type=float, default=0.001, help="Fee rate (0.001 = 0.1%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-corr", type=float, default=0.5,
                        help="Max correlation for formula clustering (lower = more diverse)")
    parser.add_argument("--wf-days", type=int, default=30,
                        help="Walk-forward window size in days")
    parser.add_argument("--pairs", type=int, default=None, help="Limit number of pairs")
    args = parser.parse_args()

    if args.csv is None:
        args.csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ohlcv_4h.csv")

    feat_names = ["RET", "VOL", "V_CHG", "PV", "TREND", "LOG_V"][:FeatureEngineer.INPUT_DIM]
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  Walk-Forward Multi-Factor Portfolio Validation")
    print("=" * 60)
    print(f"  CSV:       {args.csv}")
    print(f"  Steps:     {args.steps}")
    print(f"  Fee:       {args.fee*100:.2f}%")
    print(f"  Max corr:  {args.max_corr}")
    print(f"  WF window: {args.wf_days} days")
    print(f"  Seed:      {args.seed}")

    # --- 1. Load and split data ---
    print(f"\nLoading and splitting data (train={args.train_ratio:.0%} / test={1-args.train_ratio:.0%})...")
    train_data, test_data, time_meta = load_split_data(
        args.csv, train_ratio=args.train_ratio, limit_tokens=args.pairs
    )
    train_feat, _, _ = train_data
    test_feat, _, _ = test_data
    print(f"  Train features: {train_feat.shape}")
    print(f"  Test features:  {test_feat.shape}")

    # --- 2. Discover formulas ---
    formulas, scores = discover_formulas(args, train_data, feat_names, seed=args.seed)

    if not formulas:
        print("\n  No formulas discovered. Exiting.")
        return

    # --- 3. Cluster by signal correlation (on train data) ---
    print(f"\n--- Phase 2: Correlation Clustering (max_corr={args.max_corr}) ---")
    vm = StackVM()
    signals = compute_signals(formulas, train_feat, vm)
    print(f"  Valid signals: {len(signals)} / {len(formulas)}")

    scores_dict = {i: scores[i] for i in range(len(scores))}
    selected, clusters = greedy_cluster(formulas, scores_dict, signals, max_corr=args.max_corr)

    print(f"  Clusters: {len(clusters)}")
    for rep_idx, members in clusters:
        readable = decode_formula(formulas[rep_idx], feat_names)
        print(f"    Cluster (rep={rep_idx}, members={len(members)}, score={scores[rep_idx]:+.4f}): {readable}")

    if not selected:
        print("\n  No diverse formulas selected. Exiting.")
        return

    # --- 3b. Compute normalization stats from TRAINING data (causal) ---
    norm_stats = compute_norm_stats(formulas, selected, train_feat, vm)
    print(f"  Norm stats: computed from training set for {len(norm_stats)} formulas")

    # --- 4. Walk-forward evaluation ---
    print(f"\n--- Phase 3: Walk-Forward Evaluation ---")
    print(f"  Portfolio: {len(selected)} diverse formulas")
    print(f"  Normalization: train-set mean/std (causal, no look-ahead)")
    print(f"  Window: {args.wf_days} days")
    print(f"  Fee: {args.fee*100:.2f}% (fixed)\n")

    wf_results = walk_forward_evaluate(
        formulas, selected, train_data, test_data,
        norm_stats, time_meta, args, feat_names
    )

    # --- 5. Print results ---
    # Individual formula results
    print(f"{'Idx':<5} {'Score':>8} {'Return':>9} {'Turn':>6}  Formula")
    print("-" * 80)
    for fr in wf_results["individual_formulas"]:
        score_str = f"{fr['score']:+.4f}" if fr["score"] is not None else "   N/A"
        ret_str = f"{fr['return']:+.2%}" if fr["return"] is not None else "   N/A"
        turn_str = f"{fr['turnover']:.1f}" if fr["turnover"] is not None else " N/A"
        print(f"  {fr['index']:<3} {score_str:>8} {ret_str:>9} {turn_str:>6}  {fr['readable']}")

    # Portfolio summary
    pf = wf_results["portfolio"]
    print(f"\n  Portfolio ({pf['n_formulas']} formulas, {pf['normalization']}):")
    if pf["full_test_score"] is not None:
        print(f"    Full test score:    {pf['full_test_score']:+.4f}")
        print(f"    Full test return:   {pf['full_test_return']:+.2%}")
        print(f"    Full test turnover: {pf['full_test_turnover']:.1f}")
    else:
        print(f"    Portfolio signal failed on test set.")

    # Bar-level equity curve metrics
    eq = wf_results["equity_curve"]
    print(f"\n  === Bar-Level Equity Curve ===")
    print(f"    Cumulative return:  {eq['cumulative_return']:+.4%}" if eq["cumulative_return"] is not None else "    Cumulative return:  N/A")
    if eq["annualized_return"] is not None:
        print(f"    Annualized return:  {eq['annualized_return']:+.4%}")
    print(f"    Max drawdown (bar): {eq['max_drawdown_bar']:.4%}" if eq["max_drawdown_bar"] is not None else "    Max drawdown (bar): N/A")
    print(f"    Sharpe (annualized):{eq['sharpe_annualized']:+.3f}" if eq["sharpe_annualized"] is not None else "    Sharpe (annualized):N/A")
    print(f"    Resolution:         {eq['total_bars']} bars ({eq['bars_per_day']} bars/day)")

    # Walk-forward windows
    wf = wf_results["walk_forward"]
    print(f"\n  Walk-Forward Windows ({wf['n_windows']} x ~{args.wf_days}d):")
    print(f"  {'#':<3} {'Period':<45} {'Days':>5} {'Return':>8} {'Ret/30d':>8} {'MaxDD':>8} {'Sharpe':>7} {'Turn':>6}")
    print("  " + "-" * 95)
    for w in wf["windows"]:
        ret_str = f"{w['portfolio_return']:+.2%}" if w["portfolio_return"] is not None else "   N/A"
        ret30_str = f"{w['portfolio_return_per_30d']:+.2%}" if w["portfolio_return_per_30d"] is not None else "   N/A"
        dd_str = f"{w['portfolio_max_dd']:.2%}" if w["portfolio_max_dd"] is not None else "  N/A"
        sh_str = f"{w['portfolio_sharpe_ann']:+.2f}" if w["portfolio_sharpe_ann"] is not None else "  N/A"
        turn_str = f"{w['portfolio_turnover']:.1f}" if w["portfolio_turnover"] is not None else " N/A"
        print(f"  {w['window']:<3} {w['period']:<45} {w['days']:>5} {ret_str:>8} {ret30_str:>8} {dd_str:>8} {sh_str:>7} {turn_str:>6}")

    # Aggregate
    print(f"\n  === Walk-Forward Summary ===")
    print(f"    Test window:        {time_meta['test_days']} days ({eq['total_bars']} bars)")
    print(f"    WF windows:         {wf['n_windows']}")
    if wf["win_rate"] is not None:
        print(f"    Positive windows:   {wf['positive_windows']} / {wf['n_windows']} ({wf['win_rate']:.0%})")

    # --- 6. Save report ---
    csv_tag = os.path.splitext(os.path.basename(args.csv))[0]
    report = {
        "config": {
            "csv": os.path.basename(args.csv),
            "steps": args.steps,
            "batch": args.batch,
            "train_ratio": args.train_ratio,
            "fee": args.fee,
            "max_corr": args.max_corr,
            "wf_days": args.wf_days,
            "seed": args.seed,
        },
        "time_meta": {k: v for k, v in time_meta.items() if k != "test_timestamps"},
        "clustering": {
            "n_candidates": len(formulas),
            "n_valid_signals": len(signals),
            "n_clusters": len(clusters),
            "selected_indices": selected,
            "clusters": [
                {
                    "representative": rep_idx,
                    "representative_formula": decode_formula(formulas[rep_idx], feat_names),
                    "representative_score": scores[rep_idx],
                    "n_members": len(members),
                }
                for rep_idx, members in clusters
            ],
        },
        "portfolio": wf_results["portfolio"],
        "individual_formulas": wf_results["individual_formulas"],
        "equity_curve": wf_results["equity_curve"],
        "walk_forward": wf_results["walk_forward"],
    }

    report_path = os.path.join(out_dir, f"wf_report_{csv_tag}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()
