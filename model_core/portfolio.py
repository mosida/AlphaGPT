"""
Multi-factor portfolio construction: correlation, clustering, composite signals.
"""
import torch
from .vm import StackVM


def compute_signals(formulas, feat_tensor, vm=None):
    """Execute formulas on data, return dict of {index: signal_tensor}.

    Skips formulas that fail execution or produce degenerate signals.
    """
    if vm is None:
        vm = StackVM()
    signals = {}
    for i, formula in enumerate(formulas):
        res = vm.execute(formula, feat_tensor)
        if res is not None and res.std() > 1e-4:
            signals[i] = res
    return signals


def signal_correlation(signals):
    """Compute pairwise Pearson correlation between formula signals.

    Args:
        signals: dict of {index: tensor(n_pairs, n_timesteps)}
    Returns:
        indices: list of signal indices (sorted)
        corr_matrix: (n_signals, n_signals) correlation matrix
    """
    indices = sorted(signals.keys())
    n = len(indices)
    if n < 2:
        return indices, torch.eye(n)

    # Flatten each signal to a single vector
    flat = []
    for idx in indices:
        flat.append(signals[idx].flatten().float().cpu())

    mat = torch.stack(flat)
    # Center
    mat = mat - mat.mean(dim=1, keepdim=True)
    # Normalize
    norms = mat.norm(dim=1, keepdim=True) + 1e-8
    mat = mat / norms
    corr = mat @ mat.T
    return indices, corr


def greedy_cluster(formulas, scores, signals, max_corr=0.5):
    """Greedy correlation-based clustering.

    Picks best-scoring formula, removes highly correlated ones, repeats.

    Args:
        formulas: list of formula token lists
        scores: dict or list mapping formula index -> score
        signals: dict of {index: signal_tensor}
        max_corr: correlation threshold (formulas with |corr| >= max_corr are merged)
    Returns:
        selected: list of selected formula indices
        clusters: list of (representative_idx, [member_indices])
    """
    indices, corr = signal_correlation(signals)
    if len(indices) == 0:
        return [], []

    idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}

    # Sort by score descending
    if isinstance(scores, dict):
        scored = [(idx, scores.get(idx, -999)) for idx in indices]
    else:
        scored = [(idx, scores[idx] if idx < len(scores) else -999) for idx in indices]
    scored.sort(key=lambda x: x[1], reverse=True)

    selected = []
    clusters = []
    used = set()

    for idx, score in scored:
        if idx in used:
            continue
        pos = idx_to_pos[idx]

        cluster_members = [idx]
        for other_idx, _ in scored:
            if other_idx == idx or other_idx in used:
                continue
            other_pos = idx_to_pos.get(other_idx)
            if other_pos is not None and abs(corr[pos, other_pos].item()) >= max_corr:
                cluster_members.append(other_idx)
                used.add(other_idx)

        selected.append(idx)
        clusters.append((idx, cluster_members))
        used.add(idx)

    return selected, clusters


def compute_norm_stats(formulas, indices, feat_tensor, vm=None):
    """Compute per-formula normalization parameters from a reference dataset.

    Use this on TRAINING data, then pass the result to build_composite()
    when evaluating on test data, to avoid look-ahead bias.

    Returns:
        norm_stats: dict of {formula_index: (mean, std)} computed on feat_tensor
    """
    if vm is None:
        vm = StackVM()
    stats = {}
    for idx in indices:
        res = vm.execute(formulas[idx], feat_tensor)
        if res is not None and res.std() > 1e-4:
            stats[idx] = (res.mean().item(), res.std().item())
    return stats


def build_composite(formulas, indices, feat_tensor, vm=None, weights=None,
                    norm_stats=None):
    """Combine diverse formula signals into a single composite signal.

    Each signal is z-score normalized before combining.  When norm_stats
    is provided (from compute_norm_stats on training data), those parameters
    are used instead of computing mean/std from feat_tensor itself.
    This eliminates look-ahead bias when feat_tensor is test data.

    Args:
        formulas: list of all formulas (token lists)
        indices: which formula indices to include
        feat_tensor: feature data to execute formulas on
        vm: StackVM instance
        weights: optional per-formula weights (default: equal)
        norm_stats: dict {index: (mean, std)} from training set.
                    If None, computes from feat_tensor (look-ahead!).
    Returns:
        composite: tensor(n_pairs, n_timesteps) or None
    """
    if vm is None:
        vm = StackVM()
    if weights is None:
        weights = [1.0 / len(indices)] * len(indices)

    signals = []
    valid_weights = []
    for i, idx in enumerate(indices):
        res = vm.execute(formulas[idx], feat_tensor)
        if res is None or res.std() < 1e-4:
            continue

        if norm_stats is not None and idx in norm_stats:
            mean, std = norm_stats[idx]
        else:
            mean = res.mean().item()
            std = res.std().item()

        z = (res - mean) / (std + 1e-6)
        signals.append(z)
        valid_weights.append(weights[i])

    if not signals:
        return None

    w_sum = sum(valid_weights)
    valid_weights = [w / w_sum for w in valid_weights]

    composite = sum(w * s for w, s in zip(valid_weights, signals))
    return composite
