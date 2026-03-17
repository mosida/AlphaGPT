import torch
from .config import ModelConfig

class MemeBacktest:
    def __init__(self):
        self.trade_size = 1000.0
        self.min_liq = ModelConfig.MIN_LIQUIDITY
        self.base_fee = 0.0010

    def evaluate(self, factors, raw_data, target_ret, fee_override=None):
        liquidity = raw_data['liquidity']
        signal = torch.sigmoid(factors)
        is_safe = (liquidity > self.min_liq).float()
        position = (signal > 0.55).float() * is_safe
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        fee = fee_override if fee_override is not None else self.base_fee
        total_slippage_one_way = fee + impact_slippage
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost
        cum_ret = net_pnl.sum(dim=1)
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
        score = cum_ret - (big_drawdowns * 2.0)
        activity = position.sum(dim=1)
        score = torch.where(activity < 5, torch.tensor(-1.0, device=score.device), score)
        final_fitness = torch.median(score)
        avg_turnover = turnover.sum(dim=1).mean().item()
        return final_fitness, cum_ret.mean().item(), avg_turnover

    def evaluate_detailed(self, factors, raw_data, target_ret, fee_override=None):
        """Return per-timestep backtest results for walk-forward analysis.

        Returns dict with position, net_pnl, gross_pnl, turnover time series
        plus the same aggregate metrics as evaluate().
        """
        liquidity = raw_data['liquidity']
        signal = torch.sigmoid(factors)
        is_safe = (liquidity > self.min_liq).float()
        position = (signal > 0.55).float() * is_safe
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        fee = fee_override if fee_override is not None else self.base_fee
        total_slippage_one_way = fee + impact_slippage
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost
        cum_ret = net_pnl.sum(dim=1)
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
        score = cum_ret - (big_drawdowns * 2.0)
        activity = position.sum(dim=1)
        score = torch.where(activity < 5, torch.tensor(-1.0, device=score.device), score)
        final_fitness = torch.median(score)
        avg_turnover = turnover.sum(dim=1).mean().item()
        return {
            'position': position,
            'net_pnl': net_pnl,
            'gross_pnl': gross_pnl,
            'tx_cost': tx_cost,
            'turnover': turnover,
            'score': final_fitness,
            'cum_ret': cum_ret,
            'avg_turnover': avg_turnover,
        }