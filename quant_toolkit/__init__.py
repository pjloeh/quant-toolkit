"""
Quant Research Toolkit
======================
Personal utility library for quantitative research exercises.

pip install git+https://github.com/YOURUSERNAME/quant-toolkit.git

Usage:
    import quant_toolkit as qt
    # or
    from quant_toolkit import expanding_zscore, bootstrap_pnl, half_life
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint
import warnings



def coint_test(y, x):
    score, pvalue, crit = coint(y, x)
    # OLS hedge ratio
    beta = np.polyfit(x, y, 1)[0]
    spread = y - beta * x
    return {
        "score": score,
        "pvalue": pvalue,
        "critical_values": {"1%": crit[0], "5%": crit[1], "10%": crit[2]},
        "hedge_ratio": beta,
        "spread": spread,
    }


def half_life(spread):
    spread = spread.dropna()
    lag = spread.shift(1)
    delta = spread.diff()
    df = pd.DataFrame({"delta": delta, "lag": lag}).dropna()

    phi = np.polyfit(df["lag"], df["delta"], 1)[0]

    if phi >= 0:
        warnings.warn("Positive phi — no mean reversion detected.")
        return np.inf

    hl = -np.log(2) / np.log(1 + phi)
    return hl


def return_stats(series):
    mean = series.mean() * 252
    std = series.std() * np.sqrt(252)
    sharpe = mean / std
    downside_vol = (series[series < 0]).std() * np.sqrt(252)
    sortino = mean / downside_vol
    total_return = series.sum()
    max_dd = (series.cumsum() - series.cumsum().cummax()).min()
    no_trades = (series != 0).sum()
    wins = (series > 0).sum()
    hit_ratio = wins / no_trades
    av_win = series[series > 0].mean()
    av_loss = series[series < 0].mean()
    win_to_loss_ratio = av_win / abs(av_loss)
    perc_in_market = no_trades / len(series)
    cvar = series[series < np.percentile(series, 5)].mean()

    print(f"  Total Return:  {total_return:.2f}")
    print(f"  Ann. Vol:      {std:.2f}")
    print(f"  Sharpe:        {sharpe:.3f}")
    print(f"  Sortino:       {sortino:.3f}")
    print(f"  Max Drawdown:  {max_dd:.2f}")
    print(f"  Hit Ratio:     {hit_ratio:.2%}")
    print(f"  Win/Loss:      {win_to_loss_ratio:.2f}")
    print(f"  % in market:   {perc_in_market:.2f}")
    print(f"  CVaR:          {cvar:.2f}")

    return pd.Series({
                            'total_return': total_return,
                            'ann_vol': std,
                            'sharpe': sharpe,
                            'sortino': sortino,
                            'max_dd': max_dd,
                            'hit_ratio': hit_ratio,
                            'win_loss': win_to_loss_ratio,
                            'pct_in_market': perc_in_market,
                            'cvar': cvar,
                            }, name=series.name)


def equate_rets(df, col_name):
    df = df.copy()
    target_return = df.loc['total_return', col_name]

    columns_to_adjust = list(df.columns)

    for col in columns_to_adjust:
        return_in_column = df.loc['total_return', col]
        ret_factor = target_return / return_in_column
        df.loc['total_return', col] *= ret_factor
        df.loc['ann_vol', col] *= ret_factor
        df.loc['max_dd', col] *= ret_factor
        df.loc['cvar', col] *= ret_factor

    return df




def block_bootstrap(series, block_size, n_simulations):

    series = pd.Series(series)  # Ensure we have a Series
    n = len(series)
    n_blocks = int(np.ceil(n / block_size))  # Number of blocks needed per simulation
    simulated_data = []

    for sim in range(n_simulations):
        # Choose block start indices with replacement
        block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        bootstrapped_series = []

        # Concatenate the blocks
        for start in block_starts:
            block = series.iloc[start:start + block_size].values
            bootstrapped_series.extend(block)

        # Trim to match original series length
        bootstrapped_series = bootstrapped_series[:n]
        simulated_data.append(bootstrapped_series)

    # Convert to DataFrame
    bootstrapped_df = pd.DataFrame(simulated_data).T  # Columns = simulations
    bootstrapped_df.columns = [f"sim_{i + 1}" for i in range(n_simulations)]

    return bootstrapped_df


def evaluate_bootstrap(df_bootstrapped_simulations):
    # total pnl
    total_pnl = df_bootstrapped_simulations.sum()
    # sharpe
    bootstrapped_sharpe = df_bootstrapped_simulations.mean() * 252 / (df_bootstrapped_simulations.std() * np.sqrt(252))
    # max dd
    max_dd = (df_bootstrapped_simulations.cumsum() - df_bootstrapped_simulations.cumsum().cummax()).min()
    # vol
    vol = df_bootstrapped_simulations.std() * np.sqrt(252)

    results = {
        'total_pnl_mean': total_pnl.mean(),
        'total_pnl_median': total_pnl.median(),
        'total_pnl_p10': total_pnl.quantile(0.10),
        'total_pnl_p90': total_pnl.quantile(0.90),
        'sharpe_mean': bootstrapped_sharpe.mean(),
        'sharpe_median': bootstrapped_sharpe.median(),
        'sharpe_p10': bootstrapped_sharpe.quantile(0.10),
        'sharpe_p90': bootstrapped_sharpe.quantile(0.90),
        'max_dd_mean': max_dd.mean(),
        'max_dd_median': max_dd.median(),
        'max_dd_p10': max_dd.quantile(0.10),
        'max_dd_p90': max_dd.quantile(0.90),
        'vol_mean': vol.mean(),
        'vol_median': vol.median(),
        'vol_p10': vol.quantile(0.10),
        'vol_p90': vol.quantile(0.90),
    }

    for k, v in results.items():
        print(f"  {k:<20}: {v:.4f}")

    return pd.Series(results)



def random_positions(series, no_random_positions, returns=None):
    """ Input: series of positions
        Output: series of positions randomly shuffled and pnls if returns are given
    """
    rand_pos_df = pd.DataFrame(index=series.index)

    for i in range(no_random_positions):
        rand_pos = np.random.permutation(series.values)
        rand_pos_df[i] = pd.Series(rand_pos, index=series.index)

    if returns is not None:
        random_pnls = rand_pos_df.multiply(returns, axis=0)
        return rand_pos_df, random_pnls

    return rand_pos_df



def evaluate_random_pnl(df_random_pnls, bt_pnl_series):
    bt_total_return = bt_pnl_series.sum()
    bt_mean = bt_pnl_series.mean() * 252
    bt_std = bt_pnl_series.std() * np.sqrt(252)
    bt_sharpe = bt_mean / bt_std
    bt_max_dd = (bt_pnl_series.cumsum() - bt_pnl_series.cumsum().cummax()).min()

    # total return percentile
    percentile_total_return = (df_random_pnls.sum() < bt_total_return).mean()

    # vol percentile
    percentile_vol = (df_random_pnls.std() * np.sqrt(252) > bt_std).mean()

    # sharpe percentile
    rand_pnls_sharpe = df_random_pnls.mean() * 252 / (df_random_pnls.std() * np.sqrt(252))
    percentile_sharpe = (rand_pnls_sharpe < bt_sharpe).mean()

    # max dd percentile
    rand_max_dd = (df_random_pnls.cumsum() - df_random_pnls.cumsum().cummax()).min()
    percentile_max_dd = (rand_max_dd < bt_max_dd).mean()

    print(f"  Total Return percentile: {percentile_total_return:.2%}")
    print(f"  Vol percentile:          {percentile_vol:.2%}")
    print(f"  Sharpe percentile:       {percentile_sharpe:.2%}")
    print(f"  Max DD percentile:       {percentile_max_dd:.2%}")

    return pd.Series({
        'pct_total_return': percentile_total_return,
        'pct_vol': percentile_vol,
        'pct_sharpe': percentile_sharpe,
        'pct_max_dd': percentile_max_dd,
        }, name=bt_pnl_series.name)


def apply_buffer(target_positions, buffer):
    actual = np.zeros(len(target_positions))
    actual[0] = target_positions.iloc[0]
    for i in range(1, len(target_positions)):
        if abs(target_positions.iloc[i] - actual[i-1]) > buffer:
            actual[i] = target_positions.iloc[i]
        else:
            actual[i] = actual[i-1]
    return pd.Series(actual, index=target_positions.index, name=target_positions.name + f" {buffer} buffer")


def adjust_positions_for_holding_period(positions, holding_period):
    rebalance_mask = pd.Series([i % holding_period == 0 for i in range(len(positions))], index=positions.index)
    positions_hold_period = positions.where(rebalance_mask).ffill()
    positions_hold_period.name = f"{positions.name}_{holding_period}hold_period"
    return positions_hold_period

