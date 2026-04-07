# Quant Toolkit

Personal utility library for quantitative research.

## Install

```bash
pip install git+https://github.com/YOURUSERNAME/quant-toolkit.git
```

## Quick Start

```python
import quant_toolkit as qt

coint_test(y, x)

half_life(spread)

return_stats(series)

equate_rets(df, col_name)

block_bootstrap(series, block_size, n_simulations)

evaluate_bootstrap(df_bootstrapped_simulations)

random_positions(series, no_random_positions, returns=None)

evaluate_random_pnl(df_random_pnls, bt_pnl_series)

apply_buffer(target_positions, buffer)

adjust_positions_for_holding_period(positions, holding_period)

```

