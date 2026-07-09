# Benchmark strategies

Six naive strategies used to sanity-check the ML models. Every one of them ignores
the feature matrix entirely — they only look at the target series `y` (and sometimes
not even that). If a Random Forest or MLP can't beat these, it isn't learning anything
useful.

Implementation: [src/strategies/naive.py](../src/strategies/naive.py).

| Strategy | Predicts | Notes |
|---|---|---|
| Bullish | Always up | Baseline for an asset with a long-term upward drift. Its expected accuracy equals the training up-ratio. |
| Bearish | Always down | Mirror of Bullish. |
| Random | 50/50 coin flip | The null hypothesis — anything should beat this. |
| Frequency | Samples from the historical up/down ratio | Slightly smarter than Random: if the market went up 57% of the time in training, it predicts up 57% of the time. |
| Momentum | Repeats the last observed direction | Bet that yesterday's direction continues. |
| Mean Reversion | Opposite of the last observed direction | Bet that yesterday's direction reverses. |

## Why these and not Buy & Hold / Sharpe ratio benchmarks

This project evaluates a next-day **direction classifier**, not a trading strategy —
there's no position sizing, transaction costs, or portfolio value being tracked. So
the natural benchmarks are other ways of guessing direction, not investment strategies.
Buy & Hold, moving-average crossovers, and risk-adjusted return metrics (Sharpe,
drawdown, etc.) belong to a backtesting layer this project doesn't implement.

## Reading the comparison

If Random Forest gets 54% test accuracy and Frequency gets 53%, the model is barely
adding anything over "guess the historical average." If it's beating Bullish/Bearish/
Random by a real margin *and* holding up on the test set (not just train), that's a
more interesting result.
