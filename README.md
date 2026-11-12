# Volatility Forecasting

## Research question
Can a simple conditional-variance recursion adapt its next-day forecast when synthetic return volatility changes?

## Methodology and formula
The EWMA variance state follows v_(t+1)=lambda*v_t+(1-lambda)*r_t^2, with 0<lambda<1. The function returns the variance state before every supplied return plus one forecast after the final observation. The experiment annualizes the forecast as sqrt(252*v).

## Assumptions and limitations
- Returns are finite, equally spaced daily simple returns; the experiment has two synthetic Gaussian volatility regimes.
- The initial variance defaults to a fixed 0.0001 daily variance, independent of the full sample. A production forecast should initialize on a prior training window and be assessed out of sample.
- EWMA is not a fitted GARCH model; no mean dynamics, parameter uncertainty, market microstructure, or structural-break test is included.
- Synthetic chart values are not estimates of real market volatility or investment performance.

## Reproducibility
With Python 3.11+, run python -m pip install -e ".[test]", then python -m pytest and volatility-demo --seed 17. The fixed-seed run writes reports/ewma_volatility.png and reports/summary.json; it uses no external data or credentials.

