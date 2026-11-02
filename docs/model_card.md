# Model Card

## Intended use
Educational demonstration of tabular regression workflow and a sample prediction UI.

## Not intended for
Actual property valuation, lending, investment, or any consequential decision.

## Data
Synthetic, generated deterministically by `scripts/generate_data.py`; contains 1,000 simulated rows. It does not represent a verified housing market.

## Evaluation
The training script holds out 20% of rows, uses 5-fold CV within training data for model selection, and reports MAE, RMSE, R², and MAPE. Results vary if the data-generation process changes.

## Limitations
Synthetic data creates artificial relationships and can inflate apparent performance. Predictions may be biased or inaccurate for real locations and property types. No calibrated prediction intervals are provided.
