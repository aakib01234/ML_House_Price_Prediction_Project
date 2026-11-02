# House Price Prediction — End-to-End ML Project

A modular ML application that estimates house prices from property attributes. It demonstrates data validation, preprocessing, feature engineering, model comparison, hyperparameter tuning, cross-validation, model-agnostic interpretation, persistence/versioning, REST inference, and a responsive Flask UI.

> **Important:** The included dataset is reproducible synthetic demo data, not actual market data. Do not use the model for real financial or property decisions. For a credible real-world project, replace it with a properly licensed Kaggle/local-market dataset and document provenance.

## Quick start (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_data.py
python -m src.model_training
python -m src.web_app
```

Visit `http://127.0.0.1:5000`.

## What is implemented

- **Preprocessing:** median imputation + scaling for numeric data; most-frequent imputation + one-hot encoding for categories; unknown categories safely handled.
- **Algorithms:** Ridge Regression, Random Forest, Extra Trees, Gradient Boosting.
- **Tuning:** randomized hyperparameter search with 5-fold CV; selection by CV MAE.
- **Evaluation:** MAE, RMSE, R², MAPE on a held-out test set.
- **Interpretability:** permutation importance on held-out data (model-agnostic).
- **Reproducibility:** fixed random seeds, timestamped model/metrics/report artifacts.
- **Serving:** Flask HTML interface and JSON endpoint with validation and safe errors.
- **Testing:** unit tests for validation and API behavior.
- **Engineering:** modular source, requirements, Dockerfile, CI workflow, API docs.

## API

`POST /api/predict` with `Content-Type: application/json`:

```json
{
  "area": 1200,
  "bedrooms": 3,
  "bathrooms": 2,
  "age": 5,
  "location": "City Center",
  "property_type": "Apartment"
}
```

Success response includes `predicted_price_inr`, `model_version`, and `algorithm`. Invalid inputs return HTTP 400.

## Train / evaluate

```bash
python -m src.model_training
```

Training saves a versioned `.joblib` artifact, `current.json`, model metrics, a report, and feature-importance CSV under `models/`. Model files are ignored by Git by default; after training, preserve the desired artifact separately if needed.

## Tests

```bash
pytest -q
```

## Docker

```bash
docker build -t house-price-predictor .
docker run --rm -p 5000:5000 house-price-predictor
```

The Docker build generates demo data and trains a model. This may take several minutes.

## Repository layout

```text
data/             demo dataset + dictionary
notebooks/        exploratory analysis guide
src/              preprocessing, training, inference, Flask app
models/           generated versioned model artifacts
tests/            unit/API tests
templates/        web UI
static/           CSS
scripts/          dataset generation
docs/             technical and API documentation
.github/workflows CI tests
```

## Limitations and next steps

The sample data is synthetic; confidence intervals are not claimed. Production deployment should add real data lineage, privacy review, authentication, rate limiting, monitoring/drift detection, logging/alerting, and a model registry.
