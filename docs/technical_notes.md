# Technical notes

- Preprocessing is fitted inside the scikit-learn pipeline to prevent leakage.
- Numeric values are median-imputed and standardized.
- Categories are most-frequent-imputed and one-hot encoded; unseen categories are tolerated.
- Training uses a fixed random seed and 5-fold cross-validation.
- The selected model is the candidate with lowest mean CV MAE.
- Timestamped model artifacts and metric files support reproducibility and basic model versioning.
- Production extensions: add monitoring, authentication, rate limiting, structured logs, and a real-data validation policy.
