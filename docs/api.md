# API reference

## `POST /api/predict`

Content-Type: `application/json`

Required JSON keys: `area`, `bedrooms`, `bathrooms`, `age`, `location`, `property_type`.

Success: HTTP 200 with `predicted_price_inr`, `model_version`, and `algorithm`.

Invalid input or missing model: HTTP 400 with an `error` message. Unexpected server errors return HTTP 500 with a generic message; details are logged server-side.
