import pytest
from src.model_inference import validate_input

VALID = {"area": 1200, "bedrooms": 3, "bathrooms": 2, "age": 5,
         "location": "City Center", "property_type": "Apartment"}

def test_valid_input_returns_frame():
    frame = validate_input(VALID)
    assert frame.iloc[0]["area"] == 1200

def test_missing_field_rejected():
    with pytest.raises(ValueError, match="Missing fields"):
        validate_input({"area": 1200})

def test_negative_area_rejected():
    payload = dict(VALID, area=-1)
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_input(payload)

def test_zero_area_rejected():
    payload = dict(VALID, area=0)
    with pytest.raises(ValueError, match="greater than zero"):
        validate_input(payload)
