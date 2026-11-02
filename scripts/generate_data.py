from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "house_data.csv"

def main():
    rng = np.random.default_rng(42)
    n = 1000
    locations = rng.choice(["City Center", "Suburb", "Outskirts"], n, p=[0.3, 0.45, 0.25])
    types = rng.choice(["Apartment", "Independent House", "Villa"], n, p=[0.55, 0.3, 0.15])
    area = rng.integers(450, 3200, n)
    bedrooms = np.clip(np.round(area / 450 + rng.normal(0, 0.8, n)), 1, 7).astype(int)
    bathrooms = np.clip(np.round(bedrooms * 0.7 + rng.normal(0, 0.5, n)), 1, 6)
    age = rng.integers(0, 45, n)
    loc_premium = np.select([locations == "City Center", locations == "Suburb"], [2_500, 1_200], default=500)
    type_premium = np.select([types == "Villa", types == "Independent House"], [1_800_000, 700_000], default=0)
    price = (area * 6_500 + bedrooms * 350_000 + bathrooms * 180_000 +
             loc_premium * area / 100 + type_premium - age * 28_000 +
             rng.normal(0, 450_000, n))
    df = pd.DataFrame({"area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                       "age": age, "location": locations, "property_type": types,
                       "price": np.maximum(price, 500_000).round(0)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")

if __name__ == "__main__":
    main()
