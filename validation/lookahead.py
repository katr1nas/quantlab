import numpy as np
import pandas as pd
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.load_data import load_data_yf

def check_lookahead(
        data: pd.DataFrame, 
        features_fn,
        rtol: float = 1e-5,
        atol: float = 1e-8, 
    ) -> pd.DataFrame:
   
    full_features = features_fn(data)

    if not isinstance(full_features, pd.DataFrame):
        raise TypeError('feature_fn must return a DataFrame')
    
    violations = []

    for i in range(len(data)):
        historical_data = data.iloc[:i + 1]

        historical_features = features_fn(historical_data)

        for feature in full_features.columns:
            full_value = full_features[feature].iloc[i]
            historical_value = historical_features[feature].iloc[-1]

            if pd.isna(full_value) and pd.isna(historical_value):
                continue

            if not np.isclose(
                full_value,
                historical_value,
                rtol=rtol,
                atol=atol, 
                equal_nan=True,
            ):
                violations.append({
                    'index': i,
                    'timestamp': data.index[i],
                    'feature': feature,
                    'full_value': full_value,
                    'historical_value': historical_value,
                })
    return pd.DataFrame(violations)
   

def generate_features(data):
    features = pd.DataFrame(index=data.index)

    features["ma_10"] = data["close"].rolling(10).mean()
    features["momentum"] = data["close"].pct_change(5)

    # Intentionally introduces lookahead bias
    features["future_close"] = data["close"].shift(-1)

    return features

data = load_data_yf('ES=F', '2017-01-01', '2025-01-01')
violations = check_lookahead(
    data,
    generate_features,
)

print(violations)