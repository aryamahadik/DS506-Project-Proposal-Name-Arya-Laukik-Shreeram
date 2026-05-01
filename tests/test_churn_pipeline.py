from pathlib import Path

from src.churn_pipeline import (
    ENGINEERED_FEATURES,
    IDENTIFIER_COLUMNS,
    REMOVED_CORRELATED_FEATURES,
    clean_data,
    engineer_features,
    load_raw_data,
    train_fast_baseline,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "Churn_Modelling.csv"


def test_load_and_clean_data_removes_identifiers():
    raw = load_raw_data(DATA_PATH)
    cleaned = clean_data(raw)

    assert len(raw) == 10_000
    assert not set(IDENTIFIER_COLUMNS).intersection(cleaned.columns)
    assert cleaned["Exited"].isin([0, 1]).all()
    assert cleaned.isna().sum().sum() == 0


def test_engineer_features_creates_expected_columns_without_mutation():
    raw = load_raw_data(DATA_PATH)
    cleaned = clean_data(raw)
    original_columns = set(cleaned.columns)

    featured = engineer_features(cleaned)

    assert set(ENGINEERED_FEATURES).issubset(featured.columns)
    assert not set(REMOVED_CORRELATED_FEATURES).intersection(featured.columns)
    assert set(cleaned.columns) == original_columns
    assert (featured["Balance_Income_Ratio"] >= 0).all()


def test_engineered_features_avoid_duplicate_high_correlation_features():
    raw = load_raw_data(DATA_PATH)
    featured = engineer_features(clean_data(raw))
    base_numeric = [
        col
        for col in featured.select_dtypes("number").columns
        if col not in set(ENGINEERED_FEATURES + ["Exited"])
    ]
    correlations = featured[base_numeric + ENGINEERED_FEATURES].corr().abs()

    max_correlations = {
        feature: correlations.loc[feature, base_numeric].max()
        for feature in ENGINEERED_FEATURES
    }

    assert all(value < 0.90 for value in max_correlations.values())


def test_fast_baseline_returns_valid_metrics():
    raw = load_raw_data(DATA_PATH)
    metrics = train_fast_baseline(raw)

    assert {"accuracy", "precision", "recall", "f1", "auc_roc"} == set(metrics)
    assert all(0 <= value <= 1 for value in metrics.values())
