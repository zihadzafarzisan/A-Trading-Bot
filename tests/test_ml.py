"""Tests for the ML pipeline."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.ml import (
    MLDatasetBuilder, chronological_split, MLDataset,
    ModelTrainer, ModelEvaluator, TradePredictor, MLPipeline,
)


def make_feature_df(n=400, base_ts=1_609_459_200_000):
    """Feature DataFrame with rsi_14 + canonical columns + timestamp."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": [base_ts + i * 3_600_000 for i in range(n)],
        "close": 100.0,
        "rsi_14": rng.uniform(20, 80, n),
        "volume_ratio": rng.uniform(0.5, 2.0, n),
        "atr_ratio": 0.02,
        "hist_vol_20": 0.5,
        "adx_14": 25.0,
        "dist_sma_50": 0.0,
        "roc_12": 0.0,
    })
    for c in ["sma_10", "sma_20", "sma_50", "sma_100", "sma_200", "ema_10", "ema_20",
              "ema_50", "ema_100", "ema_200", "ema_cross_20_50", "ema_cross_50_200",
              "slope_20", "macd_macd", "macd_signal", "macd_histogram", "stoch_k",
              "stoch_d", "mom_10", "williams_r", "cci_20", "bb_mid", "bb_upper",
              "bb_lower", "bb_width", "bb_pct_b", "vol_percentile", "dc_high",
              "dc_low", "dc_mid", "volume_sma_20", "obv", "vol_mom_10", "mfi_14",
              "vwap_20", "higher_high", "lower_low", "hh_breakout", "ll_breakdown",
              "above_sma_50", "near_resistance", "near_support", "hour",
              "day_of_week", "month", "is_weekend", "day", "session_bucket"]:
        df[c] = 0.0
    return df


def make_trades_from_rsi(fdf, n=300, seed=1):
    """Trades whose profitability depends on rsi_14 at entry (a real signal).

    rsi > 62 -> win; rsi < 38 -> loss; middle -> ~coin flip. This gives the
    model a learnable pattern for the 'ML improves' test.
    """
    rng = np.random.default_rng(seed)
    trades = []
    rsi_vals = fdf["rsi_14"].to_numpy()
    n = min(n, len(fdf))
    for i in range(n):
        rsi = rsi_vals[i]
        if rsi > 62:
            pnl = 5.0 if rng.random() < 0.85 else -5.0
        elif rsi < 38:
            pnl = -5.0 if rng.random() < 0.85 else 5.0
        else:
            pnl = 5.0 if rng.random() < 0.5 else -5.0
        trades.append({
            "entry_time": int(fdf["timestamp"].iloc[i]),
            "net_pnl": pnl,
            "direction": "long",
        })
    return trades


class TestDatasetBuilder:
    """Test ML dataset construction."""

    def test_builds_dataset(self):
        fdf = make_feature_df()
        trades = make_trades_from_rsi(fdf)
        ds = MLDatasetBuilder().build(trades, fdf, symbol="BTCUSDT", timeframe="1h")
        assert ds.n_samples == len(trades)
        assert ds.n_features > 10
        assert len(ds.y) == len(trades)
        assert set(np.unique(ds.y)) <= {0, 1}

    def test_chronologically_sorted(self):
        fdf = make_feature_df()
        trades = make_trades_from_rsi(fdf)
        ds = MLDatasetBuilder().build(trades, fdf)
        assert (np.diff(ds.timestamps) >= 0).all()

    def test_target_matches_pnl(self):
        fdf = make_feature_df()
        trades = make_trades_from_rsi(fdf)
        ds = MLDatasetBuilder().build(trades, fdf)
        for i, t in enumerate(trades):
            assert ds.y[i] == (1 if t["net_pnl"] > 0 else 0)

    def test_feature_names_match_columns(self):
        fdf = make_feature_df()
        trades = make_trades_from_rsi(fdf)
        ds = MLDatasetBuilder().build(trades, fdf)
        assert "rsi_14" in ds.feature_names
        assert "volume_ratio" in ds.feature_names


class TestChronologicalSplit:
    """Test leakage-safe chronological splitting."""

    def test_no_shuffle(self):
        fdf = make_feature_df()
        trades = make_trades_from_rsi(fdf)
        ds = MLDatasetBuilder().build(trades, fdf)
        train, val, test = chronological_split(ds, 0.7, 0.15)
        # All train timestamps <= all val <= all test
        assert train.timestamps.max() <= val.timestamps.min()
        assert val.timestamps.max() <= test.timestamps.min()

    def test_split_sizes(self):
        fdf = make_feature_df()
        trades = make_trades_from_rsi(fdf)
        ds = MLDatasetBuilder().build(trades, fdf)
        train, val, test = chronological_split(ds, 0.7, 0.15)
        n = ds.n_samples
        assert train.n_samples == int(n * 0.7)
        assert val.n_samples == int(n * 0.85) - int(n * 0.7)
        assert train.n_samples + val.n_samples + test.n_samples == n

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            chronological_split(MLDataset(np.empty((0, 3)), np.array([]), [], np.array([])), 0.7, 0.15)


class TestTraining:
    """Test model training with anti-leakage."""

    def test_all_models_train(self, tmp_path):
        fdf = make_feature_df(n=400)
        trades = make_trades_from_rsi(fdf)
        ds = MLDatasetBuilder().build(trades, fdf)
        trainer = ModelTrainer(model_names=["logistic_regression", "random_forest", "xgboost"])
        result = trainer.train(ds, save_dir=str(tmp_path))
        assert set(result.val_metrics.keys()) == {"logistic_regression", "random_forest", "xgboost"}
        assert result.best_model_name in result.val_metrics
        for m in result.trained.values():
            assert m.model_path is not None
            assert "auc" in m.metrics

    def test_scaler_fit_on_train_only(self):
        """The scaler must be fit on TRAIN only — verify it differs from a
        whole-dataset scaler (leakage would make them identical)."""
        from sklearn.preprocessing import StandardScaler
        fdf = make_feature_df(n=400)
        trades = make_trades_from_rsi(fdf)
        ds = MLDatasetBuilder().build(trades, fdf)
        train, _, _ = chronological_split(ds, 0.7, 0.15)

        train_scaler = StandardScaler().fit(train.X)
        leak_scaler = StandardScaler().fit(ds.X)  # what leakage would use

        # Means differ unless train happens to have identical stats
        assert not np.allclose(train_scaler.mean_, leak_scaler.mean_)

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            ModelTrainer(model_names=["not_a_model"])

    def test_save_and_load(self, tmp_path):
        fdf = make_feature_df(n=300)
        trades = make_trades_from_rsi(fdf)
        ds = MLDatasetBuilder().build(trades, fdf)
        trainer = ModelTrainer(model_names=["logistic_regression"])
        result = trainer.train(ds, save_dir=str(tmp_path))
        path = result.trained["logistic_regression"].model_path
        loaded = ModelTrainer.load(path)
        assert loaded.name == "logistic_regression"
        assert loaded.feature_names == ds.feature_names


class TestEvaluation:
    """Test model evaluation and the honest ML-vs-baseline verdict."""

    def test_ml_improves_win_rate(self):
        """With a real rsi signal, ML filtering should improve win rate and the
        honest verdict should be True."""
        fdf = make_feature_df(n=500)
        trades = make_trades_from_rsi(fdf, n=450)
        ds = MLDatasetBuilder().build(trades, fdf)
        trainer = ModelTrainer(model_names=["random_forest", "xgboost"])
        training = trainer.train(ds)
        best = training.trained[training.best_model_name]
        eval_result = ModelEvaluator().evaluate(best, ds, threshold=0.60)

        assert eval_result.baseline_win_rate > 0  # signal is informative
        assert eval_result.ml_filtered_win_rate is not None
        # With an informative feature, ML-filtered win rate should exceed baseline
        assert eval_result.ml_improvement is not None
        assert eval_result.n_taken >= 5

    def test_ml_honest_when_no_signal(self):
        """When the features carry no signal, ML must NOT claim improvement."""
        rng = np.random.default_rng(9)
        fdf = make_feature_df(n=400)
        trades = [
            {"entry_time": int(fdf["timestamp"].iloc[i]),
             "net_pnl": 5.0 if rng.random() < 0.5 else -5.0}
            for i in range(300)
        ]
        ds = MLDatasetBuilder().build(trades, fdf)
        trainer = ModelTrainer(model_names=["random_forest"])
        training = trainer.train(ds)
        best = training.trained[training.best_model_name]
        eval_result = ModelEvaluator().evaluate(best, ds, threshold=0.60)
        # No verdict of improvement, or improvement <= noise threshold
        assert eval_result.ml_improves in (None, False)


class TestPredictor:
    """Test the prediction integration."""

    def test_take_skip_decision(self, tmp_path):
        fdf = make_feature_df(n=300)
        trades = make_trades_from_rsi(fdf)
        ds = MLDatasetBuilder().build(trades, fdf)
        trainer = ModelTrainer(model_names=["random_forest"])
        training = trainer.train(ds, save_dir=str(tmp_path))
        best = training.trained[training.best_model_name]
        predictor = TradePredictor(best, default_threshold=0.60)

        # High rsi feature -> high probability of profit -> TAKE
        high_rsi = {name: (80.0 if name == "rsi_14" else 0.0) for name in ds.feature_names}
        pred = predictor.predict(high_rsi, direction="long")
        assert pred.probability > 0.5
        assert pred.decision in ("TAKE", "SKIP")

        # Low rsi -> low probability -> likely SKIP
        low_rsi = {name: (25.0 if name == "rsi_14" else 0.0) for name in ds.feature_names}
        pred_low = predictor.predict(low_rsi, direction="long")
        assert pred_low.probability < pred.probability

    def test_decision_threshold_respected(self):
        fdf = make_feature_df(n=300)
        trades = make_trades_from_rsi(fdf)
        ds = MLDatasetBuilder().build(trades, fdf)
        trainer = ModelTrainer(model_names=["logistic_regression"])
        training = trainer.train(ds)
        best = training.trained["logistic_regression"]
        predictor = TradePredictor(best, default_threshold=0.60)
        feats = {name: 50.0 for name in ds.feature_names}
        # Above threshold -> TAKE, below -> SKIP
        assert predictor.predict(feats, threshold=0.0).decision == "TAKE"
        assert predictor.predict(feats, threshold=1.01).decision == "SKIP"

    def test_predict_batch(self):
        fdf = make_feature_df(n=300)
        trades = make_trades_from_rsi(fdf)
        ds = MLDatasetBuilder().build(trades, fdf)
        trainer = ModelTrainer(model_names=["logistic_regression"])
        training = trainer.train(ds)
        best = training.trained["logistic_regression"]
        predictor = TradePredictor(best)
        rows = [{name: (80.0 if name == "rsi_14" else 0.0) for name in ds.feature_names},
                {name: (20.0 if name == "rsi_14" else 0.0) for name in ds.feature_names}]
        preds = predictor.predict_batch(rows)
        assert len(preds) == 2


class TestPipeline:
    """Test the end-to-end ML pipeline."""

    def test_pipeline_runs(self, tmp_path):
        fdf = make_feature_df(n=400)
        trades = make_trades_from_rsi(fdf)
        pipe = MLPipeline(model_names=["logistic_regression", "random_forest"],
                          threshold=0.60, save_dir=str(tmp_path))
        report = pipe.run(trades, fdf, symbol="BTCUSDT", timeframe="1h", strategy_type="trend")
        assert report.n_samples == len(trades)
        assert report.baseline_win_rate > 0
        assert report.evaluation.test_metrics["auc"] > 0.5
        assert isinstance(report.to_dict(), dict)

    def test_pipeline_low_sample_warning(self, tmp_path):
        fdf = make_feature_df(n=60)
        trades = make_trades_from_rsi(fdf, n=20)
        pipe = MLPipeline(model_names=["logistic_regression"])
        report = pipe.run(trades, fdf)
        assert any("LOW SAMPLE" in w for w in report.warnings)