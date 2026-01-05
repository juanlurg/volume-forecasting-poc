"""Tests for evaluation metrics module."""

import numpy as np
import pandas as pd
import pytest


class TestMetrics:
    """Test suite for evaluation metrics functions."""

    def test_mae_calculation(self) -> None:
        """MAE of [10,20,30] vs [13,17,33] should be 3.0."""
        from volume_forecast.evaluation.metrics import mae

        y_true = np.array([10, 20, 30])
        y_pred = np.array([13, 17, 33])

        # Errors: |10-13|=3, |20-17|=3, |30-33|=3
        # MAE = (3 + 3 + 3) / 3 = 3.0
        result = mae(y_true, y_pred)

        assert result == pytest.approx(3.0, rel=1e-9)

    def test_mae_perfect_prediction(self) -> None:
        """MAE should be 0 when predictions are perfect."""
        from volume_forecast.evaluation.metrics import mae

        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])

        result = mae(y_true, y_pred)

        assert result == 0.0

    def test_mae_symmetric(self) -> None:
        """MAE should be symmetric (same for over/under prediction)."""
        from volume_forecast.evaluation.metrics import mae

        y_true = np.array([100, 100, 100])
        y_pred_over = np.array([110, 110, 110])
        y_pred_under = np.array([90, 90, 90])

        mae_over = mae(y_true, y_pred_over)
        mae_under = mae(y_true, y_pred_under)

        assert mae_over == mae_under == 10.0

    def test_rmse_calculation(self) -> None:
        """RMSE should be sqrt(mean(squared_errors))."""
        from volume_forecast.evaluation.metrics import rmse

        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])

        # Errors: (10-12)^2=4, (20-18)^2=4, (30-33)^2=9
        # MSE = (4 + 4 + 9) / 3 = 17/3
        # RMSE = sqrt(17/3) ~= 2.38
        expected_rmse = np.sqrt((4 + 4 + 9) / 3)
        result = rmse(y_true, y_pred)

        assert result == pytest.approx(expected_rmse, rel=1e-9)

    def test_rmse_perfect_prediction(self) -> None:
        """RMSE should be 0 when predictions are perfect."""
        from volume_forecast.evaluation.metrics import rmse

        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])

        result = rmse(y_true, y_pred)

        assert result == 0.0

    def test_rmse_penalizes_large_errors(self) -> None:
        """RMSE should penalize large errors more than MAE."""
        from volume_forecast.evaluation.metrics import mae, rmse

        y_true = np.array([100, 100, 100])
        # One large error vs three small errors
        y_pred_one_large = np.array([100, 100, 130])  # One error of 30
        y_pred_three_small = np.array([110, 110, 110])  # Three errors of 10

        # MAE: one_large = 10, three_small = 10 (same)
        mae_one = mae(y_true, y_pred_one_large)
        mae_three = mae(y_true, y_pred_three_small)
        assert mae_one == mae_three == 10.0

        # RMSE: one_large > three_small (penalizes large error)
        rmse_one = rmse(y_true, y_pred_one_large)
        rmse_three = rmse(y_true, y_pred_three_small)
        assert rmse_one > rmse_three

    def test_mape_calculation(self) -> None:
        """MAPE should handle percentage calculation correctly."""
        from volume_forecast.evaluation.metrics import mape

        y_true = np.array([100, 200, 50])
        y_pred = np.array([110, 180, 55])

        # Percentage errors: |100-110|/100=0.10, |200-180|/200=0.10, |50-55|/50=0.10
        # MAPE = mean([0.10, 0.10, 0.10]) * 100 = 10%
        result = mape(y_true, y_pred)

        assert result == pytest.approx(10.0, rel=1e-9)

    def test_mape_perfect_prediction(self) -> None:
        """MAPE should be 0 when predictions are perfect."""
        from volume_forecast.evaluation.metrics import mape

        y_true = np.array([100, 200, 50])
        y_pred = np.array([100, 200, 50])

        result = mape(y_true, y_pred)

        assert result == 0.0

    def test_mape_handles_asymmetry(self) -> None:
        """MAPE is asymmetric - over vs under predictions differ."""
        from volume_forecast.evaluation.metrics import mape

        y_true = np.array([100.0])
        y_pred_over = np.array([150.0])  # Over by 50%
        y_pred_under = np.array([50.0])  # Under by 50%

        mape_over = mape(y_true, y_pred_over)  # |100-150|/100 = 50%
        mape_under = mape(y_true, y_pred_under)  # |100-50|/100 = 50%

        # Both should be 50%
        assert mape_over == pytest.approx(50.0)
        assert mape_under == pytest.approx(50.0)

    def test_smape_calculation(self) -> None:
        """sMAPE should be symmetric."""
        from volume_forecast.evaluation.metrics import smape

        y_true = np.array([100, 200, 50])
        y_pred = np.array([110, 180, 55])

        # sMAPE = mean(2 * |y - y_hat| / (|y| + |y_hat|)) * 100
        # For (100, 110): 2*10/(100+110) = 20/210 = 0.0952
        # For (200, 180): 2*20/(200+180) = 40/380 = 0.1053
        # For (50, 55): 2*5/(50+55) = 10/105 = 0.0952
        # Mean = (0.0952 + 0.1053 + 0.0952) / 3 = 0.0986 -> 9.86%
        expected = (
            np.mean(
                [
                    2 * 10 / (100 + 110),
                    2 * 20 / (200 + 180),
                    2 * 5 / (50 + 55),
                ]
            )
            * 100
        )

        result = smape(y_true, y_pred)

        assert result == pytest.approx(expected, rel=1e-6)

    def test_smape_perfect_prediction(self) -> None:
        """sMAPE should be 0 when predictions are perfect."""
        from volume_forecast.evaluation.metrics import smape

        y_true = np.array([100, 200, 50])
        y_pred = np.array([100, 200, 50])

        result = smape(y_true, y_pred)

        assert result == 0.0

    def test_smape_is_symmetric(self) -> None:
        """sMAPE should give same result when y_true and y_pred are swapped."""
        from volume_forecast.evaluation.metrics import smape

        y_true = np.array([100.0, 200.0, 50.0])
        y_pred = np.array([110.0, 180.0, 55.0])

        smape_normal = smape(y_true, y_pred)
        smape_swapped = smape(y_pred, y_true)

        assert smape_normal == pytest.approx(smape_swapped)

    def test_smape_bounded_0_to_200(self) -> None:
        """sMAPE should be bounded between 0 and 200%."""
        from volume_forecast.evaluation.metrics import smape

        # Maximum sMAPE case: one value is 0, other is positive
        # sMAPE = 2*|a-0|/(|a|+0) = 2a/a = 2 = 200%
        y_true = np.array([100.0])
        y_pred = np.array([0.0])  # Predict 0 for positive value

        result = smape(y_true, y_pred)

        # Should be 200% (maximum)
        assert result == pytest.approx(200.0)

    def test_calculate_all_metrics(self) -> None:
        """calculate_all_metrics should return dict with all metrics."""
        from volume_forecast.evaluation.metrics import (
            calculate_all_metrics,
            mae,
            mape,
            rmse,
            smape,
        )

        y_true = np.array([100, 200, 50])
        y_pred = np.array([110, 180, 55])

        result = calculate_all_metrics(y_true, y_pred)

        assert isinstance(result, dict)
        assert "mae" in result
        assert "rmse" in result
        assert "mape" in result
        assert "smape" in result

        # Values should match individual function calls
        assert result["mae"] == pytest.approx(mae(y_true, y_pred))
        assert result["rmse"] == pytest.approx(rmse(y_true, y_pred))
        assert result["mape"] == pytest.approx(mape(y_true, y_pred))
        assert result["smape"] == pytest.approx(smape(y_true, y_pred))

    def test_metrics_with_float_arrays(self) -> None:
        """Metrics should work with float arrays."""
        from volume_forecast.evaluation.metrics import mae, rmse

        y_true = np.array([10.5, 20.3, 30.7])
        y_pred = np.array([11.0, 19.8, 31.2])

        mae_result = mae(y_true, y_pred)
        rmse_result = rmse(y_true, y_pred)

        assert isinstance(mae_result, float)
        assert isinstance(rmse_result, float)

    def test_metrics_with_single_element(self) -> None:
        """Metrics should work with single-element arrays."""
        from volume_forecast.evaluation.metrics import mae, mape, rmse, smape

        y_true = np.array([100.0])
        y_pred = np.array([110.0])

        assert mae(y_true, y_pred) == 10.0
        assert rmse(y_true, y_pred) == 10.0
        assert mape(y_true, y_pred) == pytest.approx(10.0)
        assert smape(y_true, y_pred) == pytest.approx(2 * 10 / (100 + 110) * 100)


class TestWalkForwardValidator:
    """Test suite for walk-forward validation."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame with 400 days of data."""
        dates = pd.date_range(start="2023-01-01", periods=400, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "daily_logins": np.random.randint(1000, 5000, size=400),
                "daily_deposits": np.random.randint(100, 500, size=400),
            }
        )

    def test_generates_correct_number_of_folds(self, sample_df: pd.DataFrame) -> None:
        """With 400 days, min_train=365, test=7, step=7 should give ~5 folds.

        Data: 400 days
        min_train: 365 days
        test_size: 7 days
        step_size: 7 days

        First fold: train [0:365], test [365:372] (days 1-365 train, 366-372 test)
        Second fold: train [0:372], test [372:379]
        Third fold: train [0:379], test [379:386]
        Fourth fold: train [0:386], test [386:393]
        Fifth fold: train [0:393], test [393:400]

        After fold 5, next test would be [400:407] which exceeds data, so 5 folds.
        """
        from volume_forecast.evaluation.validation import WalkForwardValidator

        validator = WalkForwardValidator(min_train_size=365, test_size=7, step_size=7)
        n_splits = validator.get_n_splits(sample_df, date_column="date")

        assert n_splits == 5

    def test_train_test_no_overlap(self, sample_df: pd.DataFrame) -> None:
        """Training and test sets should not overlap."""
        from volume_forecast.evaluation.validation import WalkForwardValidator

        validator = WalkForwardValidator(min_train_size=365, test_size=7, step_size=7)

        for train_df, test_df in validator.split(sample_df, date_column="date"):
            train_dates = set(train_df["date"])
            test_dates = set(test_df["date"])

            # No overlap between train and test
            assert train_dates.isdisjoint(test_dates), "Train and test sets overlap!"

    def test_train_expands_each_fold(self, sample_df: pd.DataFrame) -> None:
        """Training set should grow with each fold (expanding window)."""
        from volume_forecast.evaluation.validation import WalkForwardValidator

        validator = WalkForwardValidator(min_train_size=365, test_size=7, step_size=7)

        train_sizes = []
        for train_df, _test_df in validator.split(sample_df, date_column="date"):
            train_sizes.append(len(train_df))

        # Training set should grow each fold
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1], (
                f"Training set did not expand: fold {i - 1} has {train_sizes[i - 1]} samples, "
                f"fold {i} has {train_sizes[i]} samples"
            )

    def test_test_window_matches_horizon(self, sample_df: pd.DataFrame) -> None:
        """Test window should match the horizon (7 days)."""
        from volume_forecast.evaluation.validation import WalkForwardValidator

        validator = WalkForwardValidator(min_train_size=365, test_size=7, step_size=7)

        for _train_df, test_df in validator.split(sample_df, date_column="date"):
            assert len(test_df) == 7, f"Test window has {len(test_df)} days, expected 7"

    def test_split_returns_generator(self, sample_df: pd.DataFrame) -> None:
        """split() should return a generator."""
        from collections.abc import Generator

        from volume_forecast.evaluation.validation import WalkForwardValidator

        validator = WalkForwardValidator(min_train_size=365, test_size=7, step_size=7)
        result = validator.split(sample_df, date_column="date")

        assert isinstance(result, Generator)

    def test_train_precedes_test_chronologically(self, sample_df: pd.DataFrame) -> None:
        """All training dates should be before all test dates."""
        from volume_forecast.evaluation.validation import WalkForwardValidator

        validator = WalkForwardValidator(min_train_size=365, test_size=7, step_size=7)

        for train_df, test_df in validator.split(sample_df, date_column="date"):
            max_train_date = train_df["date"].max()
            min_test_date = test_df["date"].min()

            assert max_train_date < min_test_date, (
                f"Train max date {max_train_date} is not before test min date {min_test_date}"
            )

    def test_default_parameters(self) -> None:
        """Default parameters should be min_train=365, test=7, step=7."""
        from volume_forecast.evaluation.validation import WalkForwardValidator

        validator = WalkForwardValidator()

        assert validator.min_train_size == 365
        assert validator.test_size == 7
        assert validator.step_size == 7

    def test_validate_returns_fold_results(self, sample_df: pd.DataFrame) -> None:
        """validate() should return list of dicts with fold_id, metrics, predictions."""
        from unittest.mock import MagicMock

        from volume_forecast.evaluation.validation import WalkForwardValidator

        # Create mock model
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = sample_df.iloc[:7][["date", "daily_logins"]].copy()
        mock_model.predict.return_value.rename(columns={"daily_logins": "prediction"}, inplace=True)

        validator = WalkForwardValidator(min_train_size=365, test_size=7, step_size=7)
        results = validator.validate(
            mock_model, sample_df, target="daily_logins", date_column="date"
        )

        assert isinstance(results, list)
        assert len(results) == 5  # 5 folds

        # Check structure of each result
        for i, result in enumerate(results):
            assert "fold_id" in result
            assert result["fold_id"] == i
            assert "metrics" in result
            assert "predictions" in result

    def test_insufficient_data_returns_zero_folds(self) -> None:
        """When data is insufficient for even one fold, return 0."""
        import pandas as pd

        from volume_forecast.evaluation.validation import WalkForwardValidator

        # Only 100 days of data, need 365 + 7 = 372 minimum
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        small_df = pd.DataFrame(
            {
                "date": dates,
                "daily_logins": np.random.randint(1000, 5000, size=100),
            }
        )

        validator = WalkForwardValidator(min_train_size=365, test_size=7, step_size=7)
        n_splits = validator.get_n_splits(small_df, date_column="date")

        assert n_splits == 0

    def test_custom_step_size(self) -> None:
        """Custom step_size should change number of folds."""
        import pandas as pd

        from volume_forecast.evaluation.validation import WalkForwardValidator

        # 400 days with step=14 instead of step=7
        dates = pd.date_range(start="2023-01-01", periods=400, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "daily_logins": np.random.randint(1000, 5000, size=400),
            }
        )

        validator = WalkForwardValidator(min_train_size=365, test_size=7, step_size=14)
        n_splits = validator.get_n_splits(df, date_column="date")

        # With step=14:
        # Fold 1: train [0:365], test [365:372] (needs 372 samples)
        # Fold 2: train [0:379], test [379:386] (needs 386 samples)
        # Fold 3: train [0:393], test [393:400] (needs 400 samples) - exact fit!
        # Next fold would need 414 samples, so only 3 folds
        assert n_splits == 3


class TestModelComparison:
    """Test suite for model comparison utilities."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame with 100 days of data for fast testing."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "daily_logins": np.random.randint(1000, 5000, size=100),
            }
        )

    @pytest.fixture
    def validator(self) -> "WalkForwardValidator":
        """Create a validator with small sizes for fast testing."""
        from volume_forecast.evaluation.validation import WalkForwardValidator

        return WalkForwardValidator(min_train_size=50, test_size=7, step_size=7)

    @pytest.fixture
    def models(self) -> list:
        """Create simple baseline models for testing."""
        from volume_forecast.models.baselines import MovingAverageModel, NaiveModel

        return [
            NaiveModel(name="naive"),
            MovingAverageModel(window=7, name="moving_average_7"),
        ]

    def test_benchmark_returns_dataframe(
        self, sample_df: pd.DataFrame, validator: "WalkForwardValidator", models: list
    ) -> None:
        """benchmark() should return DataFrame with results."""
        from volume_forecast.evaluation.comparison import ModelBenchmark

        benchmark = ModelBenchmark(models=models, validator=validator)
        results = benchmark.benchmark(sample_df, target="daily_logins", date_column="date")

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

    def test_benchmark_includes_all_models(
        self, sample_df: pd.DataFrame, validator: "WalkForwardValidator", models: list
    ) -> None:
        """Results should include all provided models."""
        from volume_forecast.evaluation.comparison import ModelBenchmark

        benchmark = ModelBenchmark(models=models, validator=validator)
        results = benchmark.benchmark(sample_df, target="daily_logins", date_column="date")

        assert "model_name" in results.columns
        model_names = results["model_name"].tolist()
        assert "naive" in model_names
        assert "moving_average_7" in model_names
        assert len(model_names) == 2

    def test_benchmark_includes_all_metrics(
        self, sample_df: pd.DataFrame, validator: "WalkForwardValidator", models: list
    ) -> None:
        """Results should include MAE, RMSE, MAPE, sMAPE."""
        from volume_forecast.evaluation.comparison import ModelBenchmark

        benchmark = ModelBenchmark(models=models, validator=validator)
        results = benchmark.benchmark(sample_df, target="daily_logins", date_column="date")

        expected_columns = [
            "model_name",
            "mae_mean",
            "mae_std",
            "rmse_mean",
            "rmse_std",
            "mape_mean",
            "mape_std",
            "smape_mean",
            "smape_std",
        ]
        for col in expected_columns:
            assert col in results.columns, f"Missing column: {col}"

    def test_get_best_model(
        self, sample_df: pd.DataFrame, validator: "WalkForwardValidator", models: list
    ) -> None:
        """get_best_model() should return model with lowest metric."""
        from volume_forecast.evaluation.comparison import ModelBenchmark
        from volume_forecast.models.base import BaseModel

        benchmark = ModelBenchmark(models=models, validator=validator)
        benchmark.benchmark(sample_df, target="daily_logins", date_column="date")

        best_model = benchmark.get_best_model(metric="mae")

        assert isinstance(best_model, BaseModel)
        # The best model should be one of the provided models
        assert best_model.name in ["naive", "moving_average_7"]

    def test_get_best_model_with_different_metrics(
        self, sample_df: pd.DataFrame, validator: "WalkForwardValidator", models: list
    ) -> None:
        """get_best_model() should work with different metrics."""
        from volume_forecast.evaluation.comparison import ModelBenchmark

        benchmark = ModelBenchmark(models=models, validator=validator)
        benchmark.benchmark(sample_df, target="daily_logins", date_column="date")

        # Test with all supported metrics
        for metric in ["mae", "rmse", "mape", "smape"]:
            best_model = benchmark.get_best_model(metric=metric)
            assert best_model.name in ["naive", "moving_average_7"]

    def test_get_results_summary(
        self, sample_df: pd.DataFrame, validator: "WalkForwardValidator", models: list
    ) -> None:
        """get_results_summary() should return the comparison DataFrame."""
        from volume_forecast.evaluation.comparison import ModelBenchmark

        benchmark = ModelBenchmark(models=models, validator=validator)
        benchmark.benchmark(sample_df, target="daily_logins", date_column="date")

        summary = benchmark.get_results_summary()

        assert isinstance(summary, pd.DataFrame)
        assert "model_name" in summary.columns
        assert len(summary) == 2

    def test_get_best_model_raises_before_benchmark(
        self, validator: "WalkForwardValidator", models: list
    ) -> None:
        """get_best_model() should raise error if benchmark not run."""
        from volume_forecast.evaluation.comparison import ModelBenchmark

        benchmark = ModelBenchmark(models=models, validator=validator)

        with pytest.raises(ValueError, match="benchmark"):
            benchmark.get_best_model(metric="mae")

    def test_get_results_summary_raises_before_benchmark(
        self, validator: "WalkForwardValidator", models: list
    ) -> None:
        """get_results_summary() should raise error if benchmark not run."""
        from volume_forecast.evaluation.comparison import ModelBenchmark

        benchmark = ModelBenchmark(models=models, validator=validator)

        with pytest.raises(ValueError, match="benchmark"):
            benchmark.get_results_summary()


class TestWalkForwardValidatorWithFeatures:
    """Test WalkForwardValidator with feature_columns parameter."""

    def test_validator_passes_feature_df(self) -> None:
        """Validator should pass feature data to models that need it."""
        from volume_forecast.evaluation.validation import WalkForwardValidator

        # Create mock model that tracks if it received feature data
        class MockModelWithFeatures:
            def __init__(self):
                self.received_future_df = False

            def fit(self, train_df, target):
                return self

            def predict(self, horizon, future_df=None):
                if future_df is not None:
                    self.received_future_df = True
                return pd.DataFrame({
                    "date": pd.date_range("2024-06-01", periods=horizon),
                    "prediction": [100] * horizon,
                })

        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=400, freq="D"),
            "daily_logins": range(400),
            "feature_col": range(400),
        })

        validator = WalkForwardValidator(min_train_size=365, test_size=7)
        model = MockModelWithFeatures()

        # Run validation with feature columns specified
        results = validator.validate(
            model, df, target="daily_logins",
            feature_columns=["feature_col"],
        )

        assert model.received_future_df
