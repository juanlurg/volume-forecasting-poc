"""Tests for evaluation metrics module."""

import numpy as np
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
        expected = np.mean([
            2 * 10 / (100 + 110),
            2 * 20 / (200 + 180),
            2 * 5 / (50 + 55),
        ]) * 100

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
