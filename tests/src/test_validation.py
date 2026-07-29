"""Unit tests for the validation atoms in :mod:`pipetransport._validation`."""

import numpy as np
import pandas as pd
import pytest

from pipetransport._validation import (
    _validate_no_nan,
    _validate_non_negative,
    _validate_positive,
    _validate_retardation_factor,
    _validate_tedges,
)


class TestValidateTedges:
    def test_matching_parity_passes(self):
        tedges = pd.date_range("2025-06-01", periods=25, freq="h")
        _validate_tedges(tedges, np.zeros(24), tedges_name="tedges", values_name="cin")

    def test_last_axis_is_the_bin_axis(self):
        # A stacked (n_series, n_bins) array must be judged on its last axis only.
        tedges = pd.date_range("2025-06-01", periods=13, freq="h")
        _validate_tedges(tedges, np.zeros((4, 12)), tedges_name="cout_tedges", values_name="cout")

        with pytest.raises(ValueError, match="cout_tedges must have one more element than cout"):
            _validate_tedges(tedges, np.zeros((12, 4)), tedges_name="cout_tedges", values_name="cout")

    def test_too_few_edges_raises(self):
        tedges = pd.date_range("2025-06-01", periods=24, freq="h")
        with pytest.raises(ValueError, match="tedges must have one more element than cin"):
            _validate_tedges(tedges, np.zeros(24), tedges_name="tedges", values_name="cin")

    def test_too_many_edges_raises(self):
        tedges = pd.date_range("2025-06-01", periods=26, freq="h")
        with pytest.raises(ValueError, match="tedges must have one more element than cin"):
            _validate_tedges(tedges, np.zeros(24), tedges_name="tedges", values_name="cin")

    def test_names_appear_verbatim_in_the_message(self):
        tedges = pd.date_range("2025-06-01", periods=3, freq="h")
        with pytest.raises(ValueError, match="my_edges must have one more element than my_values"):
            _validate_tedges(tedges, np.zeros(3), tedges_name="my_edges", values_name="my_values")

    def test_repeated_edge_is_not_strictly_increasing(self):
        # A zero-width bin would divide by zero in the cumulative-volume mapping.
        tedges = pd.DatetimeIndex(["2025-06-01", "2025-06-02", "2025-06-02", "2025-06-03"])
        with pytest.raises(ValueError, match="tedges must be strictly increasing"):
            _validate_tedges(tedges, np.zeros(3), tedges_name="tedges", values_name="cin")

    def test_decreasing_edges_raise(self):
        tedges = pd.DatetimeIndex(["2025-06-01", "2025-06-03", "2025-06-02", "2025-06-04"])
        with pytest.raises(ValueError, match="tedges must be strictly increasing"):
            _validate_tedges(tedges, np.zeros(3), tedges_name="tedges", values_name="cin")

    def test_reversed_edges_raise(self):
        tedges = pd.date_range("2025-06-01", periods=5, freq="D")[::-1]
        with pytest.raises(ValueError, match="tedges must be strictly increasing"):
            _validate_tedges(tedges, np.zeros(4), tedges_name="tedges", values_name="cin")

    def test_non_uniform_but_increasing_edges_pass(self):
        # Only monotonicity is required; the bins may have any widths.
        tedges = pd.DatetimeIndex(["2025-06-01", "2025-06-01 00:00:01", "2025-06-09", "2025-07-01"])
        _validate_tedges(tedges, np.zeros(3), tedges_name="tedges", values_name="cin")

    def test_parity_is_checked_before_monotonicity(self):
        # Both invariants are violated; the parity message is the documented one.
        tedges = pd.DatetimeIndex(["2025-06-02", "2025-06-01"])
        with pytest.raises(ValueError, match="tedges must have one more element than cin"):
            _validate_tedges(tedges, np.zeros(4), tedges_name="tedges", values_name="cin")


class TestValidateNoNan:
    def test_finite_array_passes(self):
        _validate_no_nan(np.array([0.0, -3.5, 1e12]), name="cin")

    def test_nan_anywhere_raises(self):
        with pytest.raises(ValueError, match="cin contains NaN values, which are not allowed"):
            _validate_no_nan(np.array([1.0, np.nan, 3.0]), name="cin")

    def test_nan_in_a_higher_dimensional_array_raises(self):
        arr = np.zeros((3, 4))
        arr[2, 1] = np.nan
        with pytest.raises(ValueError, match="flow contains NaN values"):
            _validate_no_nan(arr, name="flow")

    def test_name_appears_in_the_message(self):
        with pytest.raises(ValueError, match="my_series contains NaN values"):
            _validate_no_nan([np.nan], name="my_series")

    def test_infinities_are_out_of_scope_for_this_atom(self):
        # This atom checks NaN only; magnitude is the business of the (non-)negative atoms.
        _validate_no_nan(np.array([np.inf, -np.inf]), name="cin")

    def test_accepts_list_and_pandas_input(self):
        _validate_no_nan([1.0, 2.0], name="cin")
        _validate_no_nan(pd.Series([1.0, 2.0]), name="cin")
        with pytest.raises(ValueError, match="cin contains NaN values"):
            _validate_no_nan(pd.Series([1.0, np.nan]), name="cin")


class TestValidateNonNegative:
    def test_zero_and_positive_pass(self):
        _validate_non_negative(np.array([0.0, 1.0, 1e9]), name="flow")

    def test_integer_input_passes(self):
        _validate_non_negative(np.array([0, 5, 7]), name="flow")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="flow must be non-negative"):
            _validate_non_negative(np.array([1.0, -1e-15, 2.0]), name="flow")

    def test_nan_is_rejected(self):
        # NaN passes every `< 0` comparison, so a bare inequality would let it through.
        with pytest.raises(ValueError, match="flow must be non-negative"):
            _validate_non_negative(np.array([1.0, np.nan]), name="flow")

    def test_positive_infinity_is_rejected(self):
        # +inf also passes `>= 0`; only the explicit finiteness test catches it.
        with pytest.raises(ValueError, match="flow must be non-negative"):
            _validate_non_negative(np.array([1.0, np.inf]), name="flow")

    def test_negative_infinity_is_rejected(self):
        with pytest.raises(ValueError, match="flow must be non-negative"):
            _validate_non_negative(np.array([1.0, -np.inf]), name="flow")

    def test_higher_dimensional_input(self):
        _validate_non_negative(np.zeros((4, 96)), name="flow")
        bad = np.ones((4, 96))
        bad[3, 95] = -0.5
        with pytest.raises(ValueError, match="flow must be non-negative"):
            _validate_non_negative(bad, name="flow")

    def test_custom_message_is_used_verbatim(self):
        message = "flow must be non-negative (reverse flow not supported)"
        with pytest.raises(ValueError, match=r"flow must be non-negative \(reverse flow not supported\)"):
            _validate_non_negative(np.array([-1.0]), name="flow", message=message)

    def test_custom_message_is_ignored_when_the_check_passes(self):
        _validate_non_negative(np.array([0.0]), name="flow", message="never raised")


class TestValidatePositive:
    def test_strictly_positive_passes(self):
        _validate_positive(np.array([1e-300, 1.0, 1e9]), name="segment volume")

    def test_zero_raises(self):
        # The one behaviour that separates this atom from _validate_non_negative.
        with pytest.raises(ValueError, match="segment volume must be positive"):
            _validate_positive(np.array([1.0, 0.0]), name="segment volume")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="segment diameter must be positive"):
            _validate_positive(np.array([-0.1]), name="segment diameter")

    def test_nan_is_rejected(self):
        # NaN passes every `<= 0` comparison, so a bare inequality would let it through.
        with pytest.raises(ValueError, match="segment length must be positive"):
            _validate_positive(np.array([1.0, np.nan]), name="segment length")

    def test_positive_infinity_is_rejected(self):
        with pytest.raises(ValueError, match="segment length must be positive"):
            _validate_positive(np.array([1.0, np.inf]), name="segment length")

    def test_negative_infinity_is_rejected(self):
        with pytest.raises(ValueError, match="segment length must be positive"):
            _validate_positive(np.array([-np.inf]), name="segment length")

    def test_accepts_a_pandas_column(self):
        _validate_positive(pd.Series([100.0, 40.0, 60.0], name="volume"), name="segment volume")
        with pytest.raises(ValueError, match="segment volume must be positive"):
            _validate_positive(pd.Series([100.0, 0.0]), name="segment volume")

    def test_custom_message_is_used_verbatim(self):
        with pytest.raises(ValueError, match="every pipe needs water in it"):
            _validate_positive(np.array([0.0]), name="segment volume", message="every pipe needs water in it")


class TestValidateRetardationFactor:
    @pytest.mark.parametrize("value", [1.0, 1.0000001, 2.5, 1e6])
    def test_values_at_or_above_one_pass(self, value):
        _validate_retardation_factor(value)

    @pytest.mark.parametrize("value", [0.0, 0.9999999, -1.0])
    def test_values_below_one_raise(self, value):
        # Retardation < 1 would mean the solute outruns the water, which is unphysical here.
        with pytest.raises(ValueError, match=r"retardation_factor must be >= 1\.0"):
            _validate_retardation_factor(value)

    def test_nan_is_rejected(self):
        # `NaN >= 1.0` is False, so the `not value >= 1.0` form catches it where a bare
        # `value < 1.0` would silently propagate an all-NaN transport output.
        with pytest.raises(ValueError, match=r"retardation_factor must be >= 1\.0"):
            _validate_retardation_factor(float("nan"))

    def test_the_boundary_is_inclusive(self):
        # Exactly 1.0 means no sorption at all and must be accepted; the next double below it
        # must not be.
        _validate_retardation_factor(1.0)
        with pytest.raises(ValueError, match=r"retardation_factor must be >= 1\.0"):
            _validate_retardation_factor(np.nextafter(1.0, 0.0))
