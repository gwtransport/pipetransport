"""Unit tests for :mod:`pipetransport.utils`."""

import numpy as np
import pandas as pd
import pytest

from pipetransport.utils import (
    compute_time_edges,
    cumulative_flow_volume,
    solve_inverse_transport_banded,
    step_plot_coords,
    tedges_to_days,
)


def _dense_from_banded(band_vals, col_start, n_output):
    """Materialize the dense operator that ``band_vals`` / ``col_start`` encode.

    Row ``k`` of the dense operator is ``band_vals[k]`` placed at columns
    ``[col_start[k], col_start[k] + full_band)``, with columns past ``n_output`` dropped. This
    is the reference definition the banded solver is tested against; it is deliberately a
    naive double loop so it shares no code path with the vectorized scatter under test.
    """
    n_obs, full_band = band_vals.shape
    dense = np.zeros((n_obs, n_output))
    for k in range(n_obs):
        for b in range(full_band):
            col = col_start[k] + b
            if col < n_output:
                dense[k, col] += band_vals[k, b]
    return dense


def _smoother_operator(n):
    """Build a square, symmetric [0.15, 0.7, 0.15] moving average in banded layout.

    Row ``k`` averages columns ``k-1, k, k+1``; the two boundary rows fold the missing
    neighbour weight onto the diagonal so every row sums to exactly one (a moving average
    conserves the mean of a constant signal). The resulting dense operator is strictly
    diagonally dominant, hence invertible and well conditioned (cond ~ 2.5), so a noiseless
    observation vector determines the input uniquely.
    """
    band_vals = np.zeros((n, 3))
    col_start = np.zeros(n, dtype=np.intp)
    band_vals[0] = [0.85, 0.15, 0.0]
    band_vals[1:-1] = [0.15, 0.7, 0.15]
    band_vals[-1] = [0.15, 0.85, 0.0]
    col_start[1:-1] = np.arange(n - 2)
    col_start[-1] = n - 2
    return band_vals, col_start


class TestStepPlotCoords:
    def test_numeric_edges_duplicate_interior_only(self):
        edges = np.array([0.0, 1.0, 3.0, 4.5])
        values = np.array([2.0, 5.0, -1.0])
        x, y = step_plot_coords(edges, values)

        assert x.shape == (2 * values.size,)
        assert y.shape == (2 * values.size,)
        np.testing.assert_allclose(x, [0.0, 1.0, 1.0, 3.0, 3.0, 4.5])
        np.testing.assert_allclose(y, [2.0, 2.0, 5.0, 5.0, -1.0, -1.0])
        # The outer edges appear once, every interior edge twice: that is what draws the riser.
        np.testing.assert_allclose(x[[0, -1]], edges[[0, -1]])

    def test_area_under_step_equals_bin_integral(self):
        # The step curve must enclose exactly the same area as the piecewise-constant series,
        # i.e. sum(value_i * width_i). Any mis-pairing of x and y would break this.
        edges = np.array([0.0, 0.5, 2.0, 2.25, 6.0])
        values = np.array([3.0, -2.0, 7.5, 0.25])
        x, y = step_plot_coords(edges, values)

        np.testing.assert_allclose(np.trapezoid(y, x), np.sum(values * np.diff(edges)))

    def test_dtype_preserved_for_integer_and_float_inputs(self):
        x, y = step_plot_coords(np.array([0, 1, 3], dtype=np.int64), np.array([2, 5], dtype=np.int32))

        assert x.dtype == np.int64
        assert y.dtype == np.int32
        np.testing.assert_array_equal(x, [0, 1, 1, 3])
        np.testing.assert_array_equal(y, [2, 2, 5, 5])

    def test_datetime_edges_keep_datetime_dtype(self):
        edges = pd.date_range("2025-06-01", periods=4, freq="12h")
        values = np.array([1.0, 2.0, 3.0])
        x, y = step_plot_coords(edges, values)

        assert x.dtype == edges.dtype
        assert y.dtype == values.dtype
        expected = edges.to_numpy()[[0, 1, 1, 2, 2, 3]]
        np.testing.assert_array_equal(np.asarray(x), expected)
        np.testing.assert_allclose(y, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

    def test_datetime64_array_edges(self):
        edges = pd.date_range("2025-06-01", periods=3, freq="h").to_numpy()
        x, _ = step_plot_coords(edges, np.array([1.0, 2.0]))

        assert x.dtype == edges.dtype
        np.testing.assert_array_equal(x, edges[[0, 1, 1, 2]])


class TestComputeTimeEdges:
    def test_tedges_passthrough_at_nanosecond_precision(self):
        tedges = pd.date_range("2025-06-01", periods=5, freq="D").as_unit("s")
        out = compute_time_edges(tedges=tedges, number_of_bins=4)

        assert out.dtype == np.dtype("datetime64[ns]")
        pd.testing.assert_index_equal(out, pd.DatetimeIndex(tedges).as_unit("ns"))

    def test_tedges_takes_precedence_over_tstart_and_tend(self):
        tedges = pd.date_range("2025-06-01", periods=4, freq="D")
        decoy = pd.date_range("1999-01-01", periods=3, freq="h")
        out = compute_time_edges(tedges=tedges, tstart=decoy, tend=decoy, number_of_bins=3)

        pd.testing.assert_index_equal(out, tedges.as_unit("ns"))

    def test_tstart_extrapolates_the_trailing_edge_from_the_last_interval(self):
        # Non-uniform on purpose: the documented rule uses the LAST interval only, so the new
        # trailing edge is 2025-06-05 + (2025-06-05 - 2025-06-02) = 2025-06-08.
        tstart = pd.DatetimeIndex(["2025-06-01", "2025-06-02", "2025-06-05"])
        out = compute_time_edges(tstart=tstart, number_of_bins=3)

        expected = pd.DatetimeIndex(["2025-06-01", "2025-06-02", "2025-06-05", "2025-06-08"]).as_unit("ns")
        pd.testing.assert_index_equal(out, expected)
        # The provided starts are reproduced verbatim; only the outer edge is new.
        pd.testing.assert_index_equal(out[:-1], tstart.as_unit("ns"))

    def test_tend_extrapolates_the_leading_edge_from_the_first_interval(self):
        # First interval is one day, so the new leading edge is 2025-06-01 - 1 day.
        tend = pd.DatetimeIndex(["2025-06-01", "2025-06-02", "2025-06-05"])
        out = compute_time_edges(tend=tend, number_of_bins=3)

        expected = pd.DatetimeIndex(["2025-05-31", "2025-06-01", "2025-06-02", "2025-06-05"]).as_unit("ns")
        pd.testing.assert_index_equal(out, expected)
        pd.testing.assert_index_equal(out[1:], tend.as_unit("ns"))

    def test_uniform_grids_round_trip_through_tstart_and_tend(self):
        # For uniformly spaced bins the extrapolation is exact, so both routes must rebuild
        # the very same edge array.
        tedges = pd.date_range("2025-06-01", periods=25, freq="h")
        from_start = compute_time_edges(tstart=tedges[:-1], number_of_bins=24)
        from_end = compute_time_edges(tend=tedges[1:], number_of_bins=24)

        pd.testing.assert_index_equal(from_start, tedges.as_unit("ns"))
        pd.testing.assert_index_equal(from_end, tedges.as_unit("ns"))

    def test_tedges_wrong_length_raises(self):
        tedges = pd.date_range("2025-06-01", periods=5, freq="D")
        with pytest.raises(ValueError, match="tedges must have one more element than number_of_bins"):
            compute_time_edges(tedges=tedges, number_of_bins=5)

    def test_tstart_wrong_length_raises(self):
        tstart = pd.date_range("2025-06-01", periods=5, freq="D")
        with pytest.raises(ValueError, match="tstart must have the same number of elements as number_of_bins"):
            compute_time_edges(tstart=tstart, number_of_bins=4)

    def test_tend_wrong_length_raises(self):
        tend = pd.date_range("2025-06-01", periods=5, freq="D")
        with pytest.raises(ValueError, match="tend must have the same number of elements as number_of_bins"):
            compute_time_edges(tend=tend, number_of_bins=6)

    def test_single_tstart_cannot_infer_bin_width(self):
        tstart = pd.DatetimeIndex(["2025-06-01"])
        with pytest.raises(ValueError, match="tstart must have at least 2 elements to infer the bin width"):
            compute_time_edges(tstart=tstart, number_of_bins=1)

    def test_single_tend_cannot_infer_bin_width(self):
        tend = pd.DatetimeIndex(["2025-06-01"])
        with pytest.raises(ValueError, match="tend must have at least 2 elements to infer the bin width"):
            compute_time_edges(tend=tend, number_of_bins=1)

    def test_nothing_provided_raises(self):
        with pytest.raises(ValueError, match="Either provide tedges, tstart, or tend"):
            compute_time_edges(number_of_bins=3)


class TestTedgesToDays:
    def test_default_reference_is_the_first_edge(self):
        tedges = pd.date_range("2025-06-01", periods=241, freq="h")
        days = tedges_to_days(tedges)

        assert days[0] == 0.0
        # Exact to the last bit: nanosecond counts and 86400e9 are both exactly representable,
        # so the quotient is the correctly rounded value of k/24.
        np.testing.assert_allclose(days, np.arange(241) / 24.0, rtol=0.0, atol=0.0)

    def test_explicit_reference_shifts_the_origin(self):
        tedges = pd.date_range("2025-06-02", periods=3, freq="6h")
        days = tedges_to_days(tedges, ref=pd.Timestamp("2025-06-01"))

        np.testing.assert_allclose(days, [1.0, 1.25, 1.5], rtol=0.0, atol=0.0)

    def test_reference_before_and_after_the_edges(self):
        tedges = pd.date_range("2025-06-10", periods=2, freq="D")
        after = tedges_to_days(tedges, ref=pd.Timestamp("2025-06-12"))

        np.testing.assert_allclose(after, [-2.0, -1.0], rtol=0.0, atol=0.0)

    def test_shared_reference_aligns_two_independent_grids(self):
        # Grid A is hourly from 2025-06-01, grid B is 15-minutely from 2025-06-02 12:00. The
        # instant 2025-06-02 12:00 sits at index 36 of A and index 0 of B; a shared origin must
        # map it to one and the same float on both axes.
        grid_a = pd.date_range("2025-06-01", periods=48, freq="h")
        grid_b = pd.date_range("2025-06-02 12:00", periods=8, freq="15min")
        ref = grid_a[0]
        days_a = tedges_to_days(grid_a, ref=ref)
        days_b = tedges_to_days(grid_b, ref=ref)

        assert grid_a[36] == grid_b[0]
        np.testing.assert_allclose(days_b[0], days_a[36], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(days_b[0], 1.5, rtol=0.0, atol=0.0)
        # Without the shared reference both grids start at zero and the alignment is lost.
        assert tedges_to_days(grid_b)[0] == tedges_to_days(grid_a)[0] == 0.0

    def test_a_shifted_reference_is_a_pure_offset(self):
        tedges = pd.date_range("2025-06-01", periods=97, freq="h")
        ref = pd.Timestamp("2020-01-01")
        without = tedges_to_days(tedges)
        with_ref = tedges_to_days(tedges, ref=ref)
        offset = (tedges[0] - ref) / pd.Timedelta(days=1)

        # Precision floor: the far origin lifts every value to ~1978 days, whose ulp is 2.3e-13,
        # so the shifted axis resolves the hourly spacing only to that absolute step. This is why
        # the package defaults ref to tedges[0] and only ever shares an origin between two grids
        # that overlap.
        floor = 2.0 * np.spacing(np.max(with_ref))
        assert floor < 1e-12
        np.testing.assert_allclose(with_ref - offset, without, rtol=0.0, atol=floor)
        np.testing.assert_allclose(np.diff(with_ref), np.diff(without), rtol=0.0, atol=floor)


class TestCumulativeFlowVolume:
    def test_leading_zero_and_exact_partial_sums(self):
        flow = np.array([100.0, 50.0, 0.0, 25.0])
        dt_days = np.array([1.0, 2.0, 0.5, 4.0])
        volume = cumulative_flow_volume(flow, dt_days)

        assert volume.shape == (flow.size + 1,)
        assert volume[0] == 0.0
        np.testing.assert_allclose(volume, [0.0, 100.0, 200.0, 200.0, 300.0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(volume[-1], np.sum(flow * dt_days), rtol=1e-15)

    def test_constant_flow_is_linear_in_time(self):
        q = 240.0
        dt_days = np.full(24, 1.0 / 24.0)
        volume = cumulative_flow_volume(np.full(24, q), dt_days)

        np.testing.assert_allclose(volume, q * np.concatenate([[0.0], np.cumsum(dt_days)]), rtol=1e-14)

    def test_volume_axis_inverts_to_the_analytic_travel_time(self):
        # Physics: a parcel entering a pipe of volume V at time s leaves at t where the
        # displaced volume C(t) - C(s) equals V. With Q = 100 m³/day on day one and
        # 200 m³/day on day two, a V = 150 m³ pipe is cleared 100 m³ into day one and the
        # remaining 50 m³ at 200 m³/day, i.e. 1 + 50/200 = 1.25 days after entry.
        flow = np.array([100.0, 200.0])
        dt_days = np.array([1.0, 1.0])
        volume = cumulative_flow_volume(flow, dt_days)
        edges_days = np.concatenate([[0.0], np.cumsum(dt_days)])

        exit_day = np.interp(150.0, volume, edges_days)
        np.testing.assert_allclose(exit_day, 1.25, rtol=1e-15)

    def test_leading_axes_broadcast(self):
        rng = np.random.default_rng(4)
        flow = rng.uniform(10.0, 200.0, size=(3, 5, 12))
        dt_days = rng.uniform(0.1, 2.0, size=12)
        volume = cumulative_flow_volume(flow, dt_days)

        assert volume.shape == (3, 5, 13)
        np.testing.assert_allclose(volume[..., 0], 0.0, rtol=0.0, atol=0.0)
        for i in range(3):
            for j in range(5):
                np.testing.assert_allclose(
                    volume[i, j], cumulative_flow_volume(flow[i, j], dt_days), rtol=0.0, atol=0.0
                )

    def test_zero_flow_leaves_plateaus_when_not_requested(self):
        flow = np.array([100.0, 0.0, 0.0, 50.0])
        volume = cumulative_flow_volume(flow, np.ones(4))

        np.testing.assert_allclose(volume, [0.0, 100.0, 100.0, 100.0, 150.0], rtol=0.0, atol=0.0)
        assert np.count_nonzero(np.diff(volume) == 0.0) == 2

    def test_strictly_monotone_separates_plateaus_without_moving_real_values(self):
        flow = np.array([100.0, 0.0, 0.0, 0.0, 50.0, 0.0, 80.0])
        dt_days = np.full(7, 0.5)
        raw = cumulative_flow_volume(flow, dt_days)
        bumped = cumulative_flow_volume(flow, dt_days, strictly_monotone=True)

        assert np.all(np.diff(bumped) > 0.0)
        # Genuine (non-duplicate) entries are untouched, bit for bit.
        is_dup = np.concatenate([[False], np.diff(raw) == 0.0])
        np.testing.assert_array_equal(bumped[~is_dup], raw[~is_dup])
        # The perturbation is pure numerical hygiene, far below physical relevance.
        assert np.max(np.abs(bumped - raw)) / raw.max() < 1e-12
        # Ordering relative to the surrounding genuine values is preserved.
        assert np.all(bumped[is_dup] > raw[0])
        assert bumped[-2] < raw[-1]

    def test_strictly_monotone_is_a_no_op_without_plateaus(self):
        flow = np.array([100.0, 50.0, 25.0])
        dt_days = np.array([1.0, 2.0, 0.25])

        np.testing.assert_array_equal(
            cumulative_flow_volume(flow, dt_days, strictly_monotone=True),
            cumulative_flow_volume(flow, dt_days),
        )

    def test_long_plateau_run_with_a_tight_following_gap(self):
        # 60 consecutive zero-flow bins land on one plateau, and the very next bin advances the
        # volume by only 100 ulps. An uncapped 16-ulp-per-duplicate bump would overshoot that
        # neighbour by a factor ~19 and destroy monotonicity; the per-run cap must shrink the
        # step to gap / (run_len + 1) instead.
        n_run = 60
        tiny = 100.0 * np.finfo(float).eps
        flow = np.concatenate([[1.0], np.zeros(n_run), [tiny, 1.0]])
        dt_days = np.ones(flow.size)
        raw = cumulative_flow_volume(flow, dt_days)
        bumped = cumulative_flow_volume(flow, dt_days, strictly_monotone=True)

        assert np.count_nonzero(np.diff(raw) == 0.0) == n_run
        assert np.all(np.diff(bumped) > 0.0)
        # The whole run stays strictly below the tight successor that follows it.
        successor = raw[n_run + 2]
        assert np.max(bumped[1 : n_run + 2]) < successor
        assert np.max(np.abs(bumped - raw)) / raw.max() < 1e-12

    def test_strictly_monotone_on_an_all_zero_record(self):
        # Degenerate but reachable: no water drawn at all. The scale-free fallback still has to
        # return a strictly increasing axis, here in subnormal steps around zero.
        volume = cumulative_flow_volume(np.zeros(6), np.ones(6), strictly_monotone=True)

        assert volume[0] == 0.0
        assert np.all(np.diff(volume) > 0.0)

    def test_strictly_monotone_batches_rows_bit_for_bit(self):
        # Rows with plateaus, without, and all-zero, monotonized in one 2-D call, must equal
        # the row-by-row 1-D calls exactly: the batched transfer operator relies on it.
        flow = np.array([
            [100.0, 0.0, 0.0, 50.0, 0.0, 80.0],
            [100.0, 50.0, 25.0, 10.0, 5.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        dt_days = np.full(6, 0.5)
        batched = cumulative_flow_volume(flow, dt_days, strictly_monotone=True)

        assert np.all(np.diff(batched, axis=-1) > 0.0)
        for row in range(flow.shape[0]):
            np.testing.assert_array_equal(
                batched[row], cumulative_flow_volume(flow[row], dt_days, strictly_monotone=True)
            )


class TestSolveInverseTransportBanded:
    def test_recovers_the_exact_input_from_noiseless_observations(self):
        n = 32
        band_vals, col_start = _smoother_operator(n)
        dense = _dense_from_banded(band_vals, col_start, n)
        x_true = 1.0 + np.sin(0.7 * np.arange(n)) + 0.3 * np.arange(n) / n
        observed = dense @ x_true

        recovered = solve_inverse_transport_banded(
            band_vals=band_vals,
            col_start=col_start,
            observed=observed,
            n_output=n,
            regularization_strength=1e-12,
        )

        assert not np.any(np.isnan(recovered))
        # Precision floor: the Tikhonov pull toward x_target biases the answer by
        # O(lambda * ||(WᵀW)⁻¹ (x_true - x_target)||), which is ~3e-13 here at lambda = 1e-12.
        np.testing.assert_allclose(recovered, x_true, rtol=0.0, atol=1e-11)

    def test_bias_scales_linearly_with_the_regularization_strength(self):
        # Confirms the residual error above really is the lambda-bias and not a solver
        # inaccuracy: multiplying lambda by 100 must multiply the error by ~100.
        n = 32
        band_vals, col_start = _smoother_operator(n)
        dense = _dense_from_banded(band_vals, col_start, n)
        x_true = 1.0 + np.sin(0.7 * np.arange(n))
        observed = dense @ x_true

        errors = [
            np.max(
                np.abs(
                    solve_inverse_transport_banded(
                        band_vals=band_vals,
                        col_start=col_start,
                        observed=observed,
                        n_output=n,
                        regularization_strength=lam,
                    )
                    - x_true
                )
            )
            for lam in (1e-10, 1e-8)
        ]

        np.testing.assert_allclose(errors[1] / errors[0], 100.0, rtol=1e-3)

    def test_matches_a_dense_tikhonov_reference_on_an_ill_posed_operator(self):
        # 3-wide moving average, deliberately rank-deficient (22 rows for 24 unknowns), so
        # lambda genuinely shapes the answer and the regularization target semantics are
        # exercised rather than washed out. Each row additionally carries its own surviving
        # fraction: with rows summing to exactly one the two normalizations of the target are
        # bit-identical, and the reference would agree with either convention.
        n = 24
        n_obs = n - 2
        survival = np.exp(-0.05 * np.arange(n_obs))
        band_vals = np.full((n_obs, 3), 1.0 / 3.0) * survival[:, None]
        col_start = np.arange(n_obs, dtype=np.intp)
        dense = _dense_from_banded(band_vals, col_start, n)
        x_true = 2.0 + np.cos(0.9 * np.arange(n))
        observed = dense @ x_true
        lam = 1e-3

        recovered = solve_inverse_transport_banded(
            band_vals=band_vals,
            col_start=col_start,
            observed=observed,
            n_output=n,
            regularization_strength=lam,
        )

        # Independent dense assembly of (WᵀW + lambda I) x = Wᵀ observed + lambda x_target with
        # x_target the transpose-and-normalize of W applied to observed, normalized by the
        # survival-weighted column totals so that it preserves constants.
        surv_sum = dense.T @ dense.sum(axis=1)
        assert np.all(surv_sum > 0.0)
        x_target = (dense.T @ observed) / surv_sum
        lhs = dense.T @ dense + lam * np.eye(n)
        expected = np.linalg.solve(lhs, dense.T @ observed + lam * x_target)

        np.testing.assert_allclose(recovered, expected, rtol=1e-9, atol=1e-11)
        # The reference is not simply x_true: the ill-posed inverse is visibly regularized, so
        # agreement above is a real check on the solver rather than on the data.
        assert np.max(np.abs(expected - x_true)) > 1e-3
        # ... and the two normalizations really do differ here, so the assertion above
        # discriminates the convention instead of passing under either.
        stale = np.linalg.solve(lhs, dense.T @ observed + lam * (dense.T @ observed) / dense.sum(axis=0))
        assert np.max(np.abs(stale - expected)) > 1e-6

    def test_regularization_target_is_reached_when_the_data_are_uninformative(self):
        # A single observation over three columns leaves two directions unconstrained. The
        # answer must be the contribution-weighted average of the output bins each input fed --
        # here the lone observation itself -- and it must still reproduce that observation.
        band_vals = np.array([[1 / 3, 1 / 3, 1 / 3]])
        col_start = np.array([0], dtype=np.intp)
        observed = np.array([5.0])

        recovered = solve_inverse_transport_banded(
            band_vals=band_vals,
            col_start=col_start,
            observed=observed,
            n_output=3,
            regularization_strength=1e-6,
        )

        np.testing.assert_allclose(recovered, [5.0, 5.0, 5.0], rtol=1e-12)
        np.testing.assert_allclose(band_vals @ recovered, observed, rtol=1e-12)

    @pytest.mark.parametrize("survival", [1.0, 0.6, 0.2])
    @pytest.mark.parametrize("lam", [1e-10, 1e-4, 1e-2, 1.0])
    def test_a_constant_input_survives_a_decayed_operator_at_every_lambda(self, survival, lam):
        """A decayed operator must not drag the answer toward the value it delivered.

        ``W = s I`` is the whole defect in one row: every row sums to the surviving fraction
        ``s``, so a constant input ``c`` is observed as ``s c``. Normalizing the regularization
        target by the plain column sums evaluates it at ``s c`` -- the *delivered* quality --
        and the solve returns ``c s (s + lam) / (s**2 + lam)``, which equals ``c`` for every
        ``lam`` only at ``s = 1``. At ``s = 0.2, lam = 1`` that is ``0.23 c``. The target has to
        preserve constants for the truth to annihilate both terms of the objective, which is
        what makes it the exact minimizer independently of ``lam``.
        """
        n = 50
        constant = 3.7
        band_vals = np.full((n, 1), survival)
        col_start = np.arange(n, dtype=np.intp)

        recovered = solve_inverse_transport_banded(
            band_vals=band_vals,
            col_start=col_start,
            observed=np.full(n, survival * constant),
            n_output=n,
            regularization_strength=lam,
        )

        np.testing.assert_allclose(recovered, constant, rtol=0.0, atol=1e-12)

    def test_stacked_operators_with_unsorted_rows(self):
        # Two operators with different bandwidths and different col_start conventions, stacked
        # into one system and then row-shuffled. The result must be permutation invariant and
        # still recover the exact input, since both blocks are consistent with it.
        n = 32
        band_a, col_start_a = _smoother_operator(n)
        # Second block: 2-wide 0.4/0.6 mix, zero-padded to the common band width of 3.
        band_b = np.zeros((n - 1, 3))
        band_b[:, 0] = 0.4
        band_b[:, 1] = 0.6
        col_start_b = np.arange(n - 1, dtype=np.intp)

        band_vals = np.concatenate([band_a, band_b])
        col_start = np.concatenate([col_start_a, col_start_b])
        assert not np.all(np.diff(col_start) >= 0)

        dense = _dense_from_banded(band_vals, col_start, n)
        x_true = 1.0 + np.sin(0.7 * np.arange(n)) + 0.3 * np.arange(n) / n
        observed = dense @ x_true

        stacked = solve_inverse_transport_banded(
            band_vals=band_vals,
            col_start=col_start,
            observed=observed,
            n_output=n,
            regularization_strength=1e-12,
        )
        rng = np.random.default_rng(0)
        perm = rng.permutation(observed.size)
        shuffled = solve_inverse_transport_banded(
            band_vals=band_vals[perm],
            col_start=col_start[perm],
            observed=observed[perm],
            n_output=n,
            regularization_strength=1e-12,
        )

        np.testing.assert_allclose(stacked, x_true, rtol=0.0, atol=1e-11)
        # Only the summation order changes, so the two agree to a few ulps.
        np.testing.assert_allclose(shuffled, stacked, rtol=1e-12, atol=1e-12)

    def test_nan_observations_drop_out_of_the_system(self):
        # Rows whose observation is NaN must contribute nothing at all -- not even through
        # their band weights. Here the extra rows carry weights six orders of magnitude larger
        # than the real ones, so any leakage would dominate the answer.
        n = 32
        band_vals, col_start = _smoother_operator(n)
        dense = _dense_from_banded(band_vals, col_start, n)
        x_true = 1.0 + np.sin(0.7 * np.arange(n))
        observed = dense @ x_true

        clean = solve_inverse_transport_banded(
            band_vals=band_vals,
            col_start=col_start,
            observed=observed,
            n_output=n,
            regularization_strength=1e-9,
        )
        with_gaps = solve_inverse_transport_banded(
            band_vals=np.concatenate([band_vals, np.full((5, 3), 1e6)]),
            col_start=np.concatenate([col_start, np.arange(5, dtype=np.intp)]),
            observed=np.concatenate([observed, np.full(5, np.nan)]),
            n_output=n,
            regularization_strength=1e-9,
        )

        np.testing.assert_allclose(with_gaps, clean, rtol=1e-14, atol=0.0)

    def test_gapped_rows_reduce_to_the_subsystem_that_remains(self):
        # Dropping the observation of an interior row must give exactly the same answer as
        # never supplying that row at all.
        n = 20
        band_vals, col_start = _smoother_operator(n)
        dense = _dense_from_banded(band_vals, col_start, n)
        observed = dense @ (1.0 + np.cos(0.4 * np.arange(n)))
        gap = 7

        keep = np.ones(n, dtype=bool)
        keep[gap] = False
        without_row = solve_inverse_transport_banded(
            band_vals=band_vals[keep],
            col_start=col_start[keep],
            observed=observed[keep],
            n_output=n,
            regularization_strength=1e-6,
        )
        nan_row = observed.copy()
        nan_row[gap] = np.nan
        with_nan = solve_inverse_transport_banded(
            band_vals=band_vals,
            col_start=col_start,
            observed=nan_row,
            n_output=n,
            regularization_strength=1e-6,
        )

        np.testing.assert_allclose(with_nan, without_row, rtol=1e-12, atol=1e-14)

    def test_unconstrained_columns_come_back_nan(self):
        n = 32
        band_vals, col_start = _smoother_operator(n)
        dense = _dense_from_banded(band_vals, col_start, n)
        x_true = 1.0 + np.sin(0.7 * np.arange(n))
        observed = dense @ x_true

        # Three trailing output columns no row ever touches.
        recovered = solve_inverse_transport_banded(
            band_vals=band_vals,
            col_start=col_start,
            observed=observed,
            n_output=n + 3,
            regularization_strength=1e-12,
        )

        assert recovered.shape == (n + 3,)
        assert np.all(np.isnan(recovered[n:]))
        # The constrained part is unaffected by the dead columns hanging off the end.
        np.testing.assert_allclose(recovered[:n], x_true, rtol=0.0, atol=1e-11)

    def test_all_observations_gapped_returns_all_nan(self):
        n = 8
        band_vals, col_start = _smoother_operator(n)

        recovered = solve_inverse_transport_banded(
            band_vals=band_vals,
            col_start=col_start,
            observed=np.full(n, np.nan),
            n_output=n,
            regularization_strength=1e-6,
        )

        assert np.all(np.isnan(recovered))

    @pytest.mark.parametrize("lam", [0.0, -1e-12, -1.0])
    def test_non_positive_regularization_strength_raises(self, lam):
        band_vals, col_start = _smoother_operator(4)
        with pytest.raises(ValueError, match="regularization_strength must be > 0"):
            solve_inverse_transport_banded(
                band_vals=band_vals,
                col_start=col_start,
                observed=np.ones(4),
                n_output=4,
                regularization_strength=lam,
            )
