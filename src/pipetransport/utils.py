"""
General utilities for pipe-network water quality modelling.

Time-axis construction, cumulative-volume accumulation, step-plot coordinates and the banded
Tikhonov solver that backs the reverse (deconvolution) direction. Every transport quantity in
this package is expressed on a cumulative-volume axis rather than on a time axis, so the two
helpers that build that axis -- :func:`tedges_to_days` and :func:`cumulative_flow_volume` --
are used by every module.

Available functions:

- :func:`step_plot_coords` - Expand bin edges (n+1) and bin-averaged values (n) into paired x/y arrays of 2n
  points each, so that ``ax.plot(x, y)`` draws the piecewise-constant series as a step function. Edges may be
  numeric or datetime; each output keeps the dtype of its input.

- :func:`compute_time_edges` - Build the n+1 bin edges as a nanosecond-precision DatetimeIndex from exactly one
  of explicit edges, per-bin start times, or per-bin end times, validating the length against ``number_of_bins``.
  From ``tstart`` or ``tend`` the single missing outer edge is extrapolated from the adjacent interval alone, so
  pass ``tedges`` directly when the bins are not uniformly spaced.

- :func:`tedges_to_days` - Convert a DatetimeIndex of bin edges to float64 days relative to a reference
  timestamp. The ``ref`` keyword is load-bearing: two edge arrays that are compared or interpolated against
  each other must share one origin.

- :func:`cumulative_flow_volume` - Accumulate per-bin flow rates times bin widths into the cumulative volume at
  every bin edge (n+1 values, starting at zero). With ``strictly_monotone=True`` the plateaus left by zero-flow
  bins are bumped by a few ulps, which is required before inverting the sequence from volume back to time.

- :func:`solve_inverse_transport_banded` - Recover the input signal of a forward operator stored in banded
  layout (row ``k`` is ``band_vals[k]`` placed at column ``col_start[k]``) through banded Cholesky normal
  equations plus corrected semi-normal refinement. The factorization, solve and refinement stay at
  ``O(n_output * full_band)``. The regularization strength must be strictly positive, since it is what makes
  the banded factor positive definite.

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.linalg import cho_solve_banded, cholesky_banded

# Numerical tolerance for a column/row weight sum to count as constrained.
_EPSILON_COEFF_SUM = 1e-10

# Corrected semi-normal-equation refinement steps in solve_inverse_transport_banded. One
# step reaches the QR-accurate solution; a second is a cheap, stable safety margin.
_BANDED_REFINEMENT_STEPS = 2

# Safety factor in ulps when separating plateaus; see _make_strictly_monotone.
_DUP_BUMP_ULPS = 16


def step_plot_coords(edges: npt.ArrayLike, values: npt.ArrayLike) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute step-plot coordinates from bin edges and bin-averaged values.

    Converts bin edges (n+1) and bin values (n) into paired x/y arrays suitable for plotting
    piecewise-constant (step) series with ``ax.plot(x, y)``.

    Parameters
    ----------
    edges : array-like
        Bin edges (n+1 elements for n bins). Numeric, datetime, or any type accepted by
        :func:`numpy.repeat`.
    values : array-like
        Bin-averaged values (n elements), one per bin.

    Returns
    -------
    x : ndarray
        Step x-coordinates (2n elements). Same dtype as *edges*.
    y : ndarray
        Step y-coordinates (2n elements). Same dtype as *values*.

    Examples
    --------
    >>> import numpy as np
    >>> from pipetransport.utils import step_plot_coords
    >>> x, y = step_plot_coords(np.array([0.0, 1.0, 3.0]), np.array([2.0, 5.0]))
    >>> x
    array([0., 1., 1., 3.])
    >>> y
    array([2., 2., 5., 5.])
    """
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(values, 2)
    return x, y


def compute_time_edges(
    *,
    tedges: pd.DatetimeIndex | None = None,
    tstart: pd.DatetimeIndex | None = None,
    tend: pd.DatetimeIndex | None = None,
    number_of_bins: int,
) -> pd.DatetimeIndex:
    """
    Build the n+1 time-bin edges from explicit edges, per-bin start times, or per-bin end times.

    Provide exactly one of ``tedges``, ``tstart`` or ``tend`` and leave the others at ``None``.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex or None, optional
        Explicit time edges. Must have ``number_of_bins + 1`` elements. Takes precedence over
        ``tstart`` and ``tend``.
    tstart : pandas.DatetimeIndex or None, optional
        Start time of each bin, ``number_of_bins`` elements. Used when ``tedges`` is None.
    tend : pandas.DatetimeIndex or None, optional
        End time of each bin, ``number_of_bins`` elements. Used when both others are None.
    number_of_bins : int
        Expected number of bins, validated against the provided argument.

    Returns
    -------
    pandas.DatetimeIndex
        Time edges with one more element than ``number_of_bins``, at nanosecond precision.

    Raises
    ------
    ValueError
        If the provided argument has the wrong length, if ``tstart`` / ``tend`` hold fewer
        than two elements (the bin width cannot be inferred), or if none is provided.

    Notes
    -----
    From ``tstart`` the final edge is extrapolated with the width of the last interval, and
    from ``tend`` the first edge with the width of the first interval. For non-uniformly
    spaced bins that single interval is unrelated to the rest of the series, so pass
    ``tedges`` directly whenever the bin widths vary.

    Examples
    --------
    >>> import pandas as pd
    >>> from pipetransport.utils import compute_time_edges
    >>> days = pd.date_range("2025-06-01", periods=3, freq="D")
    >>> compute_time_edges(tend=days, number_of_bins=3)
    DatetimeIndex(['2025-05-31', '2025-06-01', '2025-06-02', '2025-06-03'], dtype='datetime64[ns]', freq=None)
    """
    if tedges is not None:
        if number_of_bins != len(tedges) - 1:
            msg = "tedges must have one more element than number_of_bins"
            raise ValueError(msg)
        return pd.DatetimeIndex(tedges).as_unit("ns")

    if tstart is not None:
        tstart = pd.DatetimeIndex(tstart).as_unit("ns")
        if number_of_bins != len(tstart):
            msg = "tstart must have the same number of elements as number_of_bins"
            raise ValueError(msg)
        if len(tstart) < 2:  # noqa: PLR2004
            msg = "tstart must have at least 2 elements to infer the bin width; pass tedges for a single bin"
            raise ValueError(msg)
        return pd.DatetimeIndex([*list(tstart), tstart[-1] + (tstart[-1] - tstart[-2])], dtype=tstart.dtype)

    if tend is not None:
        tend = pd.DatetimeIndex(tend).as_unit("ns")
        if number_of_bins != len(tend):
            msg = "tend must have the same number of elements as number_of_bins"
            raise ValueError(msg)
        if len(tend) < 2:  # noqa: PLR2004
            msg = "tend must have at least 2 elements to infer the bin width; pass tedges for a single bin"
            raise ValueError(msg)
        return pd.DatetimeIndex([tend[0] - (tend[1] - tend[0]), *list(tend)], dtype=tend.dtype)

    msg = "Either provide tedges, tstart, or tend"
    raise ValueError(msg)


def tedges_to_days(tedges: pd.DatetimeIndex, *, ref: pd.Timestamp | None = None) -> npt.NDArray[np.floating]:
    """Convert time-bin edges to float64 days relative to a reference timestamp.

    Parameters
    ----------
    tedges : DatetimeIndex
        Time-bin edges to convert.
    ref : Timestamp or None, optional
        Reference timestamp mapped to day zero. Defaults to ``tedges[0]``. Pass a shared
        reference whenever a second edge array must align to the same origin.

    Returns
    -------
    ndarray
        Days since ``ref``, one value per edge.

    Examples
    --------
    >>> import pandas as pd
    >>> from pipetransport.utils import tedges_to_days
    >>> tedges_to_days(pd.date_range("2025-06-01", periods=3, freq="12h"))
    array([0. , 0.5, 1. ])
    """
    origin = tedges[0] if ref is None else ref
    return ((tedges - origin) / pd.Timedelta(days=1)).to_numpy(dtype=float)


def _make_strictly_monotone(arr: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Bump consecutive duplicates so a non-decreasing array becomes strictly monotone.

    Returns the input unchanged when it holds no consecutive duplicates. Otherwise each
    duplicate is bumped by ``k * step``, with ``k`` its 1-based position inside the duplicate
    run and ``step = min(16 * ulp(max(arr)), gap / (run_len + 1))``. The cap keeps the largest
    bump strictly below the next genuine value above the plateau; a gap narrower than the run
    length in ulps is unrepresentable and cannot be separated.

    The factor 16 is a margin against IEEE 754 rounding noise in :func:`numpy.interp`'s linear
    arithmetic, which differs subtly between x86_64 (with FMA) and ARM. A 1-ulp gap, while
    strictly monotone, can place a downstream query on the wrong side of a bracket boundary
    when the intermediate arithmetic rounds one ulp away from the exact value. The
    perturbation is relative to the array scale (``~3.5e-15 * max(arr)``), far below physical
    relevance.

    Parameters
    ----------
    arr : array-like
        1D non-decreasing array, e.g. a cumulative volume holding plateaus from zero-flow bins.

    Returns
    -------
    ndarray
        Strictly monotone array of the same length.

    Notes
    -----
    Use this before passing ``arr`` as the reference x of a volume-to-time inversion.
    Plateaus make the inverse multi-valued, and :func:`numpy.interp` would silently pick one
    of the two limits, biasing integrals over bins that span the kink.
    """
    arr = np.asarray(arr, dtype=float)
    diffs = np.diff(arr)
    if not np.any(diffs == 0):
        return arr
    ulp_max = np.nextafter(arr.max(), np.inf) - arr.max()
    n = len(arr)
    idx = np.arange(n)
    is_dup = np.concatenate(([False], diffs == 0))
    # 1-based position of each duplicate within its consecutive run.
    last_nondup = np.maximum.accumulate(np.where(is_dup, -1, idx))
    cumcount = np.where(is_dup, idx - last_nondup, 0)

    # Per-run headroom: each bumped value must stay strictly below the next genuine value above
    # the plateau, otherwise a long run can overshoot a closely-spaced successor. ``n`` marks a
    # run reaching the array end, where there is no successor and hence no overshoot risk.
    next_nondup_idx = np.minimum.accumulate(np.where(is_dup, n, idx)[::-1])[::-1]
    has_successor = next_nondup_idx < n
    gap_to_next = arr[np.clip(next_nondup_idx, 0, n - 1)] - arr[idx]
    run_len = next_nondup_idx - last_nondup - 1
    full_step = _DUP_BUMP_ULPS * ulp_max
    with np.errstate(invalid="ignore", divide="ignore"):
        capped_step = np.where(has_successor, np.minimum(full_step, gap_to_next / (run_len + 1.0)), full_step)
    return arr + np.where(is_dup, cumcount * capped_step, 0.0)


def cumulative_flow_volume(
    flow: npt.ArrayLike, dt_days: npt.ArrayLike, *, strictly_monotone: bool = False
) -> npt.NDArray[np.floating]:
    """Cumulative throughflow volume from per-bin flow rates.

    Multiplies each per-bin flow rate by its bin width and accumulates, with a leading zero
    prepended so the result carries one entry per bin edge (n+1 values for n bins).

    Parameters
    ----------
    flow : array-like
        Flow rate per bin [m³/day]. The last axis holds the n bins; leading axes broadcast.
    dt_days : array-like
        Bin widths in days, length n.
    strictly_monotone : bool, optional
        When ``True``, separate the plateaus left by zero-flow bins with a few-ulp bump so the
        result can be inverted from volume back to time; without it that inverse is
        multi-valued. Only supported for 1D ``flow``. Default is ``False``.

    Returns
    -------
    ndarray
        Cumulative volume at each edge (n+1 values along the last axis), starting at zero.

    Examples
    --------
    >>> import numpy as np
    >>> from pipetransport.utils import cumulative_flow_volume
    >>> cumulative_flow_volume(np.array([100.0, 50.0]), np.array([1.0, 2.0]))
    array([  0., 100., 200.])
    """
    flow = np.asarray(flow, dtype=float)
    volume = np.cumsum(flow * np.asarray(dt_days, dtype=float), axis=-1)
    zero = np.zeros((*volume.shape[:-1], 1))
    cumulative = np.concatenate([zero, volume], axis=-1)
    return _make_strictly_monotone(cumulative) if strictly_monotone else cumulative


def solve_inverse_transport_banded(
    *,
    band_vals: npt.NDArray[np.floating],
    col_start: npt.NDArray[np.intp],
    observed: npt.NDArray[np.floating],
    n_output: int,
    regularization_strength: float,
) -> npt.NDArray[np.floating]:
    """Recover the input signal of a banded forward operator by Tikhonov regularization.

    The forward model is ``W @ x = observed`` with ``W`` stored in banded layout: row ``k`` of
    the dense operator is ``band_vals[k]`` placed at columns
    ``[col_start[k], col_start[k] + full_band)``. Rows may appear in any order, so several
    observation series (e.g. one per endmember) can simply be stacked.

    The Tikhonov normal equations ``(WᵀW + λ D) x = Wᵀ observed + λ D x_target`` are assembled
    directly in banded form -- ``WᵀW`` is symmetric with half-bandwidth ``full_band - 1``,
    accumulated one sub-diagonal at a time so the dense operator is never materialized -- and
    factored with :func:`scipy.linalg.cholesky_banded`. Forming ``WᵀW`` squares the condition
    number, so the bare Cholesky solve loses accuracy in the under-determined directions;
    **corrected semi-normal equations** restore it by evaluating the residual through ``W``
    itself rather than through ``WᵀW``.

    The regularization target is the transpose-and-normalize of ``W`` applied to ``observed``:
    every input bin is pulled toward the contribution-weighted average of the output bins it
    fed. Columns with no forward contribution are decoupled (unit diagonal) so the system
    stays symmetric positive definite, and are returned as NaN.

    Parameters
    ----------
    band_vals : ndarray
        Banded forward weights of shape ``(n_obs, full_band)``. Rows the caller considers
        invalid must already be zeroed; zero rows contribute nothing to the normal equations.
    col_start : ndarray of int
        First output-column index of each row's band, shape ``(n_obs,)``.
    observed : ndarray
        Observed values of shape ``(n_obs,)``. NaN entries mark measurement gaps; their rows
        are excluded from the normal equations.
    n_output : int
        Length of the recovered vector.
    regularization_strength : float
        Tikhonov parameter λ. Must be strictly positive: deconvolution is generically
        rank-deficient, and λ is what makes the banded Cholesky factor positive definite.
        A good starting value for noisy data is ``(noise_std / signal_amplitude)**2``.

    Returns
    -------
    ndarray
        Recovered signal of shape ``(n_output,)``. NaN for output bins with no forward
        contribution.

    Raises
    ------
    ValueError
        If ``regularization_strength`` is not strictly positive.
    """
    if regularization_strength <= 0:
        msg = "regularization_strength must be > 0 for the banded inverse (Tikhonov positive-definiteness)"
        raise ValueError(msg)
    band_vals = np.asarray(band_vals, dtype=float)
    observed = np.asarray(observed, dtype=float)
    # Zeroed gapped rows drop out of the normal equations, and a zeroed observed value keeps
    # 0 * NaN out of Wᵀ·observed and the refinement residual.
    nan_obs = np.isnan(observed)
    if nan_obs.any():
        band_vals = np.where(nan_obs[:, None], 0.0, band_vals)
        observed = np.where(nan_obs, 0.0, observed)
    full_band = band_vals.shape[1]
    cols = col_start[:, None] + np.arange(full_band)[None, :]
    in_range = cols < n_output
    cols_clipped = np.clip(cols, 0, n_output - 1)

    # Column sums and Wᵀ observed (the reverse-target numerator) by scattering the band.
    col_sum = np.zeros(n_output)
    wt_observed = np.zeros(n_output)
    np.add.at(col_sum, cols_clipped[in_range], band_vals[in_range])
    np.add.at(wt_observed, cols_clipped[in_range], (band_vals * observed[:, None])[in_range])

    col_active = col_sum > 0
    if not np.any(col_active):
        return np.full(n_output, np.nan)

    # Reverse target: transpose-and-normalize W applied to observed. The sliver
    # 0 < col_sum <= _EPSILON_COEFF_SUM is left untargeted (filled with 0).
    with np.errstate(invalid="ignore", divide="ignore"):
        x_target = np.where(col_sum > _EPSILON_COEFF_SUM, wt_observed / col_sum, 0.0)

    # Lower-banded WᵀW, one sub-diagonal at a time. Band row d holds WᵀW[j + d, j]; row k of W
    # contributes band_vals[k, b] * band_vals[k, b + d] to column j = col_start[k] + b. Only
    # pairs whose *upper* column stays inside the output range are scattered, which also
    # implies the lower one does. Peak memory is O(n_obs * full_band), never the dense W.
    ab = np.zeros((full_band, n_output))
    for d in range(full_band):
        pair_cols = cols_clipped[:, : full_band - d]
        keep = cols[:, d:] < n_output
        np.add.at(ab[d], pair_cols[keep], (band_vals[:, : full_band - d] * band_vals[:, d:])[keep])

    lam = regularization_strength
    d_reg = lam * col_active
    ab[0] += d_reg
    # d_reg is zero off the active columns, so x_target needs no masking here or in the
    # refinement loop: the product d_reg * x_target vanishes wherever col_active is False.
    rhs = wt_observed + d_reg * x_target

    # Decouple zero (inactive, unregularized) diagonals so the matrix is SPD.
    dead = ab[0] <= 0.0
    ab[0, dead] = 1.0
    rhs[dead] = 0.0

    factor = cholesky_banded(ab, lower=True)
    x = cho_solve_banded((factor, True), rhs)

    # Corrected semi-normal equations: the residual is evaluated through W itself (in
    # observation space) rather than through WᵀW, avoiding the cancellation that makes plain
    # normal-equation refinement stall. One step reaches the QR-accurate solution; the second
    # is a safety margin (the iteration's fixed point is stable).
    for _ in range(_BANDED_REFINEMENT_STEPS):
        gathered = x[cols_clipped]
        gathered[~in_range] = 0.0
        residual = observed - (band_vals * gathered).sum(axis=1)
        gradient = np.zeros(n_output)
        np.add.at(gradient, cols_clipped[in_range], (band_vals * residual[:, None])[in_range])
        gradient += d_reg * (x_target - x)
        gradient[dead] = 0.0
        x += cho_solve_banded((factor, True), gradient)

    out = np.full(n_output, np.nan)
    out[col_active] = x[col_active]
    return out
