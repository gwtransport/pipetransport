r"""
The linear source-to-node transfer operator, built exactly for time-varying flow splits.

This module has no public API. It holds the one computation every transport, residence-time
and decay result in the package is read off: the banded matrix ``W`` with
``c_node = W @ c_source``, together with the travel times and the coverage mask that says
which output bins the record actually constrains.

The label coordinate
--------------------

Transport is not built on a time axis but on the **cumulative throughflow volume at the
reporting node**, written ``u``. A parcel keeps its label from the moment it leaves the
source until it is delivered, because the label counts the water delivered at that node
ahead of it and no water overtakes any other in plug flow. Two properties follow, and they
are what make the operator exact:

- ``du = q_node dt``, so a *uniform* average over a label interval **is** the flow-weighted
  average over the corresponding time interval. No separate weighting step is needed.
- The output bin ``[T_j, T_{j+1})`` occupies the label interval ``[u_j, u_{j+1}]`` with
  ``u_j = N(T_j)``, and the input bin ``[t_l, t_{l+1})`` occupies ``[g_l, g_{l+1}]`` with
  ``g_l = N(A(t_l))``, where ``N`` is the cumulative node throughflow and ``A`` maps a source
  departure time to the arrival time at the node. Both edge sequences are known exactly, so
  the weight of input bin ``l`` in output bin ``j`` is the plain overlap of two intervals.

Arrival times, segment by segment
---------------------------------

Within a pipe of water volume ``V`` carrying throughflow ``Q(t)``, a parcel entering at ``s``
leaves at the time ``t`` that has displaced exactly ``V``: with the pipe's own cumulative
volume ``C(t) = \int Q``, ``C(t) = C(s) + V``. Piecewise-constant ``Q`` makes ``C``
piecewise linear, so the map and its inverse are exact linear interpolations. ``A`` is the
composition of these maps along the source-to-node path. Nothing in the composition assumes
that the segments carry fixed *fractions* of the production: each segment inverts its own
cumulative volume, so a demand pattern that shifts between branches over the day is handled
without approximation.

When the splits *are* constant, ``Q_e = f_e Q_0`` collapses the composition to
``C_0(t) = C_0(s) + \sum_e V_e / f_e``: a single effective volume, in units of source
throughflow, in which a shared trunk main counts in full for every downstream path rather
than being divided among them.

The refined cell grid
---------------------

``A`` is piecewise linear in ``s``, but its breakpoints are not only at ``tedges``: a parcel
also kinks when it crosses a flow change *inside* any pipe it is still travelling through. The
grid is therefore refined with the source-time preimage of ``tedges`` taken at every node of
the path, plus the preimage of ``cout_tedges``. On the resulting cells every arrival time,
every segment travel time and the label ``g`` are exactly linear in ``s``, and each cell falls
inside exactly one input bin and one output bin. The operator is then a single scatter-add.

Decay
-----

First-order decay is carried as the dimensionless exponent ``phi = sum_e k_e tau_e``
accumulated along the path, so segments may decay at different rates -- which is the normal
case, since wall reaction scales with the surface-to-volume ratio and thin service lines
consume disinfectant far faster than trunk mains. ``phi`` is linear in the label on each cell,
so the surviving fraction integrates in closed form:
``(1/du) \int e^{-phi} du = e^{-min(phi)} (1 - e^{-|dphi|}) / |dphi|``. The zero-decay limit of
that expression is exactly ``1``, so there is no separate conservative code path.

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from pipetransport._validation import _validate_non_negative, _validate_retardation_factor, _validate_tedges
from pipetransport.utils import _make_strictly_monotone, cumulative_flow_volume, tedges_to_days

if TYPE_CHECKING:
    from pipetransport.network import PipeNetwork

# A cout bin counts as constrained once its label interval is covered to within this relative
# slack. The shortfall of a genuinely incomplete bin is a finite fraction of the bin, so the
# threshold only has to sit above the round-off of the interpolated cell boundaries.
_COVERAGE_TOLERANCE = 1e-8


class PathTransfer(NamedTuple):
    """Everything the public modules read off one source-to-node path.

    Attributes
    ----------
    band_vals : ndarray
        Forward weights of shape ``(n_cout, full_band)``: row ``j`` sits at input columns
        ``[col_start[j], col_start[j] + full_band)``. Rows sum to 1 without decay and to the
        surviving fraction with it. Rows failing :attr:`valid_out` are zero.
    col_start : ndarray of int
        First input-bin column of each output row's band, shape ``(n_cout,)``.
    valid_out : ndarray of bool
        Output bins whose label interval is fully covered by in-record parcels and carries
        throughflow, shape ``(n_cout,)``.
    residence_time_out : ndarray
        Flow-weighted mean travel time [days] of the water delivered in each output bin,
        shape ``(n_cout,)``. NaN where :attr:`valid_out` is False.
    residence_time_in : ndarray
        Volume-weighted mean travel time [days] until arrival, for the water that leaves the
        source in each input bin and is destined for this node, shape ``(n_cin,)``. NaN where
        :attr:`valid_in` is False.
    valid_in : ndarray of bool
        Input bins whose node-destined water all reaches the node inside the record,
        shape ``(n_cin,)``.
    """

    band_vals: npt.NDArray[np.floating]
    col_start: npt.NDArray[np.intp]
    valid_out: npt.NDArray[np.bool_]
    residence_time_out: npt.NDArray[np.floating]
    residence_time_in: npt.NDArray[np.floating]
    valid_in: npt.NDArray[np.bool_]


def _surviving_fraction(phi_lo: npt.NDArray[np.floating], phi_hi: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Label-averaged surviving fraction over a cell whose decay exponent runs from ``phi_lo`` to ``phi_hi``.

    Evaluates ``(e^-a - e^-b) / (b - a)`` in the form ``e^-min * (1 - e^-|b-a|) / |b-a|``, which
    is symmetric in the endpoints, never overflows for non-negative exponents, and tends to
    ``e^-a`` as the endpoints merge.

    Parameters
    ----------
    phi_lo, phi_hi : ndarray
        Dimensionless decay exponent at the two cell boundaries. Both non-negative.

    Returns
    -------
    ndarray
        Mean of ``exp(-phi)`` over the cell, elementwise. Exactly 1.0 where both exponents
        are zero.
    """
    spread = np.abs(phi_hi - phi_lo)
    with np.errstate(invalid="ignore", divide="ignore"):
        ramp = np.where(spread > 0.0, -np.expm1(-spread) / spread, 1.0)
    return np.exp(-np.minimum(phi_lo, phi_hi)) * ramp


def path_transfer(
    *,
    tedges_days: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
    path_volume: npt.NDArray[np.floating],
    path_flow: npt.NDArray[np.floating],
    path_decay: npt.NDArray[np.floating],
    node_flow: npt.NDArray[np.floating],
) -> PathTransfer:
    """Build the exact transfer operator and travel times of one source-to-node path.

    Parameters
    ----------
    tedges_days : ndarray
        Input (source) bin edges in days, strictly increasing, length ``n_cin + 1``.
    cout_tedges_days : ndarray
        Output bin edges in days on the same origin, strictly increasing, length
        ``n_cout + 1``.
    path_volume : ndarray
        Water volume [m³] of each segment on the path, ordered from the source outward and
        already multiplied by the retardation factor. Length ``m`` (0 when the reporting node
        *is* the source).
    path_flow : ndarray
        Throughflow [m³/day] of those segments, shape ``(m, n_cin)``.
    path_decay : ndarray
        First-order decay rate [1/day] of those segments, length ``m``.
    node_flow : ndarray
        Throughflow [m³/day] past the reporting node, length ``n_cin``. This is the weight of
        the output bin average and the differential of the label coordinate.

    Returns
    -------
    PathTransfer
        Banded operator, coverage mask and travel times; see :class:`PathTransfer`.
    """
    n_cin = len(tedges_days) - 1
    n_cout = len(cout_tedges_days) - 1
    n_path = len(path_volume)
    dt_days = np.diff(tedges_days)

    # Per-segment cumulative volume. Plateaus from a closed valve make the volume-to-time
    # inversion multi-valued, so they are separated before the maps below invert them.
    segment_cumulative = np.stack([
        _make_strictly_monotone(cumulative_flow_volume(path_flow[i], dt_days)) for i in range(n_path)
    ]).reshape(n_path, n_cin + 1)
    # Label axis: cumulative throughflow past the reporting node. Only ever read forward
    # (time to label), so its plateaus are meaningful and stay untouched.
    node_cumulative = cumulative_flow_volume(node_flow, dt_days)

    def travel(times: npt.NDArray[np.floating], segment: int, *, downstream: bool) -> npt.NDArray[np.floating]:
        """Map times across one segment, downstream (entry to exit) or back upstream.

        Parameters
        ----------
        times : ndarray
            Times in days at the segment's entry (``downstream=True``) or exit face.
        segment : int
            Position of the segment on the path, counted from the source.
        downstream : bool
            Direction of the map.

        Returns
        -------
        ndarray
            Times in days at the other face; NaN where the parcel falls outside the record.
        """
        cumulative = segment_cumulative[segment]
        displaced = np.interp(times, tedges_days, cumulative, left=np.nan, right=np.nan)
        target = displaced + path_volume[segment] if downstream else displaced - path_volume[segment]
        return np.interp(target, cumulative, tedges_days, left=np.nan, right=np.nan)

    # Refined source-time grid. Walking the path inwards and re-seeding with tedges at every
    # node collects, in source time, every parcel that crosses a flow change anywhere along
    # its journey -- the complete set of kinks of the arrival maps. The preimage of the output
    # edges is added so each cell also lands inside a single output bin.
    kinks = tedges_days
    for segment in range(n_path - 1, -1, -1):
        kinks = travel(np.union1d(kinks, tedges_days), segment, downstream=False)
        kinks = kinks[np.isfinite(kinks)]
    cout_preimage = cout_tedges_days
    for segment in range(n_path - 1, -1, -1):
        cout_preimage = travel(cout_preimage, segment, downstream=False)
    grid = np.unique(np.concatenate([kinks, cout_preimage[np.isfinite(cout_preimage)], tedges_days]))
    grid = grid[(grid >= tedges_days[0]) & (grid <= tedges_days[-1])]

    # Forward sweep: arrival time at every node of the path, then the decay exponent, the
    # travel time and the label of each grid parcel.
    arrival = grid
    decay_exponent = np.zeros_like(grid)
    for segment in range(n_path):
        previous, arrival = arrival, travel(arrival, segment, downstream=True)
        decay_exponent += path_decay[segment] * (arrival - previous)
    travel_time = arrival - grid
    label = np.interp(arrival, tedges_days, node_cumulative, left=np.nan, right=np.nan)

    # Output bins: label span and the two conditions that do not depend on the cells.
    edge_in_record = (cout_tedges_days >= tedges_days[0]) & (cout_tedges_days <= tedges_days[-1])
    cout_label = np.interp(cout_tedges_days, tedges_days, node_cumulative)
    cout_label_width = np.diff(cout_label)
    row_supported = edge_in_record[:-1] & edge_in_record[1:] & (cout_label_width > 0.0)

    # Cells. Each spans one grid interval; both its boundaries must reach the node inside the
    # record for it to carry information.
    cell_ok = np.isfinite(label[:-1]) & np.isfinite(label[1:])
    midpoint = 0.5 * (grid[:-1] + grid[1:])
    cin_bin = np.clip(np.searchsorted(tedges_days, midpoint, side="right") - 1, 0, n_cin - 1)

    # An input bin is constrained only if every parcel leaving in it arrives inside the record.
    label_width = np.where(cell_ok, label[1:] - label[:-1], 0.0)
    in_volume = np.bincount(cin_bin, weights=label_width, minlength=n_cin)
    in_travel = np.bincount(
        cin_bin, weights=label_width * 0.5 * np.nan_to_num(travel_time[:-1] + travel_time[1:]), minlength=n_cin
    )
    valid_in = (np.bincount(cin_bin[~cell_ok], minlength=n_cin) == 0) & (in_volume > 0.0)
    residence_time_in = np.full(n_cin, np.nan)
    residence_time_in[valid_in] = in_travel[valid_in] / in_volume[valid_in]

    keep = cell_ok & (label[1:] > label[:-1])
    cin_bin, label_width = cin_bin[keep], label_width[keep]
    label_mid = 0.5 * (label[:-1] + label[1:])[keep]
    cout_bin = np.searchsorted(cout_label, label_mid, side="right") - 1
    inside = (cout_bin >= 0) & (cout_bin < n_cout)
    cin_bin, label_width, cout_bin = cin_bin[inside], label_width[inside], cout_bin[inside]
    survived = label_width * _surviving_fraction(decay_exponent[:-1][keep][inside], decay_exponent[1:][keep][inside])
    carried_time = label_width * 0.5 * (travel_time[:-1][keep][inside] + travel_time[1:][keep][inside])

    # Cells are ordered by source time and both the label and the input-bin index increase
    # with it, so the cells of one output row are a contiguous, non-decreasing run: the band
    # bounds are read off the run's first and last cell instead of a scatter-minimum.
    n_cell = cout_bin.size
    run_lo = np.searchsorted(cout_bin, np.arange(n_cout), side="left")
    run_hi = np.searchsorted(cout_bin, np.arange(n_cout), side="right")
    populated = run_hi > run_lo
    safe_lo, safe_hi = np.clip(run_lo, 0, max(n_cell - 1, 0)), np.clip(run_hi - 1, 0, max(n_cell - 1, 0))
    col_start = np.where(populated, cin_bin[safe_lo], 0).astype(np.intp) if n_cell else np.zeros(n_cout, np.intp)
    col_stop = np.where(populated, cin_bin[safe_hi], 0) if n_cell else np.zeros(n_cout, np.intp)
    full_band = int(np.max(col_stop - col_start)) + 1 if n_cell else 1

    # Every cell contribution is a share of its output bin's label span: that division is what
    # turns the label-uniform integral into the flow-weighted bin average.
    span = np.where(cout_label_width > 0.0, cout_label_width, 1.0)
    slot = cout_bin * full_band + (cin_bin - col_start[cout_bin])
    band_vals = (
        np
        .bincount(slot, weights=survived / span[cout_bin], minlength=n_cout * full_band)
        .astype(float, copy=False)
        .reshape(n_cout, full_band)
    )
    coverage = np.bincount(cout_bin, weights=label_width, minlength=n_cout) / span
    out_travel = np.bincount(cout_bin, weights=carried_time, minlength=n_cout) / span

    valid_out = row_supported & (coverage >= 1.0 - _COVERAGE_TOLERANCE)
    band_vals[~valid_out] = 0.0
    residence_time_out = np.full(n_cout, np.nan)
    residence_time_out[valid_out] = out_travel[valid_out]
    return PathTransfer(band_vals, col_start, valid_out, residence_time_out, residence_time_in, valid_in)


def network_transfer(
    *,
    network: PipeNetwork,
    flow: npt.ArrayLike | pd.DataFrame | dict,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    nodes: list[str] | tuple[str, ...] | None,
    decay_rate: float | pd.Series,
    retardation_factor: float,
    spinup: str | None,
) -> tuple[tuple[str, ...], list[PathTransfer], int]:
    """Resolve the shared inputs of every public entry point and build one operator per node.

    Validates the time axes and physical parameters, converts the endmember demand into
    segment and node throughflow, applies the spin-up policy, and calls :func:`path_transfer`
    once per reporting node.

    Parameters
    ----------
    network : PipeNetwork
        Validated network.
    flow : DataFrame, mapping, or array-like
        Demand at every endmember [m³/day] on the ``tedges`` bins.
    tedges : DatetimeIndex
        Input bin edges, length ``n_cin + 1``.
    cout_tedges : DatetimeIndex
        Output bin edges, length ``n_cout + 1``. May differ in alignment and resolution.
    nodes : list of str or None
        Nodes to report at. ``None`` selects :attr:`~pipetransport.network.PipeNetwork.endmembers`.
    decay_rate : float or Series
        First-order decay rate [1/day], one value for every segment or a Series indexed by
        segment name.
    retardation_factor : float
        Multiplier on every segment volume; ``1.0`` is a conservative tracer.
    spinup : {"constant"} or None
        Warm-start policy, see :func:`resolve_spinup`.

    Returns
    -------
    nodes : tuple of str
        The resolved reporting nodes, in output order.
    transfers : list of PathTransfer
        One operator per reporting node, built on the padded input grid.
    n_pad : int
        Number of warm-start bins prepended to ``tedges``.

    Raises
    ------
    ValueError
        If a time axis is not strictly increasing or has the wrong length, if the retardation
        factor is below 1, if a decay rate is negative or missing for a segment, or if a
        requested node is not part of the network.
    """
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    demand = network.flow_array(flow)
    _validate_tedges(tedges, demand, tedges_name="tedges", values_name="flow")
    _validate_tedges(cout_tedges, np.empty(len(cout_tedges) - 1), tedges_name="cout_tedges", values_name="cout")
    _validate_retardation_factor(retardation_factor)

    if isinstance(decay_rate, pd.Series):
        missing = [name for name in network.segments.index if name not in decay_rate.index]
        if missing:
            msg = f"decay_rate is missing segment(s): {missing}"
            raise ValueError(msg)
        decay = decay_rate.reindex(network.segments.index).to_numpy(dtype=float)
    else:
        decay = np.asarray(decay_rate, dtype=float)
        if decay.ndim > 1 or (decay.ndim == 1 and decay.size != len(network.segments)):
            msg = f"decay_rate must be a scalar or hold one value per segment ({len(network.segments)})"
            raise ValueError(msg)
        decay = np.broadcast_to(decay, (len(network.segments),))
    _validate_non_negative(decay, name="decay_rate")

    requested = tuple(network.endmembers) if nodes is None else tuple(nodes)
    unknown = [node for node in requested if node not in network.paths]
    if unknown:
        msg = f"unknown node(s): {unknown}; network nodes are {list(network.nodes)}"
        raise ValueError(msg)
    volume = retardation_factor * network.segments["volume"].to_numpy(dtype=float)
    row_of = {name: i for i, name in enumerate(network.segments.index)}
    paths = [[row_of[segment] for segment in network.paths[node]] for node in requested]

    # Warm-start length: the longest source-to-node travel time at the leading flow rate. A
    # stagnant leading segment makes it infinite, which resolve_spinup reads as "no warm start".
    with np.errstate(divide="ignore", invalid="ignore"):
        leading = network.segment_flow(flow=demand)[:, 0]
        warm_start_days = max((np.sum(volume[path] / leading[path]) for path in paths), default=0.0)

    tedges, demand, n_pad = resolve_spinup(spinup, tedges=tedges, flow=demand, warm_start_days=float(warm_start_days))
    segment_flow = network.segment_flow(flow=demand)
    node_flow = network.node_flow(flow=demand, nodes=requested)
    tedges_days = tedges_to_days(tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=tedges[0])

    transfers = [
        path_transfer(
            tedges_days=tedges_days,
            cout_tedges_days=cout_tedges_days,
            path_volume=volume[path],
            path_flow=segment_flow[path],
            path_decay=decay[path],
            node_flow=node_flow[i],
        )
        for i, path in enumerate(paths)
    ]
    return requested, transfers, n_pad


def resolve_spinup(
    spinup: str | None,
    *,
    tedges: pd.DatetimeIndex,
    flow: npt.NDArray[np.floating],
    warm_start_days: float,
) -> tuple[pd.DatetimeIndex, npt.NDArray[np.floating], int]:
    """Validate the ``spinup`` policy and prepend the warm-start bins it implies.

    ``"constant"`` extends the record backwards by ``warm_start_days``, holding every
    endmember demand at its first observed value, so the earliest output bins are fed by a
    defined (if assumed) history instead of coming back NaN. It falls back to no padding
    whenever the warm start is undefined -- a zero or non-finite leading flow, a degenerate
    first bin, or an implied padding so long that a constant history is not a meaningful
    assumption -- leaving the strict-validity NaN in place.

    Parameters
    ----------
    spinup : {"constant"} or None
        Policy. ``None`` returns the inputs unchanged.
    tedges : DatetimeIndex
        Input bin edges, length ``n_cin + 1``.
    flow : ndarray
        Endmember demand of shape ``(n_endmembers, n_cin)``.
    warm_start_days : float
        Duration to cover, normally the longest source-to-node travel time at the leading
        flow rate.

    Returns
    -------
    tedges : DatetimeIndex
        Padded bin edges, length ``n_cin + n_pad + 1``.
    flow : ndarray
        Padded demand of shape ``(n_endmembers, n_cin + n_pad)``.
    n_pad : int
        Number of bins prepended; 0 when no padding was applied.

    Raises
    ------
    ValueError
        If ``spinup`` is neither ``None`` nor ``"constant"``.
    """
    if spinup is None:
        return tedges, flow, 0
    if spinup != "constant":
        msg = f"spinup must be None or 'constant'; got {spinup!r}"
        raise ValueError(msg)

    bin_width = tedges[1] - tedges[0]
    bin_width_days = bin_width / pd.Timedelta(days=1)
    if not (np.isfinite(warm_start_days) and warm_start_days > 0.0 and bin_width_days > 0.0):
        return tedges, flow, 0
    # One extra bin so the longest path's source window for the earliest original output bin
    # lies strictly inside the padded range rather than touching its edge.
    n_pad_float = np.ceil(warm_start_days / bin_width_days) + 1.0
    # Beyond this a constant history is not a meaningful warm start (unphysical geometry or a
    # near-stagnant leading flow), so fall through to strict validity rather than allocate.
    if n_pad_float > max(10_000, 10 * flow.shape[1]):
        return tedges, flow, 0
    n_pad = int(n_pad_float)
    offsets = pd.TimedeltaIndex(bin_width * np.arange(n_pad, 0, -1))
    padded_tedges = (tedges[0] - offsets).append(tedges)
    padded_flow = np.concatenate([np.repeat(flow[:, :1], n_pad, axis=1), flow], axis=1)
    return padded_tedges, padded_flow, n_pad
