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
also kinks when it crosses a flow change *inside* any pipe it is still travelling through. One
backward sweep along the path therefore collects the source-time preimage of ``tedges`` taken
at every node of the path, together with the preimage of ``cout_tedges``. On the resulting
cells every arrival time, every segment travel time and the label ``g`` are exactly linear in
``s``, and each cell falls inside exactly one input bin and one output bin -- so output-bin
membership is read off the arrival time of the cell midpoint. The operator is then a single
scatter-add.

All reporting nodes are built in one batched pass: the grids of every path live in the rows of
one matrix, kept the same width by tolerating duplicate points as zero-width cells instead of
deduplicating, and cells that leave the record or miss the output range drain into per-node
dustbin slots that are sliced away after the scatter-add. The Python loops that remain are the
composition of the segment maps along the path and the row-wise delegation to ``np.interp``,
whose fused C pass no combination of broadcast primitives matches.

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

from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from pipetransport._validation import (
    _per_segment,
    _validate_non_negative,
    _validate_retardation_factor,
    _validate_tedges,
)
from pipetransport.utils import cumulative_flow_volume, tedges_to_days

if TYPE_CHECKING:
    from pipetransport.network import PipeNetwork

# A cout bin counts as constrained once its label interval is covered to within this relative
# slack. The shortfall of a genuinely incomplete bin is a finite fraction of the bin, so the
# threshold only has to sit above the round-off of the interpolated cell boundaries.
_COVERAGE_TOLERANCE = 1e-8

# Slack, in ulps of the cumulative-volume scale, allowed when a displacement target misses the
# record's volume range, and the floor below which a cell's label width is plateau residue
# rather than water. Both have to clear the round-trip interpolation error (about one ulp per
# segment on the path) and the 16-ulp plateau separation in _make_strictly_monotone. At
# 1.4e-14 relative it is far below any physically meaningful volume.
_ROUNDTRIP_ULPS = 64.0


class NetworkTransfer(NamedTuple):
    """Everything the public modules read off the source-to-node paths, stacked over nodes.

    The leading axis of every field runs over the resolved reporting nodes, in output order.
    ``full_band`` is shared across nodes: narrower bands carry trailing zero slots, which
    contribute nothing in either direction.

    Attributes
    ----------
    band_vals : ndarray
        Forward weights of shape ``(n_nodes, n_cout, full_band)``: row ``j`` of node ``n``
        sits at input columns ``[col_start[n, j], col_start[n, j] + full_band)``. Rows sum to
        1 without decay and to the surviving fraction with it. Rows failing :attr:`valid_out`
        are zero.
    col_start : ndarray of int
        First input-bin column of each output row's band, shape ``(n_nodes, n_cout)``.
    valid_out : ndarray of bool
        Output bins whose label interval is fully covered by in-record parcels and carries
        throughflow, shape ``(n_nodes, n_cout)``.
    residence_time_out : ndarray
        Flow-weighted mean travel time [days] of the water delivered in each output bin,
        shape ``(n_nodes, n_cout)``. NaN where :attr:`valid_out` is False.
    residence_time_in : ndarray
        Volume-weighted mean travel time [days] until arrival, for the water that leaves the
        source in each input bin and is destined for each node, shape ``(n_nodes, n_cin)``.
        NaN where :attr:`valid_in` is False.
    valid_in : ndarray of bool
        Input bins whose node-destined water all reaches the node inside the record,
        shape ``(n_nodes, n_cin)``.
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


def _cell_edges(
    quarter: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Extrapolate a cell-affine quantity from its quarter-point samples to its two boundaries.

    Parameters
    ----------
    quarter : ndarray
        Samples at the ``1/4`` and ``3/4`` points of every cell, shape
        ``(n_nodes, 2, n_cells)``.

    Returns
    -------
    lo, hi : ndarray
        The quantity at the lower and upper cell boundary, shape ``(n_nodes, n_cells)``.
        Exact whenever the quantity is affine across the cell, which every arrival time and
        decay exponent is (a flow change inside a cell would have made it a grid point).
    """
    lo, hi = quarter[:, 0], quarter[:, 1]
    return 1.5 * lo - 0.5 * hi, 1.5 * hi - 0.5 * lo


def _interp_rows(
    x: npt.NDArray[np.floating],
    xp: npt.NDArray[np.floating],
    fp: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Row-wise ``np.interp(x, xp, fp, left=np.nan, right=np.nan)`` for batched maps.

    numpy exposes no batched interpolation, and composing one from broadcast primitives
    costs a full array pass per search, gather and blend step -- several times slower than
    :func:`numpy.interp`, which fuses them into a single C pass. So the rows delegate to
    ``np.interp`` one by one: each iteration is one fused C call, and the row count is the
    number of reporting nodes, not a data axis.

    Parameters
    ----------
    x : ndarray
        Queries of shape ``(n, m)``. Non-finite entries come back NaN.
    xp : ndarray
        Reference x, strictly increasing along the last axis: shared shape ``(k,)`` or
        per-row shape ``(n, k)``.
    fp : ndarray
        Reference y, per-row ``(n, k)`` or shared ``(k,)`` -- whichever ``xp`` is not.

    Returns
    -------
    ndarray
        Interpolated values of shape ``(n, m)``; NaN outside ``[xp[..., 0], xp[..., -1]]``.
    """
    shared_xp = xp.ndim == 1
    out = np.empty(x.shape)
    for i, queries in enumerate(x):
        out[i] = np.interp(queries, xp if shared_xp else xp[i], fp[i] if shared_xp else fp, left=np.nan, right=np.nan)
    return out


def apply_banded(transfer: NetworkTransfer, values: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Apply a banded operator to a padded input series, rows in band layout.

    Parameters
    ----------
    transfer : NetworkTransfer
        Operator whose rows are to be applied.
    values : ndarray
        Input series on the operator's padded input grid.

    Returns
    -------
    ndarray
        One value per operator row and output bin, shape ``(n_rows, n_cout)``.
    """
    columns = np.clip(transfer.col_start[..., None] + np.arange(transfer.band_vals.shape[-1]), 0, len(values) - 1)
    return np.einsum("nkb,nkb->nk", transfer.band_vals, values[columns])


def pad_paths(chains: list[npt.NDArray[np.intp]]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
    """Stack ragged path chains into the padded index matrix the operator builder takes.

    Parameters
    ----------
    chains : list of ndarray
        Segment rows of each path, source outward; may differ in length and may be empty
        (a row reporting at the source itself).

    Returns
    -------
    paths_idx : ndarray of intp
        Segment row of each path step, shape ``(len(chains), max_depth)``.
    active : ndarray of bool
        Which slots of ``paths_idx`` are real path steps.
    """
    lengths = np.array([chain.size for chain in chains], dtype=np.intp)
    max_depth = int(lengths.max(initial=0))
    active = np.arange(max_depth) < lengths[:, None]
    paths_idx = np.zeros((len(chains), max_depth), dtype=np.intp)
    if max_depth:
        paths_idx[active] = np.concatenate(chains)
    return paths_idx, active


class _CellGrid:
    """Refined source-time cells of every path, and the geometry read off them.

    Cells are what make the operator exact. Between two consecutive boundaries the travel
    time, the decay exponent and the arrival are all affine in source time, so every cell
    mean is closed-form and the operator is a single scatter-add of cell contributions into
    output bins. This object owns that construction -- the backward refinement sweep that
    collects every kink of the arrival maps, the forward displacement sweep, the per-cell
    geometry, and the band layout the scatter needs. What it deliberately does not own is
    how much of a cell survives its journey: :meth:`bands` takes that as an argument, so a
    plain reading and a reading weighted over the lag to its output bin's end share one
    grid, and the displacement arithmetic -- which carries the plateau separation, the
    round-trip snapping and the label floor -- exists once.

    Three layouts are a contract the callers depend on:

    - ``samples`` is the ``n_edge`` cell boundaries followed by the quarter and the
      three-quarter point of every cell, so an interior quantity reshapes to
      ``(n_nodes, 2, n_cells)`` and :func:`_cell_edges` extrapolates it to the boundaries.
    - Output slots carry a dustbin either side of the real bins, so ``flat_cout`` indexes
      rows of width ``n_cout + 2`` and the real bins are the ``[1:-1]`` slice.
    - Rows stay in the caller's order.

    Parameters
    ----------
    tedges_days : ndarray
        Input (source) bin edges in days, strictly increasing, length ``n_cin + 1``.
    cout_tedges_days : ndarray
        Output bin edges in days on the same origin, strictly increasing, length
        ``n_cout + 1``.
    segment_volume : ndarray
        Water volume [m³] of every segment of the network, already multiplied by the
        retardation factor. Length ``n_seg``.
    segment_flow : ndarray
        Throughflow [m³/day] of those segments, shape ``(n_seg, n_cin)``.
    segment_decay : ndarray
        First-order decay rate [1/day] of those segments, length ``n_seg``.
    node_flow : ndarray
        Throughflow [m³/day] past each reporting node, shape ``(n_nodes, n_cin)``. This is
        the weight of the output bin average and the differential of the label coordinate.
    paths_idx : ndarray of int
        Segment row of each path step, shape ``(n_nodes, max_depth)``, ordered from the
        source outward. Slots beyond a path's depth are ignored.
    active : ndarray of bool
        Which slots of ``paths_idx`` are real path steps, shape ``(n_nodes, max_depth)``. A
        path occupies the leading slots of its row, so a node reporting at the source itself
        is a row of ``False``.

    Notes
    -----
    Travel time, decay exponent and midpoint arrival are sampled at the quarter points of each
    cell rather than at its boundaries. A cell boundary may sit on a zero-flow plateau of a
    segment's cumulative volume, where the volume-to-time inverse is multi-valued and the
    16-ulp plateau separation of :func:`~pipetransport.utils._make_strictly_monotone` scales a
    one-ulp round trip up to a finite fraction of the stagnation. Interior samples are free of
    that ambiguity, and all three quantities are affine across a cell, so the pair at ``1/4``
    and ``3/4`` reproduces them exactly: the mean is the cell mean, and
    ``1.5 * phi_lo_sample - 0.5 * phi_hi_sample`` extrapolates to the boundary exponents that
    :func:`_surviving_fraction` integrates between.
    """

    def __init__(
        self,
        *,
        tedges_days: npt.NDArray[np.floating],
        cout_tedges_days: npt.NDArray[np.floating],
        segment_volume: npt.NDArray[np.floating],
        segment_flow: npt.NDArray[np.floating],
        segment_decay: npt.NDArray[np.floating],
        node_flow: npt.NDArray[np.floating],
        paths_idx: npt.NDArray[np.intp],
        active: npt.NDArray[np.bool_],
    ) -> None:
        self.tedges_days = tedges_days
        self.cout_tedges_days = cout_tedges_days
        self.segment_volume = segment_volume
        self.segment_decay = segment_decay
        self.paths_idx = paths_idx
        self.active = active
        self.n_cin = n_cin = len(tedges_days) - 1
        self.n_cout = n_cout = len(cout_tedges_days) - 1
        self.n_nodes, self.max_depth = paths_idx.shape
        n_nodes, max_depth = self.n_nodes, self.max_depth
        self.dt_days = dt_days = np.diff(tedges_days)

        # Per-segment cumulative volume, once per segment however many paths share it. Plateaus
        # from a closed valve make the volume-to-time inversion multi-valued, so they are
        # separated before the maps below invert them.
        self.segment_cumulative = cumulative_flow_volume(segment_flow, dt_days, strictly_monotone=True)
        # Label axis: cumulative throughflow past each reporting node. Only ever read forward
        # (time to label), so its plateaus are meaningful and stay untouched.
        node_cumulative = cumulative_flow_volume(node_flow, dt_days)

        # Refined source-time grid, one backward sweep for all nodes. Walking the paths inwards
        # and re-seeding with tedges at every node collects, in source time, every parcel that
        # crosses a flow change anywhere along its journey -- the complete set of kinks of the
        # arrival maps -- and carries the preimage of the output edges along, so each cell also
        # lands inside a single output bin. Points without an in-record preimage collapse onto
        # the record's end as zero-width cells, keeping every row the same width; duplicates are
        # equally harmless, so nothing is pruned.
        tedges_rows = np.broadcast_to(tedges_days, (n_nodes, n_cin + 1))
        pts = np.broadcast_to(cout_tedges_days, (n_nodes, n_cout + 1))
        for depth in range(max_depth - 1, -1, -1):
            pts = self.travel(np.concatenate([pts, tedges_rows], axis=1), depth, downstream=False)
        grid = np.concatenate([pts, tedges_rows], axis=1)
        grid = np.sort(
            np.clip(np.where(np.isfinite(grid), grid, tedges_days[-1]), tedges_days[0], tedges_days[-1]), axis=1
        )
        self.grid = grid

        # Forward sweep over the cell boundaries and the quarter points inside each cell. The
        # boundaries carry the label; the travel time, the decay exponent and the midpoint arrival
        # are read off the interior samples (see Notes).
        self.n_edge = n_edge = grid.shape[1]
        self.n_cells = grid.shape[1] - 1
        cell_width = np.diff(grid, axis=1)
        self.samples = samples = np.concatenate(
            [grid, grid[:, :-1] + 0.25 * cell_width, grid[:, :-1] + 0.75 * cell_width], axis=1
        )
        arrival = samples
        decay_exponent = np.zeros_like(samples)
        for depth in range(max_depth):
            previous, arrival = arrival, self.travel(arrival, depth, downstream=True)
            decay_exponent += segment_decay[paths_idx[:, depth], None] * (arrival - previous)
        self.quarter_arrival = quarter_arrival = arrival[:, n_edge:].reshape(n_nodes, 2, -1)
        # Output-bin membership of every cell, read off the midpoint arrival. It is needed here,
        # before any exponent factor, because a reading weighted over the lag to that bin's right
        # edge measures its weight from there; the refined grid carries the preimages of the
        # output edges, so a cell lies inside a single output bin and both its quarter samples
        # share that edge -- which is what keeps the weighted exponent affine across the cell,
        # exactly as the plain one is. Cells arriving past the output range are dustbins that the
        # slices below drop.
        self.arrival_mid = arrival_mid = quarter_arrival.mean(axis=1)
        self.cout_bin = cout_bin = np.searchsorted(cout_tedges_days, arrival_mid, side="right") - 1
        self.quarter_phi = quarter_phi = decay_exponent[:, n_edge:].reshape(n_nodes, 2, -1)
        self.cell_travel_time = cell_travel_time = (quarter_arrival - samples[:, n_edge:].reshape(n_nodes, 2, -1)).mean(
            axis=1
        )
        self.phi_lo, self.phi_hi = _cell_edges(quarter_phi)
        label = _interp_rows(arrival[:, :n_edge], tedges_days, node_cumulative)

        # Output bins: label span and the two conditions that do not depend on the cells. The
        # edges are clamped into the record exactly as np.interp clamps them.
        edge_in_record = (cout_tedges_days >= tedges_days[0]) & (cout_tedges_days <= tedges_days[-1])
        cout_label = _interp_rows(
            np.broadcast_to(np.clip(cout_tedges_days, tedges_days[0], tedges_days[-1]), (n_nodes, n_cout + 1)),
            tedges_days,
            node_cumulative,
        )
        cout_label_width = np.diff(cout_label, axis=1)
        self.row_supported = edge_in_record[:-1] & edge_in_record[1:] & (cout_label_width > 0.0)

        # Cells. Each spans one grid interval; both its boundaries must reach the node inside the
        # record for it to carry information, and it must carry more water than the plateau
        # separation of _make_strictly_monotone leaves behind. That separation is what keeps the
        # volume-to-time inversion single-valued across a closed valve, but it lands in the label
        # of exactly the cells whose parcels never departed: unfloored, their sliver of width
        # reads as carried water, and a source bin no measurement constrains comes back with a
        # finite age and a reconstruction of zero instead of NaN. The floor is relative to the
        # record's own volume because the sliver is -- it is ulps of the cumulative scale, so any
        # fixed threshold is crossed by a long enough record.
        cell_ok = np.isfinite(label[:, :-1]) & np.isfinite(label[:, 1:])
        midpoint = 0.5 * (grid[:, :-1] + grid[:, 1:])
        self.cin_bin = cin_bin = np.clip(np.searchsorted(tedges_days, midpoint, side="right") - 1, 0, n_cin - 1)
        label_floor = _ROUNDTRIP_ULPS * np.spacing(node_cumulative[:, -1:])
        self.carrying = carrying = cell_ok & (label[:, 1:] - label[:, :-1] > label_floor)
        self.label_width = label_width = np.where(carrying, label[:, 1:] - label[:, :-1], 0.0)

        # An input bin is constrained only if every parcel leaving in it arrives inside the
        # record. Every scatter-add below runs on indices flattened with a per-node offset.
        node_offset = np.arange(n_nodes)[:, None]
        flat_in = (node_offset * n_cin + cin_bin).ravel()
        in_slots = n_nodes * n_cin
        in_volume = np.bincount(flat_in, weights=label_width.ravel(), minlength=in_slots).reshape(n_nodes, n_cin)
        carried_in = label_width * np.nan_to_num(cell_travel_time)
        in_travel = np.bincount(flat_in, weights=carried_in.ravel(), minlength=in_slots).reshape(n_nodes, n_cin)
        broken = np.bincount(flat_in, weights=(~cell_ok).ravel().astype(float), minlength=in_slots).reshape(
            n_nodes, n_cin
        )
        self.valid_in = valid_in = (broken == 0.0) & (in_volume > 0.0)
        self.residence_time_in = np.where(valid_in, in_travel / np.where(valid_in, in_volume, 1.0), np.nan)

        # Cells arriving before the output range, after it, or outside the record drain into the
        # two dustbin slots wrapped around each node's real bins and are sliced away below.
        self.flat_cout = flat_cout = (node_offset * (n_cout + 2) + cout_bin + 1).ravel()
        self.out_slots = out_slots = n_nodes * (n_cout + 2)

        # Cells are ordered by source time, and arrival, label and the input-bin index all
        # increase with it, so the cells of one output slot are a contiguous, non-decreasing run
        # -- globally, since the node offsets dominate. The band bounds are read off each run's
        # first and last cell instead of a scatter-minimum, over the carrying cells only:
        # compressing a sorted array keeps it sorted, and a run of non-carrying plateau cells
        # spans a closure while contributing nothing to it, so reading them would stretch every
        # band of the operator to the length of the longest closure.
        self.cin_flat = cin_flat = cin_bin.ravel()
        carry_cout, carry_cin = flat_cout[carrying.ravel()], cin_flat[carrying.ravel()]
        n_carry = carry_cout.size
        slots = np.arange(out_slots)
        run_lo = np.searchsorted(carry_cout, slots, side="left")
        run_hi = np.searchsorted(carry_cout, slots, side="right")
        populated = run_hi > run_lo
        safe_lo, safe_hi = np.clip(run_lo, 0, max(n_carry - 1, 0)), np.clip(run_hi - 1, 0, max(n_carry - 1, 0))
        self.col_start_all = col_start_all = np.where(populated, carry_cin[safe_lo], 0).astype(np.intp)
        col_stop_all = np.where(populated, carry_cin[safe_hi], 0)
        # The band width is shared across nodes and read off the real slots only: a dustbin run
        # may span the whole input range.
        spread = (col_stop_all - col_start_all).reshape(n_nodes, n_cout + 2)[:, 1:-1]
        self.full_band = int(spread.max(initial=0)) + 1

        self.span = span = np.where(cout_label_width > 0.0, cout_label_width, 1.0)
        self.span_all = np.ones((n_nodes, n_cout + 2))
        self.span_all[:, 1:-1] = span

    def travel(self, times: npt.NDArray[np.floating], depth: int, *, downstream: bool) -> npt.NDArray[np.floating]:
        """Map times across every path's ``depth``-th segment, downstream or back upstream.

        Parameters
        ----------
        times : ndarray
            Times in days of shape ``(n_nodes, m)`` at the segment's entry
            (``downstream=True``) or exit face.
        depth : int
            Position of the segment on each path, counted from the source.
        downstream : bool
            Direction of the map.

        Returns
        -------
        ndarray
            Times in days at the other face; NaN where the parcel falls outside the record.
            Rows whose path is shorter than ``depth`` pass through unchanged.
        """
        cumulative = self.segment_cumulative[self.paths_idx[:, depth]]
        volume = self.segment_volume[self.paths_idx[:, depth], None]
        target = _interp_rows(times, self.tedges_days, cumulative)
        target = np.add(target, volume, out=target) if downstream else np.subtract(target, volume, out=target)
        # Mapping a time back upstream and forward again is not bit-exact -- each composition
        # step costs about an ulp of the cumulative volume, and the plateau separation above
        # spends up to 16 more. A bare out-of-range NaN would read that miss as "the parcel
        # left the record" and void the last output bin, so a target within _ROUNDTRIP_ULPS of
        # the range is snapped into it. A genuine excursion is still NaN, and NaN input stays
        # NaN.
        low, high = cumulative[:, :1], cumulative[:, -1:]
        slack = _ROUNDTRIP_ULPS * np.spacing(np.maximum(np.abs(low), np.abs(high)))
        with np.errstate(invalid="ignore"):
            outside = (target < low - slack) | (target > high + slack)
        mapped = _interp_rows(np.clip(target, low, high), cumulative, self.tedges_days)
        np.copyto(mapped, np.nan, where=outside)
        return np.where(self.active[:, depth, None], mapped, times)

    def bands(
        self, cell_survive: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp], npt.NDArray[np.bool_], npt.NDArray[np.floating]]:
        """Scatter the surviving cell contributions into the banded operator.

        Parameters
        ----------
        cell_survive : ndarray
            Fraction of each cell's water that reaches the node, shape
            ``(n_nodes, n_cells)``. A plain reading takes :func:`_surviving_fraction` of the
            cell's boundary exponents; a weighted reading contracts the same cell against its
            reading kernel.

        Returns
        -------
        band_vals : ndarray
            Operator bands, shape ``(n_nodes, n_cout, full_band)``.
        col_start : ndarray of intp
            First input column of each band.
        valid_out : ndarray of bool
            Output bins the source record fully constrains.
        residence_time_out : ndarray
            Flow-weighted mean travel time of each output bin, NaN where not constrained.
        """
        # Every cell contribution is a share of its output bin's label span: that division is
        # what turns the label-uniform integral into the flow-weighted bin average. Cells the
        # label does not reach have zero width; their NaN decay exponents and travel times are
        # masked out rather than multiplied by it.
        survived = np.where(self.carrying, self.label_width * cell_survive, 0.0)
        carried_out = np.where(self.carrying, self.label_width * self.cell_travel_time, 0.0)
        n_nodes, n_cout, full_band = self.n_nodes, self.n_cout, self.full_band
        flat_cout, out_slots = self.flat_cout, self.out_slots
        # Dustbin cells may spread beyond the band; the clip keeps their scatter inside their own
        # (sliced-away) rows.
        slot = flat_cout * full_band + np.clip(self.cin_flat - self.col_start_all[flat_cout], 0, full_band - 1)
        band_vals = (
            np
            .bincount(
                slot, weights=survived.ravel() / self.span_all.ravel()[flat_cout], minlength=out_slots * full_band
            )
            .astype(float, copy=False)
            .reshape(n_nodes, n_cout + 2, full_band)[:, 1:-1]
        )
        coverage = (
            np.bincount(flat_cout, weights=self.label_width.ravel(), minlength=out_slots).reshape(n_nodes, n_cout + 2)[
                :, 1:-1
            ]
            / self.span
        )
        out_travel = (
            np.bincount(flat_cout, weights=carried_out.ravel(), minlength=out_slots).reshape(n_nodes, n_cout + 2)[
                :, 1:-1
            ]
            / self.span
        )

        valid_out = self.row_supported & (coverage >= 1.0 - _COVERAGE_TOLERANCE)
        band_vals[~valid_out] = 0.0
        col_start = self.col_start_all.reshape(n_nodes, n_cout + 2)[:, 1:-1]
        return band_vals, col_start, valid_out, np.where(valid_out, out_travel, np.nan)


def paths_transfer(
    *,
    tedges_days: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
    segment_volume: npt.NDArray[np.floating],
    segment_flow: npt.NDArray[np.floating],
    segment_decay: npt.NDArray[np.floating],
    node_flow: npt.NDArray[np.floating],
    paths_idx: npt.NDArray[np.intp],
    active: npt.NDArray[np.bool_],
) -> NetworkTransfer:
    """Build the exact transfer operators and travel times of every source-to-node path.

    Parameters
    ----------
    tedges_days : ndarray
        Input (source) bin edges in days, strictly increasing, length ``n_cin + 1``.
    cout_tedges_days : ndarray
        Output bin edges in days on the same origin, strictly increasing, length
        ``n_cout + 1``.
    segment_volume : ndarray
        Water volume [m³] of every segment of the network, already multiplied by the
        retardation factor. Length ``n_seg``.
    segment_flow : ndarray
        Throughflow [m³/day] of those segments, shape ``(n_seg, n_cin)``.
    segment_decay : ndarray
        First-order decay rate [1/day] of those segments, length ``n_seg``.
    node_flow : ndarray
        Throughflow [m³/day] past each reporting node, shape ``(n_nodes, n_cin)``. This is
        the weight of the output bin average and the differential of the label coordinate.
    paths_idx : ndarray of int
        Segment row of each path step, shape ``(n_nodes, max_depth)``, ordered from the
        source outward. Slots beyond a path's depth are ignored.
    active : ndarray of bool
        Which slots of ``paths_idx`` are real path steps, shape ``(n_nodes, max_depth)``. A
        path occupies the leading slots of its row, so a node reporting at the source itself
        is a row of ``False``.

    Returns
    -------
    NetworkTransfer
        Banded operators, coverage masks and travel times; see :class:`NetworkTransfer`.

    See Also
    --------
    _CellGrid : The cells this is built on, and the quarter-point sampling they rest on.
    """
    cells = _CellGrid(
        tedges_days=tedges_days,
        cout_tedges_days=cout_tedges_days,
        segment_volume=segment_volume,
        segment_flow=segment_flow,
        segment_decay=segment_decay,
        node_flow=node_flow,
        paths_idx=paths_idx,
        active=active,
    )
    band_vals, col_start, valid_out, residence_time_out = cells.bands(_surviving_fraction(cells.phi_lo, cells.phi_hi))
    return NetworkTransfer(band_vals, col_start, valid_out, residence_time_out, cells.residence_time_in, cells.valid_in)


def network_transfer(
    *,
    network: PipeNetwork,
    flow: Mapping[str, npt.ArrayLike] | npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    report_nodes: list[str] | tuple[str, ...] | None,
    decay_rate: float | Mapping[str, float],
    retardation_factor: float | Mapping[str, float],
    spinup: str | None,
) -> tuple[tuple[str, ...], NetworkTransfer, int]:
    """Resolve the shared inputs of every public entry point and build the node operators.

    Validates the time axes and physical parameters, converts the endmember demand into
    segment and node throughflow, applies the spin-up policy, and builds the operators of
    every reporting node in one batched :func:`paths_transfer` pass.

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
    report_nodes : list of str or None
        Nodes to report at. ``None`` selects :attr:`~pipetransport.network.PipeNetwork.endmembers`.
    decay_rate : float or mapping
        First-order decay rate [1/day], shared by every segment or keyed by segment name.
    retardation_factor : float or mapping
        Multiplier on the segment volumes, shared or per segment; ``1.0`` is a conservative
        tracer.
    spinup : {"constant"} or None
        Warm-start policy, see :func:`resolve_spinup`.

    Returns
    -------
    nodes : tuple of str
        The resolved reporting nodes, in output order.
    transfer : NetworkTransfer
        The stacked operators of every reporting node, built on the padded input grid.
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
    retardation = _per_segment(retardation_factor, network.segments.index, name="retardation_factor")
    _validate_retardation_factor(retardation)
    decay = _per_segment(decay_rate, network.segments.index, name="decay_rate")
    _validate_non_negative(decay, name="decay_rate")

    requested = tuple(network.endmembers) if report_nodes is None else tuple(report_nodes)
    unknown = [node for node in requested if node not in network.paths]
    if unknown:
        msg = f"unknown node(s): {unknown}; network nodes are {list(network.nodes)}"
        raise ValueError(msg)
    volume = retardation * network.segments["volume"].to_numpy(dtype=float)
    # Padded per-node path matrix: row n holds the segment rows of node n's path, source
    # outward; `active` marks the real slots.
    paths_idx, active = pad_paths([network.segments.index.get_indexer(list(network.paths[node])) for node in requested])

    # Warm-start length: each path's source-to-node travel time at the leading flow rate.
    # resolve_spinup discards the paths it cannot warm-start one by one -- a stagnant segment
    # makes its own path's warm start infinite -- so one closed tap leaves the other nodes
    # their padding. np.where rather than multiplying by `active`, because inf * 0 is NaN.
    with np.errstate(divide="ignore"):
        ratio = volume / network.segment_flow(flow=_running_start(demand)[:, None])[:, 0]
    per_path = np.sum(np.where(active, ratio[paths_idx], 0.0), axis=1)

    tedges, demand, n_pad = resolve_spinup(spinup, tedges=tedges, flow=demand, warm_start_days=per_path)
    transfer = paths_transfer(
        tedges_days=tedges_to_days(tedges),
        cout_tedges_days=tedges_to_days(cout_tedges, ref=tedges[0]),
        segment_volume=volume,
        segment_flow=network.segment_flow(flow=demand),
        segment_decay=decay,
        node_flow=network.node_flow(flow=demand, nodes=requested),
        paths_idx=paths_idx,
        active=active,
    )
    return requested, transfer, n_pad


def _running_start(flow: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Each endmember's first running demand -- what its warm start holds.

    A record that opens flowing warm-starts at its first observed value, as it always has.
    A record that opens *idle* has nothing at its first value to fill the pipes with, yet
    it opens on water that must have been delivered somehow; its warm start holds the
    first demand the record actually shows. An endmember that never flows keeps its
    (zero) first value and its path falls through to strict validity as before.

    Parameters
    ----------
    flow : ndarray
        Endmember demand of shape ``(n_endmembers, n_cin)``.

    Returns
    -------
    ndarray
        One demand per endmember.
    """
    first_running = np.argmax(flow > 0.0, axis=1)
    lead = flow[np.arange(len(flow)), first_running]
    return np.where((flow > 0.0).any(axis=1), lead, flow[:, 0])


def resolve_spinup(
    spinup: str | None,
    *,
    tedges: pd.DatetimeIndex,
    flow: npt.NDArray[np.floating],
    warm_start_days: npt.NDArray[np.floating],
) -> tuple[pd.DatetimeIndex, npt.NDArray[np.floating], int]:
    """Validate the ``spinup`` policy and prepend the warm-start bins it implies.

    ``"constant"`` extends the record backwards far enough to cover the longest path that can
    usefully be warm-started, holding every endmember demand at its first *running* value --
    the first observed one for a record that opens flowing, the first nonzero one for a
    record that opens idle -- so the earliest output bins are fed by a defined (if assumed)
    history instead of coming back NaN. The idle case matters most to a model carrying a
    memory of its own: a record opening on standing water otherwise opens on an unseeded
    state, and the first flow resumption books a violent record-opening transient. A path
    is discarded as a candidate when its warm start is undefined -- a zero or
    non-finite leading flow -- or when the padding it implies is so long that a constant
    history is not a meaningful assumption; those paths keep their strict-validity NaN. Both
    judgements are made **per path**, so one stagnant or unreachably deep branch costs only
    itself its warm start: a node's coverage must not depend on which other nodes the caller
    happened to ask for, nor on a sibling it shares no pipe with.

    Parameters
    ----------
    spinup : {"constant"} or None
        Policy. ``None`` returns the inputs unchanged.
    tedges : DatetimeIndex
        Input bin edges, length ``n_cin + 1``.
    flow : ndarray
        Endmember demand of shape ``(n_endmembers, n_cin)``.
    warm_start_days : ndarray
        Duration to cover for each path, normally its source-to-node travel time at the
        leading flow rate. Non-positive and non-finite entries are candidates that cannot be
        warm-started at all.

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
    if not bin_width_days > 0.0:
        return tedges, flow, 0
    # One extra bin so each path's source window for the earliest original output bin lies
    # strictly inside the padded range rather than touching its edge. Beyond the cap a
    # constant history is not a meaningful warm start (unphysical geometry or a near-stagnant
    # leading flow), so that path falls through to strict validity rather than allocate; the
    # padding is the longest of the candidates that survive, and an empty survivor set is
    # "no warm start".
    duration = np.asarray(warm_start_days, dtype=float)
    with np.errstate(invalid="ignore"):
        implied = np.ceil(duration / bin_width_days) + 1.0
    usable = (duration > 0.0) & (implied <= max(10_000, 10 * flow.shape[1]))
    n_pad = int(implied[usable].max(initial=0.0))
    if n_pad == 0:
        return tedges, flow, 0
    offsets = pd.TimedeltaIndex(bin_width * np.arange(n_pad, 0, -1))
    padded_tedges = (tedges[0] - offsets).append(tedges)
    lead = _running_start(flow)
    padded_flow = np.concatenate([np.repeat(lead[:, None], n_pad, axis=1), flow], axis=1)
    return padded_tedges, padded_flow, n_pad
