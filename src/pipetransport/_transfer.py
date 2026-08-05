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

Relaxation targets
------------------

Relaxation toward a per-segment, time-varying target -- heat exchange with the soil, where
the temperature "decays" toward the soil temperature instead of toward zero -- makes the
node value affine rather than linear in the source signal: ``c_node = W @ c_source + b``.
Within segment ``e`` a parcel obeys ``dT/du = -k_e (T - Tb_e(u))`` with ``Tb_e`` piecewise
constant on the input bins, and Abel summation of the piecewise integral gives the parcel
bias as a sum over path segments of

``Tb[B_exit] e^{-phi_down} - Tb[B_entry] e^{-phi_entry} - sum_j dTb[j] e^{-k_e (A_exit - tau_j) - phi_down}``

with ``phi_down`` the exponent from the segment's exit to the node, ``phi_entry`` the one
from its entry, ``B_entry``/``B_exit`` the input bins holding the entry and exit times, and
``j`` running over the bin edges crossed inside the segment, half-open ``(A_entry, A_exit]``.
The weights on the targets are non-negative and sum to ``1 - e^{-phi_total}``, so a constant
target telescopes exactly and zero rates give zero bias with no separate code path. Every
term is ``exp`` of a function affine in the departure time, so its label average over a cell
is the same closed form as the surviving fraction. The target-independent factors are built
once (:func:`paths_transfer` with ``with_target_terms=True``) and applied to concrete target
series by :func:`apply_segment_targets`, whose only sequential work is one exponentially
forgetting scan per segment.

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import lfilter

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


class TargetTerms(NamedTuple):
    """Target-independent factors of the affine bias, built alongside the operator.

    Everything a per-sweep :func:`apply_segment_targets` call needs, precomputed on the
    refined cell grid so that applying a new target series costs one scan per segment plus
    gathers -- the operator is never rebuilt. Cell means of invalid (zero-width) cells are
    zeroed here, so the apply step is NaN-free by construction.

    The per-depth factors are stored **ragged**: a row whose path is shorter than ``max_depth``
    contributes an exact zero at its trailing slots -- its entry and exit stages are the same
    floats, so the two target readings cancel and the interior sum is empty -- and carrying
    those slots is half the depth loop on a network of mixed path depths. Since a path occupies
    the leading slots of its row, the rows active at depth ``d`` shrink monotonically with
    ``d``; the rows are therefore stored once in order of decreasing path depth, and depth
    ``d``'s factors are the leading ``n_d`` rows of their slab rather than a gather. The same
    ordering is baked into :attr:`cell_weight` and :attr:`flat_cout`, so nothing has to be
    permuted back.

    Attributes
    ----------
    mean_down : list of ndarray
        Cell means of ``exp(-phi_down)``, one slab per path stage, ``max_depth + 1`` of them.
        Slab ``0`` holds the mean of ``exp(-phi_total)`` (the surviving fraction of the
        cell); slab ``d + 1`` the mean over the exponent from the exit of the depth-``d``
        segment to the node. The depth-``d`` bias reads its entry piece from slab ``d`` and
        its exit piece from slab ``d + 1``, both restricted to depth ``d``'s rows -- so slab
        ``d + 1`` carries exactly those rows and slab ``d`` at least them.
    mean_shift : list of ndarray
        Cell means of ``exp(-(k_d (A_exit(s) - tau[bin_exit]) + phi_down(s)))``, the
        factor of the interior-edge sum, one ``(n_d, n_cells)`` slab per depth. The
        exponent is non-negative by the half-open edge convention, so it never overflows.
    bin_entry, bin_exit : list of ndarray of int32
        Input bin holding the segment entry and exit time of each cell's parcels, one
        ``(n_d, n_cells)`` slab per depth; constant over a cell because the grid seeds the
        input edges at every node. Bin numbers are below ``n_cin``, so 32 bits are exact;
        :func:`apply_segment_targets` adds :attr:`row_offset` to reach the raveled target,
        and that sum widens to the platform index type on its own.
    gap : list of ndarray
        ``exp(-k_d dt (bin_exit - bin_entry))``, the factor carrying the forgetting scan from
        the entry bin to the exit bin, one ``(n_d, n_cells)`` slab per depth. It depends
        only on the operator, so it is built once here rather than per applied target set.
    row_offset : list of ndarray of intp
        Start of the depth-``d`` segment's row in the raveled target, ``paths_idx * n_cin``,
        one length-``n_d`` vector per depth. Stored per row rather than per cell so the flat
        index is never held at operator scale.
    cell_weight : ndarray
        Label width of each cell over the label span of its output slot, shape
        ``(n_nodes, n_cells)`` -- the same weight the ``W`` scatter uses, in the stored row
        order.
    flat_cout : ndarray of intp
        Flattened output slot of each cell in the dustbin layout, shape
        ``(n_nodes * n_cells,)``, in the stored row order. The slot values still name the
        operator's own rows, so the scatter lands where it did.
    segment_rate : ndarray
        The per-segment rates [1/day] the operator was built with, length ``n_seg``.
    dt_days : float
        Uniform input-bin width [days] the scan of :func:`apply_segment_targets` assumes.
    """

    mean_down: list[npt.NDArray[np.floating]]
    mean_shift: list[npt.NDArray[np.floating]]
    bin_entry: list[npt.NDArray[np.int32]]
    bin_exit: list[npt.NDArray[np.int32]]
    gap: list[npt.NDArray[np.floating]]
    row_offset: list[npt.NDArray[np.intp]]
    cell_weight: npt.NDArray[np.floating]
    flat_cout: npt.NDArray[np.intp]
    segment_rate: npt.NDArray[np.floating]
    dt_days: float


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
    target_terms : TargetTerms or None
        Target-independent bias factors, present when the operator was built with
        ``with_target_terms=True``; see :class:`TargetTerms`.
    """

    band_vals: npt.NDArray[np.floating]
    col_start: npt.NDArray[np.intp]
    valid_out: npt.NDArray[np.bool_]
    residence_time_out: npt.NDArray[np.floating]
    residence_time_in: npt.NDArray[np.floating]
    valid_in: npt.NDArray[np.bool_]
    target_terms: TargetTerms | None = None


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
    bin_end_rate: npt.NDArray[np.floating] | None = None,
    with_target_terms: bool = False,
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
    bin_end_rate : ndarray or None, optional
        Per-row rate ``w`` [1/day] of an extra reading weight ``exp(-w (t_end - t))``, where
        ``t`` is a parcel's arrival and ``t_end`` the right edge of the output bin it lands
        in. The row then reads the exponentially weighted bin average of its input rather
        than the plain one, which is what an enthalpy balance over a bin asks for. Length
        ``n_nodes``; ``None`` (default) is a plain reading and adds exactly zero.
    with_target_terms : bool, optional
        Also build the target-independent factors of the affine relaxation bias (see
        :class:`TargetTerms`); requires uniform ``tedges_days`` spacing, which the scan of
        :func:`apply_segment_targets` assumes. The operator itself is identical either
        way. Default False.

    Returns
    -------
    NetworkTransfer
        Banded operators, coverage masks and travel times; see :class:`NetworkTransfer`.

    Raises
    ------
    ValueError
        If ``with_target_terms`` is requested on a non-uniform input grid.

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
    :func:`_surviving_fraction` integrates between. Every exponent the bias factors are built
    from is affine across a cell for the same reason, so they extrapolate the same way.
    """
    n_cin = len(tedges_days) - 1
    n_cout = len(cout_tedges_days) - 1
    n_nodes, max_depth = paths_idx.shape
    dt_days = np.diff(tedges_days)
    # The tolerance has to clear the float64 representation of the grid, not just a genuinely
    # ragged one. Days since the record start grow, their spacing grows with them, and
    # differencing an exactly uniform hourly grid wobbles by 9e-13 relative after a year and
    # 1.5e-11 after twenty; a bin width that is a dyadic fraction of a day never wobbles at
    # all. A grid that is actually non-uniform is out by orders of magnitude more than this.
    if with_target_terms and not np.allclose(dt_days, dt_days[0], rtol=1e-9, atol=0.0):
        msg = "target terms require uniformly spaced tedges (the per-segment scan assumes one bin width)"
        raise ValueError(msg)

    # Per-segment cumulative volume, once per segment however many paths share it. Plateaus
    # from a closed valve make the volume-to-time inversion multi-valued, so they are
    # separated before the maps below invert them.
    segment_cumulative = cumulative_flow_volume(segment_flow, dt_days, strictly_monotone=True)
    # Label axis: cumulative throughflow past each reporting node. Only ever read forward
    # (time to label), so its plateaus are meaningful and stay untouched.
    node_cumulative = cumulative_flow_volume(node_flow, dt_days)

    def travel(times: npt.NDArray[np.floating], depth: int, *, downstream: bool) -> npt.NDArray[np.floating]:
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
        cumulative = segment_cumulative[paths_idx[:, depth]]
        volume = segment_volume[paths_idx[:, depth], None]
        target = _interp_rows(times, tedges_days, cumulative)
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
        mapped = _interp_rows(np.clip(target, low, high), cumulative, tedges_days)
        np.copyto(mapped, np.nan, where=outside)
        return np.where(active[:, depth, None], mapped, times)

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
        pts = travel(np.concatenate([pts, tedges_rows], axis=1), depth, downstream=False)
    grid = np.concatenate([pts, tedges_rows], axis=1)
    grid = np.sort(np.clip(np.where(np.isfinite(grid), grid, tedges_days[-1]), tedges_days[0], tedges_days[-1]), axis=1)

    # Forward sweep over the cell boundaries and the quarter points inside each cell. The
    # boundaries carry the label; the travel time, the decay exponent and the midpoint arrival
    # are read off the interior samples (see Notes). The per-depth stages are kept only when
    # the bias factors are requested -- and only over the interior samples, which is all the
    # exit-to-node exponents and the entry/exit bins are read off.
    n_edge = grid.shape[1]
    cell_width = np.diff(grid, axis=1)
    samples = np.concatenate([grid, grid[:, :-1] + 0.25 * cell_width, grid[:, :-1] + 0.75 * cell_width], axis=1)
    arrival = samples
    decay_exponent = np.zeros_like(samples)
    stage_arrival: list[npt.NDArray[np.floating]] = [samples[:, n_edge:]]
    stage_phi: list[npt.NDArray[np.floating]] = [decay_exponent[:, n_edge:]]
    for depth in range(max_depth):
        previous, arrival = arrival, travel(arrival, depth, downstream=True)
        decay_exponent += segment_decay[paths_idx[:, depth], None] * (arrival - previous)
        if with_target_terms:
            # `arrival` is a fresh array each depth, so a view suffices; the exponent
            # accumulates in place, so its stage is copied.
            stage_arrival.append(arrival[:, n_edge:])
            stage_phi.append(decay_exponent[:, n_edge:].copy())
    quarter_arrival = arrival[:, n_edge:].reshape(n_nodes, 2, -1)
    # Output-bin membership of every cell, read off the midpoint arrival. It is needed here,
    # before any exponent factor, because ``bin_end_rate`` measures its weight from that bin's
    # right edge; the refined grid carries the preimages of the output edges, so a cell lies
    # inside a single output bin and both its quarter samples share that edge -- which is what
    # keeps the weighted exponent affine across the cell, exactly as the plain one is. Cells
    # arriving past the output range are dustbins that the slices below drop; clamping their
    # lag at zero keeps every exponent non-negative rather than letting a long record
    # overflow one.
    arrival_mid = quarter_arrival.mean(axis=1)
    cout_bin = np.searchsorted(cout_tedges_days, arrival_mid, side="right") - 1
    bin_end = cout_tedges_days[np.clip(cout_bin, 0, n_cout - 1) + 1]
    weight = np.zeros(n_nodes) if bin_end_rate is None else np.asarray(bin_end_rate, dtype=float)
    quarter_phi = decay_exponent[:, n_edge:].reshape(n_nodes, 2, -1) + weight[:, None, None] * np.maximum(
        bin_end[:, None, :] - quarter_arrival, 0.0
    )
    cell_travel_time = (quarter_arrival - samples[:, n_edge:].reshape(n_nodes, 2, -1)).mean(axis=1)
    phi_lo, phi_hi = _cell_edges(quarter_phi)
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
    row_supported = edge_in_record[:-1] & edge_in_record[1:] & (cout_label_width > 0.0)

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
    cin_bin = np.clip(np.searchsorted(tedges_days, midpoint, side="right") - 1, 0, n_cin - 1)
    label_floor = _ROUNDTRIP_ULPS * np.spacing(node_cumulative[:, -1:])
    carrying = cell_ok & (label[:, 1:] - label[:, :-1] > label_floor)
    label_width = np.where(carrying, label[:, 1:] - label[:, :-1], 0.0)

    # An input bin is constrained only if every parcel leaving in it arrives inside the
    # record. Every scatter-add below runs on indices flattened with a per-node offset.
    node_offset = np.arange(n_nodes)[:, None]
    flat_in = (node_offset * n_cin + cin_bin).ravel()
    in_slots = n_nodes * n_cin
    in_volume = np.bincount(flat_in, weights=label_width.ravel(), minlength=in_slots).reshape(n_nodes, n_cin)
    carried_in = label_width * np.nan_to_num(cell_travel_time)
    in_travel = np.bincount(flat_in, weights=carried_in.ravel(), minlength=in_slots).reshape(n_nodes, n_cin)
    broken = np.bincount(flat_in, weights=(~cell_ok).ravel().astype(float), minlength=in_slots).reshape(n_nodes, n_cin)
    valid_in = (broken == 0.0) & (in_volume > 0.0)
    residence_time_in = np.where(valid_in, in_travel / np.where(valid_in, in_volume, 1.0), np.nan)

    # Cells arriving before the output range, after it, or outside the record drain into the
    # two dustbin slots wrapped around each node's real bins and are sliced away below.
    flat_cout = (node_offset * (n_cout + 2) + cout_bin + 1).ravel()
    out_slots = n_nodes * (n_cout + 2)

    # Cells are ordered by source time, and arrival, label and the input-bin index all
    # increase with it, so the cells of one output slot are a contiguous, non-decreasing run
    # -- globally, since the node offsets dominate. The band bounds are read off each run's
    # first and last cell instead of a scatter-minimum, over the carrying cells only:
    # compressing a sorted array keeps it sorted, and a run of non-carrying plateau cells
    # spans a closure while contributing nothing to it, so reading them would stretch every
    # band of the operator to the length of the longest closure.
    cin_flat = cin_bin.ravel()
    carry_cout, carry_cin = flat_cout[carrying.ravel()], cin_flat[carrying.ravel()]
    n_carry = carry_cout.size
    slots = np.arange(out_slots)
    run_lo = np.searchsorted(carry_cout, slots, side="left")
    run_hi = np.searchsorted(carry_cout, slots, side="right")
    populated = run_hi > run_lo
    safe_lo, safe_hi = np.clip(run_lo, 0, max(n_carry - 1, 0)), np.clip(run_hi - 1, 0, max(n_carry - 1, 0))
    col_start_all = np.where(populated, carry_cin[safe_lo], 0).astype(np.intp)
    col_stop_all = np.where(populated, carry_cin[safe_hi], 0)
    # The band width is shared across nodes and read off the real slots only: a dustbin run
    # may span the whole input range.
    spread = (col_stop_all - col_start_all).reshape(n_nodes, n_cout + 2)[:, 1:-1]
    full_band = int(spread.max(initial=0)) + 1

    # Every cell contribution is a share of its output bin's label span: that division is
    # what turns the label-uniform integral into the flow-weighted bin average. Cells the
    # label does not reach have zero width; their NaN decay exponents and travel times are
    # masked out rather than multiplied by it.
    cell_survive = _surviving_fraction(phi_lo, phi_hi)
    survived = np.where(carrying, label_width * cell_survive, 0.0)
    carried_out = np.where(carrying, label_width * cell_travel_time, 0.0)
    span = np.where(cout_label_width > 0.0, cout_label_width, 1.0)
    span_all = np.ones((n_nodes, n_cout + 2))
    span_all[:, 1:-1] = span
    # Dustbin cells may spread beyond the band; the clip keeps their scatter inside their own
    # (sliced-away) rows.
    slot = flat_cout * full_band + np.clip(cin_flat - col_start_all[flat_cout], 0, full_band - 1)
    band_vals = (
        np
        .bincount(slot, weights=survived.ravel() / span_all.ravel()[flat_cout], minlength=out_slots * full_band)
        .astype(float, copy=False)
        .reshape(n_nodes, n_cout + 2, full_band)[:, 1:-1]
    )
    coverage = (
        np.bincount(flat_cout, weights=label_width.ravel(), minlength=out_slots).reshape(n_nodes, n_cout + 2)[:, 1:-1]
        / span
    )
    out_travel = (
        np.bincount(flat_cout, weights=carried_out.ravel(), minlength=out_slots).reshape(n_nodes, n_cout + 2)[:, 1:-1]
        / span
    )

    valid_out = row_supported & (coverage >= 1.0 - _COVERAGE_TOLERANCE)
    band_vals[~valid_out] = 0.0
    col_start = col_start_all.reshape(n_nodes, n_cout + 2)[:, 1:-1]
    residence_time_out = np.where(valid_out, out_travel, np.nan)

    target_terms = None
    if with_target_terms:
        # Target-independent bias factors, read off the stored sweep stages at the quarter
        # points and extrapolated to the cell boundaries exactly as the decay exponent is.
        # phi_down -- the exponent from a segment's exit to the node -- is the difference of
        # two accumulator stages and is non-negative, because adding non-negative increments
        # to an accumulator is monotone in floating point. Cells the label does not reach get
        # their means zeroed here, so the apply step never touches a NaN. The entry and exit
        # bins are constant over a cell (the grid seeds tedges at every node), so the cell
        # midpoint reads them off; an arrival exactly on the record's final edge is
        # re-labelled to the last real bin by the clip, whose zero-length exit piece makes
        # that exact.
        #
        # Every slab is restricted to the rows still on a path at its depth, which the
        # decreasing-depth row order makes a leading slice (see :class:`TargetTerms`). Slab
        # ``d`` of mean_down is read as the entry piece at depth d and as the exit piece at
        # depth d - 1, and the latter needs the more rows, so it is built on those.
        n_cells = grid.shape[1] - 1
        # Rows in order of decreasing path depth, so "active at depth d" is a leading slice.
        # Stage l of mean_down is read as depth l's entry piece and as depth l - 1's exit
        # piece; the latter needs the more rows, so the stage is built on those.
        path_depth = active.sum(axis=1)
        order = np.argsort(-path_depth, kind="stable")
        stage_rows = [order[: int(np.count_nonzero(path_depth > max(stage - 1, 0)))] for stage in range(max_depth + 1)]
        mean_down = [np.where(carrying[stage_rows[0]], cell_survive[stage_rows[0]], 0.0)]
        # A segment's entry face is the exit face of the one before it, so the input bin of
        # every face is read once per stage rather than once per depth per side. The row order
        # is what makes the sharing free: stage l + 1 keeps a prefix of stage l's rows, so the
        # narrower read is a slice of the wider one rather than a second gather.
        faces = [
            np.clip(
                np.searchsorted(
                    tedges_days,
                    stage_arrival[stage].reshape(n_nodes, 2, n_cells)[stage_rows[stage]].mean(axis=1),
                    side="right",
                )
                - 1,
                0,
                n_cin - 1,
            ).astype(np.int32)
            for stage in range(max_depth + 1)
        ]
        mean_shift, bin_entry, bin_exit, gap, row_offset = [], [], [], [], []
        for depth in range(max_depth):
            rows = stage_rows[depth + 1]
            here = carrying[rows]
            phi_down = quarter_phi[rows] - stage_phi[depth + 1].reshape(n_nodes, 2, n_cells)[rows]
            mean_down.append(np.where(here, _surviving_fraction(*_cell_edges(phi_down)), 0.0))
            exits = stage_arrival[depth + 1].reshape(n_nodes, 2, n_cells)[rows]
            entry_bin, exit_bin = faces[depth][: rows.size], faces[depth + 1][: rows.size]
            # The interior-edge factor, shifted by the exit bin's left edge so the exponent
            # stays non-negative (up to the operator's roundtrip rounding) however long the
            # record is.
            rate = segment_decay[paths_idx[rows, depth], None, None]
            shift = rate * (exits - tedges_days[exit_bin][:, None, :]) + phi_down
            mean_shift.append(np.where(here, _surviving_fraction(*_cell_edges(shift)), 0.0))
            gap.append(np.exp(-rate[:, 0] * float(dt_days[0]) * (exit_bin - entry_bin)))
            bin_entry.append(entry_bin)
            bin_exit.append(exit_bin)
            row_offset.append(paths_idx[rows, depth] * n_cin)
        target_terms = TargetTerms(
            mean_down=mean_down,
            mean_shift=mean_shift,
            bin_entry=bin_entry,
            bin_exit=bin_exit,
            gap=gap,
            row_offset=row_offset,
            cell_weight=(label_width / span_all.ravel()[flat_cout].reshape(n_nodes, n_cells))[order],
            flat_cout=flat_cout.reshape(n_nodes, n_cells)[order].ravel(),
            segment_rate=segment_decay,
            dt_days=float(dt_days[0]),
        )
    return NetworkTransfer(
        band_vals, col_start, valid_out, residence_time_out, residence_time_in, valid_in, target_terms
    )


def apply_segment_targets(
    transfer: NetworkTransfer, segment_target: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Apply concrete per-segment relaxation targets to a prebuilt operator's bias factors.

    The affine model is ``c_node = W @ c_source + b``; this computes ``b`` for one target
    set. The only sequential work is one exponentially forgetting scan per segment,
    ``S[j] = S[j-1] * exp(-k dt) + (Tb[j] - Tb[j-1])``, whose partial sums turn the
    interior-edge terms of every cell into two gathers; everything else is elementwise on
    the stored cell factors and one scatter-add. All exponents are non-positive, so no
    input scale can overflow. A segment whose target never moves has no interior-edge sum at
    all, so its scan is skipped -- which is most of them when the caller carries inert copies
    of its segments.

    Parameters
    ----------
    transfer : NetworkTransfer
        Operator built by :func:`paths_transfer` with ``with_target_terms=True``.
    segment_target : ndarray
        Relaxation target of every segment [same unit as the source signal], piecewise
        constant on the input bins, shape ``(n_seg, n_cin)``.

    Returns
    -------
    ndarray
        Bias of each output bin, shape ``(n_nodes, n_cout)``. Zero where
        :attr:`NetworkTransfer.valid_out` is False, so adding it to ``W @ c_source``
        keeps the NaN policy of the caller intact.

    Raises
    ------
    ValueError
        If the operator was built without target terms.
    """
    terms = transfer.target_terms
    if terms is None:
        msg = "operator was built without target terms; pass with_target_terms=True to paths_transfer"
        raise ValueError(msg)
    target = np.asarray(segment_target, dtype=float)
    n_seg, n_cin = target.shape
    n_nodes, n_cout = transfer.valid_out.shape
    dt = terms.dt_days

    # Forgetting scan over the interior input edges, one fused C pass per segment.
    steps = np.diff(target, axis=1)
    rho = np.exp(-terms.segment_rate * dt)
    scan = np.zeros((n_seg, n_cin))
    for e in np.flatnonzero(steps.any(axis=1)):
        scan[e, 1:] = lfilter([1.0], [1.0, -rho[e]], steps[e])

    # Per depth: entry piece, exit piece and the scanned interior-edge sum of each cell, over
    # the rows still on a path at that depth -- the leading ``n_rows`` of the stored order,
    # so every slab is a slice and no row is gathered. The rows the slice drops contribute an
    # exact zero: their entry and exit stages are the same floats, so the two target readings
    # cancel and the interior sum spans no bins.
    #
    # The gathers read the raveled target and scan through a flat index, the segment's row
    # offset plus the stored bin. That is the same element ``take_along_axis`` would fetch,
    # but it neither materializes the broadcast index grid that call builds internally nor
    # copies ``target[seg]`` and ``scan[seg]`` first -- together about two thirds of the work
    # of this loop, which is in turn most of the module's runtime.
    flat_target, flat_scan = target.ravel(), scan.ravel()
    cell_bias = np.zeros_like(terms.cell_weight)
    for depth, offset in enumerate(terms.row_offset):
        n_rows = offset.size
        at_entry = terms.bin_entry[depth] + offset[:, None]
        at_exit = terms.bin_exit[depth] + offset[:, None]
        interior = flat_scan[at_exit] - flat_scan[at_entry] * terms.gap[depth]
        cell_bias[:n_rows] += (
            flat_target[at_exit] * terms.mean_down[depth + 1]
            - flat_target[at_entry] * terms.mean_down[depth][:n_rows]
            - terms.mean_shift[depth] * interior
        )

    out_slots = n_nodes * (n_cout + 2)
    bias = np.bincount(terms.flat_cout, weights=(cell_bias * terms.cell_weight).ravel(), minlength=out_slots).reshape(
        n_nodes, n_cout + 2
    )[:, 1:-1]
    return np.where(transfer.valid_out, bias, 0.0)


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
        ratio = volume / network.segment_flow(flow=demand)[:, 0]
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


def resolve_spinup(
    spinup: str | None,
    *,
    tedges: pd.DatetimeIndex,
    flow: npt.NDArray[np.floating],
    warm_start_days: npt.NDArray[np.floating],
) -> tuple[pd.DatetimeIndex, npt.NDArray[np.floating], int]:
    """Validate the ``spinup`` policy and prepend the warm-start bins it implies.

    ``"constant"`` extends the record backwards far enough to cover the longest path that can
    usefully be warm-started, holding every endmember demand at its first observed value, so
    the earliest output bins are fed by a defined (if assumed) history instead of coming back
    NaN. A path is discarded as a candidate when its warm start is undefined -- a zero or
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
    padded_flow = np.concatenate([np.repeat(flow[:, :1], n_pad, axis=1), flow], axis=1)
    return padded_tedges, padded_flow, n_pad
