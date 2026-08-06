r"""
Relaxation toward a moving target, layered on the transfer operator.

This module has no public API. It extends :mod:`pipetransport._transfer` with everything a
relaxation model needs and a plain transport model does not: the target-independent bias
factors, the reading kernels that weight a bin's contribution by the lag to its end, and the
content snapshots a moment hierarchy restarts from. The refined cell grid, the displacement
maps and the band scatter are not re-implemented here -- they are
:class:`~pipetransport._transfer._CellGrid`, whose attribute layout this module treats as a
versioned internal contract.

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
once by :func:`paths_transfer` and applied to concrete target
series by :func:`apply_segment_targets`, whose only sequential work is one exponentially
forgetting scan per segment.

A target may additionally carry a component linear in position along its segment -- the
first axial mode of a relaxation field that varies along the pipe (issue #32). The same
integration by parts then leaves only two new kinds of factor, both still closed-form on
the cells: jumps weighted by the crossing edge's position in the pipe, and the parcel's own
entry position, which is affine across a cell and integrates against the exponential factors
as a ramp mean. Everything else reuses the factors above; see
:func:`apply_segment_targets`.

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter
from scipy.special import gammainc, gammaln

from pipetransport._transfer import (
    NetworkTransfer,
    _cell_edges,
    _CellGrid,
)

# A segment's content snapshot is anchored on the water it actually delivered; the anchor is
# accepted once the cells covering it account for the pipe volume to within this relative
# slack, which only has to sit above the round-off of the interpolated cell boundaries.
_ANCHOR_TOLERANCE = 1e-8

# Where the incomplete reading kernels switch from their power series in the exponent to
# their incomplete-gamma closed forms. Below the crossover the series terms fall factorially
# and the closed form cancels; above it the closed form's alternating sum is bounded because
# the exponent itself bounds the offending ratios. Both branches hold round-off accuracy in
# an overlapping decade around 2, so this is a floating-point evaluation choice, not a model
# threshold. The series length is what full float64 accuracy at the crossover takes.
_SERIES_CROSSOVER = 2.0

# Chunk length of the position frames, in pipe volumes of throughflow. Any value strictly
# above one volume plus round-off keeps a traversal inside two consecutive chunks; 1.5
# leaves that margin wide without stretching the in-frame positions the scans carry.
_CHUNK_VOLUMES = 1.5


class TargetTerms(NamedTuple):
    """Target-independent factors of the affine bias, built alongside the operator.

    Everything a per-sweep :func:`apply_segment_targets` call needs, precomputed on the
    refined cell grid so that applying a new target series costs a few scans per segment
    plus gathers -- the operator is never rebuilt. Cell means of invalid (zero-width)
    cells are zeroed here, so the apply step is NaN-free by construction.

    A parcel's bias from one segment splits at the input-bin edges of its traversal into
    an entry piece, full interior bins, and an exit piece. The interior pieces are what
    the chunk-anchored scans of :func:`apply_segment_targets` sum; the two partial pieces
    and the position weights of the interior sum are per-cell closed forms stored here.
    Every per-cell quantity involved is affine in one of two per-cell exponents -- the
    exit-side lag ``s = k_d (A_exit - tau[bin_exit])`` and the entry-side lag
    ``s' = k_d (tau[bin_entry + 1] - A_entry)`` -- so every slab is a combination of
    cell means of powers of ``s`` or ``s'`` against the reading weight and the
    downstream decay, and no stored exponent can overflow.

    The record of each segment is chunked wherever at least 1.5 pipe volumes have flowed
    through since the last chunk started. A traversal displaces exactly one volume, so a
    parcel's interior bins span at most one chunk boundary, and every position that
    enters a stored slab or a scan is measured within a chunk -- bounded by the chunk
    span plus a bin, however long the record. That bound is what keeps the high modes at
    round-off accuracy: no power of a record-scale cumulative volume ever forms.

    The per-depth factors are stored **ragged**: a row whose path is shorter than
    ``max_depth`` contributes an exact zero at its trailing slots, and carrying those
    slots is half the depth loop on a network of mixed path depths. Since a path occupies
    the leading slots of its row, the rows active at depth ``d`` shrink monotonically with
    ``d``; the rows are therefore stored once in order of decreasing path depth, and depth
    ``d``'s factors are the leading ``n_d`` rows of their slab rather than a gather. The
    same ordering is baked into :attr:`cell_weight` and :attr:`flat_cout`, so nothing has
    to be permuted back.

    Attributes
    ----------
    exit_read : list of ndarray
        Per depth, shape ``(n_modes, n_d, n_cells)``: the exit partial piece per unit of
        the monomial series ``v_i`` read at the exit bin -- the cell mean of the
        incomplete kernel ``I_i(s) = integral_0^s (1 - lam q/(k V))**i e**-lam dlam``
        times the reading weight and ``exp(-phi_down)``.
    entry_read : list of ndarray
        Per depth, same shape: the entry partial piece per unit of ``v_i`` read at the
        entry bin, with the decay from the entry bin's right edge to the parcel's
        delivery folded into the stored exponent.
    position_exit : list of ndarray
        Per depth, shape ``(n_modes, n_d, n_cells)``: cell means of ``alpha_x**u`` times
        the interior factor ``exp(-(s + phi_down))`` and the reading weight, with
        ``alpha_x`` the parcel's volume fraction offset from its exit chunk's frame
        origin. Multiplies the chunked scans gathered at the exit edge.
    position_mid : list of ndarray
        Per depth, same shape: the previous chunk's counterpart -- the frame offset
        continued across the one chunk boundary a traversal can span, times the decay
        from that boundary to the exit edge. Zero for cells whose traversal stays inside
        one chunk. Multiplies the scans gathered at the chunk-start edge.
    position_entry : list of ndarray
        Per depth, same shape: cell means of ``alpha_e**u`` (the entry-side frame offset)
        times the interior factor continued through to the entry bin's right edge.
        Multiplies the scans' entry cutoff.
    bin_entry, bin_exit : list of ndarray of int32
        Input bin holding the segment entry and exit time of each cell's parcels, one
        ``(n_d, n_cells)`` slab per depth; constant over a cell because the grid seeds
        the input edges at every node.
    bin_mid : list of ndarray of int32
        Chunk-start edge of each cell's exit chunk, one ``(n_d, n_cells)`` slab per
        depth: where the :attr:`position_mid` scans are gathered.
    segment_of : list of ndarray of intp
        Segment row of each depth-``d`` path step, one length-``n_d`` vector per depth,
        in the stored row order.
    cell_weight : ndarray
        Label width of each cell over the label span of its output slot, shape
        ``(n_nodes, n_cells)`` -- the same weight the ``W`` scatter uses, in the stored
        row order.
    flat_cout : ndarray of intp
        Flattened output slot of each cell in the dustbin layout, shape
        ``(n_nodes * n_cells,)``, in the stored row order. The slot values still name the
        operator's own rows, so the scatter lands where it did.
    segment_rate : ndarray
        The per-segment rates [1/day] the operator was built with, length ``n_seg``.
    theta : ndarray
        Segment volume fractions displaced per bin ``q dt / V``, shape ``(n_seg, n_cin)``.
    theta_in : ndarray
        Each bin's left-edge volume fraction within its own chunk's frame, shape
        ``(n_seg, n_cin)``, bounded by the chunk span -- the position factor the scans
        carry.
    chunk_edge : ndarray of int32
        Chunk-start edge of the chunk holding each edge's preceding bin, shape
        ``(n_seg, n_cin + 1)``.
    rho_into : ndarray
        ``exp(-k dt (J - chunk_edge[J]))`` per edge, same shape: the decay across each
        edge's own chunk, closing the scans' chunk cutoff.
    etilde : ndarray
        ``integral_0^1 z**r exp(-k dt (1 - z)) dz`` per segment, shape
        ``(n_modes, n_seg)``: the position moments of one fully traversed bin, anchored
        at its downstream end.
    n_modes : int
        Number of axial modes the slabs were built for.
    dt_days : float
        Uniform input-bin width [days] the scans of :func:`apply_segment_targets` assume.
    """

    exit_read: list[npt.NDArray[np.floating]]
    entry_read: list[npt.NDArray[np.floating]]
    position_exit: list[npt.NDArray[np.floating]]
    position_mid: list[npt.NDArray[np.floating]]
    position_entry: list[npt.NDArray[np.floating]]
    bin_entry: list[npt.NDArray[np.int32]]
    bin_exit: list[npt.NDArray[np.int32]]
    bin_mid: list[npt.NDArray[np.int32]]
    segment_of: list[npt.NDArray[np.intp]]
    cell_weight: npt.NDArray[np.floating]
    flat_cout: npt.NDArray[np.intp]
    segment_rate: npt.NDArray[np.floating]
    theta: npt.NDArray[np.floating]
    theta_in: npt.NDArray[np.floating]
    chunk_edge: npt.NDArray[np.int32]
    rho_into: npt.NDArray[np.floating]
    etilde: npt.NDArray[np.floating]
    n_modes: int
    dt_days: float
    snapshots: SnapshotTerms | None = None


class SnapshotTerms(NamedTuple):
    """Factors reconstructing a segment's content moments at its chunk-anchor edges.

    A parcel inside its pipe when a chunk starts is delivered later in that chunk, so the
    content moments ``y_m(t_a)`` at anchor edge ``a`` are a sum over the *delivery* cells
    whose traversal spans ``a`` -- exactly the cells the chunk construction marks as
    crossing, each touching one anchor. Every cell contributes its water volume times the
    cell mean of ``P_m(2 xi_a - 1) T(t_a)``, with ``xi_a`` the parcel's fraction at the
    anchor (the stored exit-frame offset) and ``T(t_a)`` the *first half* of its delivered
    value: the source decayed to ``t_a``, the upstream segments' bias decayed to ``t_a``,
    and the own-segment bias accrued up to the anchor -- the entry partial plus the
    interior bins before the anchor, which the chunk-anchored scans already carry. Every
    exponent is the decay accrued up to ``t_a``, non-negative and bounded, which is what
    makes the snapshots the stable restart points of the moment recursions in
    :mod:`pipetransport.heat`.

    All slab stacks lead with the kernel or frame-offset order and then the anchor-fraction
    power ``u`` (the monomials of ``P_m(2 xi_a - 1)``); rows are the snapshot rows in
    their given order, cells the operator's grid. Cells not crossing an anchor are zeroed.

    Attributes
    ----------
    segment : ndarray of intp
        Segment of each snapshot row.
    row_pos : ndarray of intp
        Position of each snapshot row in the stored slab row order, valid at every depth
        of its path.
    own_depth : ndarray of intp
        Depth of the row's own segment -- its last path step.
    alive : ndarray of bool
        Cells contributing to an anchor.
    anchor : ndarray of int32
        Chunk-start edge each cell contributes to.
    raw_volume : ndarray
        Water volume of each cell [m3].
    dep_bin : ndarray of int32
        Source bin of each cell's parcels.
    anchor_valid : ndarray of bool
        Anchor edges whose in-pipe water is fully covered by in-record cells, per row,
        shape ``(n_rows, n_cin + 1)``.
    source_read : ndarray
        ``mean[xi_a**u exp(-decay to t_a)]`` per fraction power, ``(n_pow, n_rows,
        n_cells)``.
    own_entry : ndarray
        Own-segment entry partial to the anchor per kernel order and fraction power,
        ``(n_kernel, n_pow, n_rows, n_cells)``, the gap to the anchor folded in.
    own_pos_mid, own_pos_entry : ndarray
        Frame-offset power means for the own-segment interior sum to the anchor,
        ``(n_pow, n_pow, n_rows, n_cells)`` indexed ``[offset power, fraction power]``;
        the entry cutoff carries its gap.
    up_exit, up_entry : list of ndarray
        Upstream partial-piece slabs per depth, ``(n_kernel, n_pow, n_rows, n_cells)``,
        exponents re-anchored at ``t_a``; zero for rows whose own depth is not beyond.
    up_pos_exit, up_pos_mid, up_pos_entry : list of ndarray
        Upstream interior position slabs per depth, ``(n_pow, n_pow, n_rows, n_cells)``.
    """

    segment: npt.NDArray[np.intp]
    row_pos: npt.NDArray[np.intp]
    own_depth: npt.NDArray[np.intp]
    alive: npt.NDArray[np.bool_]
    anchor: npt.NDArray[np.int32]
    raw_volume: npt.NDArray[np.floating]
    dep_bin: npt.NDArray[np.int32]
    anchor_valid: npt.NDArray[np.bool_]
    unit_read: npt.NDArray[np.floating]
    source_read: npt.NDArray[np.floating]
    own_entry: npt.NDArray[np.floating]
    own_pos_mid: npt.NDArray[np.floating]
    own_pos_entry: npt.NDArray[np.floating]
    up_exit: list[npt.NDArray[np.floating]]
    up_entry: list[npt.NDArray[np.floating]]
    up_pos_exit: list[npt.NDArray[np.floating]]
    up_pos_mid: list[npt.NDArray[np.floating]]
    up_pos_entry: list[npt.NDArray[np.floating]]


def _e_table(n_max: int, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """``E_n(x) = integral_0^1 s**n exp(-x s) ds`` for ``n = 0..n_max``, elementwise in ``x``.

    The one moment family everything in this module integrates against. The recurrence
    ``E_n = (n E_{n-1} - exp(-x)) / x`` runs upward from the closed ``E_0``, which is
    stable exactly while ``n <= x``; above that the same recurrence runs downward from a
    seed at ``n_max``, stable exactly while ``n > x``, so each entry is read from the
    branch that is stable there. The seed is the closed form ``n! P(n+1, x) / x**(n+1)``
    (``P`` the regularized lower incomplete gamma) in log space for ``x >= 1``, and the
    alternating series -- whose terms fall factorially, truncated at the float64 floor --
    below, where the closed form's power of ``x`` cancels. ``x = 0`` returns exactly
    ``1/(n+1)`` and ``x = inf`` exactly 0. Only one special-function evaluation per ``x``
    for the whole table, which is what keeps large tables affordable.

    Parameters
    ----------
    n_max : int
        Highest moment order.
    x : ndarray
        Non-negative exponents, any shape.

    Returns
    -------
    ndarray
        ``E_n(x)`` of shape ``(n_max + 1, *x.shape)``.
    """
    x = np.asarray(x, dtype=float)
    decay = np.exp(-x)
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(x > 0.0, -np.expm1(-x) / x, 1.0)
        inverse = np.where(x > 0.0, 1.0 / x, 0.0)
    if n_max == 0:
        return first[None]

    upward = np.empty((n_max + 1, *x.shape))
    upward[0] = first
    # The upward recurrence is only read where n <= x; past that it runs away (to inf for
    # small x, harmlessly, hence the errstate) and the downward branch is read instead.
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(1, n_max + 1):
            upward[n] = (n * upward[n - 1] - decay) * inverse

    small = x < 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_x = np.log(np.where(small, 1.0, x))
        seed = np.exp(gammaln(n_max + 1.0) - (n_max + 1.0) * log_x) * gammainc(n_max + 1.0, x)
    xs = np.where(small, x, 0.0)
    series = np.zeros(x.shape)
    power = np.ones(x.shape)
    for j in range(20):
        if j:
            power *= -xs / j
        series += power / (n_max + j + 1)
    seed = np.where(small, series, seed)

    downward = np.empty_like(upward)
    downward[n_max] = seed
    # x = inf lands entirely in the upward branch (exactly 0); its inf * 0 here is unread.
    with np.errstate(invalid="ignore"):
        for n in range(n_max, 0, -1):
            downward[n - 1] = (x * downward[n] + decay) / n

    n_axis = np.arange(n_max + 1).reshape((n_max + 1, *([1] * x.ndim)))
    return np.where(n_axis <= x, upward, downward)


def _affine_multiply(
    coef: npt.NDArray[np.floating],
    offset: npt.NDArray[np.floating],
    slope: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Multiply a coefficient stack by the affine factor ``offset + slope * x``.

    Parameters
    ----------
    coef : ndarray
        Polynomial coefficients over the leading axis, shape ``(d + 1, ...)``.
    offset, slope : ndarray
        The affine factor's value at ``x = 0`` and its slope, broadcastable to the
        trailing shape.

    Returns
    -------
    ndarray
        Coefficients of the product, shape ``(d + 2, ...)``.
    """
    shape = np.broadcast_shapes(coef.shape[1:], np.shape(offset), np.shape(slope))
    out = np.zeros((coef.shape[0] + 1, *shape))
    out[:-1] = coef * offset
    out[1:] += coef * slope
    return out


class _CellBasis:
    """Moments of one anchored exponent over every cell, and contractions against them.

    Every closed form in the target machinery is the cell mean of ``P(x) exp(-phi(x))``
    with ``P`` a polynomial in the cell coordinate and ``phi`` affine across the cell. The
    mean is anchored at the lower-exponent end -- the coordinate is reversed where
    ``phi_hi < phi_lo`` -- so the moments ``exp(-min(phi)) E_k(|dphi|)`` never overflow,
    and every affine factor entering a polynomial must be read in the same orientation,
    which :meth:`affine` provides. The moment table is grown on demand.

    Parameters
    ----------
    phi_lo, phi_hi : ndarray
        The exponent at the two cell boundaries, non-negative, shape ``(n_rows, n_cells)``.
    """

    def __init__(self, phi_lo: npt.NDArray[np.floating], phi_hi: npt.NDArray[np.floating]) -> None:
        self.swap = phi_hi < phi_lo
        self.delta = np.abs(phi_hi - phi_lo)
        self.scale = np.exp(-np.minimum(phi_lo, phi_hi))
        self.moments = self.scale * _e_table(0, self.delta)

    def affine(
        self, lo: npt.NDArray[np.floating], hi: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Orient an affine quantity's endpoint pair to this basis's anchored coordinate.

        Parameters
        ----------
        lo, hi : ndarray
            The quantity at the cell's lower and upper boundary.

        Returns
        -------
        offset, slope : ndarray
            Its value at the anchored origin and its slope along the anchored coordinate.
        """
        start = np.where(self.swap, hi, lo)
        return start, np.where(self.swap, lo, hi) - start

    def mean(self, coef: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Cell mean of the polynomial ``coef`` (anchored coordinate) times ``exp(-phi)``.

        Parameters
        ----------
        coef : ndarray
            Coefficients of shape ``(d + 1, n_rows, n_cells)`` (trailing axes may
            broadcast).

        Returns
        -------
        ndarray
            The mean, shape ``(n_rows, n_cells)``.
        """
        if coef.shape[0] > self.moments.shape[0]:
            # Grown geometrically: the power tables raise the degree one multiply at a
            # time, and regrowing per degree would rebuild the table quadratically often.
            self.moments = self.scale * _e_table(max(coef.shape[0], 2 * self.moments.shape[0]) - 1, self.delta)
        return np.einsum("k...,k...->...", coef, self.moments[: coef.shape[0]])


def _shape_powers(
    phi_lo: npt.NDArray[np.floating],
    phi_hi: npt.NDArray[np.floating],
    first: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    n_first: int,
    fraction: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    n_shapes: int,
) -> npt.NDArray[np.floating]:
    """Cell means of ``first**a P_m(2 fraction - 1) exp(-phi)`` for all ``a``, ``m``.

    The Legendre shapes are built by Bonnet's recursion on the cell-local coefficient
    stacks, so their values stay of order one however high the mode -- expanding them in
    powers of the fraction instead squares the shapes' monomial conditioning into every
    mean, which is measurable noise by mode five.

    Parameters
    ----------
    phi_lo, phi_hi : ndarray
        Exponent at the two cell boundaries.
    first : tuple of ndarray
        Endpoint pair of the leading affine quantity.
    n_first : int
        Its highest power.
    fraction : tuple of ndarray
        Endpoint pair of the volume fraction the shapes take.
    n_shapes : int
        Number of shapes (highest degree plus one).

    Returns
    -------
    ndarray
        Table of shape ``(n_first + 1, n_shapes, *cells)``.
    """
    basis = _CellBasis(phi_lo, phi_hi)
    shape = np.broadcast_shapes(np.shape(phi_lo), np.shape(phi_hi))
    out = np.empty((n_first + 1, n_shapes, *shape))
    f_off, f_slope = basis.affine(*first)
    u_off, u_slope = basis.affine(2.0 * fraction[0] - 1.0, 2.0 * fraction[1] - 1.0)
    shapes = [np.ones((1, *shape))]
    for m in range(1, n_shapes):
        grown = _affine_multiply(shapes[-1], u_off, u_slope) * ((2 * m - 1) / m)
        if m > 1:
            grown[: shapes[-2].shape[0]] -= (m - 1) / m * shapes[-2]
        shapes.append(grown)
    for m in range(n_shapes):
        coef = shapes[m]
        for a in range(n_first + 1):
            out[a, m] = basis.mean(coef)
            if a < n_first:
                coef = _affine_multiply(coef, f_off, f_slope)
    return out


def _series_length(max_arg: float) -> int:
    """Terms an exponential-family series needs to reach the float64 floor at ``max_arg``.

    The series in this module are entire with terms falling like ``arg**j / j!``, so this
    is pure truncation control read off the data rather than a tolerance to tune.

    Parameters
    ----------
    max_arg : float
        Largest exponent argument the series will be evaluated at.

    Returns
    -------
    int
        Smallest term count whose next term is below 1e-17 at ``max_arg``.
    """
    length, term = 1, 1.0
    arg = max(float(max_arg), 1e-3)
    while term * arg / length > 1e-17 * max(1.0, np.exp(arg)):
        term *= arg / length
        length += 1
    return length + 1


class _WeightedBasis:
    """Cell means against one base exponent, threaded with each row's reading weight.

    Every reading weight this module supports is exactly ``Wp(lag) + We(lag) exp(-w lag)``
    with ``Wp`` and ``We`` polynomials in the lag to the output-bin end:

    - a plain row is ``Wp = 1``, ``We = 0``;
    - a moment row ``lag**p exp(-w lag)`` is ``We = lag**p``;
    - an integrated row ``g_p(lag) = integral_0^lag u**p exp(-w u) du`` is, below the
      series crossover, the entire series of ``g_p`` in ``Wp`` alone, and above it the
      closed form ``p!/w**(p+1)`` in ``Wp`` with the complement polynomial in ``We`` --
      the branch masks are baked into the coefficients per cell, so both shapes coexist
      in one row.

    The mean of any polynomial ``P(x)`` against weight and base exponent is then two
    anchored contractions, one against ``base`` and one against ``base + w lag``. The
    weight polynomials are held in the lag coordinate and composed into each anchored
    cell coordinate by the :meth:`powers` tables, whose extra exponents shift the
    anchors.

    Parameters
    ----------
    base_lo, base_hi : ndarray
        Base exponent (decay and shift factors, without the weight's own exponential) at
        the two cell boundaries, shape ``(n_rows, n_cells)``.
    lag_lo, lag_hi : ndarray
        Lag to the output-bin end at the boundaries, same shape.
    rate, power, integrated : ndarray
        Per-row weight spec: exponential rate [1/day], polynomial power, and whether the
        weight is the running integral of the kernel rather than the kernel itself.
    """

    def __init__(
        self,
        base_lo: npt.NDArray[np.floating],
        base_hi: npt.NDArray[np.floating],
        lag_lo: npt.NDArray[np.floating],
        lag_hi: npt.NDArray[np.floating],
        rate: npt.NDArray[np.floating],
        power: npt.NDArray[np.integer],
        integrated: npt.NDArray[np.bool_],
        poly_scale: npt.NDArray[np.floating] | None = None,
    ) -> None:
        self.base = (base_lo, base_hi)
        self.lag = (lag_lo, lag_hi)
        self.rate = rate[:, None]

        with np.errstate(invalid="ignore"):
            weight_arg = self.rate * 0.5 * (lag_lo + lag_hi)
        series_cell = weight_arg < _SERIES_CROSSOVER
        n_terms = _series_length(min(_SERIES_CROSSOVER, float(np.nanmax(weight_arg, initial=0.0))))
        p_max = int(power.max(initial=0))

        # Weight coefficients in the lag, as (per-row scalar) x (per-cell mask) terms.
        factorial = np.cumprod(np.concatenate([[1.0], np.arange(1.0, p_max + n_terms + 2)]))
        n_cells = series_cell.shape[-1]
        w_plain = np.zeros((p_max + n_terms + 2, len(rate), n_cells))
        w_exp = np.zeros((max(p_max + 1, 1), len(rate), n_cells))
        w_exp[0] += (~integrated & (power == 0))[:, None]
        for p in range(1, p_max + 1):
            w_exp[p] += (~integrated & (power == p))[:, None]
        for row in np.flatnonzero(integrated):
            p, w = int(power[row]), float(rate[row])
            choose = series_cell[row]
            sign = 1.0
            for j in range(n_terms):
                w_plain[p + 1 + j, row] += choose * sign / (factorial[j] * (p + j + 1))
                sign *= -w
            if not choose.all() and w > 0.0:
                top = factorial[p] / w ** (p + 1)
                w_plain[0, row] += ~choose * top
                for r in range(p + 1):
                    w_exp[r, row] -= ~choose * top * w**r / factorial[r]
        if poly_scale is not None:
            # An advected kernel is the plain one times ``scale**p`` -- for the running
            # integrals too, since the scale is constant over the reading's output bin.
            # Folding it into the coefficients here keeps every downstream reading of
            # order one, so no caller has to amplify a reading's rounding floor by a
            # power of the flow to undo the kernel's own smallness.
            factor = poly_scale ** power[:, None]
            w_plain *= factor
            w_exp *= factor
        self.weight_lag = (w_plain, w_exp)

    @staticmethod
    def _compose(
        basis: _CellBasis,
        lag_coef: npt.NDArray[np.floating],
        lag_lo: npt.NDArray[np.floating],
        lag_hi: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compose a polynomial in the lag with the lag's affine run across each cell.

        Parameters
        ----------
        basis : _CellBasis
            Basis whose anchored orientation the composition must follow.
        lag_coef : ndarray
            Coefficients in the lag, shape ``(d + 1, n_rows, n_cells or 1)``.
        lag_lo, lag_hi : ndarray
            Lag endpoints per cell.

        Returns
        -------
        ndarray
            Coefficients in the anchored cell coordinate, shape ``(d + 1, n_rows, n_cells)``.
        """
        offset, slope = basis.affine(lag_lo, lag_hi)
        out = lag_coef[-1] * np.ones_like(offset)[None]
        for coef in lag_coef[-2::-1]:
            out = _affine_multiply(out, offset, slope)
            out[0] += coef
        return out

    def powers(
        self,
        qty_lo: npt.NDArray[np.floating],
        qty_hi: npt.NDArray[np.floating],
        n_powers: int,
        extra_lo: npt.NDArray[np.floating] | float = 0.0,
        extra_hi: npt.NDArray[np.floating] | float = 0.0,
    ) -> npt.NDArray[np.floating]:
        """Cell means of ``qty**m * weight * exp(-(base + extra))`` for ``m = 0..n_powers``.

        Parameters
        ----------
        qty_lo, qty_hi : ndarray
            Endpoints of the affine quantity whose powers are read.
        n_powers : int
            Highest power.
        extra_lo, extra_hi : ndarray or float, optional
            An additional affine exponent, folded exactly into the anchored base -- no
            series is involved. Default 0.

        Returns
        -------
        ndarray
            Table of shape ``(n_powers + 1, n_rows, n_cells)``.
        """
        base_lo, base_hi = self.base
        lag_lo, lag_hi = self.lag
        exponents = (
            (base_lo + extra_lo, base_hi + extra_hi),
            (base_lo + extra_lo + self.rate * lag_lo, base_hi + extra_hi + self.rate * lag_hi),
        )
        out = np.zeros((n_powers + 1, *np.broadcast_shapes(np.shape(base_lo), np.shape(base_hi))))
        for (phi_lo, phi_hi), weight in zip(exponents, self.weight_lag, strict=True):
            basis = _CellBasis(phi_lo, phi_hi)
            coef = self._compose(basis, weight, lag_lo, lag_hi)
            offset, slope = basis.affine(qty_lo, qty_hi)
            for m in range(n_powers + 1):
                out[m] += basis.mean(coef)
                if m < n_powers:
                    coef = _affine_multiply(coef, offset, slope)
        return out


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
    bin_end_power: npt.NDArray[np.integer] | None = None,
    bin_end_integrated: npt.NDArray[np.bool_] | None = None,
    n_target_modes: int = 1,
    snapshot_rows: npt.NDArray[np.intp] | None = None,
    bin_end_scale: npt.NDArray[np.floating] | None = None,
) -> tuple[NetworkTransfer, TargetTerms]:
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
    bin_end_power : ndarray of int or None, optional
        Rows whose reading weight carries the additional factor ``(t_end - t)**p`` [days**p]
        -- the ``p``-th time moment of the bin's driving, which the axial moment budgets
        read through. The factor is affine across a cell to the ``p``-th power, so every
        cell mean stays closed-form. Length ``n_nodes``; ``None`` (default) is all 0.
    bin_end_integrated : ndarray of bool or None, optional
        Rows whose weight is the *running integral* of the kernel,
        ``g_p(lag) = integral_0^lag u**p exp(-w u) du``, rather than the kernel itself --
        what the time-integrated content of a bin reads through. Evaluated by its entire
        series below the crossover of :data:`_SERIES_CROSSOVER` and its incomplete-gamma
        closed form above, per cell. Length ``n_nodes``; ``None`` (default) is all False.
    n_target_modes : int, optional
        Number of axial target modes the bias factors are built for; see
        :func:`apply_segment_targets`. Default 1, the position-uniform target.
    snapshot_rows : ndarray of intp or None, optional
        Rows -- each reading a segment's own delivery, plain weight -- for which the
        content-snapshot factors of :class:`SnapshotTerms` are also built. ``None``
        (default) builds none.
    bin_end_scale : ndarray or None, optional
        Per-row volume [m3] dividing the flow of a reading's own output bin into the
        kernel's polynomial factor, which becomes ``((t_end - t) q_bin / scale)**p`` --
        the advected moment kernels the content budgets read, of order one however fine
        the bins. Length ``n_nodes``; ``None`` (default) keeps the plain ``(t_end - t)**p``.

    Returns
    -------
    transfer : NetworkTransfer
        Banded operators, coverage masks and travel times, exactly as
        :func:`pipetransport._transfer.paths_transfer` would have built them.
    target_terms : TargetTerms
        Target-independent factors of the affine relaxation bias. The two are returned
        together because one sweep of the same cells builds both.

    Raises
    ------
    ValueError
        If the input grid is not uniformly spaced.

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
    if not np.allclose(dt_days, dt_days[0], rtol=1e-9, atol=0.0):
        msg = "target terms require uniformly spaced tedges (the per-segment scan assumes one bin width)"
        raise ValueError(msg)

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
    grid = cells.grid
    carrying, label_width = cells.carrying, cells.label_width
    cin_bin, cout_bin, flat_cout, span_all = cells.cin_bin, cells.cout_bin, cells.flat_cout, cells.span_all
    quarter_phi, phi_lo, phi_hi = cells.quarter_phi, cells.phi_lo, cells.phi_hi
    residence_time_in, valid_in = cells.residence_time_in, cells.valid_in

    # The reading weight of each row is a function of the lag to the output-bin end alone;
    # the lag is affine across a cell exactly as the decay exponent is, so every weighted
    # cell mean stays closed-form. The decay exponent is kept pure here -- the integrated
    # weights carry their own exponential inside the kernel, so the weight machinery owns
    # the split between the two.
    bin_end = cout_tedges_days[np.clip(cout_bin, 0, n_cout - 1) + 1]
    weight_rate = np.zeros(n_nodes) if bin_end_rate is None else np.asarray(bin_end_rate, dtype=float)
    weight_power = np.zeros(n_nodes, dtype=int) if bin_end_power is None else np.asarray(bin_end_power, dtype=int)
    weight_integrated = (
        np.zeros(n_nodes, dtype=bool) if bin_end_integrated is None else np.asarray(bin_end_integrated, dtype=bool)
    )
    lag_to_bin_end = np.maximum(bin_end[:, None, :] - cells.quarter_arrival, 0.0)
    lag_lo, lag_hi = _cell_edges(lag_to_bin_end)

    poly_scale = None
    if bin_end_scale is not None:
        poly_scale = (
            np.take_along_axis(node_flow, np.clip(cout_bin, 0, n_cin - 1), axis=1)
            / np.asarray(bin_end_scale, dtype=float)[:, None]
        )
    # One route, weighted or not. At zero rate and power the basis contracts against
    # ``E_0``, which is the closed form :func:`~pipetransport._transfer._surviving_fraction`
    # evaluates -- bitwise, not merely to a tolerance, so an unweighted reading needs no
    # branch of its own.
    band_weights = _WeightedBasis(
        phi_lo, phi_hi, lag_lo, lag_hi, weight_rate, weight_power, weight_integrated, poly_scale
    )
    cell_survive = band_weights.powers(np.zeros_like(phi_lo), np.zeros_like(phi_hi), 0)[0]
    band_vals, col_start, valid_out, residence_time_out = cells.bands(cell_survive)

    # The per-depth stages of the forward sweep, re-run here rather than kept by the
    # grid. Storing them for every depth costs the sample array again per segment --
    # gigabytes on a long fine record -- which a plain build must never pay, and the
    # sweep is a few per cent of what the bias factors below cost. ``travel`` returns a
    # fresh array each depth, so an arrival stage is a view; the exponent accumulates in
    # place, so its stage is copied.
    arrival = cells.samples
    phi = np.zeros_like(arrival)
    stage_arrival = [arrival[:, cells.n_edge :]]
    stage_phi = [phi[:, cells.n_edge :].copy()]
    for depth in range(max_depth):
        previous, arrival = arrival, cells.travel(arrival, depth, downstream=True)
        phi += segment_decay[paths_idx[:, depth], None] * (arrival - previous)
        stage_arrival.append(arrival[:, cells.n_edge :])
        stage_phi.append(phi[:, cells.n_edge :].copy())
    # Target-independent bias factors, read off those sweep stages at the quarter
    # points and extrapolated to the cell boundaries exactly as the decay exponent is.
    # phi_down -- the exponent from a segment's exit to the node -- is the difference of
    # two accumulator stages and is non-negative, because adding non-negative increments
    # to an accumulator is monotone in floating point. Cells the label does not reach get
    # their means zeroed here, so the apply step never touches a NaN. The entry and exit
    # bins are constant over a cell (the grid seeds tedges at every node), so the cell
    # midpoint reads them off; an arrival exactly on the record's final edge is
    # re-labelled to the last real bin by the clip, whose zero-length exit piece makes
    # that exact.
    n_modes = int(n_target_modes)
    n_cells = grid.shape[1] - 1
    binomial = np.zeros((n_modes + 1, n_modes + 1))
    binomial[:, 0] = 1.0
    for row in range(1, n_modes + 1):
        binomial[row, 1:] = binomial[row - 1, 1:] + binomial[row - 1, :-1]
    factorial = np.cumprod(np.concatenate([[1.0], np.arange(1.0, n_modes + 40.0)]))
    # Rows in order of decreasing path depth, so "active at depth d" is a leading slice.
    path_depth = active.sum(axis=1)
    order = np.argsort(-path_depth, kind="stable")
    stage_rows = [order[: int(np.count_nonzero(path_depth > max(stage - 1, 0)))] for stage in range(max_depth + 1)]
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
    # Chunk every segment's record at >= _CHUNK_VOLUMES pipe volumes of throughflow
    # since the last chunk start. A traversal displaces exactly one volume, so a
    # parcel's interior bins span at most one chunk boundary, and every position below
    # is measured within a chunk -- bounded however long the record grows.
    n_seg_all = len(segment_volume)
    dt0 = float(dt_days[0])
    theta = segment_flow * dt0 / segment_volume[:, None]
    big_theta = np.concatenate([np.zeros((n_seg_all, 1)), np.cumsum(theta, axis=1)], axis=1)
    chunk_of_bin = np.zeros((n_seg_all, n_cin), dtype=np.int32)
    for e in range(n_seg_all):
        start = 0
        row = big_theta[e]
        for j in range(1, n_cin):
            if row[j] - row[start] >= _CHUNK_VOLUMES:
                start = j
            chunk_of_bin[e, j] = start
    chunk_edge = np.concatenate([np.zeros((n_seg_all, 1), dtype=np.int32), chunk_of_bin], axis=1)
    theta_in = big_theta[:, :-1] - np.take_along_axis(big_theta, chunk_of_bin, axis=1)
    theta_in_edge = big_theta - np.take_along_axis(big_theta, chunk_edge.astype(np.intp), axis=1)
    rho_into = np.exp(-segment_decay[:, None] * dt0 * (np.arange(n_cin + 1)[None, :] - chunk_edge))
    # Snapshot rows keep one fixed position in every depth's slabs (the stored order is
    # shared), so the per-depth quantities their snapshots need are captured as row
    # slices during the loop and assembled afterwards.
    snap_pos = None
    snap_rows = np.zeros(0, dtype=np.intp) if snapshot_rows is None else np.asarray(snapshot_rows, dtype=np.intp)
    snap_depth = path_depth[snap_rows] - 1
    snap_stash: list[dict[str, npt.NDArray]] = []
    if snap_rows.size:
        pos_of_row = np.empty(n_nodes, dtype=np.intp)
        pos_of_row[order] = np.arange(n_nodes)
        snap_pos = pos_of_row[snap_rows]
    exit_read, entry_read, position_exit, position_mid, position_entry = [], [], [], [], []
    bin_entry, bin_exit, bin_mid, segment_of = [], [], [], []
    for depth in range(max_depth):
        rows = stage_rows[depth + 1]
        here = carrying[rows]
        segs = paths_idx[rows, depth]
        rate_row = segment_decay[segs][:, None]
        entry_bin, exit_bin = faces[depth][: rows.size], faces[depth + 1][: rows.size]
        phi_down = quarter_phi[rows] - stage_phi[depth + 1].reshape(n_nodes, 2, n_cells)[rows]
        exits = stage_arrival[depth + 1].reshape(n_nodes, 2, n_cells)[rows]
        entries = stage_arrival[depth].reshape(n_nodes, 2, n_cells)[rows]
        down_lo, down_hi = _cell_edges(phi_down)
        # The two anchor exponents of one traversal: the exit-side lag from the exit
        # bin's left edge, and the entry-side lag to the entry bin's right edge. Every
        # parcel-dependent quantity of the bias is proportional to one of them --
        # positions through the local flow over ``k V``, the partial-piece kernels
        # directly -- so both are computed from within-bin time differences and no
        # power of a record-scale cumulative volume ever forms.
        s_lo, s_hi = _cell_edges(rate_row[:, None, :] * (exits - tedges_days[exit_bin][:, None, :]))
        sp_lo, sp_hi = _cell_edges(rate_row[:, None, :] * (tedges_days[entry_bin + 1][:, None, :] - entries))
        with np.errstate(divide="ignore", invalid="ignore"):
            invk_exit = np.where(
                rate_row > 0.0,
                segment_flow[segs[:, None], exit_bin] / (rate_row * segment_volume[segs][:, None]),
                0.0,
            )
            invk_entry = np.where(
                rate_row > 0.0,
                segment_flow[segs[:, None], entry_bin] / (rate_row * segment_volume[segs][:, None]),
                0.0,
            )
        # The interior-piece exponent continued to the entry bin's right edge; for a
        # traversal inside one bin it runs one bin backwards, which is the exact
        # over-subtraction the two partial pieces over-count. Fold it into the stored
        # exponent so the largest scale any slab holds is ``exp(k dt)`` of one bin.
        span_bins = rate_row * float(dt_days[0]) * (exit_bin - entry_bin - 1)
        spec = (
            weight_rate[rows],
            weight_power[rows],
            weight_integrated[rows],
            None if poly_scale is None else poly_scale[rows],
        )
        # The partial-piece kernels carry their own exponential inside the lambda
        # integral, so their outer exponent is the downstream decay alone (plus, on the
        # entry side, the decay from the entry bin's right edge to the delivery); the
        # interior position slabs additionally decay through the exit-side lag itself.
        side_exit = _WeightedBasis(down_lo, down_hi, lag_lo[rows], lag_hi[rows], *spec)
        side_entry = _WeightedBasis(
            s_lo + span_bins + down_lo, s_hi + span_bins + down_hi, lag_lo[rows], lag_hi[rows], *spec
        )
        # Exit partial: gamma-kernel means per power of the local ``1/kappa``, series
        # where the exit-side lag is small and the incomplete-gamma form where it is
        # not; both branches are combinations of the same power tables.
        s_mid = 0.5 * (s_lo + s_hi)
        with np.errstate(invalid="ignore"):
            series_cells = s_mid < _SERIES_CROSSOVER
        n_terms = _series_length(min(_SERIES_CROSSOVER, float(np.nanmax(s_mid, initial=0.0))))
        t_exit = side_exit.powers(s_lo, s_hi, n_modes + n_terms)
        t_exit_shift = side_exit.powers(s_lo, s_hi, n_modes - 1, extra_lo=s_lo, extra_hi=s_hi)
        gamma_mean = np.empty((n_modes, rows.size, n_cells))
        for low in range(n_modes):
            series_val = np.zeros((rows.size, n_cells))
            sign = 1.0
            for j in range(n_terms):
                series_val += sign / (factorial[j] * (low + j + 1)) * t_exit[low + 1 + j]
                sign = -sign
            closed_val = t_exit[0].copy()
            for r in range(low + 1):
                closed_val -= t_exit_shift[r] / factorial[r]
            gamma_mean[low] = np.where(series_cells, series_val, factorial[low] * closed_val)
        # Entry partial: the kernel ``J_i(s')`` by its entire series where the
        # entry-side lag is small -- each term bounded, because ``invk s'`` is the
        # entry-edge fraction -- and by its ``i! (-1)**(i+1) (e**-s' - sum)`` closed
        # form where it is not, where the local ``invk`` cannot amplify the
        # cancellation.
        sp_mid = 0.5 * (sp_lo + sp_hi)
        with np.errstate(invalid="ignore"):
            series_entry = sp_mid < _SERIES_CROSSOVER
        n_terms_entry = _series_length(min(_SERIES_CROSSOVER, float(np.nanmax(sp_mid, initial=0.0))))
        t_entry = side_entry.powers(sp_lo, sp_hi, n_modes + n_terms_entry)
        t_entry_shift = side_entry.powers(sp_lo, sp_hi, 0, extra_lo=sp_lo, extra_hi=sp_hi)[0]
        # The exit chunk's frame data per cell: the frame offset of the parcel's exit
        # fraction, the same offset carried across the one chunk boundary a traversal
        # can span, and the entry-side frame offset closing the scans' cutoff.
        frame_exit = 1.0 - theta_in_edge[segs[:, None], exit_bin]
        chunk_start = chunk_edge[segs[:, None], exit_bin]
        crossing = entry_bin < chunk_start
        frame_mid = frame_exit - theta_in_edge[segs[:, None], chunk_start]
        decay_mid = np.where(crossing, rho_into[segs[:, None], exit_bin], 0.0)
        frame_entry = -theta_in_edge[segs[:, None], entry_bin + 1]
        # A traversal inside one bin has no interior bins: its scan terms must cancel
        # exactly, which the shared chunk frame provides -- except when the exit bin
        # itself starts a chunk, where the exit gather and the entry cutoff live in
        # different frames. There the cutoff alone books the single over-counted bin,
        # and the exit-side scan weights are zeroed.
        split_frame = (exit_bin == entry_bin) & (chunk_edge[segs[:, None], exit_bin + 1] == exit_bin)
        slab_exit = np.zeros((n_modes, rows.size, n_cells))
        slab_entry = np.zeros((n_modes, rows.size, n_cells))
        slab_pos_exit = np.zeros((n_modes, rows.size, n_cells))
        slab_pos_mid = np.zeros((n_modes, rows.size, n_cells))
        slab_pos_entry = np.zeros((n_modes, rows.size, n_cells))
        for i in range(n_modes):
            for low in range(i + 1):
                slab_exit[i] += binomial[i, low] * (-invk_exit) ** low * gamma_mean[low]
                # alpha = frame - invk s, so the position means read the power tables.
                slab_pos_exit[i] += binomial[i, low] * frame_exit ** (i - low) * (-invk_exit) ** low * t_exit_shift[low]
                slab_pos_mid[i] += binomial[i, low] * frame_mid ** (i - low) * (-invk_exit) ** low * t_exit_shift[low]
                slab_pos_entry[i] += binomial[i, low] * frame_entry ** (i - low) * invk_entry**low * t_entry[low]
            series_val = np.zeros((rows.size, n_cells))
            sign = 1.0
            for j in range(n_terms_entry):
                series_val += sign / factorial[i + 1 + j] * t_entry[i + 1 + j]
                sign = -sign
            closed_val = t_entry_shift.copy()
            for m in range(i + 1):
                closed_val -= (-1.0) ** m / factorial[m] * t_entry[m]
            closed_val *= (-1.0) ** (i + 1)
            slab_entry[i] = invk_entry**i * factorial[i] * np.where(series_entry, series_val, closed_val)
            slab_pos_mid[i] *= decay_mid
        exit_read.append(np.where(here, slab_exit, 0.0))
        entry_read.append(np.where(here, slab_entry, 0.0))
        position_exit.append(np.where(here & ~split_frame, slab_pos_exit, 0.0))
        position_mid.append(np.where(here & ~split_frame, slab_pos_mid, 0.0))
        position_entry.append(np.where(here, slab_pos_entry, 0.0))
        bin_entry.append(entry_bin)
        bin_exit.append(exit_bin)
        bin_mid.append(chunk_start)
        segment_of.append(segs)
        if snap_rows.size:
            snap_stash.append({
                "s_lo": s_lo,
                "s_hi": s_hi,
                "sp_lo": sp_lo,
                "sp_hi": sp_hi,
                "down_lo": down_lo,
                "down_hi": down_hi,
                "invk_exit": invk_exit,
                "invk_entry": invk_entry,
                "frame_exit": frame_exit,
                "frame_mid": frame_mid,
                "frame_entry": frame_entry,
                "decay_mid": decay_mid,
                "series_cells": series_cells,
                "series_entry": series_entry,
                "split_frame": split_frame,
                "here": here,
                "entry_bin": entry_bin,
                "exit_bin": exit_bin,
                "chunk_start": chunk_start,
                "span_bins": span_bins,
            })
    snapshots = None
    if snap_pos is not None:
        # Content-snapshot factors, one Python pass per snapshot row: each row's cell
        # count is the data axis, so the loops below are over mode counts and depths
        # only. A cell contributes to the unique chunk-start edge inside its traversal
        # -- either its exit chunk's start, or its exit bin itself when that bin opens
        # a chunk -- and every stored exponent is the decay accrued up to that anchor.
        n_snap = snap_rows.size
        top = n_modes - 1
        shape4 = (n_modes, n_modes, n_snap, n_cells)
        alive = np.zeros((n_snap, n_cells), dtype=bool)
        anchor = np.zeros((n_snap, n_cells), dtype=np.int32)
        raw_volume = np.zeros((n_snap, n_cells))
        snap_dep = np.zeros((n_snap, n_cells), dtype=np.int32)
        anchor_valid = np.zeros((n_snap, n_cin + 1), dtype=bool)
        unit_read = np.zeros((n_modes, n_snap, n_cells))
        source_read = np.zeros((n_modes, n_snap, n_cells))
        own_entry = np.zeros(shape4)
        own_pos_mid = np.zeros(shape4)
        own_pos_entry = np.zeros(shape4)
        up_exit: list[npt.NDArray[np.floating]] = [np.zeros(shape4) for _ in range(max_depth)]
        up_entry: list[npt.NDArray[np.floating]] = [np.zeros(shape4) for _ in range(max_depth)]
        up_pos_exit: list[npt.NDArray[np.floating]] = [np.zeros(shape4) for _ in range(max_depth)]
        up_pos_mid: list[npt.NDArray[np.floating]] = [np.zeros(shape4) for _ in range(max_depth)]
        up_pos_entry: list[npt.NDArray[np.floating]] = [np.zeros(shape4) for _ in range(max_depth)]
        flat = np.zeros(n_cells)
        for r in range(n_snap):
            own = int(snap_depth[r])
            pos = int(snap_pos[r])
            orig = int(snap_rows[r])
            st = {key: value[pos] for key, value in snap_stash[own].items()}
            seg = int(paths_idx[orig, own])
            kdt_e = float(segment_decay[seg]) * dt0
            bx, be, chunk_start = st["exit_bin"], st["entry_bin"], st["chunk_start"]
            crossing = be < chunk_start
            exit_opens = chunk_edge[seg, bx + 1] == bx
            a_edge = np.where(crossing, chunk_start, np.where(exit_opens & (be < bx), bx, -1))
            live = (a_edge >= 0) & st["here"]
            a_idx = np.maximum(a_edge, 0)

            def safe(lo: npt.NDArray, hi: npt.NDArray, keep: npt.NDArray = live) -> tuple:
                """Zero an endpoint pair outside the live cells, so no NaN or overflow forms.

                Parameters
                ----------
                lo, hi : ndarray
                    Endpoint pair.
                keep : ndarray
                    Cells to keep.

                Returns
                -------
                tuple of ndarray
                    The sanitized pair.
                """
                return (np.where(keep, lo, 0.0), np.where(keep, hi, 0.0))

            xi_shift = big_theta[seg, bx] - big_theta[seg, a_idx]
            xi = safe(1.0 - xi_shift - st["invk_exit"] * st["s_lo"], 1.0 - xi_shift - st["invk_exit"] * st["s_hi"])
            tail = safe(st["s_lo"] + kdt_e * (bx - a_idx), st["s_hi"] + kdt_e * (bx - a_idx))
            gap_entry = np.where(live, kdt_e * (a_idx - be - 1.0), 0.0)
            alpha_off = safe(xi[0] - theta_in_edge[seg, a_idx], xi[1] - theta_in_edge[seg, a_idx])
            sp = safe(st["sp_lo"], st["sp_hi"])
            unit_read[:, r] = _shape_powers(flat, flat, (flat, flat), 0, xi, n_modes)[0]
            psi = safe(phi_lo[orig] - tail[0], phi_hi[orig] - tail[1])
            source_read[:, r] = _shape_powers(psi[0], psi[1], (flat, flat), 0, xi, n_modes)[0]

            def entry_kernel(
                base_lo: npt.NDArray,
                base_hi: npt.NDArray,
                s_pair: tuple,
                xi_pair: tuple,
                invke: npt.NDArray,
                frame: npt.NDArray,
                series_mask: npt.NDArray,
            ) -> tuple[npt.NDArray, npt.NDArray]:
                """Entry-partial and entry-position slabs of one segment step to the anchor.

                Parameters
                ----------
                base_lo, base_hi : ndarray
                    Outer exponent, without the kernel's own lambda integral.
                s_pair, xi_pair : tuple
                    Endpoint pairs of the entry-side lag and the anchor fraction.
                invke : ndarray
                    Local flow over ``k V`` at the entry bin.
                frame : ndarray
                    Entry-side frame offset.
                series_mask : ndarray
                    Cells evaluated by the entire series rather than the closed form.

                Returns
                -------
                partial, position : ndarray
                    Slabs of shape ``(n_modes, n_modes, n_cells)``.
                """
                length = _series_length(
                    min(_SERIES_CROSSOVER, float(np.nanmax(0.5 * (s_pair[0] + s_pair[1]), initial=0.0)))
                )
                table = _shape_powers(base_lo, base_hi, s_pair, n_modes + length, xi_pair, n_modes)
                shifted = _shape_powers(base_lo + s_pair[0], base_hi + s_pair[1], (flat, flat), 0, xi_pair, n_modes)[0]
                partial = np.zeros((n_modes, n_modes, n_cells))
                position = np.zeros((n_modes, n_modes, n_cells))
                for i in range(n_modes):
                    series_val = np.zeros((n_modes, n_cells))
                    sign = 1.0
                    for j in range(length):
                        series_val += sign / factorial[i + 1 + j] * table[i + 1 + j]
                        sign = -sign
                    closed_val = shifted.copy()
                    for m2 in range(i + 1):
                        closed_val -= (-1.0) ** m2 / factorial[m2] * table[m2]
                    closed_val *= (-1.0) ** (i + 1)
                    partial[i] = invke**i * factorial[i] * np.where(series_mask, series_val, closed_val)
                    for r2 in range(i + 1):
                        position[i] += binomial[i, r2] * frame ** (i - r2) * invke**r2 * table[r2]
                return partial, position

            own_entry[:, :, r], own_pos_entry[:, :, r] = entry_kernel(
                gap_entry, gap_entry, sp, xi, st["invk_entry"], st["frame_entry"], st["series_entry"]
            )
            own_pos_mid[:, :, r] = _shape_powers(flat, flat, alpha_off, top, xi, n_modes)

            for d2 in range(own):
                st2 = {key: value[pos] for key, value in snap_stash[d2].items()}
                chi = safe(st2["down_lo"] - tail[0], st2["down_hi"] - tail[1])
                s2 = safe(st2["s_lo"], st2["s_hi"])
                length = _series_length(min(_SERIES_CROSSOVER, float(np.nanmax(0.5 * (s2[0] + s2[1]), initial=0.0))))
                t1s = _shape_powers(chi[0], chi[1], s2, n_modes + length, xi, n_modes)
                t1se = _shape_powers(chi[0] + s2[0], chi[1] + s2[1], s2, n_modes - 1, xi, n_modes)
                gamma = np.zeros((n_modes, n_modes, n_cells))
                for low in range(n_modes):
                    series_val = np.zeros((n_modes, n_cells))
                    sign = 1.0
                    for j in range(length):
                        series_val += sign / (factorial[j] * (low + j + 1)) * t1s[low + 1 + j]
                        sign = -sign
                    closed_val = t1s[0].copy()
                    for r2 in range(low + 1):
                        closed_val -= t1se[r2] / factorial[r2]
                    gamma[low] = np.where(st2["series_cells"], series_val, factorial[low] * closed_val)
                keep = ~st2["split_frame"]
                for i in range(n_modes):
                    for low in range(i + 1):
                        up_exit[d2][i, :, r] += binomial[i, low] * (-st2["invk_exit"]) ** low * gamma[low]
                        up_pos_exit[d2][i, :, r] += keep * (
                            binomial[i, low] * st2["frame_exit"] ** (i - low) * (-st2["invk_exit"]) ** low * t1se[low]
                        )
                        up_pos_mid[d2][i, :, r] += (
                            keep
                            * st2["decay_mid"]
                            * binomial[i, low]
                            * st2["frame_mid"] ** (i - low)
                            * (-st2["invk_exit"]) ** low
                            * t1se[low]
                        )
                chi2 = safe(
                    st2["s_lo"] + st2["span_bins"] + st2["down_lo"] - tail[0],
                    st2["s_hi"] + st2["span_bins"] + st2["down_hi"] - tail[1],
                )
                up_entry[d2][:, :, r], up_pos_entry[d2][:, :, r] = entry_kernel(
                    chi2[0],
                    chi2[1],
                    safe(st2["sp_lo"], st2["sp_hi"]),
                    xi,
                    st2["invk_entry"],
                    st2["frame_entry"],
                    st2["series_entry"],
                )
                for stack in (up_exit, up_entry, up_pos_exit, up_pos_mid, up_pos_entry):
                    stack[d2][:, :, r] = np.where(live, stack[d2][:, :, r], 0.0)
            for stack2 in (unit_read, source_read):
                stack2[:, r] = np.where(live, stack2[:, r], 0.0)
            for stack3 in (own_entry, own_pos_mid, own_pos_entry):
                stack3[:, :, r] = np.where(live, stack3[:, :, r], 0.0)
            alive[r] = live
            anchor[r] = a_idx
            raw_volume[r] = np.where(live, label_width[orig], 0.0)
            snap_dep[r] = cin_bin[orig]
            covered = np.bincount(a_idx[live], weights=label_width[orig][live], minlength=n_cin + 1)
            anchor_valid[r] = np.abs(covered - segment_volume[seg]) <= _ANCHOR_TOLERANCE * segment_volume[seg]
        snapshots = SnapshotTerms(
            segment=paths_idx[snap_rows, np.maximum(snap_depth, 0)],
            row_pos=snap_pos,
            own_depth=snap_depth,
            alive=alive,
            anchor=anchor,
            raw_volume=raw_volume,
            dep_bin=snap_dep,
            anchor_valid=anchor_valid,
            unit_read=unit_read,
            source_read=source_read,
            own_entry=own_entry,
            own_pos_mid=own_pos_mid,
            own_pos_entry=own_pos_entry,
            up_exit=up_exit,
            up_entry=up_entry,
            up_pos_exit=up_pos_exit,
            up_pos_mid=up_pos_mid,
            up_pos_entry=up_pos_entry,
        )
    kdt = segment_decay * float(dt_days[0])
    anchored = _e_table(n_modes - 1, kdt)
    etilde = np.zeros_like(anchored)
    for r in range(n_modes):
        for low in range(r + 1):
            etilde[r] += binomial[r, low] * (-1.0) ** low * anchored[low]
    target_terms = TargetTerms(
        exit_read=exit_read,
        entry_read=entry_read,
        position_exit=position_exit,
        position_mid=position_mid,
        position_entry=position_entry,
        bin_entry=bin_entry,
        bin_exit=bin_exit,
        bin_mid=bin_mid,
        segment_of=segment_of,
        cell_weight=(label_width / span_all.ravel()[flat_cout].reshape(n_nodes, n_cells))[order],
        flat_cout=flat_cout.reshape(n_nodes, n_cells)[order].ravel(),
        segment_rate=segment_decay,
        theta=theta,
        theta_in=theta_in,
        chunk_edge=chunk_edge,
        rho_into=rho_into,
        etilde=etilde,
        n_modes=n_modes,
        dt_days=float(dt_days[0]),
        snapshots=snapshots,
    )
    return (
        NetworkTransfer(band_vals, col_start, valid_out, residence_time_out, residence_time_in, valid_in),
        target_terms,
    )


def _legendre_monomial(n_modes: int) -> npt.NDArray[np.floating]:
    """Monomial coefficients of the shifted Legendre polynomials on the unit interval.

    Parameters
    ----------
    n_modes : int
        Number of modes.

    Returns
    -------
    ndarray
        ``lam`` of shape ``(n_modes, n_modes)`` with
        ``P_m(2 xi - 1) = sum_i lam[m, i] xi**i``.
    """
    lam = np.zeros((n_modes, n_modes))
    lam[0, 0] = 1.0
    if n_modes > 1:
        lam[1, :2] = [-1.0, 2.0]
    for m in range(2, n_modes):
        # Bonnet's recursion on the shifted argument: (m) P_m = (2m-1)(2 xi - 1) P_{m-1} - (m-1) P_{m-2}.
        lam[m, 1:] = 2.0 * (2 * m - 1) / m * lam[m - 1, :-1]
        lam[m] -= (2 * m - 1) / m * lam[m - 1] + (m - 1) / m * lam[m - 2]
    return lam


def _mode_scans(
    terms: TargetTerms, modes: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], list[list[npt.NDArray[np.floating]]], npt.NDArray[np.floating]]:
    """Build the monomial mode series and the chunk-anchored interior scans for one target set.

    The mode series in monomials of the volume fraction, and the per-bin interior
    integrals ``G_s`` -- what one fully traversed bin contributes per power of the
    fraction a parcel entered it at -- are built once per segment however many rows read
    them. The scans, edge-indexed, sum the bins of each edge's own chunk,
    ``sum_j G_s[j] theta_in[j]**u rho**(J-1-j)``: a plain forgetting scan of the bounded
    in-chunk positions, minus the same scan's value at the chunk start carried over the
    chunk. Contributions from before a parcel's own reach are removed, never cancelled at
    scale, which is what holds the high modes at round-off accuracy. A segment whose
    modes are identically zero scans to zero, so it is skipped.

    Parameters
    ----------
    terms : TargetTerms
        The operator's stored factors.
    modes : ndarray
        Mode coefficients, shape ``(n_given, n_seg, n_cin)`` with ``n_given`` at most
        :attr:`TargetTerms.n_modes`.

    Returns
    -------
    monomial : ndarray
        Monomial series ``v_i`` of shape ``(n_modes, n_seg, n_cin)``.
    scans : list of list of ndarray
        ``scans[s][u]`` of shape ``(n_seg, n_cin + 1)`` for ``u <= s``.
    binomial : ndarray
        Pascal's triangle up to ``n_modes``, shared by the callers.
    """
    n_modes = terms.n_modes
    n_seg, n_cin = terms.theta.shape
    lam = _legendre_monomial(n_modes)
    monomial = np.einsum("mej,mi->iej", modes, lam[: modes.shape[0]])
    kdt = terms.segment_rate * terms.dt_days
    rho = np.exp(-kdt)
    binomial = np.zeros((n_modes, n_modes))
    binomial[:, 0] = 1.0
    for row in range(1, n_modes):
        binomial[row, 1:] = binomial[row - 1, 1:] + binomial[row - 1, :-1]
    interior_bin = np.zeros((n_modes, n_seg, n_cin))
    for s in range(n_modes):
        for i in range(s, n_modes):
            interior_bin[s] += (
                binomial[i, s] * monomial[i] * terms.theta ** (i - s) * (kdt * terms.etilde[i - s])[:, None]
            )
    moving = np.flatnonzero(np.any(monomial != 0.0, axis=(0, 2)) & (kdt > 0.0))
    edge_index = np.arange(n_cin + 1)[None, :]
    scans: list[list[npt.NDArray[np.floating]]] = [[] for _ in range(n_modes)]
    for s in range(n_modes):
        for u in range(s + 1):
            forcing = np.zeros((n_seg, n_cin + 1))
            forcing[:, 1:] = interior_bin[s] * terms.theta_in**u
            scan = np.zeros_like(forcing)
            for e in moving:
                scan[e] = lfilter([1.0], [1.0, -rho[e]], forcing[e])
            at_start = np.take_along_axis(scan, terms.chunk_edge.astype(np.intp), axis=1)
            scans[s].append(scan - np.where(edge_index > terms.chunk_edge, terms.rho_into, 0.0) * at_start)
    return monomial, scans, binomial


def apply_content_snapshots(
    terms: TargetTerms,
    segment_modes: npt.NDArray[np.floating],
    source_values: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Evaluate each snapshot segment's content moments at its chunk-anchor edges.

    The affine counterpart of :func:`apply_segment_targets` for the content snapshots:
    for every cell spanning an anchor, the first half of its delivered value -- the
    source decayed to the anchor, the upstream biases decayed to it, and the own-segment
    bias accrued up to it -- is weighted by the Legendre shapes of the parcel fraction at
    the anchor and scattered by water volume. The result is what the moment recursions of
    :mod:`pipetransport.heat` restart from; see :class:`SnapshotTerms`.

    Parameters
    ----------
    terms : TargetTerms
        Bias factors of an operator built with ``snapshot_rows``.
    segment_modes : ndarray
        Axial mode coefficients as in :func:`apply_segment_targets`.
    source_values : ndarray
        Source series on the operator's padded input grid.

    Returns
    -------
    snapshot : ndarray
        Absolute content moments per mode, segment and edge, shape
        ``(n_modes, n_seg, n_cin + 1)``; nonzero only at each segment's anchors.
    unit : ndarray
        The same scatter of the shapes alone -- the moments of a unit temperature --
        with which a reference temperature's moments are subtracted.

    Raises
    ------
    ValueError
        If the operator carries no snapshot factors.
    """
    if terms is None or terms.snapshots is None:
        msg = "operator was built without content snapshots; pass snapshot_rows to paths_transfer"
        raise ValueError(msg)
    snaps = terms.snapshots
    modes = np.asarray(segment_modes, dtype=float)
    if modes.ndim == 2:  # noqa: PLR2004 -- a 2-D array is the position-uniform target
        modes = modes[None]
    monomial, scans, binomial = _mode_scans(terms, modes)
    n_modes = terms.n_modes
    n_seg, n_cin = terms.theta.shape
    flat_monomial = monomial.reshape(n_modes, -1)
    snapshot = np.zeros((n_modes, n_seg, n_cin + 1))
    unit = np.zeros_like(snapshot)
    for r in range(len(snaps.segment)):
        seg = int(snaps.segment[r])
        own = int(snaps.own_depth[r])
        pos = int(snaps.row_pos[r])
        anchors = snaps.anchor[r]
        powered = np.empty((n_modes, anchors.size))
        entry_own = terms.bin_entry[own][pos]
        v_entry = flat_monomial[:, seg * n_cin + entry_own]
        for u in range(n_modes):
            acc = snaps.source_read[u, r] * source_values[snaps.dep_bin[r]]
            for i in range(n_modes):
                acc += v_entry[i] * snaps.own_entry[i, u, r]
            for s in range(n_modes):
                for u2 in range(s + 1):
                    scan_row = scans[s][u2][seg]
                    acc += binomial[s, u2] * (
                        snaps.own_pos_mid[s - u2, u, r] * scan_row[anchors]
                        - snaps.own_pos_entry[s - u2, u, r] * scan_row[entry_own + 1]
                    )
            for d2 in range(own):
                seg2 = int(terms.segment_of[d2][pos])
                entry2 = terms.bin_entry[d2][pos]
                exit2 = terms.bin_exit[d2][pos]
                mid2 = terms.bin_mid[d2][pos]
                v_e2 = flat_monomial[:, seg2 * n_cin + entry2]
                v_x2 = flat_monomial[:, seg2 * n_cin + exit2]
                for i in range(n_modes):
                    acc += v_x2[i] * snaps.up_exit[d2][i, u, r] + v_e2[i] * snaps.up_entry[d2][i, u, r]
                for s in range(n_modes):
                    for u2 in range(s + 1):
                        scan_row = scans[s][u2][seg2]
                        acc += binomial[s, u2] * (
                            snaps.up_pos_exit[d2][s - u2, u, r] * scan_row[exit2]
                            + snaps.up_pos_mid[d2][s - u2, u, r] * scan_row[mid2]
                            - snaps.up_pos_entry[d2][s - u2, u, r] * scan_row[entry2 + 1]
                        )
            powered[u] = acc
        weights = snaps.raw_volume[r]
        for m in range(n_modes):
            snapshot[m, seg] += np.bincount(anchors, weights=weights * powered[m], minlength=n_cin + 1)
            unit[m, seg] += np.bincount(anchors, weights=weights * snaps.unit_read[m, r], minlength=n_cin + 1)
    return snapshot, unit


def apply_segment_targets(
    transfer: NetworkTransfer,
    terms: TargetTerms,
    segment_modes: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Apply concrete per-segment relaxation targets to a prebuilt operator's bias factors.

    The affine model is ``c_node = W @ c_source + b``; this computes ``b`` for one target
    set. The target of segment ``e`` at volume fraction ``xi`` through it is
    ``sum_m segment_modes[m, e] * P_m(2 xi - 1)`` with ``P_m`` the Legendre polynomials --
    mode 0 is the position-uniform target, mode 1 twice the old half-spread tilt, and the
    modes are what the axial flux moments of :mod:`pipetransport.heat` relax toward.

    A parcel's reading of that field splits at the input-bin edges of each traversal into
    an entry piece, full interior bins and an exit piece. The partial pieces are stored
    per-cell closed forms times the mode series read at the entry and exit bins. The
    interior sum runs over **chunk-anchored scans**: each segment's record is chunked at
    1.5 pipe volumes of throughflow (see :class:`TargetTerms`), a parcel's volume
    fraction at an interior edge is its per-cell frame offset plus the edge's bounded
    in-chunk position, and the scans ``sum_j G_s[j] theta_in[j]**u rho**(J-1-j)`` --
    plain forgetting scans, cut at each chunk start -- turn every cell's interior sum
    into gathers at its exit edge, its exit chunk's start and its entry edge, weighted by
    powers of the stored frame offsets. Positions never leave their chunk's frame before
    they are raised to powers, which is what holds the high modes at round-off accuracy
    for every coupling strength and record length. All the sequential work is
    ``n_modes (n_modes + 1) / 2`` forgetting scans per segment whose modes move.

    Parameters
    ----------
    transfer : NetworkTransfer
        Operator built by :func:`paths_transfer`.
    segment_modes : ndarray
        Axial mode coefficients of every segment's target [unit of the source signal],
        piecewise constant on the input bins, shape ``(n_modes, n_seg, n_cin)``; a 2-D
        array is the position-uniform target alone. Up to the operator's
        :attr:`~TargetTerms.n_modes` leading modes are read; missing trailing modes are
        zero.

    Returns
    -------
    ndarray
        Bias of each output bin, shape ``(n_nodes, n_cout)``. Zero where
        :attr:`NetworkTransfer.valid_out` is False, so adding it to ``W @ c_source``
        keeps the NaN policy of the caller intact.

    Raises
    ------
    ValueError
        If the operator carries fewer modes than passed.
    """
    modes = np.asarray(segment_modes, dtype=float)
    if modes.ndim == 2:  # noqa: PLR2004 -- a 2-D array is the position-uniform target
        modes = modes[None]
    if modes.shape[0] > terms.n_modes:
        msg = f"operator was built for {terms.n_modes} target mode(s), got {modes.shape[0]}; raise n_target_modes"
        raise ValueError(msg)
    n_modes = terms.n_modes
    n_cin = modes.shape[2]
    n_nodes, n_cout = transfer.valid_out.shape

    monomial, scans, binomial = _mode_scans(terms, modes)

    # Per depth: the mode series read at the entry and exit bins against the stored partial
    # pieces, and the scans gathered with per-cell position-power weights. The rows are the
    # leading ``n_rows`` of the stored order, so every slab is a slice and no row is
    # gathered; flat gathers avoid materializing take_along_axis's index grids.
    flat_monomial = monomial.reshape(n_modes, -1)
    cell_bias = np.zeros_like(terms.cell_weight)
    for depth, segs in enumerate(terms.segment_of):
        n_rows = segs.size
        at_exit = segs[:, None] * n_cin + terms.bin_exit[depth]
        at_entry = segs[:, None] * n_cin + terms.bin_entry[depth]
        scan_exit = segs[:, None] * (n_cin + 1) + terms.bin_exit[depth]
        scan_mid = segs[:, None] * (n_cin + 1) + terms.bin_mid[depth]
        scan_entry = segs[:, None] * (n_cin + 1) + terms.bin_entry[depth] + 1
        acc = np.zeros(at_exit.shape)
        for i in range(n_modes):
            acc += flat_monomial[i][at_exit] * terms.exit_read[depth][i]
            acc += flat_monomial[i][at_entry] * terms.entry_read[depth][i]
        for s in range(n_modes):
            for u in range(s + 1):
                flat_scan = scans[s][u].ravel()
                acc += binomial[s, u] * (
                    terms.position_exit[depth][s - u] * flat_scan[scan_exit]
                    + terms.position_mid[depth][s - u] * flat_scan[scan_mid]
                    - terms.position_entry[depth][s - u] * flat_scan[scan_entry]
                )
        cell_bias[:n_rows] += acc

    out_slots = n_nodes * (n_cout + 2)
    bias = np.bincount(terms.flat_cout, weights=(cell_bias * terms.cell_weight).ravel(), minlength=out_slots).reshape(
        n_nodes, n_cout + 2
    )[:, 1:-1]
    return np.where(transfer.valid_out, bias, 0.0)
