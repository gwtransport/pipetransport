"""
Brute-force reference implementation of source-to-node transport.

Deliberately independent of :mod:`pipetransport._transfer`: instead of composing linear
interpolations on a refined cell grid, this tracks a single parcel at a time by solving each
pipe's displacement condition with :func:`scipy.optimize.brentq` and integrates the output-bin
average with :func:`scipy.integrate.quad`. It shares no code path with the package beyond the
network topology, so agreement between the two is evidence about the physics rather than about
one implementation.

Wherever an integrand is smooth on a piece -- a parcel's relaxation toward a polynomial
target inside one flow bin, an output bin's label integral between kinks, a reading-weight
kernel -- the oracle integrates it with fixed-order Gauss-Legendre quadrature whose order is
the ``gauss_order`` of the path. The integrands are analytic on those pieces, so the error
falls factorially with the order: the default 24 sits at the float64 floor for every case the
tests reach, and raising the order is the knob that checks that claim rather than trusting it.

It is slow -- one root solve per pipe per quadrature point -- so tests keep the records short.
Flows must be strictly positive: the root solves need a strictly monotone cumulative volume,
and the package's plateau handling is checked separately.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


class _Displacement:
    """Cumulative volume displaced through one pipe, as an exact piecewise-linear function."""

    def __init__(self, tedges_days, rate):
        self.tedges = np.asarray(tedges_days, dtype=float)
        self.rate = np.asarray(rate, dtype=float)
        self.cumulative = np.concatenate([[0.0], np.cumsum(self.rate * np.diff(self.tedges))])

    def __call__(self, time):
        """Volume displaced between the record start and ``time``.

        Parameters
        ----------
        time : float
            Time in days, inside the record.

        Returns
        -------
        float
            Displaced volume.
        """
        i = int(np.clip(np.searchsorted(self.tedges, time, side="right") - 1, 0, len(self.rate) - 1))
        return float(self.cumulative[i] + self.rate[i] * (time - self.tedges[i]))


class OraclePath:
    """One source-to-node path, solved parcel by parcel.

    Parameters
    ----------
    tedges_days : ndarray
        Input bin edges in days.
    segment_flow : ndarray
        Throughflow of each path segment, shape ``(m, n_bins)``, source-first.
    segment_volume : ndarray
        Water volume of each path segment, length ``m``.
    segment_decay : ndarray
        First-order decay rate of each path segment [1/day], length ``m``.
    node_flow : ndarray
        Throughflow past the reporting node, length ``n_bins``.
    gauss_order : int, optional
        Order of every fixed Gauss-Legendre quadrature in this path -- the label integrals
        of the output-bin average and the integrated reading-weight kernel. The integrands are analytic per piece, so
        the default 24 reaches the float64 floor; raise it to verify that instead of
        assuming it. Default 24.
    """

    def __init__(
        self,
        *,
        tedges_days,
        segment_flow,
        segment_volume,
        segment_decay,
        node_flow,
        gauss_order=24,
    ):
        self.tedges = np.asarray(tedges_days, dtype=float)
        self.volume = np.asarray(segment_volume, dtype=float)
        self.decay = np.asarray(segment_decay, dtype=float)
        self.pipes = [_Displacement(self.tedges, q) for q in np.atleast_2d(segment_flow)][: len(self.volume)]
        self.node = _Displacement(self.tedges, node_flow)
        self.gauss = np.polynomial.legendre.leggauss(int(gauss_order))

    def _cross(self, pipe, volume, known, *, forward):
        """Solve the displacement condition of one pipe for the unknown face time.

        Parameters
        ----------
        pipe : _Displacement
            Cumulative displacement of the pipe.
        volume : float
            Water volume of the pipe.
        known : float
            Time in days at the known face.
        forward : bool
            True to solve for the exit time given the entry time.

        Returns
        -------
        float
            Time in days at the other face, or NaN when it falls outside the record.
        """
        target = pipe(known) + volume if forward else pipe(known) - volume
        lo, hi = self.tedges[0], self.tedges[-1]
        if not pipe(lo) <= target <= pipe(hi):
            return np.nan
        return float(brentq(lambda t: pipe(t) - target, lo, hi, xtol=1e-14))

    def arrival(self, departure):
        """Arrival time at the node and the travel time inside each segment.

        Parameters
        ----------
        departure : float
            Departure time from the source, in days.

        Returns
        -------
        arrival : float
            Arrival time at the node in days, NaN if the parcel leaves the record.
        travel : list of float
            Time spent in each segment, source-first.
        """
        time, travel = departure, []
        for pipe, volume in zip(self.pipes, self.volume, strict=True):
            nxt = self._cross(pipe, volume, time, forward=True)
            if not np.isfinite(nxt):
                return np.nan, travel
            travel.append(nxt - time)
            time = nxt
        return time, travel

    def departure(self, arrival):
        """Departure time from the source of the parcel arriving at ``arrival``.

        Parameters
        ----------
        arrival : float
            Arrival time at the node, in days.

        Returns
        -------
        float
            Departure time from the source in days, NaN if it falls before the record.
        """
        time = arrival
        for pipe, volume in zip(reversed(self.pipes), reversed(self.volume), strict=True):
            time = self._cross(pipe, volume, time, forward=False)
            if not np.isfinite(time):
                return np.nan
        return time

    def _time_at_label(self, label):
        """Invert the cumulative node throughflow for the delivery time of a label.

        Parameters
        ----------
        label : float
            Cumulative volume delivered at the node.

        Returns
        -------
        float
            Delivery time in days.
        """
        return float(brentq(lambda t: self.node(t) - label, self.tedges[0], self.tedges[-1], xtol=1e-14))

    def _weight(self, lag, *, rate, power, integrated):
        """Evaluate the reading weight at a lag before the output-bin end; scalar in, scalar out.

        Parameters
        ----------
        lag : float
            ``t_end - t`` [days], non-negative.
        rate : float
            Rate ``w`` [1/day] of the exponential factor.
        power : int
            Power ``p`` of the polynomial factor.
        integrated : bool
            False reads ``lag**p * exp(-rate * lag)``; True reads its running integral
            ``g_p(lag) = int_0^lag u**p exp(-rate u) du``, evaluated with the path's
            Gauss-Legendre rule -- the kernel the enthalpy chain's time-integrated
            content readings use.

        Returns
        -------
        float
            The weight.
        """
        if not integrated:
            return lag**power * np.exp(-rate * lag)
        nodes, weights = self.gauss
        half = 0.5 * lag
        at = half + half * nodes
        return float(half * np.sum(weights * at**power * np.exp(-rate * at)))

    def cout(self, *, cin, cout_tedges_days, bin_end_rate=0.0, bin_end_power=0, bin_end_integrated=False):
        """Bin-averaged delivered quality on the output grid.

        Splits every output bin's label interval at the labels where ``cin`` steps, then
        integrates the surviving fraction over each piece with adaptive quadrature.

        Parameters
        ----------
        cin : ndarray
            Source quality, one value per input bin.
        cout_tedges_days : ndarray
            Output bin edges in days.
        bin_end_rate : float, optional
            Rate ``w`` [1/day] of an extra reading weight ``exp(-w (t_end - t))``, with ``t``
            the parcel's delivery time and ``t_end`` the right edge of its output bin. Default
            0, the plain bin average.
        bin_end_power : int, optional
            Power ``p`` of an additional polynomial factor ``(t_end - t)**p`` [days**p] on
            the reading weight -- the p-th time-moment reading. Default 0.
        bin_end_integrated : bool, optional
            Replace the weight by its running integral ``int_0^lag u**p exp(-w u) du``.
            Default False.

        Returns
        -------
        ndarray
            Delivered quality per output bin; NaN where a parcel leaves the record.
        """
        cin = np.asarray(cin, dtype=float)
        cout_tedges_days = np.asarray(cout_tedges_days, dtype=float)
        # Label of every input-bin boundary: the parcel that leaves the source at that instant.
        edge_label = np.array([self.node(a) if np.isfinite(a := self.arrival(t)[0]) else np.nan for t in self.tedges])

        def exponent(label):
            travel = self.arrival(self.departure(self._time_at_label(label)))[1]
            return float(np.sum(self.decay[: len(travel)] * np.asarray(travel)))

        out = np.full(len(cout_tedges_days) - 1, np.nan)
        for j in range(len(out)):
            lo, hi = self.node(cout_tedges_days[j]), self.node(cout_tedges_days[j + 1])
            if not hi > lo:
                continue
            # Both ends of the bin must be fed by a parcel that left inside the record; this
            # is the same coverage condition the package applies, checked parcel by parcel.
            if not all(np.isfinite(self.departure(self._time_at_label(edge))) for edge in (lo, hi)):
                continue
            interior = edge_label[np.isfinite(edge_label) & (edge_label > lo) & (edge_label < hi)]
            bounds = np.concatenate([[lo], interior, [hi]])
            total = 0.0
            bin_end = cout_tedges_days[j + 1]

            def integrand(label, end=bin_end):
                lag = end - self._time_at_label(label)
                weight = self._weight(lag, rate=bin_end_rate, power=bin_end_power, integrated=bin_end_integrated)
                return np.exp(-exponent(label)) * weight

            for a, b in itertools.pairwise(bounds):
                source_time = self.departure(self._time_at_label(0.5 * (a + b)))
                bin_index = int(np.clip(np.searchsorted(self.tedges, source_time, side="right") - 1, 0, len(cin) - 1))
                integral, _ = quad(integrand, a, b, limit=200)
                total += cin[bin_index] * integral
            out[j] = total / (hi - lo)
        return out
