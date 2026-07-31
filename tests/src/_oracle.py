"""
Brute-force reference implementation of source-to-node transport.

Deliberately independent of :mod:`pipetransport._transfer`: instead of composing linear
interpolations on a refined cell grid, this tracks a single parcel at a time by solving each
pipe's displacement condition with :func:`scipy.optimize.brentq` and integrates the output-bin
average with :func:`scipy.integrate.quad`. It shares no code path with the package beyond the
network topology, so agreement between the two is evidence about the physics rather than about
one implementation.

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
    """

    def __init__(self, *, tedges_days, segment_flow, segment_volume, segment_decay, node_flow, segment_target=None):
        self.tedges = np.asarray(tedges_days, dtype=float)
        self.volume = np.asarray(segment_volume, dtype=float)
        self.decay = np.asarray(segment_decay, dtype=float)
        self.pipes = [_Displacement(self.tedges, q) for q in np.atleast_2d(segment_flow)][: len(self.volume)]
        self.node = _Displacement(self.tedges, node_flow)
        self.target = None if segment_target is None else np.atleast_2d(segment_target)[: len(self.volume)]

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

    def cout(self, *, cin, cout_tedges_days):
        """Bin-averaged delivered quality on the output grid.

        Splits every output bin's label interval at the labels where ``cin`` steps, then
        integrates the surviving fraction over each piece with adaptive quadrature.

        Parameters
        ----------
        cin : ndarray
            Source quality, one value per input bin.
        cout_tedges_days : ndarray
            Output bin edges in days.

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
            for a, b in itertools.pairwise(bounds):
                source_time = self.departure(self._time_at_label(0.5 * (a + b)))
                bin_index = int(np.clip(np.searchsorted(self.tedges, source_time, side="right") - 1, 0, len(cin) - 1))
                integral, _ = quad(lambda label: np.exp(-exponent(label)), a, b, limit=200)
                total += cin[bin_index] * integral
            out[j] = total / (hi - lo)
        return out

    def deliver(self, departure, t_source):
        """Deliver one parcel with per-segment relaxation toward the piecewise-constant targets.

        Walks the path sequentially: within each segment the parcel relaxes toward that
        segment's target, exactly, one target bin at a time -- deliberately the naive
        piece-by-piece algorithm, sharing no arithmetic with the package's Abel-summed form.

        Parameters
        ----------
        departure : float
            Departure time from the source, in days.
        t_source : float
            Temperature of the parcel when it leaves the source.

        Returns
        -------
        arrival : float
            Arrival time at the node, NaN if the parcel leaves the record.
        temperature : float
            Delivered temperature, NaN with the arrival.
        """
        if self.target is None:
            msg = "deliver() needs per-segment relaxation targets; pass segment_target to OraclePath"
            raise ValueError(msg)
        time, temperature = departure, t_source
        for pipe, volume, decay, target in zip(self.pipes, self.volume, self.decay, self.target, strict=True):
            exit_time = self._cross(pipe, volume, time, forward=True)
            if not np.isfinite(exit_time):
                return np.nan, np.nan
            while time < exit_time:
                j = int(np.clip(np.searchsorted(self.tedges, time, side="right") - 1, 0, len(target) - 1))
                step_end = min(self.tedges[j + 1], exit_time)
                if step_end <= time:
                    break
                temperature = target[j] + (temperature - target[j]) * np.exp(-decay * (step_end - time))
                time = step_end
            time = exit_time
        return time, temperature

    def _departure_through(self, count, face_time):
        """Departure time of the parcel that crosses the ``count``-th segment face at ``face_time``.

        Parameters
        ----------
        count : int
            Number of segments between the source and the face, ``1 <= count <= m``.
        face_time : float
            Time in days at which the parcel passes the face.

        Returns
        -------
        float
            Departure time from the source, NaN if it falls before the record.
        """
        time = face_time
        for pipe, volume in zip(reversed(self.pipes[:count]), reversed(self.volume[:count]), strict=True):
            time = self._cross(pipe, volume, time, forward=False)
            if not np.isfinite(time):
                return np.nan
        return time

    def tout(self, *, tin, cout_tedges_days):
        """Bin-averaged delivered temperature on the output grid, with relaxation targets.

        The integrand over the label is smooth except where a parcel crosses *any* segment
        face exactly at a target-bin edge, so every output bin's label interval is split at
        the labels of those crossings (found by the same root solves the oracle is built
        on) and each kink-free piece is integrated with adaptive quadrature.

        Parameters
        ----------
        tin : ndarray
            Source temperature, one value per input bin.
        cout_tedges_days : ndarray
            Output bin edges in days.

        Returns
        -------
        ndarray
            Delivered temperature per output bin; NaN where a parcel leaves the record.
        """
        tin = np.asarray(tin, dtype=float)
        cout_tedges_days = np.asarray(cout_tedges_days, dtype=float)
        # Labels of every kink: a parcel crossing any face (source counts as face 0) at a
        # bin edge of the shared time grid.
        kinks = []
        for count in range(len(self.pipes) + 1):
            for edge in self.tedges:
                depart = edge if count == 0 else self._departure_through(count, edge)
                if not np.isfinite(depart):
                    continue
                arrival, _ = self.deliver(depart, 0.0)
                if np.isfinite(arrival):
                    kinks.append(self.node(arrival))
        kinks = np.array(sorted(kinks))

        def temperature(label):
            depart = self.departure(self._time_at_label(label))
            j = int(np.clip(np.searchsorted(self.tedges, depart, side="right") - 1, 0, len(tin) - 1))
            return self.deliver(depart, tin[j])[1]

        # Each piece between consecutive kinks is analytic, so fixed-order Gauss-Legendre
        # converges immediately; adaptive quadrature only rediscovers that at much greater
        # cost, and complains about the kinks when a piece happens to straddle one.
        nodes, weights = np.polynomial.legendre.leggauss(12)

        out = np.full(len(cout_tedges_days) - 1, np.nan)
        for j in range(len(out)):
            lo, hi = self.node(cout_tedges_days[j]), self.node(cout_tedges_days[j + 1])
            if not hi > lo:
                continue
            if not all(np.isfinite(self.departure(self._time_at_label(edge))) for edge in (lo, hi)):
                continue
            interior = kinks[(kinks > lo) & (kinks < hi)]
            bounds = np.concatenate([[lo], interior, [hi]])
            total = 0.0
            for a, b in itertools.pairwise(bounds):
                if not b > a:
                    continue
                middle, half = 0.5 * (a + b), 0.5 * (b - a)
                total += half * sum(w * temperature(middle + half * x) for x, w in zip(nodes, weights, strict=True))
            out[j] = total / (hi - lo)
        return out
