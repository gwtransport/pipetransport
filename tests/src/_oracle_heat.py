"""
Relaxation targets for the brute-force oracle.

The same parcel-by-parcel reference as :mod:`_oracle`, extended with the one thing the
relaxation model adds: within a segment a parcel decays toward a per-segment target rather
than toward zero, so its delivered value is an integral along the path instead of a single
surviving fraction. Kept apart from the transport oracle because the relaxation model ships
as its own package; the transport oracle must stay usable without it.
"""

from __future__ import annotations

import itertools

import numpy as np
from _oracle import OraclePath
from numpy.polynomial import legendre


class RelaxingOraclePath(OraclePath):
    """One source-to-node path whose segments relax toward piecewise-constant targets.

    Parameters
    ----------
    segment_target : ndarray
        Relaxation target of each path segment, piecewise constant on the input bins,
        shape ``(m, n_bins)``.
    segment_target_modes : ndarray or None, optional
        Higher axial modes of the target, shape ``(n_extra, m, n_bins)``: the target at
        volume fraction ``xi = x/L`` through a segment is
        ``target[j] + sum_k modes[k - 1][j] * P_k(2 xi - 1)`` with ``P_k`` the Legendre
        polynomial of degree ``k``. ``None`` (default) is the position-uniform target.
    **kwargs
        Everything :class:`_oracle.OraclePath` takes.
    """

    def __init__(self, *, segment_target, segment_target_modes=None, **kwargs):
        super().__init__(**kwargs)
        self.target = np.atleast_2d(segment_target)[: len(self.volume)]
        self.modes = None
        if segment_target_modes is not None:
            self.modes = np.asarray(segment_target_modes, dtype=float)[:, : len(self.volume)]

    def deliver(self, departure, t_source):
        """Deliver one parcel with per-segment relaxation toward the piecewise-constant targets.

        Walks the path sequentially: within each segment the parcel relaxes toward that
        segment's target, one target bin at a time -- deliberately the naive piece-by-piece
        algorithm, sharing no arithmetic with the package's scanned closed forms.

        Within a piece the flow is constant, so the parcel's volume fraction is affine in
        time and the target it sees is a polynomial in time of the modes' degree. The exact
        update over the piece is the decaying transient plus the relaxation integral
        ``int k Tb(t) exp(-k (end - t)) dt``, whose integrand is analytic; it is evaluated
        with the path's fixed Gauss-Legendre rule rather than a closed form, which is what
        keeps this reference numerically independent of the package's expressions.

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
        nodes, weights = self.gauss
        time, temperature = departure, t_source
        for e, (pipe, volume, decay, target) in enumerate(
            zip(self.pipes, self.volume, self.decay, self.target, strict=True)
        ):
            entry_volume = pipe(time)
            exit_time = self._cross(pipe, volume, time, forward=True)
            if not np.isfinite(exit_time):
                return np.nan, np.nan
            while time < exit_time:
                j = int(np.clip(np.searchsorted(self.tedges, time, side="right") - 1, 0, len(target) - 1))
                step_end = min(self.tedges[j + 1], exit_time)
                if step_end <= time:
                    break
                span = step_end - time
                if decay > 0.0:
                    middle, half = 0.5 * (time + step_end), 0.5 * span
                    at = middle + half * nodes
                    tb = np.full(len(at), target[j])
                    if self.modes is not None:
                        fraction = (pipe(middle) + pipe.rate[j] * half * nodes - entry_volume) / volume
                        coefficients = np.concatenate([[0.0], self.modes[:, e, j]])
                        tb += legendre.legval(2.0 * fraction - 1.0, coefficients)
                    relax = half * np.sum(weights * decay * tb * np.exp(-decay * (step_end - at)))
                    temperature = temperature * np.exp(-decay * span) + relax
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

    def tout(self, *, tin, cout_tedges_days, bin_end_rate=0.0, bin_end_power=0, bin_end_integrated=False):
        """Bin-averaged delivered temperature on the output grid, with relaxation targets.

        The integrand over the label is smooth except where a parcel crosses *any* segment
        face exactly at a target-bin edge, so every output bin's label interval is split at
        the labels of those crossings (found by the same root solves the oracle is built
        on) and each kink-free piece is integrated with the path's Gauss-Legendre rule.

        Parameters
        ----------
        tin : ndarray
            Source temperature, one value per input bin.
        cout_tedges_days : ndarray
            Output bin edges in days.
        bin_end_rate : float, optional
            Rate ``w`` [1/day] of an extra reading weight ``exp(-w (t_end - t))``, with ``t``
            the parcel's delivery time and ``t_end`` the right edge of its output bin.
            Default 0, the plain bin average.
        bin_end_power : int, optional
            Power ``p`` of an additional polynomial factor ``(t_end - t)**p`` on the
            reading weight -- the p-th time-moment reading. Default 0.
        bin_end_integrated : bool, optional
            Replace the weight by its running integral ``int_0^lag u**p exp(-w u) du``.
            Default False.

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

        def temperature(label, bin_end):
            depart = self.departure(self._time_at_label(label))
            j = int(np.clip(np.searchsorted(self.tedges, depart, side="right") - 1, 0, len(tin) - 1))
            value = self.deliver(depart, tin[j])[1]
            lag = bin_end - self._time_at_label(label)
            return value * self._weight(lag, rate=bin_end_rate, power=bin_end_power, integrated=bin_end_integrated)

        # Each piece between consecutive kinks is analytic, so fixed-order Gauss-Legendre
        # converges immediately; adaptive quadrature only rediscovers that at much greater
        # cost, and complains about the kinks when a piece happens to straddle one.
        nodes, weights = self.gauss

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
            bin_end = cout_tedges_days[j + 1]
            for a, b in itertools.pairwise(bounds):
                if not b > a:
                    continue
                middle, half = 0.5 * (a + b), 0.5 * (b - a)
                total += half * sum(
                    w * temperature(middle + half * x, bin_end) for x, w in zip(nodes, weights, strict=True)
                )
            out[j] = total / (hi - lo)
        return out
