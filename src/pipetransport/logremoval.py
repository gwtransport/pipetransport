"""
First-order reactions in the distribution network: disinfectant decay and log removal.

Everything that decays in a pipe under first-order kinetics is one exponential in contact
time. Two currencies express it, and this module holds the conversions between them and the
two rules that matter in a network:

- **Fraction remaining** ``C/C0 = exp(-k * t)`` with the rate constant ``k`` [1/day]. This is
  how chlorine residual is reported and how :func:`pipetransport.transport.source_to_endmember`
  takes its ``decay_rate``.
- **Log removal** ``LR = mu * t`` with ``mu = k / ln(10)`` [log10/day]. This is how
  disinfection credit and pathogen inactivation are reported: LR 1 is a 90 % reduction, LR 2
  is 99 %, LR 3 is 99.9 %.

Chlorine decay in a pipe
------------------------

A pipe consumes disinfectant in the bulk water and at the wall, and the two combine into one
apparent first-order rate (Rossman, Clark and Grayman, 1994):

    k = k_b + k_w * k_f / (R_h * (k_w + k_f)),    R_h = D / 4

with ``k_b`` [1/day] the bulk decay rate, ``k_w`` [m/day] the wall reaction rate constant,
``k_f`` [m/day] the mass transfer coefficient between bulk and wall, and ``R_h`` [m] the
hydraulic radius of the full pipe. The wall term scales with the surface-to-volume ratio
``1 / R_h = 4 / D``, so a 100 mm service line loses residual roughly four times faster per
unit contact time than a 400 mm trunk main of the same water quality. That diameter
dependence is why :func:`segment_decay_rate` returns a rate per segment rather than one rate
for the whole network, and why the transport operator carries a per-segment exponent.

``k_f`` depends on velocity and therefore on time. The transport operator takes a
time-constant rate per segment, so pass a representative value (or leave it out, which assumes
mass transfer is not limiting) rather than a per-time-step one.

Available functions:

- :func:`residence_time_to_log_removal` - Multiply travel times [days] by a log10 decay rate
  [log10/day] to get log removal, keeping the shape of the input.

- :func:`decay_rate_to_log10_decay_rate` / :func:`log10_decay_rate_to_decay_rate` - Convert
  between the natural-log rate constant ``k`` [1/day] and the log10 rate ``mu`` [log10/day]
  via ``mu = k / ln(10)``.

- :func:`log_removal_to_fraction_remaining` / :func:`fraction_remaining_to_log_removal` -
  Convert between log removal and the surviving fraction ``10 ** (-LR)``, i.e. between the
  disinfection-credit and the residual-concentration view of the same number.

- :func:`parallel_mean` - Combine the log removals of streams that blend into one, as
  ``-log10(sum(F_i * 10 ** (-LR_i)))``. Concentrations mix, not log removals, so the blend is
  always dominated by the least-treated stream.

- :func:`segment_decay_rate` - Per-segment first-order decay rate [1/day] from a bulk rate, a
  wall rate and each segment's diameter, returned as a Series ready to pass as
  ``decay_rate`` to :mod:`pipetransport.transport`.

References
----------
Rossman, L. A., Clark, R. M., & Grayman, W. M. (1994). Modeling chlorine residuals in
drinking-water distribution systems. Journal of Environmental Engineering, 120(4), 803-820.

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from pipetransport._validation import _validate_non_negative, _validate_positive
from pipetransport.network import PipeNetwork  # noqa: TC001 -- runtime dependency of the signature


def residence_time_to_log_removal(
    *, residence_times: npt.ArrayLike, log10_decay_rate: float
) -> npt.NDArray[np.floating]:
    """Compute log removal from travel times and a log10 decay rate.

    ``LR = log10_decay_rate * residence_time``, equivalent to ``C/C0 = 10 ** (-LR)``.

    Parameters
    ----------
    residence_times : array-like
        Travel times [days], of any shape -- typically the output of
        :func:`pipetransport.residence_time.endmember_to_source`. NaN propagates.
    log10_decay_rate : float
        Log10 decay rate mu [log10/day].

    Returns
    -------
    ndarray
        Log removal, same shape as ``residence_times``.

    See Also
    --------
    decay_rate_to_log10_decay_rate : Convert a natural-log rate constant to mu.
    log_removal_to_fraction_remaining : Express the result as a surviving fraction.
    pipetransport.residence_time.endmember_to_source : Travel times to feed in.

    Examples
    --------
    >>> from pipetransport.logremoval import residence_time_to_log_removal
    >>> residence_time_to_log_removal(
    ...     residence_times=[0.5, 1.0, 2.0], log10_decay_rate=0.2
    ... )
    array([0.1, 0.2, 0.4])
    """
    return log10_decay_rate * np.asarray(residence_times, dtype=float)


def decay_rate_to_log10_decay_rate(decay_rate: float) -> float:
    """Convert a natural-log rate constant to a log10 decay rate: ``mu = k / ln(10)``.

    Parameters
    ----------
    decay_rate : float
        Natural-log first-order rate constant k [1/day], e.g. ``np.log(2) / half_life``.

    Returns
    -------
    float
        Log10 decay rate mu [log10/day].

    See Also
    --------
    log10_decay_rate_to_decay_rate : Inverse conversion.

    Examples
    --------
    >>> from pipetransport.logremoval import decay_rate_to_log10_decay_rate
    >>> float(round(decay_rate_to_log10_decay_rate(0.5), 6))
    0.217147
    """
    return decay_rate / np.log(10)


def log10_decay_rate_to_decay_rate(log10_decay_rate: float) -> float:
    """Convert a log10 decay rate to a natural-log rate constant: ``k = mu * ln(10)``.

    Parameters
    ----------
    log10_decay_rate : float
        Log10 decay rate mu [log10/day].

    Returns
    -------
    float
        Natural-log first-order rate constant k [1/day], ready to pass as ``decay_rate`` to
        :func:`pipetransport.transport.source_to_endmember`.

    See Also
    --------
    decay_rate_to_log10_decay_rate : Inverse conversion.

    Examples
    --------
    >>> from pipetransport.logremoval import log10_decay_rate_to_decay_rate
    >>> float(round(log10_decay_rate_to_decay_rate(0.2), 6))
    0.460517
    """
    return log10_decay_rate * np.log(10)


def log_removal_to_fraction_remaining(log_removal: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Convert log removal to the surviving fraction ``10 ** (-LR)``.

    Parameters
    ----------
    log_removal : array-like
        Log removal values, any shape.

    Returns
    -------
    ndarray
        Fraction of the original concentration that remains, same shape as the input.

    See Also
    --------
    fraction_remaining_to_log_removal : Inverse conversion.

    Examples
    --------
    >>> from pipetransport.logremoval import log_removal_to_fraction_remaining
    >>> log_removal_to_fraction_remaining([0.0, 1.0, 2.0])
    array([1.  , 0.1 , 0.01])
    """
    return 10.0 ** (-np.asarray(log_removal, dtype=float))


def fraction_remaining_to_log_removal(fraction_remaining: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Convert a surviving fraction to log removal ``-log10(C/C0)``.

    Parameters
    ----------
    fraction_remaining : array-like
        Fraction of the original concentration that remains, strictly positive.

    Returns
    -------
    ndarray
        Log removal, same shape as the input.

    Raises
    ------
    ValueError
        If any value is not strictly positive: a zero or negative residual has no finite log
        removal.

    See Also
    --------
    log_removal_to_fraction_remaining : Inverse conversion.

    Examples
    --------
    >>> from pipetransport.logremoval import fraction_remaining_to_log_removal
    >>> fraction_remaining_to_log_removal([0.1, 0.01, 0.001])
    array([1., 2., 3.])
    """
    fraction_remaining = np.asarray(fraction_remaining, dtype=float)
    _validate_positive(fraction_remaining, name="fraction_remaining")
    return -np.log10(fraction_remaining)


def parallel_mean(
    *, log_removals: npt.ArrayLike, flow_fractions: npt.ArrayLike | None = None, axis: int | None = None
) -> np.floating | npt.NDArray[np.floating]:
    """Combine the log removals of streams that blend into one.

    Concentrations mix, log removals do not, so the blend is
    ``LR = -log10(sum(F_i * 10 ** (-LR_i)))``. It always sits below the arithmetic mean and
    close to the least-treated stream: a single short-circuiting path dominates the mixture.

    Parameters
    ----------
    log_removals : array-like
        Log removal of each stream.
    flow_fractions : array-like or None, optional
        Fraction of the blended flow carried by each stream. Must sum to 1 along ``axis``.
        Defaults to an equal split.
    axis : int or None, optional
        Axis to blend over. ``None`` (default) blends the flattened input and returns a
        scalar, matching how :func:`numpy.mean` treats ``axis=None``.

    Returns
    -------
    numpy.floating or ndarray
        Blended log removal; a scalar for ``axis=None``, otherwise the input with ``axis``
        removed.

    Raises
    ------
    ValueError
        If ``flow_fractions`` does not sum to 1 along ``axis``.

    See Also
    --------
    residence_time_to_log_removal : Per-stream log removal from travel time.

    Examples
    --------
    A district fed 70 % by a well-treated main and 30 % by a short branch:

    >>> import numpy as np
    >>> from pipetransport.logremoval import parallel_mean
    >>> float(
    ...     round(
    ...         parallel_mean(
    ...             log_removals=np.array([3.0, 1.0]),
    ...             flow_fractions=np.array([0.7, 0.3]),
    ...         ),
    ...         6,
    ...     )
    ... )
    1.512862
    """
    log_removals = np.asarray(log_removals, dtype=float)
    remaining = 10.0 ** (-log_removals)
    if flow_fractions is None:
        return -np.log10(np.mean(remaining, axis=axis))
    flow_fractions = np.asarray(flow_fractions, dtype=float)
    if not np.all(np.isclose(np.sum(flow_fractions, axis=axis), 1.0)):
        msg = "flow_fractions must sum to 1.0 (along the specified axis)"
        raise ValueError(msg)
    return -np.log10(np.sum(flow_fractions * remaining, axis=axis))


def segment_decay_rate(
    *,
    network: PipeNetwork,
    bulk_decay_rate: float = 0.0,
    wall_decay_rate: float = 0.0,
    mass_transfer_coefficient: float | None = None,
) -> pd.Series:
    """Per-segment first-order decay rate from bulk and wall reaction.

    Combines a bulk rate with a wall rate scaled by the surface-to-volume ratio of each pipe
    (Rossman, Clark and Grayman, 1994):

        ``k = k_b + k_w * k_f / (R_h * (k_w + k_f))``  with ``R_h = D / 4``

    Leaving ``mass_transfer_coefficient`` at ``None`` takes the limit of fast mass transfer,
    ``k = k_b + 4 * k_w / D``, which is the wall-reaction-controlled case and an upper bound
    on the wall contribution.

    Parameters
    ----------
    network : PipeNetwork
        Network whose segments carry a ``"diameter"`` column [m].
    bulk_decay_rate : float, optional
        Bulk decay rate k_b [1/day], non-negative. Default 0.0.
    wall_decay_rate : float, optional
        Wall reaction rate constant k_w [m/day], non-negative. Default 0.0.
    mass_transfer_coefficient : float or None, optional
        Mass transfer coefficient k_f [m/day] between bulk and wall, strictly positive.
        ``None`` (default) assumes mass transfer is not limiting. It depends on velocity, so
        pass a value representative of the operating range rather than a per-time-step one.

    Returns
    -------
    pandas.Series
        Decay rate [1/day] indexed by segment name, ready to pass as ``decay_rate`` to
        :func:`pipetransport.transport.source_to_endmember`.

    Raises
    ------
    ValueError
        If the network was built from ``volume`` alone (no ``"diameter"`` column) while a
        wall rate is requested, if either rate is negative, or if
        ``mass_transfer_coefficient`` is not strictly positive.

    See Also
    --------
    pipetransport.transport.source_to_endmember : Consumes the returned Series.

    Examples
    --------
    The wall term is four times larger in the 100 mm branch than in the 400 mm trunk:

    >>> from pipetransport.examples import example_network
    >>> from pipetransport.logremoval import segment_decay_rate
    >>> network = example_network()
    >>> rates = segment_decay_rate(
    ...     network=network, bulk_decay_rate=0.3, wall_decay_rate=0.02
    ... )
    >>> float(rates["Plant-A"].round(3)), float(rates["C-T4"].round(3))
    (0.5, 1.1)
    """
    _validate_non_negative(bulk_decay_rate, name="bulk_decay_rate")
    _validate_non_negative(wall_decay_rate, name="wall_decay_rate")
    if wall_decay_rate == 0.0:
        return pd.Series(bulk_decay_rate, index=network.segments.index, dtype=float, name="decay_rate")
    if "diameter" not in network.segments.columns:
        msg = "a wall reaction needs the segment diameter; build the network from length and diameter"
        raise ValueError(msg)
    hydraulic_radius = network.segments["diameter"].to_numpy(dtype=float) / 4.0
    if mass_transfer_coefficient is None:
        wall = wall_decay_rate / hydraulic_radius
    else:
        _validate_positive(mass_transfer_coefficient, name="mass_transfer_coefficient")
        wall = (
            wall_decay_rate
            * mass_transfer_coefficient
            / (hydraulic_radius * (wall_decay_rate + mass_transfer_coefficient))
        )
    return pd.Series(bulk_decay_rate + wall, index=network.segments.index, name="decay_rate")
