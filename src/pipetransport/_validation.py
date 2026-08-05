"""
Composable input-validation atoms for the public entry points.

The public functions in :mod:`pipetransport.network`, :mod:`pipetransport.transport` and
:mod:`pipetransport.residence_time` share a small set of input invariants (bin-edge parity,
NaN-free arrays, non-negative flow, positive geometry). The atoms here factor those
invariants once so that each module composes them with its own error-message wording.

Each atom is keyword-only (except for the value under test) and raises ``ValueError`` with a
default message; the optional ``message`` keyword lets a caller supply its own wording
verbatim, which downstream ``pytest.raises(..., match=...)`` tests pin.

This module has no public API; importers are the package modules themselves plus
``tests/src/test_validation.py``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd  # noqa: TC002  -- pandas is a hard runtime dependency; import unconditionally


def _validate_tedges(tedges: pd.DatetimeIndex, values: npt.ArrayLike, *, tedges_name: str, values_name: str) -> None:
    """Validate bin-edge parity and strict monotonicity of a time-edge array.

    Parameters
    ----------
    tedges : DatetimeIndex
        Bin edges (length ``n + 1``).
    values : array-like
        Bin-constant values whose last axis has length ``n``.
    tedges_name, values_name : str
        Names used in the error messages, e.g. ``"tedges"`` and ``"cin"``.

    Raises
    ------
    ValueError
        If ``len(tedges) != values.shape[-1] + 1`` or ``tedges`` is not strictly increasing.
    """
    n_values = np.asarray(values).shape[-1]
    if len(tedges) != n_values + 1:
        msg = f"{tedges_name} must have one more element than {values_name}"
        raise ValueError(msg)
    # Non-monotonic edges would silently corrupt the cumulative-volume mapping.
    if np.any(np.diff(tedges.asi8) <= 0):
        msg = f"{tedges_name} must be strictly increasing"
        raise ValueError(msg)


def _validate_no_nan(arr: npt.ArrayLike, *, name: str) -> None:
    """Validate that ``arr`` contains no NaN values.

    Parameters
    ----------
    arr : array-like
        Array to check.
    name : str
        Variable name used in the error message.

    Raises
    ------
    ValueError
        If any element of ``arr`` is NaN.
    """
    if np.any(np.isnan(np.asarray(arr, dtype=float))):
        msg = f"{name} contains NaN values, which are not allowed"
        raise ValueError(msg)


def _validate_non_negative(arr: npt.ArrayLike, *, name: str, message: str | None = None) -> None:
    """Validate that every element of ``arr`` is finite and non-negative.

    Zeros are allowed; the companion :func:`_validate_positive` rejects them too. NaN and
    ``+inf`` are rejected explicitly: both pass every ``< 0`` comparison, so a bare
    inequality would let them slip through and poison the downstream computation.

    Parameters
    ----------
    arr : array-like
        Array to check (any shape).
    name : str
        Variable name used in the default error message.
    message : str, optional
        Override the default ``"{name} must be non-negative"`` wording.

    Raises
    ------
    ValueError
        If any element of ``arr`` is negative or non-finite.
    """
    a = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(a) & (a >= 0.0)):
        msg = message if message is not None else f"{name} must be non-negative"
        raise ValueError(msg)


def _validate_positive(arr: npt.ArrayLike, *, name: str, message: str | None = None) -> None:
    """Validate that every element of ``arr`` is finite and strictly positive.

    NaN and ``+inf`` are rejected explicitly: both pass every ``<= 0`` comparison, so a bare
    inequality would let them slip through and poison the downstream computation.

    Parameters
    ----------
    arr : array-like
        Array to check (any shape).
    name : str
        Variable name used in the default error message.
    message : str, optional
        Override the default ``"{name} must be positive"`` wording.

    Raises
    ------
    ValueError
        If any element of ``arr`` is ``<= 0`` or non-finite.
    """
    a = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(a) & (a > 0.0)):
        msg = message if message is not None else f"{name} must be positive"
        raise ValueError(msg)


def _validate_retardation_factor(value: npt.ArrayLike) -> None:
    """Validate that every retardation factor is ``>= 1`` (anti-retardation is unphysical).

    The check is written as ``not (value >= 1.0).all()`` rather than ``(value < 1.0).any()``
    so that NaN is rejected too: ``NaN >= 1.0`` is False, so the bare ``< 1.0`` form would let
    NaN pass and silently propagate an all-NaN transport output.

    Parameters
    ----------
    value : array-like
        Retardation factor, scalar or one per segment.

    Raises
    ------
    ValueError
        If any entry is NaN or below 1.
    """
    if not np.all(np.asarray(value, dtype=float) >= 1.0):
        msg = "retardation_factor must be >= 1.0"
        raise ValueError(msg)


def _per_segment(value: float | Mapping[str, float], index: pd.Index, *, name: str) -> npt.NDArray[np.floating]:
    """Coerce a scalar or a segment-keyed mapping to an array in segment-table order.

    Parameters
    ----------
    value : float or mapping
        One value shared by every segment, or a mapping from segment name to its own.
    index : pandas.Index
        Segment names, in the order the returned array is to follow.
    name : str
        Parameter name used in the error messages.

    Returns
    -------
    ndarray
        One value per segment, in ``index`` order.

    Raises
    ------
    ValueError
        If the mapping misses a segment or holds a key that is not one.
    """
    if not isinstance(value, Mapping):
        return np.full(len(index), float(value))
    named = dict(zip(map(str, value), np.asarray(list(value.values()), dtype=float), strict=True))
    names = [str(segment) for segment in index]
    missing = [segment for segment in names if segment not in named]
    if missing:
        msg = f"{name} is missing segment(s): {missing}"
        raise ValueError(msg)
    unknown = sorted(set(named) - set(names))
    if unknown:
        msg = f"{name} holds key(s) that are not segments: {unknown}"
        raise ValueError(msg)
    return np.array([named[segment] for segment in names], dtype=float)
