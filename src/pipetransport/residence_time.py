"""
Water age: how long the delivered water has been in the network.

Age is the single most-used water quality indicator in a distribution network. It sets the
contact time for every reaction the water undergoes -- disinfectant decay, disinfection
by-product formation, nitrification, biofilm growth -- and it is what a utility can actually
influence through network design and flushing.

The travel time here is the same quantity the transport operator is built on: the time between
a parcel leaving the source and being delivered at the reporting node, obtained by inverting
each pipe's cumulative throughflow in turn. Under a diurnal demand pattern the age at a given
tap swings by hours over the day, because the pipes hold a fixed water volume that a varying
flow pushes through at a varying rate.

Available functions:

- :func:`full` - Bin-averaged travel time [days] for every reporting node. The
  ``"endmember_to_source"`` direction answers "how old is the water delivered now", averaged
  over each ``cout_tedges`` bin and reported on that grid; ``"source_to_endmember"`` answers
  "how long until the water produced now arrives", averaged over each ``tedges`` bin and
  reported on that grid. Both averages are volume-weighted, matching the bin averaging of
  :mod:`pipetransport.transport`, and both are NaN where the record does not constrain the bin.

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from pipetransport._transfer import network_transfer
from pipetransport.network import PipeNetwork  # noqa: TC001 -- runtime dependency of the signature

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

_DIRECTIONS = ("endmember_to_source", "source_to_endmember")


def full(
    *,
    flow: npt.ArrayLike | pd.DataFrame | dict,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex | None = None,
    network: PipeNetwork,
    nodes: list[str] | tuple[str, ...] | None = None,
    direction: str = "endmember_to_source",
    retardation_factor: float = 1.0,
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """Compute the bin-averaged travel time from the source to each reporting node.

    Parameters
    ----------
    flow : DataFrame, mapping, or array-like
        Demand at every endmember [m³/day], constant over each ``tedges`` bin. A DataFrame or
        mapping is keyed by endmember name; an array must have shape
        ``(n_endmembers, len(tedges) - 1)`` ordered as ``network.endmembers``.
    tedges : pandas.DatetimeIndex
        Time edges of the ``flow`` bins. Length ``n_flow + 1``.
    cout_tedges : pandas.DatetimeIndex or None, optional
        Output grid for ``direction="endmember_to_source"``. Defaults to ``tedges``. Ignored
        by the other direction, which always reports on ``tedges``.
    network : PipeNetwork
        The distribution network.
    nodes : list of str or None, optional
        Nodes to report at, in output row order. Any node is allowed; a junction reports the
        age of the water passing through it. Defaults to ``network.endmembers``.
    direction : {"endmember_to_source", "source_to_endmember"}, optional
        Which question to answer.

        - ``"endmember_to_source"`` (default): how long ago the water delivered at the node
          during each ``cout_tedges`` bin left the source. Averaged with the node's
          throughflow as weight.
        - ``"source_to_endmember"``: how long the water produced during each ``tedges`` bin
          takes to reach the node. Averaged with the node-destined volume as weight, so
          production that mostly serves other branches contributes little.

    retardation_factor : float, optional
        Multiplier on every segment volume, ``>= 1``. Default 1.0.
    spinup : {"constant"} or None, optional
        ``"constant"`` (default) warm-starts the record by extending it backwards at the
        first observed demand, so the earliest bins carry a value instead of NaN. ``None``
        keeps strict validity.

    Returns
    -------
    numpy.ndarray
        Mean travel time [days] of shape ``(len(nodes), n_bins)``, with ``n_bins`` taken from
        ``cout_tedges`` for ``"endmember_to_source"`` and from ``tedges`` for
        ``"source_to_endmember"``. NaN marks bins the record does not constrain.

    Raises
    ------
    ValueError
        If ``direction`` is not one of the two accepted values, if a time axis is not
        strictly increasing or has the wrong length, if ``flow`` holds NaN or negative
        values, if ``retardation_factor < 1``, or if a requested node is not part of the
        network.

    See Also
    --------
    pipetransport.transport.source_to_endmember : Transport behind these same travel times.
    pipetransport.logremoval.residence_time_to_log_removal : Turn age into first-order removal.
    :ref:`concept-water-age` : What age means in a distribution network.

    Notes
    -----
    Chlorine residual follows from age only when the produced concentration is constant, and
    even then ``exp(-k * mean_age)`` is not the bin average of ``exp(-k * age)``. Pass
    ``decay_rate`` to :func:`pipetransport.transport.source_to_endmember` for the exact
    residual; use the age here as the diagnostic it is.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pipetransport.examples import example_network, example_demand
    >>> from pipetransport.residence_time import full
    >>>
    >>> network = example_network()
    >>> tedges = pd.date_range("2025-06-01", "2025-06-08", freq="h")
    >>> demand = example_demand(tedges=tedges, network=network)
    >>> age = full(flow=demand, tedges=tedges, network=network)
    >>> age.shape
    (4, 168)

    T4 sits at the end of a long, thin, low-demand branch, so its water is the oldest:

    >>> bool(np.nanmean(age[3]) > np.nanmean(age[0]))
    True
    """
    if direction not in _DIRECTIONS:
        msg = f"direction must be one of {_DIRECTIONS}; got {direction!r}"
        raise ValueError(msg)
    tedges = pd.DatetimeIndex(tedges)
    _, transfer, n_pad = network_transfer(
        network=network,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges if cout_tedges is None else cout_tedges,
        nodes=nodes,
        decay_rate=0.0,
        retardation_factor=retardation_factor,
        spinup=spinup,
    )
    if direction == "endmember_to_source":
        return transfer.residence_time_out
    # The warm-start prefix is an assumed history, not a result; drop it so the rows align
    # with the user-provided tedges.
    return transfer.residence_time_in[:, n_pad:]
