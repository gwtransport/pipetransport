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

Available functions, the two directions of the same travel time:

- :func:`source_to_endmember` - How long the water produced during each ``tedges`` bin takes to
  reach each reporting node, reported on ``tedges``.

- :func:`endmember_to_source` - How old the water delivered during each ``cout_tedges`` bin is,
  reported on ``cout_tedges``.

Both averages are volume-weighted, matching the bin averaging of :mod:`pipetransport.transport`,
and both are NaN where the record does not constrain the bin. The names match the transport and
heat pairs, but the asymmetry there does not apply: neither direction takes an observation, so
both report per node and both keep ``report_nodes``. In particular this
:func:`endmember_to_source` is a per-node report looking backward in time, not a reconstruction
of a source series.

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd

from pipetransport._transfer import network_transfer
from pipetransport.network import PipeNetwork  # noqa: TC001 -- runtime dependency of the signature

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


def source_to_endmember(
    *,
    flow: Mapping[str, npt.ArrayLike],
    tedges: pd.DatetimeIndex,
    network: PipeNetwork,
    report_nodes: list[str] | tuple[str, ...] | None = None,
    retardation_factor: float = 1.0,
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """Compute how long the water produced in each bin takes to reach each reporting node.

    The average over a ``tedges`` bin is weighted by the volume destined for that node, so
    production that mostly serves other branches contributes little to the bin's mean.

    Parameters
    ----------
    flow : mapping
        Demand at every endmember [m³/day], keyed by endmember name: one bin-constant array
        per endmember on the ``tedges`` bins.
    tedges : pandas.DatetimeIndex
        Time edges of the ``flow`` bins, and of the output. Length ``n_flow + 1``.
    network : PipeNetwork
        The distribution network.
    report_nodes : list of str or None, optional
        Nodes to report at, in output row order. Any node is allowed; a junction reports the
        age of the water passing through it. Defaults to ``network.endmembers``.
    retardation_factor : float, optional
        Multiplier on every segment volume, ``>= 1``. Default 1.0.
    spinup : {"constant"} or None, optional
        ``"constant"`` (default) warm-starts the record by extending it backwards at the
        first observed demand, so the earliest bins carry a value instead of NaN. ``None``
        keeps strict validity.

    Returns
    -------
    numpy.ndarray
        Mean travel time [days] of shape ``(len(report_nodes), len(tedges) - 1)``. NaN marks
        bins the record does not constrain.

    Raises
    ------
    ValueError
        If a time axis is not strictly increasing or has the wrong length, if ``flow`` holds
        NaN or negative values, if ``retardation_factor < 1``, or if a requested node is not
        part of the network.

    See Also
    --------
    endmember_to_source : The other direction -- the age of the water delivered now.
    pipetransport.transport.source_to_endmember : Transport behind these same travel times.
    :ref:`concept-water-age` : What age means in a distribution network.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pipetransport.examples import example_network, example_demand
    >>> from pipetransport.residence_time import source_to_endmember
    >>>
    >>> network = example_network()
    >>> tedges = pd.date_range("2025-06-01", "2025-06-08", freq="h")
    >>> demand = example_demand(tedges=tedges, network=network)
    >>> lead = source_to_endmember(flow=demand, tedges=tedges, network=network)
    >>> lead.shape
    (4, 168)

    T4 sits at the end of a long, thin, low-demand branch, so its water takes longest:

    >>> bool(np.nanmean(lead[3]) > np.nanmean(lead[0]))
    True
    """
    tedges = pd.DatetimeIndex(tedges)
    _, transfer, n_pad = network_transfer(
        network=network,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        report_nodes=report_nodes,
        decay_rate=0.0,
        retardation_factor=retardation_factor,
        spinup=spinup,
    )
    # The warm-start prefix is an assumed history, not a result; drop it so the rows align
    # with the user-provided tedges.
    return transfer.residence_time_in[:, n_pad:]


def endmember_to_source(
    *,
    flow: Mapping[str, npt.ArrayLike],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    network: PipeNetwork,
    report_nodes: list[str] | tuple[str, ...] | None = None,
    retardation_factor: float = 1.0,
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """Compute how long ago the water delivered in each output bin left the source.

    The average over a ``cout_tedges`` bin is weighted by the node's own throughflow, which
    is what makes it the age of the water the tap actually delivered.

    Parameters
    ----------
    flow : mapping
        Demand at every endmember [m³/day], keyed by endmember name: one bin-constant array
        per endmember on the ``tedges`` bins.
    tedges : pandas.DatetimeIndex
        Time edges of the ``flow`` bins. Length ``n_flow + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges of the output bins; alignment and resolution are free.
    network : PipeNetwork
        The distribution network.
    report_nodes : list of str or None, optional
        Nodes to report at, in output row order. Any node is allowed; a junction reports the
        age of the water passing through it. Defaults to ``network.endmembers``.
    retardation_factor : float, optional
        Multiplier on every segment volume, ``>= 1``. Default 1.0.
    spinup : {"constant"} or None, optional
        ``"constant"`` (default) warm-starts the record by extending it backwards at the
        first observed demand, so the earliest bins carry a value instead of NaN. ``None``
        keeps strict validity.

    Returns
    -------
    numpy.ndarray
        Mean age [days] of shape ``(len(report_nodes), len(cout_tedges) - 1)``. NaN marks
        bins the record does not constrain.

    Raises
    ------
    ValueError
        If a time axis is not strictly increasing or has the wrong length, if ``flow`` holds
        NaN or negative values, if ``retardation_factor < 1``, or if a requested node is not
        part of the network.

    See Also
    --------
    source_to_endmember : The other direction -- the lead time of the water produced now.
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
    >>> from pipetransport.residence_time import endmember_to_source
    >>>
    >>> network = example_network()
    >>> tedges = pd.date_range("2025-06-01", "2025-06-08", freq="h")
    >>> demand = example_demand(tedges=tedges, network=network)
    >>> age = endmember_to_source(
    ...     flow=demand, tedges=tedges, cout_tedges=tedges, network=network
    ... )
    >>> age.shape
    (4, 168)

    T4 sits at the end of a long, thin, low-demand branch, so its water is the oldest:

    >>> bool(np.nanmean(age[3]) > np.nanmean(age[0]))
    True
    """
    _, transfer, _ = network_transfer(
        network=network,
        flow=flow,
        tedges=pd.DatetimeIndex(tedges),
        cout_tedges=cout_tedges,
        report_nodes=report_nodes,
        decay_rate=0.0,
        retardation_factor=retardation_factor,
        spinup=spinup,
    )
    return transfer.residence_time_out
