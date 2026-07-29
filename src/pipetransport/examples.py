"""
A ready-made distribution network and demand pattern for documentation and tests.

The example is a treatment plant feeding a 400 mm trunk main that splits into two district
mains, each serving two monitoring points. It is deliberately unbalanced: T4 sits at the end
of a 2.5 km, 100 mm branch that carries a tenth of the production, so it holds the *least*
water of any path yet delivers by far the *oldest* water. That contrast is what makes the
network worth modelling.

The demand pattern is deliberately **not** proportional. Every endmember peaks at a different
hour of the day -- households in the morning and evening, a commercial block at midday, an
industrial user on the night shift -- so the fraction of production each pipe carries moves
over the day. A model that assumes fixed flow fractions cannot represent that; this package
recomputes the split at every time step.

Available functions:

- :func:`example_network` - The seven-segment, four-endmember :class:`~pipetransport.network.PipeNetwork`.

- :func:`example_demand` - Diurnal demand for its endmembers on a given time grid, as a
  DataFrame indexed by bin midpoint with one column per endmember.

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipetransport.network import PipeNetwork

# Mean demand [m³/day], relative diurnal amplitude [-] and peak hour [h] of each endmember of
# example_network. An endmember of some other network falls back to a deterministic profile
# derived from its position, so the function stays usable outside the canned example.
_PROFILES = {
    "T1": (240.0, 0.55, 8.0),  # residential, morning peak
    "T2": (360.0, 0.45, 19.0),  # residential, evening peak
    "T3": (120.0, 0.35, 12.0),  # commercial block, midday
    "T4": (80.0, 0.70, 2.0),  # industrial, night shift
}


def example_network() -> PipeNetwork:
    """Build the example distribution network: one plant, three junctions, four endmembers.

    Returns
    -------
    PipeNetwork
        A tree holding 1114 m³ of water across seven segments, with endmembers
        ``("T1", "T2", "T3", "T4")``.

    See Also
    --------
    example_demand : The diurnal demand pattern that goes with it.
    pipetransport.network.PipeNetwork : The class returned here.

    Examples
    --------
    >>> from pipetransport.examples import example_network
    >>> network = example_network()
    >>> network
    PipeNetwork(source='Plant', segments=7, endmembers=4, volume=1113.9 m3)
    >>> network.paths["T4"]
    ('Plant-A', 'A-C', 'C-T4')
    """
    segments = pd.DataFrame(
        {
            "from": ["Plant", "A", "A", "B", "B", "C", "C"],
            "to": ["A", "B", "C", "T1", "T2", "T3", "T4"],
            "length": [2000.0, 1500.0, 1200.0, 800.0, 400.0, 600.0, 2500.0],  # m
            "diameter": [0.40, 0.30, 0.25, 0.15, 0.20, 0.15, 0.10],  # m
        },
        index=["Plant-A", "A-B", "A-C", "B-T1", "B-T2", "C-T3", "C-T4"],
    )
    return PipeNetwork(segments=segments, source="Plant")


def example_demand(*, tedges: pd.DatetimeIndex, network: PipeNetwork) -> pd.DataFrame:
    """Build a diurnal, deliberately non-proportional demand pattern for a network.

    Each endmember follows ``mean * (1 + amplitude * cos(2*pi*(hour - peak) / 24))`` with its
    own amplitude and peak hour, so the flow fraction every pipe carries changes over the day.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time bin edges, ``n + 1`` edges for ``n`` bins.
    network : PipeNetwork
        Network whose endmembers the columns follow.

    Returns
    -------
    pandas.DataFrame
        Demand [m³/day] with ``n`` rows indexed by bin midpoint and one column per endmember,
        in ``network.endmembers`` order. Ready to pass as ``flow``.

    See Also
    --------
    example_network : The network these profiles are tuned for.
    pipetransport.network.PipeNetwork.segment_flow : Turns this demand into per-pipe flow.

    Examples
    --------
    >>> import pandas as pd
    >>> from pipetransport.examples import example_network, example_demand
    >>> network = example_network()
    >>> demand = example_demand(tedges=pd.date_range("2025-06-01", periods=25, freq="h"),
    ...                         network=network)
    >>> list(demand.columns)
    ['T1', 'T2', 'T3', 'T4']

    Averaged over the day every endmember hits its mean, but the split is far from constant:

    >>> float(demand["T4"].mean().round(6))
    80.0
    >>> share = demand["T4"] / demand.sum(axis=1)
    >>> bool(share.max() > 1.6 * share.min())
    True
    """
    tedges = pd.DatetimeIndex(tedges)
    midpoint = tedges[:-1] + (tedges[1:] - tedges[:-1]) / 2
    hour = midpoint.hour + midpoint.minute / 60.0 + midpoint.second / 3600.0
    n_endmember = len(network.endmembers)
    return pd.DataFrame(
        {
            name: mean * (1.0 + amplitude * np.cos(2.0 * np.pi * (hour - peak) / 24.0))
            for name, (mean, amplitude, peak) in (
                (name, _PROFILES.get(name, (200.0, 0.4, 6.0 + 24.0 * i / n_endmember)))
                for i, name in enumerate(network.endmembers)
            )
        },
        index=midpoint,
    )
