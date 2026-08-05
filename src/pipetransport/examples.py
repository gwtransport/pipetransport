"""
A ready-made distribution network and demand pattern for documentation and tests.

The example is a treatment plant feeding a 400 mm trunk main that splits into two district
mains, each serving two monitoring points. It is deliberately unbalanced: T4 sits at the end
of a 2.5 km, 100 mm branch that carries only a tenth of the production, so its path holds
*less* water than T1's yet delivers by far the *oldest* water. That contrast is what makes the
network worth modelling.

The demand pattern is deliberately **not** proportional. Every endmember peaks at a different
hour of the day -- households in the morning and evening, a commercial block at midday, an
industrial user on the night shift -- so the fraction of production each pipe carries moves
over the day. A model that assumes fixed flow fractions cannot represent that; this package
recomputes the split at every time step.

Available functions:

- :func:`example_network` - The seven-segment, four-endmember :class:`~pipetransport.network.PipeNetwork`.

- :func:`example_demand` - Diurnal demand for its endmembers on a given time grid, as a mapping
  from endmember name to a bin-constant array.

- :func:`example_heat_network` - The same network dressed for :mod:`pipetransport.heat`: land
  cover, burial depth and soil properties per segment.

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from pipetransport.heat import HeatNetwork
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
        A tree holding 473 m³ of water across seven segments, with endmembers
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
    PipeNetwork(source='Plant', segments=7, endmembers=4, volume=473.2 m3)
    >>> network.paths["T4"]
    ('Plant-A', 'A-C', 'C-T4')
    """
    segments = {
        "Plant-A": {"from": "Plant", "to": "A", "length": 2000.0, "diameter": 0.40},  # m
        "A-B": {"from": "A", "to": "B", "length": 1500.0, "diameter": 0.30},
        "A-C": {"from": "A", "to": "C", "length": 1200.0, "diameter": 0.25},
        "B-T1": {"from": "B", "to": "T1", "length": 800.0, "diameter": 0.15},
        "B-T2": {"from": "B", "to": "T2", "length": 400.0, "diameter": 0.20},
        "C-T3": {"from": "C", "to": "T3", "length": 600.0, "diameter": 0.15},
        "C-T4": {"from": "C", "to": "T4", "length": 2500.0, "diameter": 0.10},
    }
    return PipeNetwork(segments=segments, source="Plant")


def example_demand(*, tedges: pd.DatetimeIndex, network: PipeNetwork) -> dict[str, npt.NDArray[np.floating]]:
    """Build a diurnal, deliberately non-proportional demand pattern for a network.

    Each endmember follows ``mean * (1 + amplitude * cos(2*pi*(hour - peak) / 24))`` with its
    own amplitude and peak hour, so the flow fraction every pipe carries changes over the day.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time bin edges, ``n + 1`` edges for ``n`` bins.
    network : PipeNetwork
        Network whose endmembers the mapping is keyed by.

    Returns
    -------
    dict of str to ndarray
        Demand [m³/day] keyed by endmember name, in ``network.endmembers`` order, each an
        array of ``n`` bin-constant values on ``tedges``. Ready to pass as ``flow``.

    See Also
    --------
    example_network : The network these profiles are tuned for.
    pipetransport.network.PipeNetwork.segment_flow : Turns this demand into per-pipe flow.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pipetransport.examples import example_network, example_demand
    >>> network = example_network()
    >>> demand = example_demand(
    ...     tedges=pd.date_range("2025-06-01", periods=25, freq="h"), network=network
    ... )
    >>> list(demand)
    ['T1', 'T2', 'T3', 'T4']

    Averaged over the day every endmember hits its mean, but the split is far from constant:

    >>> float(demand["T4"].mean().round(6))
    80.0
    >>> share = demand["T4"] / np.sum(list(demand.values()), axis=0)
    >>> bool(share.max() > 1.6 * share.min())
    True
    """
    tedges = pd.DatetimeIndex(tedges)
    midpoint = tedges[:-1] + (tedges[1:] - tedges[:-1]) / 2
    hour = midpoint.hour + midpoint.minute / 60.0 + midpoint.second / 3600.0
    n_endmember = len(network.endmembers)
    return {
        name: np.asarray(mean * (1.0 + amplitude * np.cos(2.0 * np.pi * (hour - peak) / 24.0)), dtype=float)
        for name, (mean, amplitude, peak) in (
            (name, _PROFILES.get(name, (200.0, 0.4, 6.0 + 24.0 * i / n_endmember)))
            for i, name in enumerate(network.endmembers)
        )
    }


def example_heat_network() -> HeatNetwork:
    """Build the example network dressed for the heat pair: cover, burial and soil per segment.

    The same seven segments as :func:`example_network`, plus the columns
    :class:`~pipetransport.heat.HeatNetwork` needs. The trunk and the branches to T1, T2 and
    T4 run under grass, the rest under pavement, and the burial depths vary a little from
    pipe to pipe so no two pipes share a soil field by accident. The wall, film and surface
    defaults are left alone: a bare pipe under a prescribed-temperature surface.

    Returns
    -------
    HeatNetwork
        The example network with ``cover``, ``depth``, ``alpha`` and ``kappa_soil`` per
        segment and ``eta = 0.41`` throughout.

    See Also
    --------
    example_network : The transport-only network these segments come from.
    pipetransport.heat.source_to_endmember : Consumes this network.

    Examples
    --------
    >>> from pipetransport.examples import example_heat_network
    >>> network = example_heat_network()
    >>> network.endmembers
    ('T1', 'T2', 'T3', 'T4')
    >>> row = network.segments.loc["C-T4"]
    >>> row["cover"], float(row["depth"]), float(row["kappa_soil"])
    ('grass', 0.8, 0.025)
    """
    soil = {
        "grass": {"cover": "grass", "alpha": 0.05, "kappa_soil": 0.025, "eta": 0.41},
        "paved": {"cover": "paved", "alpha": 0.075, "kappa_soil": 0.035, "eta": 0.41},
    }
    buried = {
        "Plant-A": ("grass", 1.2),
        "A-B": ("grass", 1.0),
        "A-C": ("paved", 1.0),
        "B-T1": ("paved", 0.9),
        "B-T2": ("grass", 1.0),
        "C-T3": ("paved", 1.0),
        "C-T4": ("grass", 0.8),
    }
    pipes = example_network().segments.drop(columns="volume").to_dict(orient="index")
    segments = {
        name: {**pipe, **soil[cover], "depth": depth}
        for (name, pipe), (cover, depth) in zip(pipes.items(), buried.values(), strict=True)
    }
    return HeatNetwork(segments=segments, source="Plant")
