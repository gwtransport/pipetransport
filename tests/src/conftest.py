"""
Shared fixtures for the pipetransport unit tests.

Three networks cover the range the tests need: a single pipe where every quantity has a
one-line closed form, a two-branch tree that is the smallest topology in which the flow split
matters, and the seven-segment example network. Demand comes either constant (so the
proportional-split closed form applies and results can be checked analytically) or from the
deliberately non-proportional diurnal profile of :mod:`pipetransport.examples`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipetransport.examples import example_demand, example_network
from pipetransport.network import PipeNetwork

# Make tests-only helper modules (the brute-force _oracle) importable by bare name under the
# importlib import mode.
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# Networks
# ============================================================================


@pytest.fixture
def single_pipe():
    """One 100 m³ pipe from Plant to T1: travel time is exactly volume / demand."""
    segments = pd.DataFrame(
        {"from": ["Plant"], "to": ["T1"], "volume": [100.0]},
        index=["Plant-T1"],
    )
    return PipeNetwork(segments=segments, source="Plant")


@pytest.fixture
def single_pipe_35():
    """Return a 35.34 m³ service line: at 600 m³/day the transit is well under two bins."""
    segments = pd.DataFrame({"from": ["Plant"], "to": ["T1"], "volume": [35.34]}, index=["Plant-T1"])
    return PipeNetwork(segments=segments, source="Plant")


@pytest.fixture
def two_branch():
    """Plant -> A, then A -> T1 and A -> T2: the smallest network with a flow split."""
    segments = pd.DataFrame(
        {
            "from": ["Plant", "A", "A"],
            "to": ["A", "T1", "T2"],
            "volume": [300.0, 40.0, 60.0],
        },
        index=["Plant-A", "A-T1", "A-T2"],
    )
    return PipeNetwork(segments=segments, source="Plant")


@pytest.fixture
def network():
    """Return the seven-segment, four-endmember example network."""
    return example_network()


# ============================================================================
# Time grids
# ============================================================================


@pytest.fixture
def hourly_tedges():
    """Ten days of hourly bins (240 bins)."""
    return pd.date_range("2025-06-01", periods=241, freq="h")


@pytest.fixture
def short_tedges():
    """Four days of hourly bins (96 bins), short enough for the brute-force oracle."""
    return pd.date_range("2025-06-01", periods=97, freq="h")


# ============================================================================
# Demand
# ============================================================================


@pytest.fixture
def constant_demand():
    """Build a factory for a constant per-endmember demand array of shape (n_endmembers, n_bins).

    Returns
    -------
    callable
        ``make(network, tedges, means=None)`` -> ndarray. ``means`` defaults to
        ``100 * (1 + i)`` m³/day for endmember ``i``, so the endmembers carry visibly
        different shares of the production.
    """

    def _make(network, tedges, means=None):
        n_bins = len(tedges) - 1
        if means is None:
            means = [100.0 * (1 + i) for i in range(len(network.endmembers))]
        return np.asarray(means, dtype=float)[:, None] * np.ones(n_bins)

    return _make


@pytest.fixture
def diurnal_demand():
    """Build a factory for the non-proportional diurnal demand of :func:`pipetransport.examples.example_demand`.

    Returns
    -------
    callable
        ``make(network, tedges)`` -> DataFrame with one column per endmember.
    """

    def _make(network, tedges):
        return example_demand(tedges=tedges, network=network)

    return _make


@pytest.fixture
def analytic_travel_time():
    """Build a factory for the closed-form travel time of a path under a constant, proportional split.

    With every segment carrying a fixed fraction ``f`` of the production, the composition of
    the per-pipe displacement conditions collapses to a single effective volume
    ``sum(V_i / f_i)`` expressed in source throughflow, so the travel time is that volume
    divided by the production rate.

    Returns
    -------
    callable
        ``make(network, demand, node)`` -> float travel time in days, where ``demand`` is a
        constant ``(n_endmembers, n_bins)`` array.
    """

    def _make(network, demand, node):
        segment_flow = network.segment_flow(flow=demand)[:, 0]
        production = float(np.sum(demand[:, 0]))
        row_of = {name: i for i, name in enumerate(network.segments.index)}
        rows = [row_of[name] for name in network.paths[node]]
        volume = network.segments["volume"].to_numpy(dtype=float)
        effective = float(np.sum(volume[rows] / (segment_flow[rows] / production)))
        return effective / production

    return _make
