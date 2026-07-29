"""Tests for pipetransport.examples: the canned network and its diurnal demand pattern.

The example exists to carry a specific claim: the demand is *not* proportional, so the flow
fraction every pipe carries moves over the day and a fixed-fraction model cannot represent it.
These tests pin the geometry and topology against the documented table, recover the mean,
amplitude and peak hour of every profile exactly from its daily harmonic, and check that the
resulting endmember shares and segment flow fractions genuinely vary.
"""

import numpy as np
import pandas as pd
import pytest

from pipetransport.examples import _PROFILES, example_demand
from pipetransport.network import PipeNetwork

# The documented segment table of example_network: name, from, to, length [m], diameter [m].
DOCUMENTED_SEGMENTS = [
    ("Plant-A", "Plant", "A", 2000.0, 0.40),
    ("A-B", "A", "B", 1500.0, 0.30),
    ("A-C", "A", "C", 1200.0, 0.25),
    ("B-T1", "B", "T1", 800.0, 0.15),
    ("B-T2", "B", "T2", 400.0, 0.20),
    ("C-T3", "C", "T3", 600.0, 0.15),
    ("C-T4", "C", "T4", 2500.0, 0.10),
]

DOCUMENTED_PATHS = {
    "Plant": (),
    "A": ("Plant-A",),
    "B": ("Plant-A", "A-B"),
    "C": ("Plant-A", "A-C"),
    "T1": ("Plant-A", "A-B", "B-T1"),
    "T2": ("Plant-A", "A-B", "B-T2"),
    "T3": ("Plant-A", "A-C", "C-T3"),
    "T4": ("Plant-A", "A-C", "C-T4"),
}


def _hour_of_day(index):
    """Hour of day of a DatetimeIndex as a float array, so a daily harmonic can be projected onto it."""
    return np.asarray(index.hour + index.minute / 60.0 + index.second / 3600.0, dtype=float)


def _daily_harmonic(series):
    """Recover (mean, mean*amplitude, peak hour) of ``mean * (1 + amplitude * cos(2 pi (h - peak) / 24))``.

    Exact whenever the samples cover a whole number of days on a uniform grid of at least three
    bins per day: the constant term and the second harmonic both sum to zero over a full period,
    leaving ``sum(v_k exp(-2 pi i h_k / 24)) = N/2 * mean * amplitude * exp(-2 pi i peak / 24)``.
    """
    values = np.asarray(series, dtype=float)
    coefficient = np.sum(values * np.exp(-2j * np.pi * _hour_of_day(series.index) / 24.0))
    peak = float(-np.angle(coefficient) * 24.0 / (2.0 * np.pi)) % 24.0
    return float(values.mean()), 2.0 * abs(coefficient) / len(values), peak


# ============================================================================
# example_network: geometry and topology
# ============================================================================


def test_example_network_geometry_matches_documented_table(network):
    names, upstream, downstream, length, diameter = (list(column) for column in zip(*DOCUMENTED_SEGMENTS, strict=True))
    assert list(network.segments.index) == names
    assert list(network.segments["from"]) == upstream
    assert list(network.segments["to"]) == downstream
    np.testing.assert_allclose(network.segments["length"].to_numpy(float), length, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(network.segments["diameter"].to_numpy(float), diameter, rtol=0.0, atol=0.0)

    # Volume is the cylinder volume of the inner diameter, and the total is the documented 473.2 m3.
    expected_volume = np.pi / 4.0 * np.asarray(diameter) ** 2 * np.asarray(length)
    np.testing.assert_allclose(network.segments["volume"].to_numpy(float), expected_volume, rtol=1e-15)
    np.testing.assert_allclose(network.segments["volume"].sum(), 473.2023934469627, rtol=1e-13)
    assert repr(network) == "PipeNetwork(source='Plant', segments=7, endmembers=4, volume=473.2 m3)"


def test_example_network_topology_matches_documented_tree(network):
    assert network.source == "Plant"
    assert network.nodes == ("Plant", "A", "B", "C", "T1", "T2", "T3", "T4")
    assert network.endmembers == ("T1", "T2", "T3", "T4")
    assert network.paths == DOCUMENTED_PATHS
    # Every endmember is reached by exactly one path, which is what makes the single-cin model exact.
    assert len({network.paths[e] for e in network.endmembers}) == len(network.endmembers)


def test_example_network_t4_branch_is_the_documented_thin_long_tenth(network):
    # "T4 sits at the end of a 2.5 km, 100 mm branch that carries a tenth of the production."
    trunk = network.segments.loc["Plant-A"]
    branch = network.segments.loc["C-T4"]
    assert (float(trunk["diameter"]), float(branch["length"]), float(branch["diameter"])) == (0.40, 2500.0, 0.10)

    mean_demand = np.array([_PROFILES[e][0] for e in network.endmembers])
    np.testing.assert_allclose(mean_demand[-1] / mean_demand.sum(), 0.10, rtol=0.0, atol=0.0)


def test_example_network_delivers_the_oldest_water_at_t4(network, constant_demand, hourly_tedges, analytic_travel_time):
    # Under the profile means the split is constant, so travel time is the closed form
    # sum(V_i / f_i) / production. T4 must come out oldest "by far" -- the point of the example.
    means = [_PROFILES[e][0] for e in network.endmembers]
    demand = constant_demand(network, hourly_tedges, means=means)
    ages = {e: analytic_travel_time(network, demand, e) for e in network.endmembers}

    # Effective volumes sum(V_i / f_i) of 439.82, 420.62, 557.63 and 683.30 m3, over 800 m3/day.
    np.testing.assert_allclose(
        [ages[e] for e in ("T1", "T2", "T3", "T4")],
        np.array([439.8229715025711, 420.62434973063347, 557.6326960121884, 683.29640215578]) / 800.0,
        rtol=1e-12,
    )
    assert max(ages, key=ages.get) == "T4"
    assert ages["T4"] > 1.2 * max(ages[e] for e in ("T1", "T2", "T3"))


# ============================================================================
# example_demand: layout, means, amplitudes and peak hours
# ============================================================================


def test_example_demand_columns_follow_endmembers_and_index_is_bin_midpoint(network, hourly_tedges):
    demand = example_demand(tedges=hourly_tedges, network=network)
    assert list(demand.columns) == list(network.endmembers)
    assert len(demand) == len(hourly_tedges) - 1
    expected_index = hourly_tedges[:-1] + (hourly_tedges[1:] - hourly_tedges[:-1]) / 2
    np.testing.assert_array_equal(demand.index.to_numpy(), expected_index.to_numpy())
    # The DataFrame is ready to pass straight in as `flow`.
    np.testing.assert_allclose(network.flow_array(demand), demand.to_numpy(float).T, rtol=0.0, atol=0.0)


def test_example_demand_column_order_follows_a_reordered_endmember_tuple():
    # Breadth-first order puts T2 before T1 here; the columns must follow that, not the alphabet.
    segments = pd.DataFrame(
        {"from": ["Plant", "A", "A"], "to": ["A", "T2", "T1"], "volume": [300.0, 60.0, 40.0]},
        index=["Plant-A", "A-T2", "A-T1"],
    )
    reordered = PipeNetwork(segments=segments, source="Plant")
    assert reordered.endmembers == ("T2", "T1")

    demand = example_demand(tedges=pd.date_range("2025-06-01", periods=25, freq="h"), network=reordered)
    assert list(demand.columns) == ["T2", "T1"]
    # Each column still carries its own profile, so the means did not swap along with the order.
    np.testing.assert_allclose([demand["T2"].mean(), demand["T1"].mean()], [360.0, 240.0], rtol=1e-12)


@pytest.mark.parametrize(
    ("periods", "freq"),
    [(25, "h"), (241, "h"), (97, "15min"), (9, "3h")],
)
def test_example_demand_averages_to_the_documented_mean_over_whole_days(network, periods, freq):
    # Over a whole number of days the cosine sums to zero on any uniform grid, so the bin mean is
    # the documented mean. Floating-point cancellation of the 24-term cosine sum is the only error.
    tedges = pd.date_range("2025-06-01", periods=periods, freq=freq)
    demand = example_demand(tedges=tedges, network=network)
    np.testing.assert_allclose(
        demand.mean().to_numpy(float),
        [_PROFILES[e][0] for e in network.endmembers],
        rtol=1e-12,
    )


def test_example_demand_is_strictly_positive_and_within_its_profile_envelope(network, hourly_tedges):
    demand = example_demand(tedges=hourly_tedges, network=network)
    assert np.all(demand.to_numpy(float) > 0.0)
    for name in network.endmembers:
        mean, amplitude, _ = _PROFILES[name]
        assert amplitude < 1.0  # what makes the profile strictly positive in the first place
        assert demand[name].min() >= mean * (1.0 - amplitude) - 1e-12
        assert demand[name].max() <= mean * (1.0 + amplitude) + 1e-12


def test_example_demand_amplitude_and_peak_hour_match_the_documented_profiles(network, hourly_tedges):
    demand = example_demand(tedges=hourly_tedges, network=network)
    for name in network.endmembers:
        mean, amplitude, peak = _PROFILES[name]
        recovered_mean, recovered_amplitude, recovered_peak = _daily_harmonic(demand[name])
        np.testing.assert_allclose(recovered_mean, mean, rtol=1e-12)
        np.testing.assert_allclose(recovered_amplitude, mean * amplitude, rtol=1e-12)
        np.testing.assert_allclose(recovered_peak, peak, rtol=1e-12)


def test_example_demand_reproduces_the_documented_cosine_exactly(network, hourly_tedges):
    demand = example_demand(tedges=hourly_tedges, network=network)
    hour = _hour_of_day(demand.index)
    for name in network.endmembers:
        mean, amplitude, peak = _PROFILES[name]
        expected = mean * (1.0 + amplitude * np.cos(2.0 * np.pi * (hour - peak) / 24.0))
        np.testing.assert_allclose(demand[name].to_numpy(float), expected, rtol=0.0, atol=0.0)


# ============================================================================
# example_demand: the split is genuinely time-varying
# ============================================================================


def test_example_demand_endmember_shares_swing_over_the_day(network):
    tedges = pd.date_range("2025-06-01", periods=25, freq="h")
    demand = example_demand(tedges=tedges, network=network)
    production = demand.sum(axis=1).to_numpy(float)
    share = demand.to_numpy(float) / production[:, None]

    np.testing.assert_allclose(share.sum(axis=1), 1.0, rtol=1e-14)
    # A proportional pattern would give a flat share. Every endmember moves by far more than the
    # "few percent" bar; the smallest swing here is T3 at ~9.8 percentage points.
    swing = share.max(axis=0) - share.min(axis=0)
    assert np.all(swing > 0.09)
    np.testing.assert_allclose(swing, [0.33196869, 0.39440383, 0.09829846, 0.14455858], rtol=1e-6)


def test_example_demand_segment_flow_fractions_are_not_constant(network):
    tedges = pd.date_range("2025-06-01", periods=25, freq="h")
    demand = example_demand(tedges=tedges, network=network)
    production = network.node_flow(flow=demand, nodes=[network.source])[0]
    fraction = network.segment_flow(flow=demand) / production

    # The trunk main carries the whole production by mass conservation, so its fraction is the one
    # constant in the network; every other segment's fraction moves over the day.
    np.testing.assert_allclose(fraction[0], 1.0, rtol=0.0, atol=1e-15)
    swing = np.ptp(fraction, axis=1)
    assert np.all(swing[1:] > 0.08)
    np.testing.assert_allclose(
        swing[1:], [0.08332109, 0.08332109, 0.33196869, 0.39440383, 0.09829846, 0.14455858], rtol=1e-6
    )

    # Mass conservation: the summed demand is the production past the source at every bin.
    np.testing.assert_allclose(production, demand.sum(axis=1).to_numpy(float), rtol=1e-13)


def test_example_demand_split_beats_the_proportional_model_by_more_than_rounding(network, two_branch):
    # The A-B / A-C sibling pair is where a fixed-fraction model would be wrong: their shares are
    # complementary and both move, so no single pair of constants reproduces them.
    tedges = pd.date_range("2025-06-01", periods=25, freq="h")
    demand = example_demand(tedges=tedges, network=network)
    production = network.node_flow(flow=demand, nodes=[network.source])[0]
    fraction = pd.DataFrame((network.segment_flow(flow=demand) / production).T, columns=network.segments.index)
    np.testing.assert_allclose(fraction["A-B"] + fraction["A-C"], 1.0, rtol=1e-14)
    assert fraction["A-B"].std() > 0.02

    # By contrast, a two-branch network fed by two copies of the same profile does split
    # proportionally, so the fixed-fraction reduction applies exactly there.
    single = example_demand(tedges=tedges, network=two_branch)["T1"].to_numpy(float)
    proportional = np.stack([single, 1.5 * single])
    fixed = two_branch.segment_flow(flow=proportional) / proportional.sum(axis=0)
    np.testing.assert_allclose(fixed, np.array([1.0, 0.4, 0.6])[:, None] * np.ones(len(single)), rtol=1e-14)


# ============================================================================
# example_demand: fallback for networks outside the canned example
# ============================================================================


def _star_network(n_endmember):
    """Build a source feeding ``n_endmember`` leaves named X0, X1, ... -- none of them in _PROFILES."""
    names = [f"X{i}" for i in range(n_endmember)]
    segments = pd.DataFrame(
        {"from": ["S"] * n_endmember, "to": names, "volume": [10.0 * (i + 1) for i in range(n_endmember)]},
        index=[f"S-{name}" for name in names],
    )
    return PipeNetwork(segments=segments, source="S")


@pytest.mark.parametrize("n_endmember", [2, 3, 5, 8])
def test_example_demand_falls_back_to_positional_profiles(n_endmember):
    other = _star_network(n_endmember)
    tedges = pd.date_range("2025-06-01", periods=241, freq="h")
    demand = example_demand(tedges=tedges, network=other)

    assert list(demand.columns) == list(other.endmembers)
    hour = _hour_of_day(demand.index)
    for i, name in enumerate(other.endmembers):
        peak = 6.0 + 24.0 * i / n_endmember
        expected = 200.0 * (1.0 + 0.4 * np.cos(2.0 * np.pi * (hour - peak) / 24.0))
        np.testing.assert_allclose(demand[name].to_numpy(float), expected, rtol=0.0, atol=0.0)
    assert np.all(demand.to_numpy(float) > 0.0)


@pytest.mark.parametrize("n_endmember", [2, 3, 5, 8])
def test_example_demand_fallback_peak_hours_stay_distinct(n_endmember):
    other = _star_network(n_endmember)
    tedges = pd.date_range("2025-06-01", periods=241, freq="h")
    demand = example_demand(tedges=tedges, network=other)

    recovered = np.array([_daily_harmonic(demand[name])[2] for name in other.endmembers])
    expected = 6.0 + 24.0 * np.arange(n_endmember) / n_endmember
    # Peak hour is an angle, so compare it as one: midnight is 0 and 24 alike.
    np.testing.assert_allclose((recovered - expected + 12.0) % 24.0 - 12.0, 0.0, rtol=0.0, atol=1e-9)

    # Evenly spread over the day, so no two endmembers peak together and the split still moves.
    separation = np.abs(recovered[:, None] - recovered[None, :])
    separation = np.minimum(separation, 24.0 - separation)[~np.eye(n_endmember, dtype=bool)]
    np.testing.assert_allclose(separation.min(), 24.0 / n_endmember, rtol=1e-9)

    production = demand.sum(axis=1).to_numpy(float)
    share = demand.to_numpy(float) / production[:, None]
    assert np.all(share.max(axis=0) - share.min(axis=0) > 0.03)


def test_example_demand_mixes_canned_and_fallback_profiles():
    # A network holding one canned endmember and one unknown one gets each from its own source.
    segments = pd.DataFrame(
        {"from": ["S", "S"], "to": ["T3", "Z"], "volume": [10.0, 20.0]},
        index=["S-T3", "S-Z"],
    )
    mixed = PipeNetwork(segments=segments, source="S")
    demand = example_demand(tedges=pd.date_range("2025-06-01", periods=25, freq="h"), network=mixed)

    np.testing.assert_allclose(_daily_harmonic(demand["T3"]), (120.0, 120.0 * 0.35, 12.0), rtol=1e-12)
    np.testing.assert_allclose(_daily_harmonic(demand["Z"]), (200.0, 80.0, 18.0), rtol=1e-12)
