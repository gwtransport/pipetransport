"""Tests for :mod:`pipetransport.residence_time`.

Every check is anchored on something outside the function under test: the closed-form
``sum(V_i / f_i) / Q`` travel time of a constant, proportional split, the brute-force
Lagrangian oracle, or the transport operator the same travel times are read off.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from _oracle import OraclePath
from scipy.integrate import quad

from pipetransport.residence_time import full
from pipetransport.transport import source_to_endmember
from pipetransport.utils import tedges_to_days

NETWORK_FIXTURES = ["single_pipe", "two_branch", "network"]


def _path_oracles(network, demand, node, tdays):
    """Brute-force path solvers for the full path to ``node`` and for each of its tails.

    The tail solvers exist only to locate the kinks of the travel-time function: a parcel
    kinks whenever it crosses a flow change inside any pipe it is still travelling through,
    so the arrival times of the tail paths are exactly the extra breakpoints.

    Parameters
    ----------
    network : PipeNetwork
        The network.
    demand : DataFrame or ndarray
        Endmember demand on the ``tdays`` bins.
    node : str
        Reporting node.
    tdays : ndarray
        Bin edges in days.

    Returns
    -------
    list of OraclePath
        Solver for segments ``k:`` of the path, for ``k = 0 ... len(path) - 1``.
    """
    row_of = {name: i for i, name in enumerate(network.segments.index)}
    rows = [row_of[name] for name in network.paths[node]]
    segment_flow = network.segment_flow(flow=demand)[rows]
    segment_volume = network.segments["volume"].to_numpy(dtype=float)[rows]
    node_flow = network.node_flow(flow=demand, nodes=[node])[0]
    return [
        OraclePath(
            tedges_days=tdays,
            segment_flow=segment_flow[k:],
            segment_volume=segment_volume[k:],
            segment_decay=np.zeros(len(rows) - k),
            node_flow=node_flow,
        )
        for k in range(len(rows))
    ]


# ============================================================================
# Constant demand: the closed form
# ============================================================================


@pytest.mark.parametrize("network_name", NETWORK_FIXTURES)
def test_constant_demand_reproduces_closed_form(
    request, network_name, hourly_tedges, constant_demand, analytic_travel_time
):
    network = request.getfixturevalue(network_name)
    demand = constant_demand(network, hourly_tedges)
    age = full(flow=demand, tedges=hourly_tedges, network=network)

    assert age.shape == (len(network.endmembers), len(hourly_tedges) - 1)
    assert np.all(np.isfinite(age)), "the warm start should make every output bin of a steady record valid"
    for i, node in enumerate(network.endmembers):
        expected = analytic_travel_time(network, demand, node)
        np.testing.assert_allclose(age[i], expected, rtol=1e-12)
        # A steady demand gives a steady age; the spread is pure interpolation round-off.
        np.testing.assert_allclose(age[i], age[i, 0], rtol=1e-12)


def test_single_pipe_travel_time_is_volume_over_demand(single_pipe, hourly_tedges):
    # 100 m3 of pipe emptied by 100 m3/day is exactly one day, with no split to complicate it.
    demand = np.full((1, len(hourly_tedges) - 1), 100.0)
    age = full(flow=demand, tedges=hourly_tedges, network=single_pipe)
    np.testing.assert_allclose(age, 1.0, rtol=1e-12)


def test_two_branch_effective_volume_counts_the_trunk_in_full(two_branch, hourly_tedges):
    # Plant-A (300 m3) carries all 300 m3/day, A-T1 (40 m3) a third, A-T2 (60 m3) two thirds.
    demand = np.array([[100.0], [200.0]]) * np.ones(len(hourly_tedges) - 1)
    age = full(flow=demand, tedges=hourly_tedges, network=two_branch)
    np.testing.assert_allclose(age[0], (300.0 + 40.0 * 3.0) / 300.0, rtol=1e-12)
    np.testing.assert_allclose(age[1], (300.0 + 60.0 * 1.5) / 300.0, rtol=1e-12)


def test_shared_trunk_contributes_its_full_volume_to_every_path(network, hourly_tedges, constant_demand):
    """The trunk main enters T1's and T3's effective volume undivided, not split by branch share."""
    demand = constant_demand(network, hourly_tedges)
    production = float(np.sum(demand[:, 0]))
    volume = network.segments["volume"]
    segment_flow = pd.Series(network.segment_flow(flow=demand)[:, 0], index=network.segments.index)
    fraction = segment_flow / production

    age = full(flow=demand, tedges=hourly_tedges, network=network, nodes=["T1", "T3"])
    effective = age[:, 0] * production  # effective volume in units of source throughflow

    branch_t1 = volume["A-B"] / fraction["A-B"] + volume["B-T1"] / fraction["B-T1"]
    branch_t3 = volume["A-C"] / fraction["A-C"] + volume["C-T3"] / fraction["C-T3"]
    trunk_in_t1 = effective[0] - branch_t1
    trunk_in_t3 = effective[1] - branch_t3

    np.testing.assert_allclose(trunk_in_t1, volume["Plant-A"], rtol=1e-11)
    np.testing.assert_allclose(trunk_in_t3, volume["Plant-A"], rtol=1e-11)
    np.testing.assert_allclose(trunk_in_t1, trunk_in_t3, rtol=1e-11)
    # Guard against a vacuous check: sharing the trunk out over the two branches instead of
    # counting it in full would move both numbers by tens of percent.
    assert abs(volume["Plant-A"] * fraction["A-B"] - trunk_in_t1) > 0.5 * volume["Plant-A"]
    assert abs(volume["Plant-A"] * fraction["A-C"] - trunk_in_t3) > 0.2 * volume["Plant-A"]


@pytest.mark.parametrize("network_name", ["two_branch", "network"])
def test_both_directions_give_the_same_constant_under_constant_flow(
    request, network_name, hourly_tedges, constant_demand, analytic_travel_time
):
    network = request.getfixturevalue(network_name)
    demand = constant_demand(network, hourly_tedges)
    backward = full(flow=demand, tedges=hourly_tedges, network=network, direction="endmember_to_source")
    forward = full(flow=demand, tedges=hourly_tedges, network=network, direction="source_to_endmember")

    assert forward.shape == backward.shape
    for i, node in enumerate(network.endmembers):
        expected = analytic_travel_time(network, demand, node)
        # The forward direction cannot answer for production that arrives past the record end.
        finite = np.isfinite(forward[i])
        assert finite.sum() > 0.5 * finite.size
        np.testing.assert_allclose(forward[i][finite], expected, rtol=1e-12)
        np.testing.assert_allclose(backward[i], expected, rtol=1e-12)


def test_retardation_factor_scales_the_travel_time_linearly(network, hourly_tedges, constant_demand):
    # Every segment volume is multiplied by R and every flow is untouched, so under a constant
    # proportional split sum(R * V_i / f_i) / Q is exactly R times the unretarded travel time.
    demand = constant_demand(network, hourly_tedges)
    base = full(flow=demand, tedges=hourly_tedges, network=network)
    for factor in (2.5, 7.0):
        scaled = full(flow=demand, tedges=hourly_tedges, network=network, retardation_factor=factor)
        np.testing.assert_allclose(scaled, factor * base, rtol=1e-12)


# ============================================================================
# Diurnal demand: independent oracle and cross-direction consistency
# ============================================================================


@pytest.mark.parametrize("node", ["T1", "T4"])
def test_diurnal_travel_time_matches_the_brute_force_oracle(network, short_tedges, diurnal_demand, node):
    """Cross-check the bin-averaged age against parcel-by-parcel root solves plus quadrature."""
    demand = diurnal_demand(network, short_tedges)
    tdays = tedges_to_days(short_tedges)
    age = full(flow=demand, tedges=short_tedges, network=network, nodes=[node])[0]

    oracles = _path_oracles(network, demand, node, tdays)
    kinks = np.unique(np.concatenate([[tail.arrival(t)[0] for t in tdays] for tail in oracles]))
    kinks = kinks[np.isfinite(kinks)]

    # The output grid equals the input grid here, so the node throughflow is constant inside an
    # output bin and the label-uniform average the package reports is a plain time average.
    for j in range(40, 52):
        lo, hi = tdays[j], tdays[j + 1]
        bounds = np.concatenate([[lo], kinks[(kinks > lo) & (kinks < hi)], [hi]])
        integral = sum(
            quad(lambda t: t - oracles[0].departure(t), a, b, limit=200)[0] for a, b in itertools.pairwise(bounds)
        )
        np.testing.assert_allclose(age[j], integral / (hi - lo), rtol=1e-11)


def test_diurnal_directions_agree_on_the_matched_arrival_grid(network, short_tedges, diurnal_demand):
    """Reporting the backward direction on the arrival image of ``tedges`` recovers the forward one.

    Output bin ``[A(t_l), A(t_{l+1})]`` holds exactly the parcels that left the source in input
    bin ``l``, so both directions then average the same travel time over the same parcels and
    must agree to round-off -- unlike on a shared clock grid, where they average different
    parcel populations.
    """
    node = "T4"
    demand = diurnal_demand(network, short_tedges)
    tdays = tedges_to_days(short_tedges)
    oracle = _path_oracles(network, demand, node, tdays)[0]

    arrivals = np.array([oracle.arrival(t)[0] for t in tdays])
    arrivals = arrivals[np.isfinite(arrivals)]
    assert len(arrivals) > 50
    assert np.all(np.diff(arrivals) > 0)
    arrival_tedges = short_tedges[0] + pd.to_timedelta(arrivals, unit="D")

    forward = full(flow=demand, tedges=short_tedges, network=network, nodes=[node], direction="source_to_endmember")[0]
    backward = full(
        flow=demand,
        tedges=short_tedges,
        cout_tedges=arrival_tedges,
        network=network,
        nodes=[node],
        direction="endmember_to_source",
    )[0]

    matched = forward[: len(arrivals) - 1]
    assert np.all(np.isfinite(matched))
    assert np.all(np.isfinite(backward))
    np.testing.assert_allclose(matched, backward, rtol=1e-11)


def test_diurnal_directions_bracket_each_other(network, hourly_tedges, diurnal_demand):
    """On a shared clock grid the two directions cannot match bin by bin, but must interleave.

    Each direction reports averages of one travel-time function over its own windows, so the
    record mean of either has to sit strictly inside the range the other sweeps over the same
    parcels. The two record means agree only to ~1e-4 relative -- that residual is the
    different parcel population at the ends of the record, not a numerical floor.
    """
    demand = diurnal_demand(network, hourly_tedges)
    tdays = tedges_to_days(hourly_tedges)
    forward = full(flow=demand, tedges=hourly_tedges, network=network, direction="source_to_endmember")
    backward = full(flow=demand, tedges=hourly_tedges, network=network, direction="endmember_to_source")

    for i, node in enumerate(network.endmembers):
        oracle = _path_oracles(network, demand, node, tdays)[0]
        answered = np.flatnonzero(np.isfinite(forward[i]))
        # Arrival span of the parcels the forward direction answers for.
        first = oracle.arrival(tdays[answered[0]])[0]
        last = oracle.arrival(tdays[answered[-1] + 1])[0]
        window = backward[i][(tdays[:-1] >= first) & (tdays[1:] <= last)]
        window = window[np.isfinite(window)]
        assert window.size > 100

        forward_values = forward[i][answered]
        assert window.min() < forward_values.mean() < window.max()
        assert forward_values.min() < window.mean() < forward_values.max()
        # Not vacuous: the diurnal demand swings the age by at least 5% of its mean, so the
        # bracket is a real interval rather than a repeated constant.
        assert np.ptp(window) > 0.05 * window.mean()
        np.testing.assert_allclose(forward_values.mean(), window.mean(), rtol=1e-3)


def test_age_ordering_under_diurnal_demand(network, hourly_tedges, diurnal_demand):
    demand = diurnal_demand(network, hourly_tedges)
    age = full(flow=demand, tedges=hourly_tedges, network=network, nodes=["T1", "T2", "T3", "T4"])
    assert np.all(np.isfinite(age))
    t1, t2, t3, t4 = age

    # T4 sits behind a 2.5 km, 100 mm branch carrying a tenth of the production: oldest water
    # in every single bin, by a wide margin.
    assert np.all(t4 > np.max(age[:3], axis=0) + 0.01)
    # T2 is the youngest on average. It is not the youngest in every bin -- T1 peaks in the
    # morning and T2 in the evening, so their hourly ages cross -- but it stays below the two
    # long-branch nodes everywhere.
    assert t2.mean() < min(t1.mean(), t3.mean(), t4.mean())
    assert np.all(t2 < t3)
    assert np.all(t2 < t4)

    # A junction carries the water its endmembers will later receive, so it is strictly younger.
    junction = full(flow=demand, tedges=hourly_tedges, network=network, nodes=["B"])[0]
    assert np.all(junction < t1)
    assert np.all(junction < t2)


# ============================================================================
# Consistency with the transport operator
# ============================================================================


def test_constant_demand_decay_equals_exp_minus_rate_times_age(network, hourly_tedges, constant_demand):
    """A steady demand gives every parcel in a bin the same age, so the residual is exp(-k tau)."""
    demand = constant_demand(network, hourly_tedges)
    decay_rate = 0.4
    age = full(flow=demand, tedges=hourly_tedges, network=network)
    residual = source_to_endmember(
        cin=np.ones(len(hourly_tedges) - 1),
        flow=demand,
        tedges=hourly_tedges,
        cout_tedges=hourly_tedges,
        network=network,
        decay_rate=decay_rate,
    )
    assert np.all(np.isfinite(residual))
    np.testing.assert_allclose(residual, np.exp(-decay_rate * age), rtol=1e-12)


def test_diurnal_demand_leaves_a_jensen_gap(network, hourly_tedges, diurnal_demand):
    """With the age varying inside a bin, mean(exp(-k tau)) > exp(-k mean(tau)) by convexity."""
    demand = diurnal_demand(network, hourly_tedges)
    decay_rate = 0.6
    # Six-hourly reporting bins so each average spans a real spread of ages.
    cout_tedges = pd.date_range(hourly_tedges[0], hourly_tedges[-1], freq="6h")
    age = full(flow=demand, tedges=hourly_tedges, cout_tedges=cout_tedges, network=network)
    residual = source_to_endmember(
        cin=np.ones(len(hourly_tedges) - 1),
        flow=demand,
        tedges=hourly_tedges,
        cout_tedges=cout_tedges,
        network=network,
        decay_rate=decay_rate,
    )
    gap = residual - np.exp(-decay_rate * age)
    assert np.all(np.isfinite(gap))
    # Direction of the inequality, with a round-off allowance of a few ulps of exp(-k tau).
    assert gap.min() > -1e-12
    # And it is a real gap, not an artefact: T4's age swings by hours inside a six-hour bin.
    assert gap[3].max() > 1e-4
    # Which also means the age is not a shortcut to the residual.
    assert not np.allclose(residual, np.exp(-decay_rate * age), rtol=1e-9)


# ============================================================================
# API behaviour
# ============================================================================


def test_source_node_has_zero_travel_time(network, hourly_tedges, constant_demand):
    demand = constant_demand(network, hourly_tedges)
    for direction in ("endmember_to_source", "source_to_endmember"):
        age = full(flow=demand, tedges=hourly_tedges, network=network, nodes=["Plant"], direction=direction)
        assert np.all(np.isfinite(age))
        np.testing.assert_allclose(age, 0.0, atol=0.0)


def test_nodes_order_is_honoured(network, hourly_tedges, constant_demand, analytic_travel_time):
    demand = constant_demand(network, hourly_tedges)
    order = ["T3", "B", "T1", "C"]
    age = full(flow=demand, tedges=hourly_tedges, network=network, nodes=order)
    assert age.shape == (len(order), len(hourly_tedges) - 1)
    for i, node in enumerate(order):
        # Same row whether or not the other nodes are asked for. Not bit-identical: the
        # warm-start length is the longest travel time over the requested set, which shifts
        # the interpolation grid by a few ulps.
        alone = full(flow=demand, tedges=hourly_tedges, network=network, nodes=[node])[0]
        answered = np.isfinite(age[i])
        np.testing.assert_array_equal(answered, np.isfinite(alone))
        np.testing.assert_allclose(age[i][answered], alone[answered], rtol=1e-12)
        np.testing.assert_allclose(age[i][answered], analytic_travel_time(network, demand, node), rtol=1e-12)
    # The rows genuinely differ, so a permuted output would be caught.
    assert len({round(float(row[0]), 9) for row in age}) == len(order)


def test_every_node_answers_the_final_output_bin(network, hourly_tedges, constant_demand):
    # The last output bin ends exactly at the record end and is fed only by parcels that left
    # inside the record, so it is constrained at every node.
    demand = constant_demand(network, hourly_tedges)
    age = full(flow=demand, tedges=hourly_tedges, network=network, nodes=list(network.nodes[1:]))
    assert np.all(np.isfinite(age[:, -1]))


def test_invalid_direction_raises(network, hourly_tedges, constant_demand):
    demand = constant_demand(network, hourly_tedges)
    with pytest.raises(ValueError, match="direction must be one of"):
        full(flow=demand, tedges=hourly_tedges, network=network, direction="source_to_source")


def test_spinup_none_leaves_the_first_travel_time_worth_of_bins_nan(
    network, hourly_tedges, constant_demand, analytic_travel_time
):
    """Without a warm start, an output bin is answered only once no parcel in it predates the record."""
    demand = constant_demand(network, hourly_tedges)
    strict = full(flow=demand, tedges=hourly_tedges, network=network, spinup=None)
    warm = full(flow=demand, tedges=hourly_tedges, network=network, spinup="constant")

    for i, node in enumerate(network.endmembers):
        expected = analytic_travel_time(network, demand, node)
        missing = np.isnan(strict[i])
        # Bin j is answerable once j hourly bins cover the travel time.
        n_missing = int(np.ceil(expected * 24.0))
        assert missing.sum() == n_missing
        assert np.all(missing[:n_missing]), "the unanswered bins must be the leading ones"
        np.testing.assert_allclose(strict[i][~missing], expected, rtol=1e-12)
    assert np.all(np.isfinite(warm))
