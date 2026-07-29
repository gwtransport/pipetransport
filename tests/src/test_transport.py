"""
Unit tests for :mod:`pipetransport.transport`.

Every expectation is anchored to something the implementation cannot fake: the
constant-flow-fraction closed form ``tau = sum(V_i / f_i) / Q`` (through the
``analytic_travel_time`` fixture), mass conservation, linearity of the operator, the
brute-force Lagrangian reference of :mod:`_oracle`, or a composition invariant that splits one
source-to-endmember path into two consecutive transports through an internal node.
"""

import numpy as np
import pandas as pd
import pytest
from _oracle import OraclePath

from pipetransport.network import PipeNetwork
from pipetransport.transport import endmember_to_source, source_to_endmember
from pipetransport.utils import tedges_to_days

# The output of an hourly record is compared bin by bin, so travel times are expressed in
# hours throughout; one bin then equals one unit of the ramp below.
_HOURS_PER_DAY = 24.0


def _step_ramp(n_bins, step_bin, tau_hours):
    """Bin averages of a unit step delayed by ``tau_hours`` on unit-hour bins."""
    return np.clip(np.arange(1, n_bins + 1) - (step_bin + tau_hours), 0.0, 1.0)


def _path_rows(network, node):
    """Row positions of the source-to-``node`` path in ``network.segments``."""
    row_of = {name: i for i, name in enumerate(network.segments.index)}
    return [row_of[name] for name in network.paths[node]]


def _oracle_path(*, network, demand, node, tedges, decay=None):
    """Build the brute-force reference for one source-to-node path."""
    rows = _path_rows(network, node)
    rates = np.zeros(len(network.segments)) if decay is None else np.asarray(decay, dtype=float)
    return OraclePath(
        tedges_days=tedges_to_days(tedges),
        segment_flow=network.segment_flow(flow=demand)[rows],
        segment_volume=network.segments["volume"].to_numpy(dtype=float)[rows],
        segment_decay=rates[rows],
        node_flow=network.node_flow(flow=demand, nodes=[node])[0],
    )


def _arrival_days(*, network, demand, node, tedges):
    """Arrival time at ``node`` of the parcel leaving the source at each edge of ``tedges``.

    Solved parcel by parcel with the oracle's root finder, so it is independent of the
    operator under test. NaN marks an edge whose parcel has not arrived when the record ends.
    """
    path = _oracle_path(network=network, demand=demand, node=node, tedges=tedges)
    return np.array([path.arrival(t)[0] for t in tedges_to_days(tedges)])


# ============================================================================
# Mass conservation
# ============================================================================


@pytest.mark.parametrize("demand_kind", ["constant", "diurnal"])
@pytest.mark.parametrize("grid_kind", ["aligned", "coarse_offset"])
def test_constant_source_is_delivered_unchanged(
    network, hourly_tedges, constant_demand, diurnal_demand, demand_kind, grid_kind
):
    """A constant produced quality comes back unchanged: the operator rows sum to exactly 1."""
    demand = (
        constant_demand(network, hourly_tedges) if demand_kind == "constant" else diurnal_demand(network, hourly_tedges)
    )
    cout_tedges = hourly_tedges if grid_kind == "aligned" else pd.date_range("2025-06-01 00:20", periods=41, freq="5h")
    nodes = list(network.nodes[1:])  # every node but the source: junctions report too
    cin = np.full(len(hourly_tedges) - 1, 1.0)

    cout = source_to_endmember(
        cin=cin, flow=demand, tedges=hourly_tedges, cout_tedges=cout_tedges, network=network, nodes=nodes
    )

    assert cout.shape == (len(nodes), len(cout_tedges) - 1)
    valid = ~np.isnan(cout)
    # Node C loses its final aligned bin to a one-ulp overshoot in the arrival round trip (see
    # the notes on _transfer.travel); that is conservative, so allow one node to fall short.
    assert valid.all(axis=1).sum() >= len(nodes) - 1
    np.testing.assert_allclose(cout[valid], 1.0, rtol=0.0, atol=1e-12)


# ============================================================================
# Pure delay against the constant-flow-fraction closed form
# ============================================================================


def test_step_arrives_as_analytic_ramp_at_every_endmember(
    network, hourly_tedges, constant_demand, analytic_travel_time
):
    """Constant demand turns transport into a pure delay: a step becomes a one-bin ramp at ``tau``."""
    demand = constant_demand(network, hourly_tedges)
    n_bins = len(hourly_tedges) - 1
    step_bin = 100
    cin = np.where(np.arange(n_bins) >= step_bin, 1.0, 0.0)

    cout = source_to_endmember(cin=cin, flow=demand, tedges=hourly_tedges, cout_tedges=hourly_tedges, network=network)

    assert not np.isnan(cout).any()
    for i, node in enumerate(network.endmembers):
        tau_hours = analytic_travel_time(network, demand, node) * _HOURS_PER_DAY
        np.testing.assert_allclose(cout[i], _step_ramp(n_bins, step_bin, tau_hours), rtol=0.0, atol=1e-12)


def test_single_pipe_travel_time_is_volume_over_flow(single_pipe, hourly_tedges, constant_demand, analytic_travel_time):
    """One 100 m³ pipe at 250 m³/day delays by exactly 0.4 day; nothing else can be right."""
    demand = constant_demand(single_pipe, hourly_tedges, means=[250.0])
    n_bins = len(hourly_tedges) - 1
    tau_days = 100.0 / 250.0
    assert analytic_travel_time(single_pipe, demand, "T1") == pytest.approx(tau_days, rel=0.0, abs=1e-15)

    step_bin = 72
    cin = np.where(np.arange(n_bins) >= step_bin, 1.0, 0.0)
    cout = source_to_endmember(
        cin=cin, flow=demand, tedges=hourly_tedges, cout_tedges=hourly_tedges, network=single_pipe
    )

    np.testing.assert_allclose(cout[0], _step_ramp(n_bins, step_bin, tau_days * _HOURS_PER_DAY), rtol=0.0, atol=1e-13)


def test_retardation_factor_scales_the_travel_time(two_branch, hourly_tedges, constant_demand, analytic_travel_time):
    """R multiplies every segment volume, so the arrival of a step is delayed by exactly R * tau."""
    demand = constant_demand(two_branch, hourly_tedges)
    n_bins = len(hourly_tedges) - 1
    step_bin = 40
    cin = np.where(np.arange(n_bins) >= step_bin, 1.0, 0.0)
    # Plant-A carries the full production (300 m³/day) and A-T1 a third of it, so the
    # effective volume is 300 + 3 * 40 = 420 m³ and tau is exactly 1.4 day.
    assert analytic_travel_time(two_branch, demand, "T1") == pytest.approx(1.4, rel=0.0, abs=1e-14)

    for retardation in (1.0, 2.0, 2.5):
        cout = source_to_endmember(
            cin=cin,
            flow=demand,
            tedges=hourly_tedges,
            cout_tedges=hourly_tedges,
            network=two_branch,
            retardation_factor=retardation,
        )
        for i, node in enumerate(two_branch.endmembers):
            tau_hours = retardation * analytic_travel_time(two_branch, demand, node) * _HOURS_PER_DAY
            np.testing.assert_allclose(cout[i], _step_ramp(n_bins, step_bin, tau_hours), rtol=0.0, atol=1e-12)


# ============================================================================
# Linearity
# ============================================================================


def test_transport_is_linear_in_the_source_signal(network, hourly_tedges, diurnal_demand):
    """cout(a*c1 + b*c2) == a*cout(c1) + b*cout(c2): the model is one linear operator."""
    demand = diurnal_demand(network, hourly_tedges)
    n_bins = len(hourly_tedges) - 1
    rng = np.random.default_rng(4242)
    c1 = rng.uniform(0.0, 1.0, n_bins)
    c2 = np.sin(np.arange(n_bins) / 9.0)
    a, b = -1.7, 2.3
    shared = {
        "flow": demand,
        "tedges": hourly_tedges,
        "cout_tedges": pd.date_range("2025-06-01 00:20", periods=41, freq="5h"),
        "network": network,
        "decay_rate": 0.35,
        "spinup": None,
    }

    out1 = source_to_endmember(cin=c1, **shared)
    out2 = source_to_endmember(cin=c2, **shared)
    combined = source_to_endmember(cin=a * c1 + b * c2, **shared)

    assert np.array_equal(np.isnan(combined), np.isnan(out1))
    np.testing.assert_allclose(combined, a * out1 + b * out2, rtol=0.0, atol=1e-13)


# ============================================================================
# First-order decay
# ============================================================================


@pytest.mark.parametrize("scalar", [True, False])
def test_decay_residual_matches_the_closed_form(network, hourly_tedges, constant_demand, scalar):
    """With constant flow the surviving fraction is exp(-sum k_e V_e / Q_e), per segment."""
    demand = constant_demand(network, hourly_tedges)
    n_segments = len(network.segments)
    rates = np.full(n_segments, 0.3) if scalar else np.linspace(0.1, 0.9, n_segments)
    decay_rate = 0.3 if scalar else pd.Series(rates, index=network.segments.index)

    cout = source_to_endmember(
        cin=np.ones(len(hourly_tedges) - 1),
        flow=demand,
        tedges=hourly_tedges,
        cout_tedges=hourly_tedges,
        network=network,
        decay_rate=decay_rate,
    )

    volume = network.segments["volume"].to_numpy(dtype=float)
    segment_flow = network.segment_flow(flow=demand)[:, 0]
    assert not np.isnan(cout).any()
    for i, node in enumerate(network.endmembers):
        rows = _path_rows(network, node)
        residual = np.exp(-np.sum(rates[rows] * volume[rows] / segment_flow[rows]))
        assert 0.5 < residual < 1.0  # a meaningful amount of decay, not a no-op
        np.testing.assert_allclose(cout[i], residual, rtol=0.0, atol=1e-12)


def test_zero_decay_reproduces_the_conservative_result(network, hourly_tedges, diurnal_demand):
    """The decay-weighted cell integral must reduce to the plain cell width at k = 0."""
    demand = diurnal_demand(network, hourly_tedges)
    rng = np.random.default_rng(99)
    cin = rng.uniform(0.5, 2.5, len(hourly_tedges) - 1)
    shared = {
        "cin": cin,
        "flow": demand,
        "tedges": hourly_tedges,
        "cout_tedges": hourly_tedges,
        "network": network,
    }

    conservative = source_to_endmember(**shared)
    zero_scalar = source_to_endmember(decay_rate=0.0, **shared)
    zero_series = source_to_endmember(
        decay_rate=pd.Series(np.zeros(len(network.segments)), index=network.segments.index), **shared
    )

    np.testing.assert_allclose(zero_scalar, conservative, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(zero_series, conservative, rtol=0.0, atol=1e-15)


# ============================================================================
# Independent brute-force oracle
# ============================================================================


def _oracle_case(network, tedges):
    """Shared setup of the oracle comparisons: random source signal and a coarse, offset grid."""
    rng = np.random.default_rng(20250729)
    cin = rng.uniform(0.5, 2.5, len(tedges) - 1)
    cout_tedges = pd.date_range("2025-06-01 01:40", periods=17, freq="5h")
    decay = pd.Series(np.linspace(0.05, 0.5, len(network.segments)), index=network.segments.index)
    return cin, cout_tedges, decay


@pytest.mark.parametrize("node", ["T4", "B"])
def test_matches_brute_force_oracle_without_decay(network, short_tedges, diurnal_demand, node):
    """Conservative transport agrees with the parcel-by-parcel reference to machine precision."""
    demand = diurnal_demand(network, short_tedges)
    cin, cout_tedges, _ = _oracle_case(network, short_tedges)

    cout = source_to_endmember(
        cin=cin,
        flow=demand,
        tedges=short_tedges,
        cout_tedges=cout_tedges,
        network=network,
        nodes=[node],
        spinup=None,
    )
    reference = _oracle_path(network=network, demand=demand, node=node, tedges=short_tedges).cout(
        cin=cin, cout_tedges_days=tedges_to_days(cout_tedges, ref=short_tedges[0])
    )

    # spinup=None: the reference leaves the earliest bins unconstrained, and only those.
    assert np.isnan(reference).any()
    assert not np.isnan(reference).all()
    assert np.array_equal(np.isnan(cout[0]), np.isnan(reference))
    np.testing.assert_allclose(cout[0], reference, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("node", ["T4", "B"])
def test_matches_brute_force_oracle_with_per_segment_decay(network, short_tedges, diurnal_demand, node):
    """Per-segment decay under a non-proportional split agrees with the parcel-by-parcel reference."""
    demand = diurnal_demand(network, short_tedges)
    cin, cout_tedges, decay = _oracle_case(network, short_tedges)

    cout = source_to_endmember(
        cin=cin,
        flow=demand,
        tedges=short_tedges,
        cout_tedges=cout_tedges,
        network=network,
        nodes=[node],
        decay_rate=decay,
        spinup=None,
    )
    reference = _oracle_path(
        network=network, demand=demand, node=node, tedges=short_tedges, decay=decay.to_numpy()
    ).cout(cin=cin, cout_tedges_days=tedges_to_days(cout_tedges, ref=short_tedges[0]))

    assert np.array_equal(np.isnan(cout[0]), np.isnan(reference))
    # Precision floor: the oracle integrates exp(-phi) with scipy.integrate.quad at its default
    # epsabs = 1.49e-8 per sub-interval, and an output bin here spans about six of them. Set the
    # decay rates to zero (the test above) and the same comparison closes to 1e-12, which pins
    # the residual on the reference's quadrature rather than on the operator.
    np.testing.assert_allclose(cout[0], reference, rtol=0.0, atol=3e-8)


# ============================================================================
# Composition through an internal node
# ============================================================================


def test_transport_composes_through_an_internal_node(network, hourly_tedges, diurnal_demand):
    """Source -> B followed by B -> T1 reproduces source -> T1 exactly.

    The intermediate grid is the union of the input edges with their arrival times at B, both
    computed independently of the package: the demand is piecewise constant on it (it refines
    ``tedges``) and so is the quality passing B (it jumps only at those arrival times), so the
    hand-off between the two transports loses nothing and the composition must be exact.
    """
    demand = diurnal_demand(network, hourly_tedges)
    n_bins = len(hourly_tedges) - 1
    rng = np.random.default_rng(7)
    # A constant leading stretch makes the warm start of both routes identical.
    cin = np.concatenate([np.full(48, 1.25), rng.uniform(0.5, 2.0, n_bins - 48)])

    edges_days = tedges_to_days(hourly_tedges)
    arrival = _arrival_days(network=network, demand=demand, node="B", tedges=hourly_tedges)
    nanoseconds = np.unique(
        np.round(np.concatenate([edges_days, arrival[np.isfinite(arrival)]]) * 86_400e9).astype("int64")
    )
    mid_tedges = hourly_tedges[0] + pd.to_timedelta(nanoseconds[nanoseconds <= round(edges_days[-1] * 86_400e9)])
    assert len(mid_tedges) > len(hourly_tedges)  # the arrival edges genuinely refine the grid

    quality_at_b = source_to_endmember(
        cin=cin, flow=demand, tedges=hourly_tedges, cout_tedges=mid_tedges, network=network, nodes=["B"]
    )[0]
    assert not np.isnan(quality_at_b).any()

    sub_network = PipeNetwork(segments=network.segments.loc[["B-T1", "B-T2"]], source="B")
    midpoints = 0.5 * (
        tedges_to_days(mid_tedges, ref=hourly_tedges[0])[:-1] + tedges_to_days(mid_tedges, ref=hourly_tedges[0])[1:]
    )
    source_bin = np.clip(np.searchsorted(edges_days, midpoints, side="right") - 1, 0, n_bins - 1)
    sub_demand = network.flow_array(demand)[[network.endmembers.index(e) for e in sub_network.endmembers]][
        :, source_bin
    ]
    # The sub-network's production is exactly the throughflow past B, so every segment below B
    # carries the same flow as it does in the full network.
    np.testing.assert_allclose(
        sub_demand.sum(axis=0), network.node_flow(flow=demand, nodes=["B"])[0][source_bin], rtol=0.0, atol=1e-12
    )

    cout_tedges = pd.date_range(
        hourly_tedges[0] + pd.Timedelta(days=2), hourly_tedges[0] + pd.Timedelta(days=9), freq="2h"
    )
    direct = source_to_endmember(
        cin=cin, flow=demand, tedges=hourly_tedges, cout_tedges=cout_tedges, network=network, nodes=["T1"]
    )[0]
    composed = source_to_endmember(
        cin=quality_at_b, flow=sub_demand, tedges=mid_tedges, cout_tedges=cout_tedges, network=sub_network, nodes=["T1"]
    )[0]

    assert not np.isnan(direct).any()
    assert not np.isnan(composed).any()
    assert direct.std() > 0.1  # the signal actually varies, so agreement is not trivial
    np.testing.assert_allclose(composed, direct, rtol=0.0, atol=1e-12)


# ============================================================================
# Spin-up policy
# ============================================================================


def test_spinup_none_marks_the_leading_bins_nan(network, hourly_tedges, constant_demand, analytic_travel_time):
    """Under constant demand the strict-validity boundary is exactly ceil(tau) hourly bins."""
    demand = constant_demand(network, hourly_tedges)
    rng = np.random.default_rng(11)
    shared = {
        "cin": rng.uniform(0.0, 2.0, len(hourly_tedges) - 1),
        "flow": demand,
        "tedges": hourly_tedges,
        "cout_tedges": hourly_tedges,
        "network": network,
    }

    strict = source_to_endmember(spinup=None, **shared)
    warm = source_to_endmember(spinup="constant", **shared)

    assert not np.isnan(warm).any()
    for i, node in enumerate(network.endmembers):
        tau_hours = analytic_travel_time(network, demand, node) * _HOURS_PER_DAY
        # Output bin j spans source times [j - tau, j + 1 - tau); it is fed entirely by the
        # record once j >= tau, so exactly ceil(tau) leading bins stay unconstrained.
        expected_nan = np.arange(len(hourly_tedges) - 1) < np.ceil(tau_hours)
        assert np.array_equal(np.isnan(strict[i]), expected_nan)
        both = ~np.isnan(strict[i])
        np.testing.assert_allclose(strict[i][both], warm[i][both], rtol=0.0, atol=1e-12)


# ============================================================================
# Reverse direction
# ============================================================================


def _reverse_signal(n_bins):
    """A two-frequency source signal whose recovery is not achievable by smoothing alone."""
    hours = np.arange(n_bins)
    return 2.0 + 0.8 * np.sin(2.0 * np.pi * hours / 48.0) + 0.3 * np.cos(2.0 * np.pi * hours / 17.0)


@pytest.mark.parametrize("decay_kind", ["none", "scalar", "per_segment"])
def test_round_trip_recovers_the_source_signal(network, hourly_tedges, diurnal_demand, decay_kind):
    """Forward transport to two endmembers and back reproduces the produced quality."""
    demand = diurnal_demand(network, hourly_tedges)
    cin = _reverse_signal(len(hourly_tedges) - 1)
    decay_rate = {
        "none": 0.0,
        "scalar": 0.4,
        "per_segment": pd.Series(np.linspace(0.1, 0.8, len(network.segments)), index=network.segments.index),
    }[decay_kind]
    nodes = ["T1", "T4"]
    shared = {
        "flow": demand,
        "tedges": hourly_tedges,
        "cout_tedges": hourly_tedges,
        "network": network,
        "nodes": nodes,
        "decay_rate": decay_rate,
    }

    measured = source_to_endmember(cin=cin, **shared)
    recovered = endmember_to_source(cout=measured, **shared)

    interior = slice(24, -24)
    np.testing.assert_allclose(recovered[interior], cin[interior], rtol=0.0, atol=1e-9)
    # The tail is genuinely unconstrained: that water has not reached either node yet.
    unarrived = np.ones(len(cin), dtype=bool)
    for node in nodes:
        unarrived &= ~np.isfinite(_arrival_days(network=network, demand=demand, node=node, tedges=hourly_tedges)[:-1])
    assert unarrived.any()
    assert np.array_equal(np.isnan(recovered), unarrived)


def test_round_trip_tolerates_a_measurement_outage(network, hourly_tedges, diurnal_demand):
    """A 36 h outage at both sampled endmembers loses exactly the water delivered inside it."""
    demand = diurnal_demand(network, hourly_tedges)
    n_bins = len(hourly_tedges) - 1
    cin = _reverse_signal(n_bins)
    nodes = ["T1", "T4"]
    shared = {
        "flow": demand,
        "tedges": hourly_tedges,
        "cout_tedges": hourly_tedges,
        "network": network,
        "nodes": nodes,
    }

    measured = source_to_endmember(cin=cin, **shared)
    edges_days = tedges_to_days(hourly_tedges)
    outage_lo, outage_hi = 4.0, 5.5  # days since the record start
    blanked = (edges_days[:-1] >= outage_lo) & (edges_days[1:] <= outage_hi)
    assert blanked.sum() == 36
    gapped = np.where(blanked, np.nan, measured)

    recovered = endmember_to_source(cout=gapped, **shared)

    # Classify every source bin from arrival times solved independently of the package.
    unconstrained = np.ones(n_bins, dtype=bool)
    disturbed = np.zeros(n_bins, dtype=bool)
    for node in nodes:
        arrival = _arrival_days(network=network, demand=demand, node=node, tedges=hourly_tedges)
        delivered = np.isfinite(arrival[:-1]) & np.isfinite(arrival[1:])
        inside = delivered & (arrival[:-1] >= outage_lo) & (arrival[1:] <= outage_hi)
        never = ~np.isfinite(arrival[:-1])
        unconstrained &= inside | never
        disturbed |= never | (delivered & (arrival[1:] > outage_lo) & (arrival[:-1] < outage_hi))
    assert unconstrained.sum() > 20
    assert np.array_equal(np.isnan(recovered), unconstrained)

    # A source bin sharing an output row with a blanked one is only partly constrained, so the
    # comparison keeps one bin of margin around the disturbed window.
    safe = ~(disturbed | np.roll(disturbed, 1) | np.roll(disturbed, -1))
    safe[:24] = False
    safe[-24:] = False
    assert safe.sum() > 100
    np.testing.assert_allclose(recovered[safe], cin[safe], rtol=0.0, atol=1e-9)


# ============================================================================
# Input validation
# ============================================================================


def test_source_to_endmember_rejects_invalid_input(network, hourly_tedges, constant_demand):
    """Every documented ValueError of the forward direction fires with its own message."""
    demand = constant_demand(network, hourly_tedges)
    n_bins = len(hourly_tedges) - 1
    shared = {"flow": demand, "tedges": hourly_tedges, "cout_tedges": hourly_tedges, "network": network}

    with pytest.raises(ValueError, match=r"tedges must have one more element than cin"):
        source_to_endmember(cin=np.ones(n_bins - 1), **shared)
    with pytest.raises(ValueError, match=r"cin contains NaN values"):
        source_to_endmember(cin=np.where(np.arange(n_bins) == 5, np.nan, 1.0), **shared)
    with pytest.raises(ValueError, match=r"tedges must be strictly increasing"):
        source_to_endmember(
            cin=np.ones(n_bins),
            flow=demand,
            tedges=hourly_tedges[::-1],
            cout_tedges=hourly_tedges,
            network=network,
        )
    with pytest.raises(ValueError, match=r"unknown node\(s\): \['Reservoir'\]"):
        source_to_endmember(cin=np.ones(n_bins), nodes=["T1", "Reservoir"], **shared)
    with pytest.raises(ValueError, match=r"retardation_factor must be >= 1\.0"):
        source_to_endmember(cin=np.ones(n_bins), retardation_factor=0.5, **shared)
    with pytest.raises(ValueError, match=r"decay_rate must be non-negative"):
        source_to_endmember(cin=np.ones(n_bins), decay_rate=-0.1, **shared)
    with pytest.raises(ValueError, match=r"decay_rate is missing segment\(s\): \['C-T4'\]"):
        source_to_endmember(
            cin=np.ones(n_bins),
            decay_rate=pd.Series(np.full(6, 0.1), index=network.segments.index[:6]),
            **shared,
        )


def test_endmember_to_source_rejects_invalid_input(network, hourly_tedges, constant_demand):
    """Every documented ValueError of the reverse direction fires with its own message."""
    demand = constant_demand(network, hourly_tedges)
    n_bins = len(hourly_tedges) - 1
    shared = {"flow": demand, "tedges": hourly_tedges, "cout_tedges": hourly_tedges, "network": network}
    measured = source_to_endmember(cin=np.ones(n_bins), **shared)

    with pytest.raises(ValueError, match=r"cout must hold one row per reporting node \(3\), got shape \(2, 240\)"):
        endmember_to_source(cout=measured[:2], nodes=["T1", "T2", "T3"], **shared)
    with pytest.raises(ValueError, match=r"cout_tedges must have one more element than cout"):
        endmember_to_source(cout=measured[:, :-1], **shared)
    with pytest.raises(ValueError, match=r"cout is missing node\(s\): \['T2'\]"):
        endmember_to_source(cout={"T1": measured[0]}, nodes=["T1", "T2"], **shared)
    for strength in (0.0, -1e-6):
        with pytest.raises(ValueError, match=r"regularization_strength must be > 0"):
            endmember_to_source(cout=measured, regularization_strength=strength, **shared)
