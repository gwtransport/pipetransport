"""Unit tests for :mod:`pipetransport.logremoval`.

Every expectation is a textbook anchor (LR 1 is a 90 % reduction), a hand-evaluated value of
the Rossman wall-reaction formula, a mass balance on blended streams, or the closed-form
residual ``exp(-sum_e k_e V_e / Q_e)`` that the transport operator must reproduce exactly when
the demand is constant.
"""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from pipetransport.logremoval import (
    decay_rate_to_log10_decay_rate,
    fraction_remaining_to_log_removal,
    log10_decay_rate_to_decay_rate,
    log_removal_to_fraction_remaining,
    parallel_mean,
    residence_time_to_log_removal,
    segment_decay_rate,
)
from pipetransport.residence_time import endmember_to_source as residence_time_at_taps
from pipetransport.transport import source_to_endmember

LN10 = 2.302585092994046  # ln(10), correctly rounded to double precision
LOG10_2 = 0.30102999566398120  # log10(2), correctly rounded to double precision


# ============================================================================
# residence_time_to_log_removal
# ============================================================================


def test_residence_time_to_log_removal_is_the_same_exponential_as_the_rate_constant():
    log10_decay_rate = 0.37
    residence_times = np.linspace(0.0, 3.0, 24).reshape(2, 3, 4)

    log_removal = residence_time_to_log_removal(residence_times=residence_times, log10_decay_rate=log10_decay_rate)

    assert log_removal.shape == residence_times.shape
    np.testing.assert_allclose(log_removal, log10_decay_rate * residence_times, rtol=1e-15)
    # 10 ** (-mu t) and exp(-k t) with k = mu ln(10) are the same decay curve.
    np.testing.assert_allclose(
        log_removal_to_fraction_remaining(log_removal),
        np.exp(-log10_decay_rate * LN10 * residence_times),
        rtol=1e-14,
    )
    # One decade of removal takes exactly 1 / mu days.
    np.testing.assert_allclose(
        residence_time_to_log_removal(residence_times=[1.0 / log10_decay_rate], log10_decay_rate=log10_decay_rate),
        [1.0],
        rtol=1e-15,
    )


def test_residence_time_to_log_removal_propagates_unconstrained_bins(network, short_tedges, diurnal_demand):
    demand = diurnal_demand(network, short_tedges)
    log10_decay_rate = 0.4
    # spinup=None leaves the earliest bins unconstrained, so the age array carries NaN.
    age = residence_time_at_taps(
        flow=demand,
        tedges=short_tedges,
        cout_tedges=short_tedges,
        network=network,
        report_nodes=["T1", "T4"],
        spinup=None,
    )

    log_removal = residence_time_to_log_removal(residence_times=age, log10_decay_rate=log10_decay_rate)

    unconstrained = np.isnan(age)
    assert unconstrained.any()
    assert not unconstrained.all()
    assert log_removal.shape == age.shape
    np.testing.assert_array_equal(np.isnan(log_removal), unconstrained)
    np.testing.assert_allclose(log_removal[~unconstrained], log10_decay_rate * age[~unconstrained], rtol=1e-15)
    # T4 sits at the end of the long, thin, low-demand branch: its water is the oldest and
    # therefore the most inactivated.
    both_known = ~unconstrained[0] & ~unconstrained[1]
    assert np.all(log_removal[1][both_known] > log_removal[0][both_known])


# ============================================================================
# Rate conversions
# ============================================================================


def test_rate_conversions_are_exact_inverses_and_carry_the_ln10_factor():
    decay_rates = np.array([0.0, 1e-6, 0.05, 0.5, 2.0, 17.3])
    log10_rates = np.array([decay_rate_to_log10_decay_rate(k) for k in decay_rates])

    np.testing.assert_allclose(log10_rates, decay_rates / LN10, rtol=1e-15)
    np.testing.assert_allclose([log10_decay_rate_to_decay_rate(mu) for mu in log10_rates], decay_rates, rtol=1e-15)
    # And the other way round, starting from mu.
    log10_rates = np.array([0.0, 1e-6, 0.1, 1.0, 4.2])
    decay_rates = np.array([log10_decay_rate_to_decay_rate(mu) for mu in log10_rates])
    np.testing.assert_allclose(decay_rates, log10_rates * LN10, rtol=1e-15)
    np.testing.assert_allclose([decay_rate_to_log10_decay_rate(k) for k in decay_rates], log10_rates, rtol=1e-15)
    # mu = 1 log10/day is by definition k = ln(10) per day.
    np.testing.assert_allclose(log10_decay_rate_to_decay_rate(1.0), LN10, rtol=1e-15)


def test_rate_conversions_agree_on_a_half_life():
    half_life = 3.0  # days
    decay_rate = np.log(2.0) / half_life

    log10_decay_rate = decay_rate_to_log10_decay_rate(decay_rate)

    # Halving in 3 days is log10(2) / 3 decades per day.
    np.testing.assert_allclose(log10_decay_rate, LOG10_2 / half_life, rtol=1e-15)
    # Both currencies leave exactly half the residual after one half-life.
    np.testing.assert_allclose(
        log_removal_to_fraction_remaining(
            residence_time_to_log_removal(residence_times=half_life, log10_decay_rate=log10_decay_rate)
        ),
        0.5,
        rtol=1e-15,
    )


# ============================================================================
# Log removal <-> fraction remaining
# ============================================================================


def test_log_removal_to_fraction_remaining_hits_the_textbook_anchors():
    np.testing.assert_allclose(
        log_removal_to_fraction_remaining([0.0, 1.0, 2.0, 3.0]), [1.0, 0.1, 0.01, 0.001], rtol=1e-15
    )
    np.testing.assert_allclose(
        fraction_remaining_to_log_removal([1.0, 0.1, 0.01, 0.001]), [0.0, 1.0, 2.0, 3.0], atol=1e-15
    )


def test_log_removal_and_fraction_remaining_round_trip():
    fractions = np.geomspace(1e-9, 1.0, 37)
    np.testing.assert_allclose(
        log_removal_to_fraction_remaining(fraction_remaining_to_log_removal(fractions)), fractions, rtol=1e-14
    )

    log_removals = np.linspace(0.0, 9.0, 37).reshape(37, 1)
    round_tripped = fraction_remaining_to_log_removal(log_removal_to_fraction_remaining(log_removals))
    assert round_tripped.shape == log_removals.shape
    np.testing.assert_allclose(round_tripped, log_removals, rtol=1e-14, atol=1e-15)


def test_fraction_remaining_to_log_removal_rejects_a_non_positive_residual():
    for bad in (0.0, -1e-12, -0.5):
        with pytest.raises(ValueError, match="fraction_remaining must be positive"):
            fraction_remaining_to_log_removal([0.5, bad])


# ============================================================================
# parallel_mean
# ============================================================================


def test_parallel_mean_equal_fractions_matches_a_hand_computed_blend():
    source = 5.0  # mg/L leaving the plant
    # Two equal streams at LR 1 and LR 3 deliver 0.5 and 0.005 mg/L; the 50/50 blend carries
    # (0.5 + 0.005) / 2 = 0.2525 mg/L, i.e. 0.0505 of the source, LR = -log10(0.0505).
    blended = parallel_mean(log_removals=[1.0, 3.0])

    np.testing.assert_allclose(blended, 1.2967086218813386, rtol=1e-14)
    mixed = np.mean(source * np.array([0.1, 0.001]))
    np.testing.assert_allclose(mixed, 0.2525, rtol=1e-15)
    np.testing.assert_allclose(blended, -np.log10(mixed / source), rtol=1e-15)


def test_parallel_mean_unequal_fractions_matches_a_hand_computed_blend():
    # 70 % of the flow at LR 3 (1e-3 remaining) and 30 % at LR 1 (1e-1 remaining):
    # 0.7 * 0.001 + 0.3 * 0.1 = 0.0307 remaining, LR = -log10(0.0307).
    blended = parallel_mean(log_removals=np.array([3.0, 1.0]), flow_fractions=np.array([0.7, 0.3]))

    np.testing.assert_allclose(blended, 1.5128616245228135, rtol=1e-14)
    # Same number from a mass balance in m³/day and mg/L rather than in fractions.
    flows = np.array([700.0, 300.0])
    concentrations = 2.0 * np.array([1e-3, 1e-1])
    np.testing.assert_allclose(blended, -np.log10(np.sum(flows * concentrations) / (np.sum(flows) * 2.0)), rtol=1e-14)


def test_parallel_mean_axis_none_flattens_and_axis_one_blends_rows():
    log_removals = np.array([[1.0, 3.0], [2.0, 4.0]])

    flattened = parallel_mean(log_removals=log_removals)

    # mean(1e-1, 1e-3, 1e-2, 1e-4) = 0.0277775 remaining.
    assert np.ndim(flattened) == 0
    np.testing.assert_allclose(flattened, 1.5563459323870947, rtol=1e-14)

    flow_fractions = np.array([[0.7, 0.3], [0.25, 0.75]])
    rows = parallel_mean(log_removals=log_removals, flow_fractions=flow_fractions, axis=1)

    # Row 0: 0.7 * 1e-1 + 0.3 * 1e-3 = 0.0703. Row 1: 0.25 * 1e-2 + 0.75 * 1e-4 = 0.002575.
    assert rows.shape == (2,)
    np.testing.assert_allclose(rows, [1.153044674980176, 2.58922276662279], rtol=1e-14)

    # Mixing is associative: blending the two row blends by their flow shares must equal
    # blending all four streams at once.
    flows = np.array([[70.0, 30.0], [25.0, 75.0]])
    row_shares = flows.sum(axis=1) / flows.sum()
    np.testing.assert_allclose(
        parallel_mean(log_removals=rows, flow_fractions=row_shares),
        parallel_mean(log_removals=log_removals.ravel(), flow_fractions=(flows / flows.sum()).ravel()),
        rtol=1e-14,
    )


def test_parallel_mean_rejects_flow_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match=r"flow_fractions must sum to 1\.0"):
        parallel_mean(log_removals=[1.0, 3.0], flow_fractions=[0.5, 0.4])
    with pytest.raises(ValueError, match=r"flow_fractions must sum to 1\.0"):
        parallel_mean(
            log_removals=np.array([[1.0, 3.0], [2.0, 4.0]]),
            flow_fractions=np.array([[0.7, 0.3], [0.25, 0.70]]),
            axis=1,
        )


def test_parallel_mean_is_bounded_by_the_streams_and_below_the_arithmetic_mean():
    rng = np.random.default_rng(20250729)
    log_removals = rng.uniform(0.2, 6.0, size=(200, 5))
    weights = rng.uniform(0.05, 1.0, size=(200, 5))
    flow_fractions = weights / weights.sum(axis=1, keepdims=True)

    blended = parallel_mean(log_removals=log_removals, flow_fractions=flow_fractions, axis=1)

    assert np.all(blended > log_removals.min(axis=1))
    assert np.all(blended < log_removals.max(axis=1))
    # Concentrations mix, log removals do not: 10 ** (-LR) is strictly convex, so the blend
    # sits strictly below the flow-weighted arithmetic mean of the log removals.
    assert np.all(blended < np.sum(flow_fractions * log_removals, axis=1))
    # Every single stream caps the blend: the mixture cannot be cleaner than what stream j
    # alone contributes, LR <= LR_j - log10(F_j).
    assert np.all(blended <= np.min(log_removals - np.log10(flow_fractions), axis=1))

    equal_split = parallel_mean(log_removals=log_removals, axis=1)
    assert np.all(equal_split < log_removals.mean(axis=1))
    assert np.all(equal_split > log_removals.min(axis=1))


# ============================================================================
# segment_decay_rate
# ============================================================================


def test_segment_decay_rate_without_a_wall_reaction_is_the_bulk_rate(network, single_pipe):
    rates = segment_decay_rate(network=network, bulk_decay_rate=0.42)

    pd.testing.assert_index_equal(rates.index, network.segments.index)
    np.testing.assert_allclose(rates.to_numpy(), np.full(len(network.segments), 0.42), rtol=0.0)
    # No diameter is needed when nothing reacts at the wall, so a volume-only network works.
    volume_only = segment_decay_rate(network=single_pipe, bulk_decay_rate=0.42)
    pd.testing.assert_index_equal(volume_only.index, single_pipe.segments.index)
    np.testing.assert_allclose(volume_only.to_numpy(), [0.42], rtol=0.0)
    # The default is a conservative tracer.
    np.testing.assert_allclose(segment_decay_rate(network=network).to_numpy(), 0.0, atol=0.0)


def test_segment_decay_rate_wall_term_scales_with_four_over_diameter(network):
    bulk_decay_rate, wall_decay_rate = 0.3, 0.02

    rates = segment_decay_rate(network=network, bulk_decay_rate=bulk_decay_rate, wall_decay_rate=wall_decay_rate)

    # k = k_b + 4 k_w / D, hand-evaluated for the example diameters
    # [0.40, 0.30, 0.25, 0.15, 0.20, 0.15, 0.10] m: 4 * 0.02 / D = 0.08 / D.
    expected = np.array([0.5, 0.3 + 4 / 15, 0.62, 0.3 + 8 / 15, 0.7, 0.3 + 8 / 15, 1.1])
    pd.testing.assert_index_equal(rates.index, network.segments.index)
    np.testing.assert_allclose(rates.to_numpy(), expected, rtol=1e-14)
    # The wall contribution is a pure surface-to-volume effect: (k - k_b) * D == 4 k_w.
    np.testing.assert_allclose(
        (rates.to_numpy() - bulk_decay_rate) * network.segments["diameter"].to_numpy(),
        4.0 * wall_decay_rate,
        rtol=1e-14,
    )
    # The 100 mm branch to T4 loses residual four times faster per unit contact time than the
    # 400 mm trunk main it hangs off.
    np.testing.assert_allclose(rates["C-T4"] - bulk_decay_rate, 4.0 * (rates["Plant-A"] - bulk_decay_rate), rtol=1e-14)


def test_segment_decay_rate_mass_transfer_limits_the_wall_term(network):
    bulk_decay_rate, wall_decay_rate = 0.3, 0.02
    no_limit = segment_decay_rate(network=network, bulk_decay_rate=bulk_decay_rate, wall_decay_rate=wall_decay_rate)
    limited = segment_decay_rate(
        network=network,
        bulk_decay_rate=bulk_decay_rate,
        wall_decay_rate=wall_decay_rate,
        mass_transfer_coefficient=0.5,
    )

    assert np.all(limited.to_numpy() < no_limit.to_numpy())
    assert np.all(limited.to_numpy() > bulk_decay_rate)
    pd.testing.assert_index_equal(limited.index, network.segments.index)
    # Reaction and transport resistances add in series: 1 / (k - k_b) = R_h * (1/k_w + 1/k_f).
    hydraulic_radius = network.segments["diameter"].to_numpy() / 4.0
    resistance = hydraulic_radius * (1.0 / wall_decay_rate + 1.0 / 0.5)
    np.testing.assert_allclose(limited.to_numpy() - bulk_decay_rate, 1.0 / resistance, rtol=1e-14)
    # Hand value for the 400 mm trunk: 0.02 * 0.5 / (0.1 * 0.52) = 5 / 26.
    np.testing.assert_allclose(limited["Plant-A"], bulk_decay_rate + 5 / 26, rtol=1e-14)

    # Faster mass transfer means less limitation, converging on the no-limit case.
    faster = [
        segment_decay_rate(
            network=network,
            bulk_decay_rate=bulk_decay_rate,
            wall_decay_rate=wall_decay_rate,
            mass_transfer_coefficient=coefficient,
        ).to_numpy()
        for coefficient in (0.5, 5.0, 50.0, 1e9)
    ]
    for slower, quicker in pairwise(faster):
        assert np.all(quicker > slower)
    assert np.all(faster[-1] < no_limit.to_numpy())
    np.testing.assert_allclose(faster[-1], no_limit.to_numpy(), rtol=1e-9)


def test_segment_decay_rate_requires_a_diameter_for_a_wall_reaction(single_pipe):
    with pytest.raises(ValueError, match="wall reaction needs the segment diameter"):
        segment_decay_rate(network=single_pipe, bulk_decay_rate=0.1, wall_decay_rate=0.02)


def test_segment_decay_rate_rejects_unphysical_parameters(network):
    with pytest.raises(ValueError, match="bulk_decay_rate must be non-negative"):
        segment_decay_rate(network=network, bulk_decay_rate=-1e-12)
    with pytest.raises(ValueError, match="wall_decay_rate must be non-negative"):
        segment_decay_rate(network=network, bulk_decay_rate=0.3, wall_decay_rate=-0.02)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="mass_transfer_coefficient must be positive"):
            segment_decay_rate(network=network, wall_decay_rate=0.02, mass_transfer_coefficient=bad)


def test_segment_decay_rate_drives_transport_to_the_closed_form_residual(network, hourly_tedges, constant_demand):
    demand = constant_demand(network, hourly_tedges, means=[240.0, 360.0, 120.0, 80.0])
    rates = segment_decay_rate(network=network, bulk_decay_rate=0.3, wall_decay_rate=0.02)
    nodes = ["T3", "T4"]

    cout = source_to_endmember(
        cin=np.ones(len(hourly_tedges) - 1),
        flow=demand,
        tedges=hourly_tedges,
        cout_tedges=hourly_tedges,
        network=network,
        report_nodes=nodes,
        decay_rate=rates,
    )

    # Constant demand freezes every segment travel time at V_e / Q_e, so the delivered
    # residual of a constant unit source is exactly exp(-sum_e k_e V_e / Q_e).
    segment_flow = network.segment_flow(flow=demand)[:, 0]
    volume = network.segments["volume"].to_numpy(dtype=float)
    row_of = {name: i for i, name in enumerate(network.segments.index)}
    expected = np.array([
        np.exp(
            -np.sum(
                rates.to_numpy()[[row_of[s] for s in network.paths[node]]]
                * volume[[row_of[s] for s in network.paths[node]]]
                / segment_flow[[row_of[s] for s in network.paths[node]]]
            )
        )
        for node in nodes
    ])
    assert not np.isnan(cout).any()
    # Precision floor ~1e-13: the operator accumulates the surviving fraction cell by cell in
    # the label coordinate, so the row sum carries the round-off of the interpolated cell
    # boundaries rather than reproducing exp(-phi) to the last bit.
    np.testing.assert_allclose(cout, np.broadcast_to(expected[:, None], cout.shape), rtol=1e-12)
    # T3 and T4 share every upstream pipe, so the difference is the last segment alone: the
    # 2.5 km, 100 mm branch to T4 holds its water longer and reacts faster at the wall.
    assert np.all(cout[1] < cout[0])
    np.testing.assert_allclose(expected, [0.6614533737911086, 0.5435327458198652], rtol=1e-12)

    # The Series is aligned by segment name, not by position: reversing it changes nothing.
    reversed_rates = rates.iloc[::-1]
    np.testing.assert_allclose(
        source_to_endmember(
            cin=np.ones(len(hourly_tedges) - 1),
            flow=demand,
            tedges=hourly_tedges,
            cout_tedges=hourly_tedges,
            network=network,
            report_nodes=nodes,
            decay_rate=reversed_rates,
        ),
        cout,
        rtol=0.0,
        atol=0.0,
    )
