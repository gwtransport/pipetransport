"""
Tests for :mod:`pipetransport.heat`.

The battery is organised by what it pins: the three soil kernels against independently
written references (Gaussian line-source quadrature, adaptive quadrature of the step
response, a radial finite-difference solve), the affine-bias machinery against the
brute-force oracle and against its own non-negativity invariant, the coupled fixed point
against the analytic steady buried-pipe law, and the public API's contracts.

Two conventions worth stating once. Where the package promises exactness, the tolerance is
machine precision; where a stated approximation is being measured, the test pins the
approximation's *magnitude* rather than hiding it behind a loose tolerance. And references
are written from the physical parameters (``alpha`` diffusing, ``kappa`` scaling), never by
reusing the package's own grouped constants, so that a swap between two parameters which
share a unit cannot pass unnoticed.
"""

import numpy as np
import pandas as pd
import pytest
from _oracle import OraclePath
from scipy.integrate import quad
from scipy.linalg import solve_banded
from scipy.special import erfc, erfcx, exp1

from pipetransport import heat, transport
from pipetransport._transfer import apply_segment_targets, paths_transfer
from pipetransport.network import PipeNetwork
from pipetransport.utils import tedges_to_days

GRASS = {"alpha": 0.05, "kappa": 0.025, "eta": 0.41}
PAVED = {"alpha": 0.075, "kappa": 0.035, "eta": 0.41}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def soil():
    """Two cover classes: grass and paved."""
    return pd.DataFrame([GRASS, PAVED], index=["grass", "paved"])


@pytest.fixture
def heat_pipe():
    """One 100 mm, 1 km pipe under grass: every quantity has a closed form."""
    segments = pd.DataFrame(
        {"from": ["Plant"], "to": ["T1"], "length": [1000.0], "diameter": [0.1], "cover": ["grass"]},
        index=["Plant-T1"],
    )
    return PipeNetwork(segments=segments, source="Plant")


@pytest.fixture
def heat_network(network):
    """The example network with land cover assigned and a mix of depths."""
    network.segments["cover"] = ["grass", "grass", "paved", "paved", "grass", "paved", "grass"]
    network.segments["depth"] = [1.2, 1.0, 1.0, 0.9, 1.0, 1.0, 0.8]
    return network


@pytest.fixture
def surface():
    """Build a per-cover sol-air DataFrame on a time grid."""

    def _make(tedges, *, grass=20.0, paved=28.0, amplitude=0.0):
        n = len(tedges) - 1
        hours = np.arange(n)
        wave = amplitude * np.sin(2.0 * np.pi * hours / 24.0)
        return pd.DataFrame({"grass": grass + wave, "paved": paved + wave})

    return _make


def _rate(network, **kwargs):
    """Steady exchange rate of every segment, with the fixture soil parameters."""
    return heat.segment_heat_rate(network=network, kappa=GRASS["kappa"], eta=GRASS["eta"], **kwargs)


# ============================================================================
# Kernels: the undisturbed soil field
# ============================================================================


def _robin_step(lag, *, depth, alpha, kappa, eta):
    """Textbook Robin step response, written from the physical parameters."""
    u = depth / (2.0 * np.sqrt(alpha * lag))
    h_over_k = eta / kappa
    return erfc(u) - np.exp(-(u**2)) * erfcx(u + h_over_k * np.sqrt(alpha * lag))


def test_step_response_integral_matches_quadrature_of_the_step_response():
    """The closed-form bin integral is the integral of the Robin step response.

    The reference is adaptive quadrature of the step response written from ``alpha``,
    ``kappa`` and ``eta`` separately, so grouping them differently cannot agree by
    construction.
    """
    for cover in (GRASS, PAVED):
        radiation_length = cover["kappa"] / cover["eta"]
        for lag in (0.05, 0.5, 5.0, 50.0, 500.0):
            closed = heat._step_response_integral(
                np.array([lag]), depth=1.0, alpha=cover["alpha"], radiation_length=radiation_length
            )[0]
            reference, _ = quad(
                lambda s, c=cover: _robin_step(s, depth=1.0, alpha=c["alpha"], kappa=c["kappa"], eta=c["eta"]),
                0.0,
                lag,
                limit=400,
            )
            # Below ~1e-30 day the response has underflowed; quadrature has no relative
            # accuracy left there and neither quantity is physically distinguishable from 0.
            np.testing.assert_allclose(closed, reference, rtol=5e-12, atol=1e-30, err_msg=f"lag={lag}")


def test_step_response_integral_reduces_to_the_dirichlet_kernel():
    """A vanishing radiation length is the prescribed-temperature surface, with no branch."""
    lag = np.array([0.1, 1.0, 10.0, 100.0])
    alpha, depth = 0.05, 1.0
    u = depth / (2.0 * np.sqrt(alpha * lag))
    dirichlet = (lag + depth**2 / (2.0 * alpha)) * erfc(u) - depth / np.sqrt(np.pi * alpha) * np.sqrt(lag) * np.exp(
        -(u**2)
    )
    np.testing.assert_allclose(
        heat._step_response_integral(lag, depth=depth, alpha=alpha, radiation_length=0.0), dirichlet, rtol=1e-14
    )


def test_step_response_integral_is_zero_and_finite_at_the_guards(recwarn):
    """Non-positive lags return exactly zero, and no lag emits a warning."""
    lag = np.array([-5.0, -1e-30, 0.0, 1e-300, 1e-8, 1.0])
    for radiation_length in (0.0, 0.061):
        out = heat._step_response_integral(lag, depth=1.0, alpha=0.05, radiation_length=radiation_length)
        np.testing.assert_array_equal(out[:3], 0.0)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)
    assert not recwarn.list


def test_soil_temperature_superposes_the_kernel_exactly():
    """A piecewise-constant surface series maps to depth by plain kernel superposition."""
    rng = np.random.default_rng(11)
    n = 60
    tedges = pd.date_range("2025-01-01", periods=n + 1, freq="D")
    series = 12.0 + np.cumsum(rng.normal(0.0, 1.5, n))
    t_pre = 7.0

    actual = heat.soil_temperature(
        surface_temperature=series,
        tedges=tedges,
        depth=1.0,
        t_pre=t_pre,
        **{k: GRASS[k] for k in ("alpha", "kappa")},
        eta=GRASS["eta"],
    )

    # Hand-assembled superposition: each surface step contributes its bin-averaged response.
    days = tedges_to_days(tedges)
    steps = np.diff(series, prepend=t_pre)
    radiation_length = GRASS["kappa"] / GRASS["eta"]
    integrals = heat._step_response_integral(
        days[:, None] - days[None, :-1], depth=1.0, alpha=GRASS["alpha"], radiation_length=radiation_length
    )
    expected = t_pre + np.diff(integrals @ steps) / np.diff(days)
    np.testing.assert_allclose(actual, expected, rtol=1e-13)


def _crank_nicolson(surface, *, n_days, alpha, kappa, eta, depth, dz, sub):
    """Soil temperature at ``depth``, bin-averaged per day, by a Crank-Nicolson solve.

    An arbiter written from the physical parameters: ``alpha`` diffuses, and the surface
    exchanges through the Robin condition set by ``eta / kappa``. Insulated far below.
    """
    n_z = int(8.0 / dz)
    dt = 1.0 / sub
    temperature = np.full(n_z, 10.0)
    mu = alpha * dt / dz**2
    robin = eta / kappa * dz
    main = np.full(n_z, 1.0 + mu)
    main[0] = 1.0 + 0.5 * mu * (1.0 + robin)
    main[-1] = 1.0 + 0.5 * mu
    ab = np.zeros((3, n_z))
    ab[0, 1:], ab[1], ab[2, :-1] = np.full(n_z - 1, -0.5 * mu), main, np.full(n_z - 1, -0.5 * mu)

    depth_index = round(depth / dz) - 1
    samples = []
    for step in range(n_days * sub):
        forcing = surface[int(step * dt)]
        rhs = temperature.copy()
        rhs[1:-1] += 0.5 * mu * (temperature[:-2] - 2.0 * temperature[1:-1] + temperature[2:])
        rhs[0] += 0.5 * mu * (temperature[1] - (1.0 + robin) * temperature[0] + 2.0 * robin * forcing)
        rhs[-1] += 0.5 * mu * (temperature[-2] - temperature[-1])
        temperature = solve_banded((1, 1), ab, rhs)
        samples.append(temperature[depth_index])
    return np.asarray(samples).reshape(n_days, sub).mean(axis=1)


def test_soil_temperature_matches_a_finite_difference_solve():
    """An independent Crank-Nicolson solve of the heat equation reproduces the field at depth.

    The arbiter is a PDE solve with the physical Robin boundary condition, not a rearranged
    kernel, so it constrains the diffusivity and the surface coupling separately.
    """
    alpha, kappa, eta, depth = GRASS["alpha"], GRASS["kappa"], GRASS["eta"], 1.0
    n_days = 40
    tedges = pd.date_range("2025-01-01", periods=n_days + 1, freq="D")
    surface = np.where(np.arange(n_days) < 7, 25.0, 10.0)

    package = heat.soil_temperature(
        surface_temperature=surface, tedges=tedges, depth=depth, alpha=alpha, kappa=kappa, eta=eta, t_pre=10.0
    )
    settings = dict(n_days=n_days, alpha=alpha, kappa=kappa, eta=eta, depth=depth)
    coarse = _crank_nicolson(surface, **settings, dz=0.01, sub=96)
    fine = _crank_nicolson(surface, **settings, dz=0.005, sub=384)

    # The gap is the scheme's own error, not the kernel's: refining the grid shrinks it, so
    # the solve is converging onto the closed form rather than merely sitting near it.
    assert np.abs(package - fine).max() < 0.6 * np.abs(package - coarse).max()
    np.testing.assert_allclose(package, fine, atol=5e-2)


@pytest.mark.parametrize(
    ("cover", "peak_fraction", "peak_day", "step_year", "weekly_amplitude"),
    [("grass", 0.246, 8.90, 0.861, 0.0419), ("paved", 0.318, 8.10, 0.883, 0.0707)],
)
def test_soil_temperature_reproduces_the_published_attenuation(
    cover, peak_fraction, peak_day, step_year, weekly_amplitude
):
    """Pinned attenuation of a heatwave, a step and a weekly wave at 1 m, per cover class.

    These are the numbers the module documents, and they depend on the diffusivity and the
    surface coupling jointly: they move by several percentage points between the Robin and
    the prescribed-temperature surface, and by more between the two cover classes.
    """
    parameters = GRASS if cover == "grass" else PAVED

    hourly = pd.date_range("2025-01-01", periods=60 * 24 + 1, freq="h")
    pulse = np.zeros(60 * 24)
    pulse[: 7 * 24] = 1.0
    response = heat.soil_temperature(surface_temperature=pulse, tedges=hourly, depth=1.0, t_pre=0.0, **parameters)
    assert response.max() == pytest.approx(peak_fraction, abs=5e-4)
    assert (np.argmax(response) + 0.5) / 24.0 == pytest.approx(peak_day, abs=0.02)

    daily = pd.date_range("2025-01-01", periods=366, freq="D")
    step = heat.soil_temperature(surface_temperature=np.ones(365), tedges=daily, depth=1.0, t_pre=0.0, **parameters)
    assert step[-1] == pytest.approx(step_year, abs=5e-4)

    long_hourly = pd.date_range("2025-01-01", periods=140 * 24 + 1, freq="h")
    days = (np.arange(140 * 24) + 0.5) / 24.0
    wave = heat.soil_temperature(
        surface_temperature=np.sin(2.0 * np.pi * days / 7.0), tedges=long_hourly, depth=1.0, t_pre=0.0, **parameters
    )
    tail = wave[-28 * 24 :]
    assert 0.5 * (tail.max() - tail.min()) == pytest.approx(weekly_amplitude, abs=5e-4)


def test_soil_temperature_holds_the_pre_history_before_the_surface_record():
    """Output bins preceding the surface record are exactly the pre-history temperature."""
    surface_tedges = pd.date_range("2025-02-01", periods=31, freq="D")
    tedges = pd.date_range("2025-01-01", periods=62, freq="D")
    out = heat.soil_temperature(
        surface_temperature=np.full(30, 25.0),
        tedges=tedges,
        surface_tedges=surface_tedges,
        depth=1.0,
        t_pre=9.0,
        **GRASS,
    )
    np.testing.assert_array_equal(out[:31], 9.0)
    assert out[-1] > 9.0


@pytest.mark.parametrize(
    "surface_freq",
    ["12h", "D"],
    ids=["finer than the output grid", "coarser than the output grid"],
)
def test_soil_temperature_rejects_a_surface_grid_of_a_different_width(surface_freq):
    """The superposition is a convolution only while both grids share one bin width.

    The lag of an (output edge, surface step) pair is then a function of their index
    difference alone, which is what turns a quadratic matrix product into a transform. A
    surface record on its own spacing has to be resampled by the caller, who knows whether
    interpolating or bin-averaging it is the right thing to do.
    """
    tedges = pd.date_range("2025-01-01", periods=25, freq="h")
    surface_tedges = pd.date_range("2025-01-01", periods=5, freq=surface_freq)
    with pytest.raises(ValueError, match="uniform bin width"):
        heat.soil_temperature(
            surface_temperature=np.full(len(surface_tedges) - 1, 20.0),
            tedges=tedges,
            surface_tedges=surface_tedges,
            depth=1.0,
            **GRASS,
        )


# ============================================================================
# Kernels: the halo
# ============================================================================


def _halo_response_by_quadrature(lag, *, r_outer, d_eff, alpha, kappa):
    """Wall-temperature step response by quadrature of the instantaneous Gaussian kernel.

    ``alpha`` sets the diffusion and ``kappa`` the amplitude, written separately: the
    package's ``E1`` form groups them into ``c = r**2/(4 alpha)`` and a ``1/(4 pi kappa)``
    prefactor, so a swap between two parameters that share a unit shows up here.
    """

    def integrand(s):
        return (np.exp(-(r_outer**2) / (4.0 * alpha * s)) - np.exp(-((2.0 * d_eff) ** 2) / (4.0 * alpha * s))) / (
            4.0 * np.pi * kappa * s
        )

    value, _ = quad(integrand, 0.0, lag, limit=400)
    return value


def test_deficit_kernel_matches_the_gaussian_line_source():
    """The bin-averaged deficit is the saturation gap of the line-source-plus-image response."""
    r_outer, d_eff, dt = 0.05, 1.0605, 1.0
    alpha, kappa = GRASS["alpha"], GRASS["kappa"]
    dbar = heat._deficit_kernel(
        6, dt, r_o=np.array([r_outer]), d_eff=np.array([d_eff]), alpha=np.array([alpha]), kappa=np.array([kappa])
    )[0]
    r_inf = np.log(2.0 * d_eff / r_outer) / (2.0 * np.pi * kappa)

    for m in range(6):
        averaged, _ = quad(
            lambda s: _halo_response_by_quadrature(s, r_outer=r_outer, d_eff=d_eff, alpha=alpha, kappa=kappa),
            m * dt,
            (m + 1) * dt,
            limit=200,
        )
        np.testing.assert_allclose(dbar[m], r_inf - averaged / dt, rtol=1e-9, err_msg=f"lag bin {m}")


def test_deficit_kernel_saturates_at_the_steady_buried_pipe_resistance():
    """The deficit vanishes at long lag: the halo stops growing because of the mirror image."""
    r_outer, d_eff = 0.05, 1.0605
    dbar = heat._deficit_kernel(
        2, 4000.0, r_o=np.array([r_outer]), d_eff=np.array([d_eff]), alpha=np.array([0.05]), kappa=np.array([0.025])
    )[0]
    r_inf = np.log(2.0 * d_eff / r_outer) / (2.0 * np.pi * 0.025)
    assert dbar[-1] / r_inf < 1e-3


def test_deficit_kernel_tail_follows_the_physical_law():
    """The deficit decays as ``d_eff**2 / (4 pi kappa alpha tau)``, from physical parameters.

    Reading the constant off the tail constrains the diffusivity and the conductivity
    separately -- unlike the saturation limit, whose argument ratio is diffusivity-free.
    """
    r_outer, d_eff, alpha, kappa = 0.05, 1.0605, 0.05, 0.025
    dt = 200.0
    dbar = heat._deficit_kernel(
        60, dt, r_o=np.array([r_outer]), d_eff=np.array([d_eff]), alpha=np.array([alpha]), kappa=np.array([kappa])
    )[0]
    lag = dt * (np.arange(60) + 0.5)
    predicted = ((2.0 * d_eff) ** 2 - r_outer**2) / (16.0 * np.pi * kappa * alpha)
    np.testing.assert_allclose(dbar[-1] * lag[-1], predicted, rtol=2e-3)


def test_deficit_kernel_is_finite_at_zero_lag(recwarn):
    """The first lag bin is the largest deficit, and evaluating it emits no warning."""
    dbar = heat._deficit_kernel(
        3, 1.0 / 24.0, r_o=np.array([0.05]), d_eff=np.array([1.0605]), alpha=np.array([0.05]), kappa=np.array([0.025])
    )[0]
    assert np.all(np.isfinite(dbar))
    assert np.all(np.diff(dbar) < 0.0)
    assert not recwarn.list


# ============================================================================
# The exchange rate
# ============================================================================


def test_segment_heat_rate_reproduces_the_documented_values():
    """Pinned rates: a 100 mm service line equilibrates ten times faster than a 400 mm main."""
    segments = pd.DataFrame(
        {
            "from": ["P", "P"],
            "to": ["A", "B"],
            "length": [1000.0, 1000.0],
            "diameter": [0.1, 0.4],
            "wall_thickness": [0.0065, 0.0235],
        },
        index=["service", "trunk"],
    )
    network = PipeNetwork(segments=segments, source="P")

    bare = _rate(network)
    assert bare["service"] == pytest.approx(5.3361, abs=5e-4)
    assert bare["trunk"] == pytest.approx(0.5293, abs=5e-4)

    # The PE wall adds about a tenth of the soil resistance, so the rate drops by ~6 %.
    walled = _rate(network, kappa_pipe=0.008)
    assert walled["service"] / bare["service"] == pytest.approx(0.935, abs=2e-3)

    # Fully developed laminar flow in the 100 mm pipe: the film is 29 % of the soil term.
    film = _rate(network, film_coefficient=0.4539)
    d_eff = 1.0 + GRASS["kappa"] / GRASS["eta"]
    r_soil = np.log(2.0 * d_eff / 0.05) / (2.0 * np.pi * GRASS["kappa"])
    r_film = 1.0 / (2.0 * np.pi * 0.05 * 0.4539)
    assert r_film / r_soil == pytest.approx(0.294, abs=2e-3)
    np.testing.assert_allclose(film["service"], 1.0 / ((r_film + r_soil) * np.pi * 0.05**2), rtol=1e-12)


def test_segment_heat_rate_approaches_the_exact_buried_cylinder_resistance():
    """The line-source-plus-image soil resistance is the exact cylinder one to ``O((r/d)**2)``.

    A cylinder of radius ``r`` whose axis lies at depth ``d`` below an isothermal plane has the
    exact steady shape-factor resistance ``acosh(d/r) / (2 pi kappa)``. The package uses
    ``ln(2 d / r) / (2 pi kappa)``, which is its large-``d/r`` limit, so the dimensionless gap
    is ``1 / (4 (d/r)**2)``. Pinning the gap against that expression -- rather than merely
    observing that it is small at one geometry -- is what distinguishes the intended
    approximation from an algebra slip, and burying the pipe deeper must quarter it.
    """
    kappa = GRASS["kappa"]
    segments = pd.DataFrame(
        {"from": ["P", "P"], "to": ["A", "B"], "length": [1000.0, 1000.0], "diameter": [0.1, 0.4]},
        index=["service", "trunk"],
    )
    network = PipeNetwork(segments=segments, source="P")
    radius = segments["diameter"].to_numpy() / 2.0

    gaps = {}
    for depth in (1.0, 2.0, 4.0):
        # eta=None keeps d_eff = depth, so the geometry is exactly the textbook one.
        rate = heat.segment_heat_rate(network=network, kappa=kappa, depth=depth).to_numpy()
        soil_resistance = 1.0 / (rate * np.pi * radius**2)
        gap = 2.0 * np.pi * kappa * soil_resistance - np.arccosh(depth / radius)
        # The next term of the expansion is 3/(32 z**4), which is why the trunk main -- at
        # z = 5 rather than 20 -- needs the looser tolerance.
        np.testing.assert_allclose(gap, 1.0 / (4.0 * (depth / radius) ** 2), rtol=0.02, err_msg=f"depth {depth}")
        gaps[depth] = gap

    np.testing.assert_allclose(gaps[1.0] / gaps[2.0], 4.0, rtol=0.02)
    np.testing.assert_allclose(gaps[2.0] / gaps[4.0], 4.0, rtol=0.02)


def test_segment_heat_rate_resistances_add_in_series(heat_pipe):
    """Film, wall and soil are one series sum: adding a term can only lower the rate."""
    heat_pipe.segments["wall_thickness"] = 0.0065
    bare = _rate(heat_pipe)["Plant-T1"]
    walled = _rate(heat_pipe, kappa_pipe=0.008)["Plant-T1"]
    both = _rate(heat_pipe, kappa_pipe=0.008, film_coefficient=0.4539)["Plant-T1"]
    assert both < walled < bare
    resistance = lambda rate: 1.0 / (rate * np.pi * 0.05**2)  # noqa: E731
    np.testing.assert_allclose(resistance(both) - resistance(walled), 1.0 / (2.0 * np.pi * 0.05 * 0.4539), rtol=1e-12)


def test_segment_heat_rate_accepts_per_segment_series(heat_network):
    """Per-segment parameters are honoured, and a scalar is the uniform case of them."""
    per_segment = pd.Series(GRASS["kappa"], index=heat_network.segments.index)
    per_segment.iloc[0] = 0.04
    rates = heat.segment_heat_rate(
        network=heat_network, kappa=per_segment, depth=heat_network.segments["depth"], eta=GRASS["eta"]
    )
    uniform = heat.segment_heat_rate(
        network=heat_network, kappa=GRASS["kappa"], depth=heat_network.segments["depth"], eta=GRASS["eta"]
    )
    assert rates.iloc[0] != uniform.iloc[0]
    np.testing.assert_allclose(rates.iloc[1:], uniform.iloc[1:], rtol=1e-14)


# ============================================================================
# The affine bias operator
# ============================================================================


def _build_operator(network, demand, tedges, rates, *, nodes=None, with_target_terms=True):
    """Build one operator directly, bypassing the spin-up policy."""
    requested = list(network.endmembers if nodes is None else nodes)
    paths = [network.segments.index.get_indexer(list(network.paths[node])) for node in requested]
    lengths = np.array([path.size for path in paths], dtype=np.intp)
    max_depth = int(lengths.max(initial=0))
    active = np.arange(max_depth) < lengths[:, None]
    paths_idx = np.zeros((len(requested), max_depth), dtype=np.intp)
    paths_idx[active] = np.concatenate(paths)
    days = tedges_to_days(tedges)
    return paths_transfer(
        tedges_days=days,
        cout_tedges_days=days,
        segment_volume=network.segments["volume"].to_numpy(dtype=float),
        segment_flow=network.segment_flow(flow=demand),
        segment_decay=np.asarray(rates, dtype=float),
        node_flow=network.node_flow(flow=demand, nodes=requested),
        paths_idx=paths_idx,
        active=active,
        with_target_terms=with_target_terms,
    )


def test_target_terms_leave_the_transport_operator_bit_identical(heat_network, hourly_tedges, diurnal_demand):
    """Requesting the bias factors changes nothing about ``W`` itself."""
    demand = heat_network.flow_array(diurnal_demand(heat_network, hourly_tedges))
    rates = _rate(heat_network, depth=heat_network.segments["depth"]).to_numpy()
    plain = _build_operator(heat_network, demand, hourly_tedges, rates, with_target_terms=False)
    with_terms = _build_operator(heat_network, demand, hourly_tedges, rates)

    np.testing.assert_array_equal(plain.band_vals, with_terms.band_vals)
    np.testing.assert_array_equal(plain.col_start, with_terms.col_start)
    np.testing.assert_array_equal(plain.valid_out, with_terms.valid_out)
    assert plain.target_terms is None


def test_bias_weights_are_non_negative_and_complete_the_row_sum(heat_network, short_tedges, diurnal_demand):
    """Every target bin enters with a non-negative weight, and the weights close the budget.

    Applying the operator to the unit-impulse basis exposes the weight of each (segment,
    bin) pair. A negative weight means an index convention that double-counts a bin edge --
    the failure mode of this construction -- and the row sums must complete ``W``'s to one,
    which is what makes a spatially uniform temperature a fixed point.
    """
    demand = heat_network.flow_array(diurnal_demand(heat_network, short_tedges))
    rates = _rate(heat_network, depth=heat_network.segments["depth"]).to_numpy()
    transfer = _build_operator(heat_network, demand, short_tedges, rates)
    n_seg, n_bins = len(heat_network.segments), len(short_tedges) - 1

    total = np.zeros_like(transfer.band_vals[..., 0])
    for segment in range(n_seg):
        for bin_index in range(n_bins):
            impulse = np.zeros((n_seg, n_bins))
            impulse[segment, bin_index] = 1.0
            weights = apply_segment_targets(transfer, impulse)
            assert weights.min() >= -1e-15, f"negative weight at segment {segment}, bin {bin_index}"
            total += weights

    row_sum = transfer.band_vals.sum(axis=2)
    np.testing.assert_allclose(total[transfer.valid_out], (1.0 - row_sum)[transfer.valid_out], atol=1e-12)


def test_constant_target_telescopes_to_the_surviving_fraction(heat_network, hourly_tedges, diurnal_demand):
    """A spatially and temporally constant target is delivered as ``c (1 - W row sum)``."""
    demand = heat_network.flow_array(diurnal_demand(heat_network, hourly_tedges))
    rates = _rate(heat_network, depth=heat_network.segments["depth"]).to_numpy()
    transfer = _build_operator(heat_network, demand, hourly_tedges, rates)

    constant = 17.3
    bias = apply_segment_targets(transfer, np.full((len(heat_network.segments), len(hourly_tedges) - 1), constant))
    expected = constant * (1.0 - transfer.band_vals.sum(axis=2))
    np.testing.assert_allclose(bias[transfer.valid_out], expected[transfer.valid_out], atol=1e-11)


def test_zero_rates_give_zero_bias_and_conservative_transport(heat_network, hourly_tedges, diurnal_demand):
    """With no exchange the bias vanishes and the operator is the conservative one."""
    demand = heat_network.flow_array(diurnal_demand(heat_network, hourly_tedges))
    n_seg, n_bins = len(heat_network.segments), len(hourly_tedges) - 1
    transfer = _build_operator(heat_network, demand, hourly_tedges, np.zeros(n_seg))

    rng = np.random.default_rng(3)
    targets = rng.uniform(5.0, 25.0, size=(n_seg, n_bins))
    bias = apply_segment_targets(transfer, targets)
    # Exact in real arithmetic; in floating point the telescoping of computed differences
    # leaves a residue of order eps times the target scale.
    assert np.abs(bias).max() < 1e-13 * np.abs(targets).max()
    np.testing.assert_allclose(transfer.band_vals.sum(axis=2)[transfer.valid_out], 1.0, rtol=1e-12)


@pytest.mark.parametrize("node", ["T1", "T4", "B"])
def test_bias_matches_the_brute_force_oracle(heat_network, short_tedges, diurnal_demand, node):
    """Per-segment time-varying targets, non-proportional demand, against the parcel oracle.

    The oracle integrates each parcel sequentially, one (segment, bin) piece at a time, and
    splits every output bin's label interval at the crossings of *every* segment face with
    a bin edge -- so the comparison exercises the depth indexing, the half-open edge
    convention and the cell averaging together.
    """
    demand = heat_network.flow_array(diurnal_demand(heat_network, short_tedges))
    rng = np.random.default_rng(17)
    n_seg, n_bins = len(heat_network.segments), len(short_tedges) - 1
    rates = _rate(heat_network, depth=heat_network.segments["depth"]).to_numpy()
    targets = rng.uniform(8.0, 24.0, size=(n_seg, n_bins))
    tin = rng.uniform(6.0, 14.0, size=n_bins)

    transfer = _build_operator(heat_network, demand, short_tedges, rates, nodes=[node])
    columns = np.clip(transfer.col_start[..., None] + np.arange(transfer.band_vals.shape[-1]), 0, n_bins - 1)
    actual = np.einsum("nkb,nkb->nk", transfer.band_vals, tin[columns])[0]
    actual += apply_segment_targets(transfer, targets)[0]
    actual = np.where(transfer.valid_out[0], actual, np.nan)

    rows = heat_network.segments.index.get_indexer(list(heat_network.paths[node]))
    days = tedges_to_days(short_tedges)
    oracle = OraclePath(
        tedges_days=days,
        segment_flow=heat_network.segment_flow(flow=demand)[rows],
        segment_volume=heat_network.segments["volume"].to_numpy(dtype=float)[rows],
        segment_decay=rates[rows],
        node_flow=heat_network.node_flow(flow=demand, nodes=[node])[0],
        segment_target=targets[rows],
    )
    expected = oracle.tout(tin=tin, cout_tedges_days=days)

    both = np.isfinite(actual) & np.isfinite(expected)
    assert both.sum() > 0.5 * len(actual)
    np.testing.assert_allclose(actual[both], expected[both], atol=1e-11)


def test_bias_handles_transits_landing_exactly_on_bin_edges():
    """A segment face crossing exactly at a bin edge is the convention's failure mode.

    Volumes are chosen so that every face of a three-segment path lands on an input-bin
    edge for every parcel: with a constant flow the transit of each segment is an exact
    whole number of bins. The naive open-interval convention is wrong by O(1) here.
    """
    segments = pd.DataFrame(
        {
            "from": ["Plant", "A", "B"],
            "to": ["A", "B", "T1"],
            "volume": [100.0, 200.0, 300.0],
        },
        index=["Plant-A", "A-B", "B-T1"],
    )
    network = PipeNetwork(segments=segments, source="Plant")
    tedges = pd.date_range("2025-06-01", periods=97, freq="h")
    n_bins = len(tedges) - 1
    demand = np.full((1, n_bins), 2400.0)  # 100 m3/h: transits are 1, 2 and 3 whole bins

    rng = np.random.default_rng(23)
    rates = np.array([1.5, 0.7, 3.1])
    targets = rng.uniform(10.0, 20.0, size=(3, n_bins))
    tin = rng.uniform(5.0, 15.0, size=n_bins)

    transfer = _build_operator(network, demand, tedges, rates)
    columns = np.clip(transfer.col_start[..., None] + np.arange(transfer.band_vals.shape[-1]), 0, n_bins - 1)
    actual = np.einsum("nkb,nkb->nk", transfer.band_vals, tin[columns])[0]
    actual += apply_segment_targets(transfer, targets)[0]

    days = tedges_to_days(tedges)
    oracle = OraclePath(
        tedges_days=days,
        segment_flow=network.segment_flow(flow=demand),
        segment_volume=network.segments["volume"].to_numpy(dtype=float),
        segment_decay=rates,
        node_flow=network.node_flow(flow=demand, nodes=["T1"])[0],
        segment_target=targets,
    )
    expected = oracle.tout(tin=tin, cout_tedges_days=days)
    valid = transfer.valid_out[0] & np.isfinite(expected)
    np.testing.assert_allclose(actual[valid], expected[valid], atol=1e-11)


def test_batched_nodes_of_mixed_depth_agree_with_single_node_calls(heat_network, short_tedges, diurnal_demand):
    """A shallow node padded to a deeper path's width must contribute nothing extra."""
    demand = heat_network.flow_array(diurnal_demand(heat_network, short_tedges))
    rates = _rate(heat_network, depth=heat_network.segments["depth"]).to_numpy()
    rng = np.random.default_rng(29)
    targets = rng.uniform(8.0, 24.0, size=(len(heat_network.segments), len(short_tedges) - 1))

    batched = _build_operator(heat_network, demand, short_tedges, rates, nodes=["A", "T4"])
    batched_bias = apply_segment_targets(batched, targets)
    for row, node in enumerate(["A", "T4"]):
        alone = _build_operator(heat_network, demand, short_tedges, rates, nodes=[node])
        np.testing.assert_allclose(apply_segment_targets(alone, targets)[0], batched_bias[row], rtol=1e-13)
        np.testing.assert_array_equal(alone.valid_out[0], batched.valid_out[row])


def test_target_terms_carry_only_the_rows_still_on_a_path(heat_network, short_tedges, diurnal_demand):
    """The per-depth factors are stored ragged, over the rows that reach that depth.

    A row padded out to a deeper path's width contributes an exact zero at its trailing
    slots, and the bias operator is ``O(max_depth**2 * n_rows * n_cin)``, so carrying them
    is both the depth loop's and the operator's largest avoidable cost. The row order is an
    implementation detail; the counts are the contract.
    """
    demand = heat_network.flow_array(diurnal_demand(heat_network, short_tedges))
    rates = _rate(heat_network, depth=heat_network.segments["depth"]).to_numpy()
    nodes = ["A", "B", "T4"]  # paths of depth 1, 2 and 3
    terms = _build_operator(heat_network, demand, short_tedges, rates, nodes=nodes).target_terms

    depths = np.array([len(heat_network.paths[node]) for node in nodes])
    for depth, offset in enumerate(terms.row_offset):
        expected = int(np.count_nonzero(depths > depth))
        for name in ("mean_shift", "bin_entry", "bin_exit", "gap"):
            assert len(getattr(terms, name)[depth]) == expected, f"{name}[{depth}]"
        assert len(offset) == expected
        # Stage ``d + 1`` is also depth ``d + 1``'s entry piece, so it carries depth d's rows.
        assert len(terms.mean_down[depth + 1]) == expected
    assert len(terms.mean_down) == len(terms.row_offset) + 1
    assert min(len(slab) for slab in terms.mean_shift) < len(nodes), "no depth was compacted"


# ============================================================================
# The coupled model
# ============================================================================


def _uniform_case(network, tedges, *, tin, sol_air, flow):
    """Run the coupled model with constant inputs."""
    n = len(tedges) - 1
    return heat.source_to_endmember(
        tin=np.full(n, tin),
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        network=network,
        soil=pd.DataFrame([GRASS], index=["grass"]),
        surface_temperature=pd.DataFrame({"grass": np.full(n, sol_air)}),
    )


def _local_reference(*, tin, t_inf, dt, tau, n_slug, r_inner, d_eff, alpha, kappa, r_other):
    """Integrate the coupled model directly, with one soil memory per axial cell.

    The package keeps one wall-flux history per pipe; the physics keeps one per axial
    position, because the soil columns are independent and the flux falls along the pipe.
    This reference resolves that: a CFL-1 Lagrangian slug of ``n_slug`` cells, each with its
    own flux history, an exact exponential parcel update over each sub-step, the exact
    time-mean flux over it, and the wall closed in one line per step rather than iterated. It
    shares no arithmetic with the package -- no operator, no Abel summation, no scan.

    The wall temperature is eliminated first, so the parcel relaxes toward
    ``T_eff = T_inf - memory`` through the *full* series resistance. That arrangement stays
    well behaved as the film and wall resistance shrink, and it reduces exactly to the one-way
    relaxation when the halo is switched off.

    Parameters
    ----------
    tin, t_inf : ndarray
        Source temperature and undisturbed soil temperature at pipe depth, one value per
        output bin.
    dt, tau : float
        Output bin width and pipe transit time [days].
    n_slug : int
        Axial cells; ``dt * n_slug / tau`` must be an integer, or a bin average of the fine
        output would not be a bin average of the same grid.
    r_inner, d_eff, alpha, kappa : float
        Inner radius [m], effective burial depth [m], soil diffusivity and conductivity ratio
        [m²/day]. The line source is read at ``r_inner`` (bare pipe).
    r_other : float
        Film plus pipe-wall resistance [day/m²] -- everything in series except the soil.

    Returns
    -------
    ndarray
        Bin-averaged delivered temperature, one value per input bin.
    """
    per = round(dt * n_slug / tau)
    assert abs(dt * n_slug / tau - per) < 1e-9, "n_slug must make dt/dtf an integer"
    area = np.pi * r_inner**2
    r_inf = np.log(2.0 * d_eff / r_inner) / (2.0 * np.pi * kappa)
    step = tau / n_slug
    n_steps = len(tin) * per

    # Bin-averaged deficit on the sub-step grid, from the physical parameters: the line
    # source at the wall minus its mirror image at 2 d_eff, integrated in closed form.
    lag = step * np.arange(n_steps + 2)
    integral = np.zeros((2, len(lag)))
    for row, c in enumerate((r_inner**2 / (4.0 * alpha), (2.0 * d_eff) ** 2 / (4.0 * alpha))):
        with np.errstate(divide="ignore", over="ignore"):
            x = c / lag[1:]
            integral[row, 1:] = (lag[1:] + c) * exp1(x) - lag[1:] * np.exp(-x)
    deficit = np.diff(r_inf * lag - (integral[0] - integral[1]) / (4.0 * np.pi * kappa)) / step

    rate = 1.0 / ((r_other + r_inf) * area)
    survive = np.exp(-rate * step)
    # The cell filled at the end of a sub-step holds the water that entered *during* it.
    sample = np.clip(((np.arange(n_steps) + 0.5) * step / dt).astype(int), 0, len(tin) - 1)
    tin_fine, tinf_fine = np.asarray(tin)[sample], np.asarray(t_inf)[sample]

    gain = rate * area * (1.0 - survive) / (rate * step)  # psi = gain * (T_parcel - T_eff)
    slug = np.full(n_slug, tin_fine[0])
    flux = np.zeros((n_steps, n_slug))
    increment = np.zeros((n_steps, n_slug))
    delivered = np.zeros(n_steps)
    for j in range(n_steps):
        memory = deficit[j:0:-1] @ increment[:j] if j else np.zeros(n_slug)
        previous = flux[j - 1] if j else np.zeros(n_slug)
        settled = tinf_fine[j] - memory + deficit[0] * previous
        flux[j] = gain * (slug - settled) / (1.0 - gain * deficit[0])
        target = settled - deficit[0] * flux[j]
        slug = target + (slug - target) * survive
        increment[j] = flux[j] - previous
        delivered[j] = slug[0]
        slug = np.roll(slug, -1)
        slug[-1] = tin_fine[j]
    return delivered.reshape(-1, per).mean(axis=1)


def _series_pipe(n_sub, *, transit_hours, tin, tedges, film_coefficient=0.454):
    """Run the package on one 100 mm, 2 km grass pipe cut into ``n_sub`` series pieces.

    The model carries one wall-flux history per pipe, so declaring the pipe as a chain of
    shorter segments is how a caller refines the soil memory along it. That refinement is
    the subject of the tests below.

    Returns
    -------
    ndarray
        Delivered temperature at the far end, one value per bin.
    """
    nodes = ["Plant", *[f"n{i}" for i in range(1, n_sub)], "T1"]
    segments = pd.DataFrame(
        {
            "from": nodes[:-1],
            "to": nodes[1:],
            "length": [2000.0 / n_sub] * n_sub,
            "diameter": [0.1] * n_sub,
            "cover": ["grass"] * n_sub,
        },
        index=[f"s{i}" for i in range(n_sub)],
    )
    network = PipeNetwork(segments=segments, source="Plant")
    n_bins = len(tedges) - 1
    volume = float(network.segments["volume"].sum())
    return heat.source_to_endmember(
        tin=tin,
        flow=np.full((1, n_bins), volume / (transit_hours / 24.0)),
        tedges=tedges,
        cout_tedges=tedges,
        network=network,
        soil=pd.DataFrame([GRASS], index=["grass"]),
        surface_temperature=pd.DataFrame({"grass": np.full(n_bins, 18.0)}),
        film_coefficient=film_coefficient,
    )[0]


@pytest.mark.parametrize(("transit_hours", "n_slug", "tolerance"), [(2.0, 8, 0.60), (6.0, 12, 1.15)])
def test_two_way_model_agrees_with_the_local_fine_step_reference(transit_hours, n_slug, tolerance):
    """The coupled fixed point against an independently integrated local reference.

    The reference keeps one soil memory per axial cell; the package keeps one per pipe. The
    gap between them is therefore that one modelling difference, and the test that they are
    the same physics rather than two answers that happen to land near each other is that the
    gap *closes* as the pipe is modelled as more series pieces. It does so at first order in
    the number of pieces, not second: the flux of a piece is read off the water it delivers,
    which smears the flux history over one piece transit.

    A film resistance is supplied deliberately. At the bare-pipe limit the reference's
    closed-form per-step solve is singular and it diverges below about 2 % of the soil
    resistance; the laminar film of a 100 mm service line is 29 % of it. Every anchor below
    scales with the forcing, which is 13 +- 3 K of produced water into soil held at 18 C.
    """
    tedges = pd.date_range("2025-06-01", periods=4 * 24 + 1, freq="h")
    n_bins = len(tedges) - 1
    tin = 13.0 + 3.0 * np.sin(2.0 * np.pi * (np.arange(n_bins) + 0.5) / 24.0)
    reference = _local_reference(
        tin=tin,
        t_inf=np.full(n_bins, 18.0),
        dt=1.0 / 24.0,
        tau=transit_hours / 24.0,
        n_slug=n_slug,
        r_inner=0.05,
        d_eff=1.0 + GRASS["kappa"] / GRASS["eta"],
        alpha=GRASS["alpha"],
        kappa=GRASS["kappa"],
        r_other=1.0 / (2.0 * np.pi * 0.05 * 0.454),
    )

    settled = slice(2 * 24, None)  # both models start with no halo at all
    gaps = [
        float(
            np.max(
                np.abs(_series_pipe(n_sub, transit_hours=transit_hours, tin=tin, tedges=tedges) - reference)[settled]
            )
        )
        for n_sub in (1, 4, 8)
    ]
    assert gaps[0] < tolerance, gaps
    assert gaps[1] < 0.4 * gaps[0], gaps  # two doublings; measured ratio 0.21-0.25
    assert gaps[2] < 0.75 * gaps[1], gaps  # one more; measured 0.50-0.56


def test_uniform_temperature_is_a_fixed_point(heat_network, hourly_tedges, diurnal_demand):
    """Water at the soil temperature everywhere stays there: no flux, no halo, no drift."""
    heat_network.segments["cover"] = "grass"
    out = _uniform_case(
        heat_network, hourly_tedges, tin=14.0, sol_air=14.0, flow=diurnal_demand(heat_network, hourly_tedges)
    )
    np.testing.assert_allclose(out[np.isfinite(out)], 14.0, atol=1e-11)


@pytest.mark.parametrize("transit_hours", [2.0, 6.0])
def test_converged_model_satisfies_the_steady_buried_pipe_law(heat_pipe, transit_hours):
    """With constant forcing the fixed point is the analytic steady relaxation.

    This is the discriminating test of the discretization: relaxing through the *full*
    series resistance gives the distributed steady solution, approached from above as the
    remaining ``D(tau) ~ 3/tau`` deficit tail decays over the record. Folding the first lag
    bin's deficit into an early-time resistance instead would sit 5-20 % *below* this law.
    """
    tedges = pd.date_range("2025-01-01", periods=400 * 24 + 1, freq="h")
    volume = float(heat_pipe.segments.loc["Plant-T1", "volume"])
    flow = np.full((1, len(tedges) - 1), volume / (transit_hours / 24.0))

    out = _uniform_case(heat_pipe, tedges, tin=8.0, sol_air=20.0, flow=flow)

    rate = _rate(heat_pipe)["Plant-T1"]
    expected = 20.0 + (8.0 - 20.0) * np.exp(-rate * transit_hours / 24.0)
    delivered = float(np.nanmean(out[0, -24:]))
    # The remaining gap is the undecayed deficit tail, which still lets a little more heat
    # through than the fully developed halo would: delivered sits just past the steady law,
    # never short of it. The early-time-resistance split would land several percent short.
    approach = (delivered - expected) / (20.0 - 8.0)
    assert 0.0 < approach < 0.01, f"delivered {delivered} against steady law {expected}"


@pytest.mark.parametrize("r_inner", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("transit_bins", [2, 6, 24])
def test_the_quasi_steady_limit_is_the_steady_buried_pipe_law_across_the_geometry(r_inner, transit_bins):
    """Once the halo is developed the fixed point is the analytic law, for every geometry.

    The test above pins the same law at one radius; this sweeps the radius and the transit,
    which is what separates a discretization that happens to be right at one operating point
    from one that is right. Daily bins, because the record has to be long: the deficit tail
    decays like ``3-5 / t`` in days, so a record that reaches the quasi-steady limit is a few
    thousand bins on a daily grid and a hundred thousand on an hourly one.

    The tolerance is one-sided as well as two-sided. The remaining deficit still lets a
    little more heat through than a fully developed halo would, so the fixed point sits just
    *past* the law and never short of it --- which is the direction that discriminates: the
    rejected early-time-resistance split lands several percent below.
    """
    tedges = pd.date_range("2025-01-01", periods=1501, freq="D")
    segments = pd.DataFrame(
        {"from": ["Plant"], "to": ["T1"], "length": [1000.0], "diameter": [2.0 * r_inner], "cover": ["grass"]},
        index=["Plant-T1"],
    )
    network = PipeNetwork(segments=segments, source="Plant")
    volume = float(network.segments.loc["Plant-T1", "volume"])
    flow = np.full((1, 1500), volume / transit_bins)  # one bin is one day, so transit_bins days

    delivered = float(_uniform_case(network, tedges, tin=8.0, sol_air=20.0, flow=flow)[0, -1])
    expected = 20.0 + (8.0 - 20.0) * np.exp(-_rate(network)["Plant-T1"] * transit_bins)
    assert delivered > expected
    # The worst measured excess over the nine cases is 9.4e-4, so 1e-3 would be 6 % of margin.
    np.testing.assert_allclose(delivered, expected, rtol=2e-3)


def test_one_way_model_is_the_first_iterate(heat_network, hourly_tedges, diurnal_demand, soil, surface):
    """``max_sweeps=1`` is exactly the operator applied to the undisturbed soil field.

    Assembling that by hand is the regression guard on the one-way promise: the returned
    series must be ``W @ tin`` plus the bias of the *undisturbed* targets, with no trace of
    a halo. It also fixes what the coupled sweeps are a correction to.
    """
    n = len(hourly_tedges) - 1
    tin = 8.0 + 0.5 * np.sin(2.0 * np.pi * np.arange(n) / 24.0)
    shared = dict(
        flow=diurnal_demand(heat_network, hourly_tedges),
        tedges=hourly_tedges,
        cout_tedges=hourly_tedges,
        network=heat_network,
        soil=soil,
        surface_temperature=surface(hourly_tedges, amplitude=3.0),
    )
    one_way = heat.source_to_endmember(tin=tin, **shared, max_sweeps=1)

    system = heat._build_system(
        **shared,
        surface_tedges=None,
        nodes=None,
        kappa_pipe=None,
        film_coefficient=None,
        spinup="constant",
    )
    padded = np.concatenate([np.full(system.n_pad, tin[0]), tin])
    expected = heat._apply(system.reporting, padded) + apply_segment_targets(
        system.reporting, heat._extended(system.t_inf)
    )
    expected[~system.reporting.valid_out] = np.nan

    np.testing.assert_array_equal(np.isnan(one_way), np.isnan(expected))
    finite = np.isfinite(one_way)
    np.testing.assert_allclose(one_way[finite], expected[finite], rtol=1e-14)

    # And the coupling is a correction to it, not a replacement: the two agree once the
    # halo has had no chance to build, and differ where it has.
    two_way = heat.source_to_endmember(tin=tin, **shared)
    assert np.nanmax(np.abs(two_way - one_way)) > 0.1


def test_each_segment_relaxes_toward_its_own_cover_and_depth(
    heat_network, hourly_tedges, diurnal_demand, soil, surface
):
    """Every pipe must be given the soil field of *its* land cover at *its* burial depth.

    ``_build_system`` fans the undisturbed field out over segments by matching on the
    ``(cover, depth)`` pair and reading the matching surface column, and that plumbing is
    invisible in the delivered temperature: a mix-up hands every pipe a wrong but perfectly
    smooth soil field, so nothing downstream looks anomalous. The fixture network is built for
    this check --- two covers against four distinct depths --- and the field is re-derived here
    from each segment's own parameters, which is exact rather than merely close.
    """
    surface_temperature = surface(hourly_tedges, amplitude=4.0)
    system = heat._build_system(
        flow=diurnal_demand(heat_network, hourly_tedges),
        tedges=hourly_tedges,
        cout_tedges=hourly_tedges,
        network=heat_network,
        soil=soil,
        surface_temperature=surface_temperature,
        surface_tedges=None,
        nodes=None,
        kappa_pipe=None,
        film_coefficient=None,
        spinup="constant",
    )
    # The field is built on the padded grid, which the spin-up prepends to the caller's.
    width = hourly_tedges[1] - hourly_tedges[0]
    padded = (hourly_tedges[0] - pd.TimedeltaIndex(width * np.arange(system.n_pad, 0, -1))).append(hourly_tedges)

    segments = heat_network.segments
    assert len(set(zip(segments["cover"], segments["depth"], strict=True))) > 1, "fixture must mix covers and depths"
    for row, name in enumerate(segments.index):
        cover, depth = segments.loc[name, "cover"], float(segments.loc[name, "depth"])
        expected = heat.soil_temperature(
            surface_temperature=surface_temperature[cover].to_numpy(dtype=float),
            tedges=padded,
            depth=depth,
            alpha=float(soil.loc[cover, "alpha"]),
            kappa=float(soil.loc[cover, "kappa"]),
            eta=float(soil.loc[cover, "eta"]),
            surface_tedges=hourly_tedges,
        )
        np.testing.assert_array_equal(system.t_inf[row], expected, err_msg=f"segment {name} ({cover}, {depth} m)")


def test_the_halo_stores_heat_and_gives_it_back(heat_pipe):
    """The memory reverses sign when the flux does, which is what makes it a memory.

    Warm water into cold soil pours heat into a halo that has not yet developed, so it
    cools faster than the fully-developed-halo model says. When the source then drops to
    the soil temperature the flux reverses: the stored heat comes back out, and the
    delivered water is now *warmer* than that model says. The two signs cannot both come
    from a mis-signed kernel, and the return must fade as the halo drains to the surface.
    """
    tedges = pd.date_range("2025-01-01", periods=60 * 24 + 1, freq="h")
    n = len(tedges) - 1
    days = (np.arange(n) + 0.5) / 24.0
    volume = float(heat_pipe.segments.loc["Plant-T1", "volume"])
    shared = dict(
        tin=np.where(days < 30.0, 25.0, 10.0),
        flow=np.full((1, n), volume / (6.0 / 24.0)),
        tedges=tedges,
        cout_tedges=tedges,
        network=heat_pipe,
        soil=pd.DataFrame([GRASS], index=["grass"]),
        surface_temperature=pd.DataFrame({"grass": np.full(n, 10.0)}),
    )
    difference = heat.source_to_endmember(**shared)[0] - heat.source_to_endmember(**shared, max_sweeps=1)[0]

    warm = slice(20 * 24, 30 * 24)
    after = slice(31 * 24, 45 * 24)
    assert np.all(difference[warm] < 0.0), "an undeveloped halo must take up heat more readily"
    assert np.all(difference[after] > 0.0), "the stored heat must come back out"
    assert difference[after][0] > 3.0 * difference[after][-1], "the return must fade"


def test_the_delivered_temperature_overshoots_the_forcing_range_by_a_measured_amount(heat_pipe):
    """The two-way answer leaves the range of its own inputs, and by how much is pinned here.

    It would be comfortable to assert that relaxation is a convex combination and so the
    delivered temperature lies between the produced water and the soil. It is not true. The
    relaxation target is an effective driving temperature: the rate carries the *steady* soil
    resistance, which is too large while the halo is still developing, so the target is driven
    past the undisturbed soil to reproduce the faster early exchange. The delivered
    temperature is a genuine weighted average of ``tin`` and those targets, so it inherits a
    share of the excursion.

    Pinning the size of it is the test that can fail. Loosening it into a range check with a
    few kelvin of slack --- which is what a hand-written pair of literals amounts to --- would
    pass whether the overshoot were 0.2 K or 4 K.
    """
    tedges = pd.date_range("2025-06-01", periods=40 * 24 + 1, freq="h")
    n = len(tedges) - 1
    volume = float(heat_pipe.segments.loc["Plant-T1", "volume"])
    shared = dict(
        # A step in the produced temperature 20 days in, into soil held flat: the hull is
        # [21, 45] and every excursion past it is the halo answering the step.
        tin=np.where(np.arange(n) < 20 * 24, 21.0, 45.0),
        flow=np.full((1, n), volume / (12.0 / 24.0)),
        tedges=tedges,
        cout_tedges=tedges,
        network=heat_pipe,
        soil=pd.DataFrame([GRASS], index=["grass"]),
        surface_temperature=pd.DataFrame({"grass": np.full(n, 21.0)}),
    )
    contrast = 45.0 - 21.0

    two_way = heat.source_to_endmember(**shared)[0]
    finite = two_way[np.isfinite(two_way)]
    # The hot water pours heat into a halo that does not exist yet, so the target is driven
    # *below* the soil and the delivered water follows it down: the excursion is on the cold
    # side, and there is none on the warm side.
    assert finite.max() <= 45.0 + 1e-9
    excursion = (21.0 - finite.min()) / contrast
    assert 0.15 < excursion < 0.25, excursion

    # Two things bound it. The one-way model has a fixed target and is exactly inside the
    # hull; and declaring the same pipe as two segments, which refines the flux history the
    # halo is built from, cuts the excursion severalfold without removing it.
    one_way = heat.source_to_endmember(**shared, max_sweeps=1)[0]
    assert np.nanmin(one_way) >= 21.0 - 1e-9
    halves = PipeNetwork(
        segments=pd.DataFrame(
            {
                "from": ["Plant", "mid"],
                "to": ["mid", "T1"],
                "length": [500.0, 500.0],
                "diameter": [0.1, 0.1],
                "cover": ["grass", "grass"],
            },
            index=["a", "b"],
        ),
        source="Plant",
    )
    split = heat.source_to_endmember(**{**shared, "network": halves})[0]
    assert (21.0 - np.nanmin(split)) / contrast < 0.5 * excursion


def test_the_model_is_linear_in_every_temperature_input(heat_network, hourly_tedges, diurnal_demand, soil, surface):
    """Scaling every temperature input scales the output, to the accuracy of the fixed point.

    Exactly, in the model. In the answer, to whatever the iteration was asked to reach: the
    tolerance is absolute, so doubling the inputs doubles the increments and the two runs stop
    at different distances from their own fixed points. Both are within ``atol`` of it, and the
    map's contraction turns that into a bound about an order of magnitude larger --- which is
    what is asserted here, and why a tighter ``atol`` recovers a tighter agreement.
    """
    n = len(hourly_tedges) - 1
    shared = dict(
        flow=diurnal_demand(heat_network, hourly_tedges),
        tedges=hourly_tedges,
        cout_tedges=hourly_tedges,
        network=heat_network,
        soil=soil,
    )
    base = heat.source_to_endmember(
        tin=np.full(n, 9.0), surface_temperature=surface(hourly_tedges, amplitude=4.0), **shared
    )
    scaled = heat.source_to_endmember(
        tin=np.full(n, 18.0), surface_temperature=2.0 * surface(hourly_tedges, amplitude=4.0), **shared
    )
    np.testing.assert_allclose(scaled, 2.0 * base, rtol=1e-12, atol=1e-7, equal_nan=True)

    tight = dict(shared, atol=1e-11)
    base = heat.source_to_endmember(
        tin=np.full(n, 9.0), surface_temperature=surface(hourly_tedges, amplitude=4.0), **tight
    )
    scaled = heat.source_to_endmember(
        tin=np.full(n, 18.0), surface_temperature=2.0 * surface(hourly_tedges, amplitude=4.0), **tight
    )
    np.testing.assert_allclose(scaled, 2.0 * base, rtol=1e-12, atol=1e-9, equal_nan=True)


def test_the_answer_does_not_depend_on_the_temperature_origin(
    heat_network, hourly_tedges, diurnal_demand, soil, surface
):
    """Celsius or kelvin gives the same answer, shifted --- the converged fixed point included.

    Every coefficient of the model is geometry, so adding a constant to every temperature
    input must add exactly that constant to the output. The one thing that can break it is the
    convergence rule: a tolerance measured *relative* to the size of the iterate is 273 times
    looser in kelvin than in Celsius, so the two runs stop at different points and disagree by
    the difference. That is why the tolerance is absolute.
    """
    n = len(hourly_tedges) - 1
    tin = 8.0 + 3.0 * np.sin(2.0 * np.pi * np.arange(n) / 24.0)
    shared = dict(
        flow=diurnal_demand(heat_network, hourly_tedges),
        tedges=hourly_tedges,
        cout_tedges=hourly_tedges,
        network=heat_network,
        soil=soil,
    )
    base = heat.source_to_endmember(tin=tin, surface_temperature=surface(hourly_tedges, amplitude=4.0), **shared)
    shifted = heat.source_to_endmember(
        tin=tin + 273.15, surface_temperature=surface(hourly_tedges, amplitude=4.0) + 273.15, **shared
    )
    np.testing.assert_allclose(shifted - 273.15, base, atol=1e-10, equal_nan=True)


def test_inert_copy_makes_the_entry_row_the_same_water(heat_network, short_tedges, diurnal_demand, surface):
    """With nothing to exchange, the entry and delivery rows must coincide exactly.

    The flux pass differences two operator rows: the delivery temperature at a segment's
    downstream node, and the temperature of that same water where it entered the segment,
    obtained by replacing the segment with an inert copy that carries the parcel across
    without exchanging. If the two rows ever described different parcels, driving every
    exchange rate to zero -- which makes the real segment and its inert copy identical --
    would not collapse them onto each other.
    """
    heat_network.segments["cover"] = "grass"
    inert = pd.DataFrame([{"alpha": GRASS["alpha"], "kappa": 1e-9, "eta": GRASS["eta"]}], index=["grass"])
    system = heat._build_system(
        flow=diurnal_demand(heat_network, short_tedges),
        tedges=short_tedges,
        cout_tedges=short_tedges,
        network=heat_network,
        soil=inert,
        surface_temperature=surface(short_tedges, amplitude=3.0),
        surface_tedges=None,
        nodes=None,
        kappa_pipe=None,
        film_coefficient=None,
        spinup=None,
    )
    rng = np.random.default_rng(41)
    tin = 9.0 + rng.normal(0.0, 1.0, system.n_bins)
    t_int = heat._internal_pass(system, heat._apply(system.internal, tin), system.t_inf)
    n_seg = len(heat_network.segments)
    entry, delivered = t_int[:n_seg], t_int[n_seg:]

    both = np.isfinite(entry) & np.isfinite(delivered)
    assert both.sum() > 0.5 * entry.size
    np.testing.assert_allclose(entry[both], delivered[both], atol=1e-6)


def test_leaf_delivery_rows_agree_with_the_transport_module(heat_network, short_tedges, diurnal_demand, soil, surface):
    """A segment feeding an endmember delivers what the public transport model delivers.

    For those segments the segment throughflow *is* the node throughflow, so the internal
    delivery row and :func:`pipetransport.transport.source_to_endmember` average the same
    water over the same bins. With the relaxation targets set to zero the bias vanishes and
    what remains is pure decay -- which the transport module computes independently.
    """
    n = len(short_tedges) - 1
    system = heat._build_system(
        flow=diurnal_demand(heat_network, short_tedges),
        tedges=short_tedges,
        cout_tedges=short_tedges,
        network=heat_network,
        soil=soil,
        surface_temperature=surface(short_tedges, amplitude=3.0),
        surface_tedges=None,
        nodes=None,
        kappa_pipe=None,
        film_coefficient=None,
        spinup=None,
    )
    tin = np.full(n, 9.0)
    n_seg = len(heat_network.segments)
    # The rates the operator was built with, per cover class; what they should be is pinned
    # separately, and reusing them here keeps this test about the operator rows alone.
    rates = pd.Series(system.internal.target_terms.segment_rate[:n_seg], index=heat_network.segments.index)
    bare = heat._apply(system.internal, tin) + apply_segment_targets(system.internal, np.zeros((2 * n_seg, n)))
    bare = np.where(system.internal.valid_out, bare, np.nan)

    checked = 0
    for row, (name, segment) in enumerate(heat_network.segments.iterrows()):
        node = str(segment["to"])
        if node not in heat_network.endmembers:
            continue
        expected = transport.source_to_endmember(
            cin=tin,
            flow=diurnal_demand(heat_network, short_tedges),
            tedges=short_tedges,
            cout_tedges=short_tedges,
            network=heat_network,
            nodes=[node],
            decay_rate=rates,
            spinup=None,
        )[0]
        actual = bare[n_seg + row]
        both = np.isfinite(actual) & np.isfinite(expected)
        assert both.sum() > 0, f"no comparable bin for {name}"
        np.testing.assert_allclose(actual[both], expected[both], rtol=1e-12, err_msg=f"segment {name}")
        checked += 1
    assert checked == len(heat_network.endmembers)


def test_wall_flux_vanishes_without_a_temperature_difference(heat_network, hourly_tedges, diurnal_demand, soil):
    """Water already at the soil temperature exchanges nothing, so the halo never forms."""
    heat_network.segments["cover"] = "grass"
    n = len(hourly_tedges) - 1
    system = heat._build_system(
        flow=diurnal_demand(heat_network, hourly_tedges),
        tedges=hourly_tedges,
        cout_tedges=hourly_tedges,
        network=heat_network,
        soil=soil,
        surface_temperature=pd.DataFrame({"grass": np.full(n, 14.0), "paved": np.full(n, 14.0)}),
        surface_tedges=None,
        nodes=None,
        kappa_pipe=None,
        film_coefficient=None,
        spinup="constant",
    )
    uniform = np.full(system.n_bins, 14.0)
    targets = heat._update_targets(
        system, heat._internal_pass(system, heat._apply(system.internal, uniform), system.t_inf)
    )
    np.testing.assert_allclose(targets, system.t_inf, atol=1e-11)


def test_pre_history_transient_decays_with_a_longer_lead_in(heat_pipe):
    """The halo starts undisturbed, and that assumption is expensive until it is paid off.

    The model knows no flux history before the record, so a pipe that has in reality been
    running for years is modelled as one switched on at the first bin -- meeting soil that
    accepts heat almost without resistance. The delivered temperature is wrong by several
    kelvin at first and the error decays as the record is extended backwards. Pinning the
    decay puts a number on the lead-in a user has to supply.
    """
    volume = float(heat_pipe.segments.loc["Plant-T1", "volume"])
    window_days = 20

    def run(lead_days):
        total = lead_days + window_days
        tedges = pd.date_range(
            pd.Timestamp("2025-03-01") - pd.Timedelta(days=lead_days), periods=total * 24 + 1, freq="h"
        )
        n = len(tedges) - 1
        out = heat.source_to_endmember(
            tin=np.full(n, 8.0),
            flow=np.full((1, n), volume / (2.0 / 24.0)),
            tedges=tedges,
            cout_tedges=tedges,
            network=heat_pipe,
            soil=pd.DataFrame([GRASS], index=["grass"]),
            surface_temperature=pd.DataFrame({"grass": np.full(n, 20.0)}),
        )
        return out[0, lead_days * 24 :]

    reference = run(180)
    errors = [float(np.nanmax(np.abs(run(lead) - reference))) for lead in (0, 7, 30, 90)]
    assert errors[0] > errors[1] > errors[2] > errors[3]
    # Starting cold is worth kelvins, a week of lead-in buys back most of it, and a season
    # brings it under a twentieth of a kelvin.
    assert 3.0 < errors[0] < 12.0
    assert errors[1] < 1.5
    assert errors[2] < 0.4
    assert errors[3] < 0.1


def test_declaring_a_pipe_as_series_segments_leaves_the_transport_operator_alone(
    heat_network, hourly_tedges, diurnal_demand, soil, surface
):
    """Refining the soil memory must not perturb the transport: ``W``, its coverage, its travel times.

    The model carries one wall-flux history per pipe, so a caller who needs the flux resolved
    along a pipe declares it as a chain of shorter segments. That is admissible only because
    ``k`` series pieces of volume ``V/k`` at the same flow compose to the same arrival map and
    their exchange exponents add to the whole pipe's. Applying the operator with the targets
    left out isolates ``W @ tin`` from the bias, which is the quantity that must not move ---
    otherwise refining the memory would silently perturb conservative transport, the residence
    times and the ``h -> 0`` reduction.
    """

    def chained(k):
        rows = {"from": [], "to": [], "length": [], "diameter": [], "volume": [], "cover": [], "depth": []}
        index = []
        for name, row in heat_network.segments.iterrows():
            previous = row["from"]
            for i in range(k):
                nxt = row["to"] if i == k - 1 else f"{name}~{i}"
                rows["from"].append(previous)
                rows["to"].append(nxt)
                rows["length"].append(row["length"] / k)
                rows["volume"].append(row["volume"] / k)
                rows["diameter"].append(row["diameter"])
                rows["cover"].append(row["cover"])
                rows["depth"].append(row["depth"])
                index.append(f"{name}~p{i}")
                previous = nxt
        return PipeNetwork(segments=pd.DataFrame(rows, index=index), source="Plant")

    def system(network):
        return heat._build_system(
            flow=diurnal_demand(heat_network, hourly_tedges),
            tedges=hourly_tedges,
            cout_tedges=hourly_tedges,
            network=network,
            soil=soil,
            surface_temperature=surface(hourly_tedges, amplitude=3.0),
            surface_tedges=None,
            nodes=None,
            kappa_pipe=None,
            film_coefficient=None,
            spinup="constant",
        )

    whole = system(heat_network)
    assert len(whole.length) == len(heat_network.segments), "one flux history per pipe"

    tin = np.random.default_rng(7).normal(10.0, 2.0, whole.n_bins)
    reference = heat._apply(whole.reporting, tin)
    # The deviation is round-off in the composed displacement maps and grows with the record
    # (about 64 ulps of the cumulative volume over the per-bin node volume), so it is pinned
    # absolutely rather than bit-exactly; it measures ~2e-13 K on these ten days.
    for k in (2, 3, 4):
        split = system(chained(k))
        assert len(split.length) == k * len(heat_network.segments)
        np.testing.assert_allclose(heat._apply(split.reporting, tin), reference, atol=1e-11)
        np.testing.assert_array_equal(split.reporting.valid_out, whole.reporting.valid_out)
        np.testing.assert_allclose(
            split.reporting.residence_time_out, whole.reporting.residence_time_out, rtol=1e-10, equal_nan=True
        )


def test_the_fixed_point_does_not_depend_on_how_long_the_record_runs(heat_pipe):
    """Extending the record forwards cannot move a bin that was already inside it.

    The target map is strictly lower-triangular in the bin index --- the halo at bin ``n`` is
    a convolution over lags ``j <= n`` --- so the converged answer for a bin is a property of
    the record *before* it and of nothing else. An iteration that stopped short of its fixed
    point, or a kernel that leaked a lag from the wrong end of the convolution, would drift
    between records of different length; both failures are invisible in any single run.
    """
    volume = float(heat_pipe.segments.loc["Plant-T1", "volume"])

    def run(n_days):
        tedges = pd.date_range("2025-06-01", periods=n_days * 24 + 1, freq="h")
        n = len(tedges) - 1
        return heat.source_to_endmember(
            tin=9.0 + 3.0 * np.sin(2.0 * np.pi * np.arange(n) / 24.0),
            flow=np.full((1, n), volume / (2.0 / 24.0)),
            tedges=tedges,
            cout_tedges=tedges,
            network=heat_pipe,
            soil=pd.DataFrame([GRASS], index=["grass"]),
            surface_temperature=pd.DataFrame({"grass": np.full(n, 21.0)}),
        )[0]

    short, medium, longest = run(10), run(30), run(90)
    for other in (medium, longest):
        window = len(short)
        both = np.isfinite(short) & np.isfinite(other[:window])
        assert both.sum() > 0.9 * window
        np.testing.assert_allclose(other[:window][both], short[both], atol=1e-9)
    # And the halo really is still developing over those 90 days, so the agreement above is
    # not the trivial one of a signal that has already settled.
    assert abs(float(np.nanmean(longest[-24:]) - np.nanmean(short[-24:]))) > 0.1


def test_an_over_cap_branch_does_not_void_the_warm_start_of_the_whole_network(soil, surface):
    """One branch too long to warm-start must cost only itself its padding.

    ``_build_system`` takes its warm start over every endmember path, so a single branch
    behind a volume large enough to trip the padding cap used to leave the entire call on
    strict validity -- including nodes whose own history is a few hours long. Unlike the
    forward transport module the failure is unconditional here: there is no request order to
    notice it by.
    """
    segments = pd.DataFrame(
        {
            "from": ["Plant", "A", "A"],
            "to": ["A", "T1", "T2"],
            "length": [1000.0, 200.0, 2.0e5],
            "diameter": [0.3, 0.15, 1.0],
            "cover": ["grass", "grass", "grass"],
        },
        index=["Plant-A", "A-T1", "A-T2"],
    )
    network = PipeNetwork(segments=segments, source="Plant")
    tedges = pd.date_range("2025-06-01", periods=4 * 24 + 1, freq="h")
    n = len(tedges) - 1
    shared = dict(
        tin=np.full(n, 9.0),
        flow=pd.DataFrame({"T1": np.full(n, 400.0), "T2": np.full(n, 300.0)}),
        tedges=tedges,
        cout_tedges=tedges,
        network=network,
        soil=soil,
        surface_temperature=surface(tedges),
    )

    out = heat.source_to_endmember(nodes=["T1", "T2"], **shared)

    assert not np.isnan(out[0]).any(), "T1 sits behind 74 m3 and is warm-startable"
    assert np.all(np.isnan(out[1])), "T2 sits behind 1.6e5 m3, well over a year of transit"


def test_halo_memory_converges_under_bin_refinement():
    """The bin-averaged deficit convolution converges to the continuous Duhamel integral.

    The halo memory is the one place the model reads a continuous history through bins, so
    this is where the bin discretization has to be shown to converge. A smooth flux history
    is convolved with the bin-averaged deficit at successively finer bins and compared with
    a finely resolved sum over the same kernel; halving the bin must shrink the gap.
    """
    r_outer, d_eff, alpha, kappa = 0.05, 1.0 + GRASS["kappa"] / GRASS["eta"], GRASS["alpha"], GRASS["kappa"]
    span = 40.0
    geometry = dict(r_o=np.array([r_outer]), d_eff=np.array([d_eff]), alpha=np.array([alpha]), kappa=np.array([kappa]))

    def flux(t):
        return 1.0 - np.exp(-t / 6.0) * np.cos(2.0 * np.pi * t / 11.0)

    def response(n_bins):
        dt = span / n_bins
        deficit = heat._deficit_kernel(n_bins, dt, **geometry)[0]
        binned = flux((np.arange(n_bins) + 0.5) * dt)
        return float(np.sum(np.diff(binned, prepend=0.0) * deficit[::-1]))

    reference = response(40000)
    errors = [abs(response(n) - reference) for n in (250, 500, 1000)]
    assert errors[1] < 0.6 * errors[0], f"halving the bin did not shrink the gap: {errors}"
    assert errors[2] < 0.6 * errors[1], f"halving the bin did not shrink the gap: {errors}"


# ============================================================================
# The reverse direction
# ============================================================================


def _reverse_case(network, tedges, demand, soil, surface_frame, *, nodes, tin, **kwargs):
    """Forward then reverse through the same configuration."""
    shared = dict(
        flow=demand,
        tedges=tedges,
        cout_tedges=tedges,
        network=network,
        soil=soil,
        surface_temperature=surface_frame,
        nodes=nodes,
        # What these test is the deconvolution and its refusals, none of which the internal
        # split touches; carrying it here would multiply their cost to re-measure what
        # test_the_reverse_direction_survives_splitting_the_pipes already covers.
    )
    measured = heat.source_to_endmember(tin=tin, **shared)
    recovered = heat.endmember_to_source(tout=measured, **shared, **kwargs)
    return measured, recovered


def test_the_reconstruction_reproduces_the_measurements_it_was_built_from(heat_pipe):
    """Pushed forward again, the reconstruction has to land back on the data.

    The forward map is affine, ``tout = (W + B) tin + b0``, and the reverse deconvolves
    against ``W`` alone with the bias held at the current targets, iterating on the outside.
    That the outer loop converged is not by itself evidence that it converged to the right
    thing: a bias evaluated at the wrong point of the loop converges just as happily to a
    reconstruction that no longer explains the measurements. Feeding it back through the
    forward model is the residual that catches it.

    Only the interior is asserted. The deconvolution is Tikhonov-regularized and the record's
    two ends are constrained by a fraction of a source window each, so the residual there is a
    property of the regularization rather than of the coupling, and it reaches a few tenths of
    a kelvin in the first transits.
    """
    tedges = pd.date_range("2025-06-01", periods=8 * 24 + 1, freq="h")
    n = len(tedges) - 1
    volume = float(heat_pipe.segments.loc["Plant-T1", "volume"])
    shared = dict(
        flow=np.full((1, n), volume / (2.0 / 24.0)),
        tedges=tedges,
        cout_tedges=tedges,
        network=heat_pipe,
        soil=pd.DataFrame([GRASS], index=["grass"]),
        surface_temperature=pd.DataFrame({"grass": np.full(n, 20.0)}),
    )
    tin = 10.0 + 2.5 * np.sin(2.0 * np.pi * np.arange(n) / 30.0)
    measured = heat.source_to_endmember(tin=tin, **shared)
    recovered = heat.endmember_to_source(tout=measured, **shared)

    assert np.isfinite(recovered).sum() > 0.8 * n
    forward = heat.source_to_endmember(tin=np.where(np.isfinite(recovered), recovered, tin), **shared)
    inner = slice(24, -24)
    residual = np.abs(forward[0, inner] - measured[0, inner])
    assert np.nanmax(residual) < 1e-6, np.nanmax(residual)


def test_reverse_recovers_the_production_temperature(heat_network, short_tedges, diurnal_demand, soil, surface):
    """Round trip through every endmember: the deconvolution inverts the affine model.

    The tolerance is the banded solver's own, not a concession to the coupling: the outer
    iteration is driven to the same target increment as the forward direction, so the extra
    error it contributes stays below the deconvolution's.
    """
    n = len(short_tedges) - 1
    tin = 10.0 + 2.0 * np.sin(2.0 * np.pi * np.arange(n) / 72.0)
    _, recovered = _reverse_case(
        heat_network,
        short_tedges,
        diurnal_demand(heat_network, short_tedges),
        soil,
        surface(short_tedges, amplitude=4.0),
        nodes=None,
        tin=tin,
    )
    inner = slice(36, -36)
    np.testing.assert_allclose(recovered[inner], tin[inner], atol=1e-8)


def test_reverse_tolerates_a_measurement_outage(heat_network, short_tedges, diurnal_demand, soil, surface):
    """One sensor dropping out leaves the reconstruction to the endmembers still reporting."""
    n = len(short_tedges) - 1
    tin = 10.0 + 2.0 * np.sin(2.0 * np.pi * np.arange(n) / 72.0)
    shared = dict(
        flow=diurnal_demand(heat_network, short_tedges),
        tedges=short_tedges,
        cout_tedges=short_tedges,
        network=heat_network,
        soil=soil,
        surface_temperature=surface(short_tedges, amplitude=4.0),
    )
    measured = heat.source_to_endmember(tin=tin, **shared)
    gapped = measured.copy()
    gapped[1, 40:60] = np.nan
    recovered = heat.endmember_to_source(tout=gapped, **shared)

    inner = slice(36, -36)
    np.testing.assert_allclose(recovered[inner], tin[inner], atol=1e-7)


def test_reverse_refuses_an_answer_it_cannot_stand_behind(heat_network, short_tedges, diurnal_demand, soil, surface):
    """Losing every sensor at once makes the coupled inverse ill-posed, and it says so.

    With no endmember reporting over a window, the production during it is unconstrained
    while the halo still couples it to the bins that follow. The fixed point then amplifies
    rather than contracts, and returning the iterate would mean returning a number the data
    does not support.
    """
    n = len(short_tedges) - 1
    tin = 10.0 + 2.0 * np.sin(2.0 * np.pi * np.arange(n) / 72.0)
    shared = dict(
        flow=diurnal_demand(heat_network, short_tedges),
        tedges=short_tedges,
        cout_tedges=short_tedges,
        network=heat_network,
        soil=soil,
        surface_temperature=surface(short_tedges, amplitude=4.0),
    )
    blinded = heat.source_to_endmember(tin=tin, **shared)
    blinded[:, 40:70] = np.nan
    with pytest.raises(RuntimeError, match="did not converge"):
        heat.endmember_to_source(tout=blinded, **shared, max_sweeps=40)


def test_one_way_reverse_is_a_single_banded_solve(heat_network, short_tedges, diurnal_demand, soil, surface):
    """Without the coupling the reverse is the existing deconvolution, and it is exact."""
    n = len(short_tedges) - 1
    tin = 11.0 + 1.5 * np.sin(2.0 * np.pi * np.arange(n) / 60.0)
    shared = dict(
        flow=diurnal_demand(heat_network, short_tedges),
        tedges=short_tedges,
        cout_tedges=short_tedges,
        network=heat_network,
        soil=soil,
        surface_temperature=surface(short_tedges, amplitude=3.0),
    )
    measured = heat.source_to_endmember(tin=tin, **shared, max_sweeps=1)
    recovered = heat.endmember_to_source(tout=measured, **shared, max_sweeps=1)
    inner = slice(36, -36)
    np.testing.assert_allclose(recovered[inner], tin[inner], atol=1e-9)


# ============================================================================
# API behaviour
# ============================================================================


def test_non_convergence_raises_rather_than_returning_a_partial_answer(
    heat_network, hourly_tedges, diurnal_demand, soil, surface
):
    """A truncated iteration is not an answer, so the cap raises."""
    n = len(hourly_tedges) - 1
    with pytest.raises(RuntimeError, match="did not converge"):
        heat.source_to_endmember(
            tin=np.full(n, 8.0),
            flow=diurnal_demand(heat_network, hourly_tedges),
            tedges=hourly_tedges,
            cout_tedges=hourly_tedges,
            network=heat_network,
            soil=soil,
            surface_temperature=surface(hourly_tedges),
            max_sweeps=3,
        )


def test_a_year_of_hourly_data_converges_and_stays_physical(heat_network, soil, diurnal_demand):
    """Convergence does not degrade with the record length, and the answer stays bounded.

    What this test is about --- that the Picard iteration reaches its fixed point and the
    convolution stays bounded over 8760 bins --- is a property of the driver and the kernel,
    and it complements
    :func:`test_the_fixed_point_does_not_depend_on_how_long_the_record_runs`, which pins that
    the answer for a bin does not move as the record grows around it.
    """
    tedges = pd.date_range("2025-01-01", "2026-01-01", freq="h")
    n = len(tedges) - 1
    seasonal = np.sin(2.0 * np.pi * np.arange(n) / n)
    out = heat.source_to_endmember(
        tin=8.0 + 4.0 * seasonal,
        flow=diurnal_demand(heat_network, tedges),
        tedges=tedges,
        cout_tedges=tedges,
        network=heat_network,
        soil=soil,
        surface_temperature=pd.DataFrame({"grass": 12.0 + 10.0 * seasonal, "paved": 16.0 + 12.0 * seasonal}),
    )
    finite = out[np.isfinite(out)]
    assert finite.size > 0.9 * out.size
    assert finite.min() >= 4.0 - 1e-9
    assert finite.max() <= 28.0 + 1e-9


def test_strict_validity_marks_the_same_bins_as_transport(heat_network, hourly_tedges, diurnal_demand, soil, surface):
    """``spinup=None`` invalidates exactly the bins the conservative model invalidates.

    Not run at ``max_sweeps=1``, tempting as that is for a test that compares only NaN
    masks: ``spinup=None`` is what makes non-finite entries reach ``_update_targets``, so
    this is the one test in the file that exercises the halo's NaN scrub, and skipping the
    sweeps would leave that line unguarded.
    """
    n = len(hourly_tedges) - 1
    strict = heat.source_to_endmember(
        tin=np.full(n, 8.0),
        flow=diurnal_demand(heat_network, hourly_tedges),
        tedges=hourly_tedges,
        cout_tedges=hourly_tedges,
        network=heat_network,
        soil=soil,
        surface_temperature=surface(hourly_tedges),
        spinup=None,
    )
    conservative = transport.source_to_endmember(
        cin=np.full(n, 8.0),
        flow=diurnal_demand(heat_network, hourly_tedges),
        tedges=hourly_tedges,
        cout_tedges=hourly_tedges,
        network=heat_network,
        spinup=None,
    )
    np.testing.assert_array_equal(np.isnan(strict), np.isnan(conservative))


@pytest.mark.parametrize(("freq", "periods"), [("h", 3 * 8760), ("15min", 2 * 35040)])
def test_a_multi_year_sub_daily_grid_counts_as_uniform(heat_pipe, freq, periods):
    """An exactly uniform grid stays uniform however far into the record it runs.

    The uniformity check runs on days-since-the-record-start, whose float64 spacing coarsens
    as the record lengthens: differencing an exactly uniform *hourly* grid wobbles by 9e-13
    relative after a year and 1.5e-11 after twenty, and a 15-minute grid four times faster. A
    tolerance tight enough to catch that wobble rejects every multi-year sub-daily record ---
    with a message saying the grid is not uniform, which it demonstrably is. Bin widths that
    are dyadic fractions of a day never wobble, so only the sub-daily ones fail.
    """
    tedges = pd.date_range("2025-01-01", periods=periods + 1, freq=freq)
    days = tedges_to_days(tedges)
    spacing = np.diff(days)
    assert np.ptp(tedges.asi8[1:] - tedges.asi8[:-1]) == 0, "the grid must be exactly uniform to begin with"
    assert np.ptp(spacing) > 0.0, "this record is too short to exercise the float64 wobble"

    volume = float(heat_pipe.segments.loc["Plant-T1", "volume"])
    paths_transfer(
        tedges_days=days,
        cout_tedges_days=days,
        segment_volume=np.array([volume]),
        segment_flow=np.full((1, periods), volume * 12.0),
        segment_decay=np.array([5.0]),
        node_flow=np.full((1, periods), volume * 12.0),
        paths_idx=np.zeros((1, 1), dtype=np.intp),
        active=np.ones((1, 1), dtype=bool),
        with_target_terms=True,
    )


def test_output_grid_may_differ_from_the_input_grid(heat_network, hourly_tedges, diurnal_demand, soil, surface):
    """``cout_tedges`` is free in alignment and resolution.

    Run at ``max_sweeps=1``: the targets are iterated on the *internal* operator, which is
    built on ``tedges`` whatever the output grid is, so the sweeps cannot reach the assertion
    and the shape is settled by the reporting operator alone.
    """
    n = len(hourly_tedges) - 1
    cout_tedges = pd.date_range(hourly_tedges[0], hourly_tedges[-1], freq="6h")
    out = heat.source_to_endmember(
        tin=np.full(n, 8.0),
        flow=diurnal_demand(heat_network, hourly_tedges),
        tedges=hourly_tedges,
        cout_tedges=cout_tedges,
        network=heat_network,
        soil=soil,
        surface_temperature=surface(hourly_tedges),
        max_sweeps=1,
    )
    assert out.shape == (len(heat_network.endmembers), len(cout_tedges) - 1)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda kwargs: kwargs.update(
                tedges=pd.DatetimeIndex([*kwargs["tedges"][:-1], kwargs["tedges"][-1] + pd.Timedelta(hours=3)])
            ),
            "uniformly spaced",
        ),
        (lambda kwargs: kwargs.update(soil=kwargs["soil"].drop(index="grass")), "cover class"),
        (lambda kwargs: kwargs.update(soil=kwargs["soil"].drop(columns="eta")), "column 'eta'"),
        (
            lambda kwargs: kwargs.update(surface_temperature=kwargs["surface_temperature"].drop(columns="grass")),
            "cover class",
        ),
        (lambda kwargs: kwargs.update(nodes=["nowhere"]), "unknown node"),
    ],
)
def test_invalid_input_is_rejected(heat_network, hourly_tedges, diurnal_demand, soil, surface, mutate, message):
    """Each malformed input names what is wrong with it."""
    n = len(hourly_tedges) - 1
    kwargs = dict(
        tin=np.full(n, 8.0),
        flow=diurnal_demand(heat_network, hourly_tedges),
        tedges=hourly_tedges,
        cout_tedges=hourly_tedges,
        network=heat_network,
        soil=soil,
        surface_temperature=surface(hourly_tedges),
    )
    mutate(kwargs)
    with pytest.raises(ValueError, match=message):
        heat.source_to_endmember(**kwargs)


def test_missing_geometry_columns_are_reported(heat_network, hourly_tedges, diurnal_demand, soil, surface):
    """The heat pair needs the pipe geometry the line-source kernel is built on."""
    n = len(hourly_tedges) - 1
    heat_network.segments = heat_network.segments.drop(columns="diameter")
    with pytest.raises(ValueError, match="diameter"):
        heat.source_to_endmember(
            tin=np.full(n, 8.0),
            flow=diurnal_demand(heat_network, hourly_tedges),
            tedges=hourly_tedges,
            cout_tedges=hourly_tedges,
            network=heat_network,
            soil=soil,
            surface_temperature=surface(hourly_tedges),
        )


def test_wall_conductivity_requires_the_wall_thickness(heat_pipe):
    """``kappa_pipe`` without a thickness column is a configuration error, not a default."""
    with pytest.raises(ValueError, match="wall_thickness"):
        _rate(heat_pipe, kappa_pipe=0.008)


def test_sol_air_temperature_combines_the_surface_energy_budget():
    """Absorbed shortwave warms the surface and the loss terms cool it, both over ``eta``."""
    warmed = heat.sol_air_temperature(air_temperature=20.0, solar_irradiance=6.2, absorptivity=0.9, eta=0.41)
    np.testing.assert_allclose(warmed, 20.0 + 0.9 * 6.2 / 0.41, rtol=1e-14)

    cooled = heat.sol_air_temperature(
        air_temperature=20.0, solar_irradiance=6.2, absorptivity=0.9, eta=0.41, heat_loss=2.0
    )
    np.testing.assert_allclose(cooled, warmed - 2.0 / 0.41, rtol=1e-14)

    # The normalized irradiance is what makes the ratio a temperature: 1 W/m2 is 0.0207.
    np.testing.assert_allclose(
        heat.sol_air_temperature(
            air_temperature=0.0, solar_irradiance=300.0 * 86400 / 4.18e6, absorptivity=1.0, eta=0.41
        ),
        300.0 * 86400 / 4.18e6 / 0.41,
        rtol=1e-14,
    )
