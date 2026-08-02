"""
Drinking water temperature in the network: two-way heat exchange with the soil.

Water warms toward the soil around the pipe, and the soil warms back. Newton's law of
cooling makes the water temperature relax toward the soil temperature at a rate set by the
pipe's thermal resistance -- first-order decay toward a moving target instead of toward
zero -- so the delivered temperature is an *affine* reading of the same transport operator
the rest of this package uses: ``T_out = W @ T_in + b``, with ``W`` built with the
per-segment exchange rate in the decay-rate slot and ``b`` the soil's contribution
accumulated along each path. Three exact, closed-form kernels carry all the soil physics:

- **Surface to depth**: the undisturbed soil temperature at pipe depth is the superposition
  of Robin-boundary step responses of the half space, driven by the piecewise-constant
  sol-air temperature per land cover. Bin averages use the closed-form time integral of the
  step response, so a surface series that is constant per time bin maps exactly.
- **Wall flux to wall temperature**: the warm halo the pipe builds in the soil is a
  continuous line source plus a mirror-image sink above the surface (the image is what
  makes the halo saturate at the steady buried-pipe resistance). Splitting its response
  into *steady part + transient deficit* turns the two-way model into the one-way model
  with a memory-shifted target: one short convolution of the wall-flux history per segment.
- **Exchange rate**: the series resistance of the water-side film, the pipe wall and the
  soil, divided by the water column's heat capacity, gives the relaxation rate [1/day] --
  the same ``1/(D**2 ln D)``-flavoured diameter law as chlorine wall decay: service lines
  equilibrate much faster than trunk mains.

The two-way coupling is a fixed point of two vectorized passes (one transport evaluation,
one convolution per segment). Sweep 1 relaxes toward the undisturbed soil temperature at
the steady buried-pipe rate -- exactly the classical one-way model, returned by
``max_sweeps=1``; every further sweep shifts the targets by the deficit convolution of the
latest wall-flux history. Two choices make that iteration usable. The relaxation rate keeps
the *steady* resistance (folding the lag-0 deficit into an early-time resistance -- the
borehole-model move -- lands 5-20 % below the analytic steady buried-pipe law). And the wall
flux is read off the *delivered water*: each parcel's heat loss in a segment, attributed to
its delivery bin, so a target shift can never extract more heat than the parcels carry. That
is what keeps the same-bin loop gain small and, decisively, *shrinking* as the bins narrow --
measured 0.26 at 6 h bins down to 0.014 at 10 min bins for a 100 mm line. A flux read from an
enthalpy budget instead is the more accurate one, and its gain runs the other way, to 0.98
and beyond, where this iteration stops converging to its own fixed point. The price of the
delivered-water flux is that the halo sees the wall-flux history smeared over one pipe
transit, and that error does *not* vanish as ``tedges`` is refined; the ``Validity`` notes
below put numbers on what is left. The
whole model stays linear in the produced water temperature and the surface temperatures, so
the reverse problem reuses the existing banded solver.

Units
-----

Everything closes in {m, day, K}; no energy unit appears. The two soil parameters are
``alpha``, the thermal diffusivity [m²/day], and ``kappa = k_soil / (rho c_w)``, the soil
conductivity divided by the volumetric heat capacity of *water* [m²/day]. The surface
coefficient is ``eta = h_s / (rho c_w)`` [m/day]. With ``rho c_w = 4.18e6 J/(m³ K)``:
``kappa = 0.0207 * k_soil[W/(m K)]``, ``eta = 0.0207 * h_s[W/(m² K)]``. Typical values:
grass ``kappa`` 0.02-0.03, ``alpha`` 0.04-0.06; paved ``kappa`` 0.02-0.04, ``alpha``
0.05-0.08 m²/day; pipe wall ``kappa_pipe`` 0.008 (PE), 0.0035 (PVC) m²/day; ``eta`` 0.41
m/day (``h_s`` = 20 W/(m² K)).

Available functions:

- :func:`sol_air_temperature` - The surface forcing per land cover: air temperature plus
  the absorbed solar radiation, less the longwave and latent losses, over the surface film.
- :func:`soil_temperature` - Exact bin-averaged undisturbed soil temperature at depth from
  a piecewise-constant surface series (Robin surface; a Dirichlet surface is ``eta=inf``).
- :func:`segment_heat_rate` - Per-segment exchange rate [1/day] from the wall and soil
  resistances, analogous to :func:`pipetransport.logremoval.segment_decay_rate`.
- :func:`source_to_endmember` - Delivered temperature at the reporting nodes, two-way by
  default; ``max_sweeps=1`` is the one-way model.
- :func:`endmember_to_source` - Reverse: production temperature from delivered
  temperatures, the same fixed point wrapped around the banded deconvolution.

The heat pair requires uniformly spaced ``tedges`` (the halo memory is a convolution), and
segments must carry ``length``, ``diameter`` and a ``cover`` land-class column; ``depth``
[m, to axis, default 1] and ``wall_thickness`` [m, with ``kappa_pipe``] are optional.

Validity
--------

- The halo is a *line* source read at the pipe wall. It is exact once the heat has
  diffused well past the pipe itself, and it overstates the deficit of the first lag bins
  before that, by up to about 9 % of the steady resistance around a Fourier number
  ``alpha dt / r_o**2`` of 0.3. The deficit is the correction term rather than the leading
  resistance, so the delivered temperature moves far less: a few hundredths of a kelvin
  for a service line on hourly bins, and up to a few tenths for a trunk main at sub-daily
  bins -- comparable there to the bin discretization itself. Replacing it with the
  constant-flux cylinder response would need numerical evaluation, which is why the
  closed form is kept. Its other cost is stiffness: the line source sends
  ``Dbar[0]/R_soil`` to 1 for a wide pipe on short bins (1.0000 for a 400 mm main at
  hourly bins), and that ratio is what sets the sweep count.
- **One wall-flux history per pipe.** The soil columns along a pipe are independent and the
  wall flux falls along it like ``exp(-h tau)`` -- by a factor 1.6 over a 2 h transit on a
  100 mm service line -- but the model gives every parcel in a pipe the same soil memory.
  That is the model's spatial resolution along a pipe, and it is a stated assumption rather
  than a parameter: under 24 h diurnal forcing at hourly bins it is worth around 0.5-0.9 K
  on a 100 mm line. Refining ``tedges`` does not reduce it; it is a property of the transit.
  The flux itself is read off the water the pipe delivers and attributed to the delivery
  bin, which additionally smears that history over one transit.

  Declare a pipe as a chain of shorter segments if you need it resolved -- splitting is
  exact for the transport, which is unchanged to round-off by it -- but check that the
  answer settles as you refine. It does under continuous flow; under intermittent demand it
  does not reliably converge, and refining can move the delivered temperature further
  outside the range of its inputs rather than closer to the truth.
- The relaxation target is an *effective driving temperature*, not a wall temperature. The
  rate keeps the steady soil resistance, which overstates the resistance while the halo is
  still developing, so the target has to be pushed past the undisturbed soil to reproduce
  the faster early exchange -- measured excursions of several times the driving contrast in
  the bins after a sharp change in the wall flux. The delivered temperature is a weighted
  average of ``tin`` and those targets, so it can leave the range of its own inputs, and
  there is no general bound on by how much. A step in ``tin`` into a *continuously flowing*
  pipe is the mild case and the one the figure usually quoted describes: 20 % of the
  instantaneous plant-to-soil contrast, 1.9 transits after the step, back inside the range
  after nine days, falling to 1-4 % once the pipe is split. **Intermittent demand is far
  worse and does not decay.** A line idle 16 h a day delivers water 66 % of the contrast past
  the soil, settling at a third of the contrast and recurring every duty cycle for as long as
  the duty cycle lasts; at 20-22 h idle it reaches 81-85 %. Splitting does not rescue it --
  on such a branch the criterion below usually declines to split at all, and where it does
  split the excursion can *grow* (47 % unsplit against 75 % at four pieces on one 12 h-idle
  line). Only ``max_sweeps=1`` is guaranteed inside the range of its inputs.
  For wide pipes there is a further regime in which this becomes unusable rather than merely
  large; see the bin-width note above.
- Bins in which a segment has no throughflow contribute zero wall flux to the halo, so the
  soil around a stagnant branch is treated as undisturbed while the water in it goes on
  relaxing exactly. Overnight stagnation of a service line reaches ``h tau`` of order 2,
  so the first water delivered afterwards meets a halo the model has not built.
- The bin width must resolve the forcing: a signal varying within a few bins is carried by
  the transport operator exactly but enters the flux memory only as its bin average.
- The soil around each segment is its own set of independent radial columns -- axial
  conduction over a month reaches about a metre against variation scales of hundreds of
  metres -- and segments do not share a halo, so parallel mains in one trench are outside
  the model.

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import fftconvolve
from scipy.special import erfc, erfcx, exp1

from pipetransport._transfer import NetworkTransfer, apply_segment_targets, paths_transfer, resolve_spinup
from pipetransport._validation import _validate_no_nan, _validate_positive, _validate_tedges
from pipetransport.network import PipeNetwork  # noqa: TC001 -- runtime dependency of the signatures
from pipetransport.utils import solve_inverse_transport_banded, tedges_to_days


def _step_response_integral(
    lag: npt.NDArray[np.floating], *, depth: float, alpha: float, radiation_length: float
) -> npt.NDArray[np.floating]:
    """Time integral of the Robin-surface step response at depth, elementwise in the lag.

    The half space is initially uniform; at ``lag = 0`` the sol-air temperature steps by
    one unit behind the surface film. The response at depth ``z`` is
    ``S(t) = erfc(u) - exp(-u**2) erfcx(u + sqrt(alpha t)/rl)`` with
    ``u = z / (2 sqrt(alpha t))`` and ``rl = kappa/eta`` the radiation length, and its
    integral is closed-form (the ``erfcx`` arrangement is what keeps every factor bounded;
    the textbook ``exp(...) * erfc(...)`` form overflows beyond a few weeks). At
    ``rl = 0`` both reduce exactly to the Dirichlet half-space forms -- the division by
    zero lands on ``erfcx(inf) = 0`` -- so there is no separate code path.

    Parameters
    ----------
    lag : ndarray
        Time since the step [days], any shape. Non-positive lags return 0.
    depth : float
        Depth below the surface [m], positive.
    alpha : float
        Thermal diffusivity of the soil [m²/day], positive.
    radiation_length : float
        ``kappa / eta`` [m], non-negative; 0 is a prescribed-temperature surface.

    Returns
    -------
    ndarray
        ``integral_0^lag S(s) ds`` [days], same shape as ``lag``, non-negative.
    """
    lag = np.asarray(lag, dtype=float)
    out = np.zeros_like(lag)
    m = lag > 0.0
    t = lag[m]
    z, rl = depth, radiation_length
    root = np.sqrt(alpha * t)
    u = z / (2.0 * root)
    gauss = np.exp(-np.square(u))
    with np.errstate(divide="ignore"):
        tail = erfcx(u + root / rl)
    out[m] = (
        (t + ((z + rl) ** 2 + rl**2) / (2.0 * alpha)) * erfc(u)
        - (z + 2.0 * rl) / np.sqrt(np.pi * alpha) * np.sqrt(t) * gauss
        - rl**2 / alpha * gauss * tail
    )
    # The two exponentially small terms share their leading asymptotics deep in the tail,
    # and their float difference can land a few units below zero at the subnormal floor.
    return np.maximum(out, 0.0)


def _halo_integral(c: npt.NDArray[np.floating], lag: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """``integral_0^lag E1(c/s) ds`` elementwise: ``(lag + c) E1(c/lag) - lag exp(-c/lag)``.

    ``E1`` underflows cleanly to zero for large arguments, so short lags (down to and
    including 0, where ``c/0 = inf``) return exactly 0 without special-casing.

    Parameters
    ----------
    c : ndarray
        Diffusion time scale ``r**2 / (4 alpha)`` [days], positive; broadcasts against
        ``lag``.
    lag : ndarray
        Time since the flux step [days]; non-positive lags return 0.

    Returns
    -------
    ndarray
        The integral [days / (resistance unit)], broadcast shape of ``c`` and ``lag``.
    """
    lag = np.asarray(lag, dtype=float)
    c, lag = np.broadcast_arrays(np.asarray(c, dtype=float), lag)
    out = np.zeros(lag.shape)
    m = lag > 0.0
    with np.errstate(divide="ignore", over="ignore"):
        x = c[m] / lag[m]
        out[m] = (lag[m] + c[m]) * exp1(x) - lag[m] * np.exp(-x)
    return out


def _deficit_kernel(
    n_bins: int,
    dt_days: float,
    *,
    r_o: npt.NDArray[np.floating],
    d_eff: npt.NDArray[np.floating],
    alpha: npt.NDArray[np.floating],
    kappa: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Bin-averaged transient deficit ``Dbar[m]`` of every segment, shape ``(n_seg, n_bins)``.

    The wall-temperature step response of a continuous line source at the outer pipe radius,
    with its mirror image at ``2 d_eff``, is ``G(t) = [E1(c_o/t) - E1(c_i/t)]/(4 pi kappa)``,
    saturating at the steady buried-pipe resistance ``R_inf = ln(2 d_eff/r_o)/(2 pi kappa)``.
    The deficit ``D = R_inf - G`` is what has *not yet arrived*; its exact bin average over
    lag bin ``m`` comes from the closed-form time integral of ``G``.

    Parameters
    ----------
    n_bins : int
        Number of lag bins.
    dt_days : float
        Bin width [days].
    r_o, d_eff, alpha, kappa : ndarray
        Outer radius [m], effective burial depth [m], diffusivity and conductivity ratio
        [m²/day] of every segment, length ``n_seg``.

    Returns
    -------
    ndarray
        ``Dbar`` [day/m²] per segment and lag bin; ``Dbar[:, 0]`` is ``R_inf - Gbar(dt)``.
    """
    c_o = (r_o**2 / (4.0 * alpha))[:, None]
    c_i = ((2.0 * d_eff) ** 2 / (4.0 * alpha))[:, None]
    r_inf = (np.log(2.0 * d_eff / r_o) / (2.0 * np.pi * kappa))[:, None]
    lag = dt_days * np.arange(n_bins + 1)[None, :]
    cumulative = r_inf * lag - (_halo_integral(c_o, lag) - _halo_integral(c_i, lag)) / (4.0 * np.pi * kappa[:, None])
    return np.diff(cumulative, axis=1) / dt_days


def sol_air_temperature(
    *,
    air_temperature: npt.ArrayLike,
    solar_irradiance: npt.ArrayLike,
    absorptivity: float,
    eta: float,
    heat_loss: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute the sol-air temperature: the surface forcing including absorbed solar radiation.

    ``T_sa = T_air + (absorptivity * solar_irradiance - heat_loss) / eta``. A sunlit
    surface acts as if the air were warmer by the absorbed shortwave flux over the surface
    film conductance -- much warmer over asphalt (absorptivity ~0.9) than over grass.

    ``heat_loss`` collects the fluxes that cool the surface relative to the air, and it is
    not optional for a summer study. Longwave exchange with a clear sky costs
    ``emissivity * 63 W/m²`` for a horizontal surface (about 2.8 K at ``eta = 0.41``), and
    evapotranspiration removes 100 W/m² and more from vegetation (about 5 K) while a paved
    surface loses none -- which is the greater part of why pavement runs hotter than grass,
    more than the absorptivity contrast is.

    Parameters
    ----------
    air_temperature : array-like
        Air temperature [K or °C], any shape.
    solar_irradiance : array-like
        Shortwave irradiance normalized by the volumetric heat capacity of water
        [K m/day]: multiply W/m² by ``86400 / 4.18e6 = 0.0207``.
    absorptivity : float
        Shortwave absorptivity of the surface [-], non-negative.
    eta : float
        Surface film coefficient ``h_s / (rho c_w)`` [m/day], positive: multiply
        W/(m² K) by 0.0207. ``inf`` is the prescribed-temperature surface, as elsewhere in
        this module, and returns the air temperature unchanged.
    heat_loss : array-like, optional
        Longwave and latent flux leaving the surface, in the same normalized unit as
        ``solar_irradiance`` [K m/day], broadcast against it. Default 0.0, which models a
        dry surface under an overcast sky and biases the surface warm.

    Returns
    -------
    ndarray
        Sol-air temperature, in the unit of ``air_temperature``.

    Raises
    ------
    ValueError
        If ``eta`` is not positive or ``absorptivity`` is negative.

    See Also
    --------
    soil_temperature : Propagates this forcing down to pipe depth.

    Examples
    --------
    Midday over asphalt (300 W/m² absorbed at 0.9, clear-sky longwave only) against grass,
    which also transpires 120 W/m²:

    >>> from pipetransport.heat import sol_air_temperature
    >>> sol_air_temperature(
    ...     air_temperature=25.0,
    ...     solar_irradiance=6.2,
    ...     absorptivity=0.9,
    ...     eta=0.41,
    ...     heat_loss=1.17,
    ... ).round(1)
    np.float64(35.8)
    >>> sol_air_temperature(
    ...     air_temperature=25.0,
    ...     solar_irradiance=6.2,
    ...     absorptivity=0.7,
    ...     eta=0.41,
    ...     heat_loss=1.17 + 2.48,
    ... ).round(1)
    np.float64(26.7)
    """
    # ``inf`` is the prescribed-temperature surface here too, as it is in soil_temperature and
    # segment_heat_rate: an infinitely conductive film pins the surface to the air temperature,
    # which is what dividing the absorbed flux by it gives. NaN still fails the comparison.
    if not eta > 0.0:
        msg = "eta must be positive (inf is a prescribed-temperature surface)"
        raise ValueError(msg)
    if not absorptivity >= 0.0:
        msg = "absorptivity must be non-negative"
        raise ValueError(msg)
    air = np.asarray(air_temperature, dtype=float)
    absorbed = absorptivity * np.asarray(solar_irradiance, dtype=float)
    return air + (absorbed - np.asarray(heat_loss, dtype=float)) / eta


def soil_temperature(
    *,
    surface_temperature: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    depth: float,
    alpha: float,
    kappa: float | None = None,
    eta: float | None = None,
    t_pre: float | None = None,
    surface_tedges: pd.DatetimeIndex | None = None,
) -> npt.NDArray[np.floating]:
    """Compute the bin-averaged undisturbed soil temperature at depth from a surface series.

    Exact superposition of half-space step responses: the piecewise-constant surface
    (sol-air) series steps at each of its bin edges, each step arrives at depth as an
    erfc-family response, and the average over every output bin uses the closed-form time
    integral -- no quadrature, no grid. Before ``surface_tedges[0]`` the soil is uniformly
    at ``t_pre``; after the last surface bin the series holds its final value. A one-week
    heatwave arrives at 1 m attenuated to roughly a quarter to a third of its surface
    amplitude, peaking a day or two after it ends; the annual wave arrives at about
    two-thirds amplitude, three to four weeks delayed.

    Parameters
    ----------
    surface_temperature : array-like
        Sol-air temperature per surface bin (see :func:`sol_air_temperature`), length
        ``len(surface_tedges) - 1``.
    tedges : pandas.DatetimeIndex
        Output bin edges; may start before ``surface_tedges[0]`` (those bins return
        ``t_pre``) and end after its last edge.
    depth : float
        Depth below the surface [m], positive.
    alpha : float
        Soil thermal diffusivity [m²/day], positive.
    kappa, eta : float or None, optional
        Soil conductivity ratio [m²/day] and surface film coefficient [m/day]. Given
        together they impose the physical Robin surface condition with radiation length
        ``kappa/eta``; both ``None`` (default, equivalent to ``eta=inf``) is a
        prescribed-temperature (Dirichlet) surface.
    t_pre : float or None, optional
        Uniform soil temperature before the surface record. Defaults to the first surface
        value.
    surface_tedges : pandas.DatetimeIndex or None, optional
        Bin edges of the surface series; defaults to ``tedges``. Supply a record starting
        well before the period of interest -- after one year roughly 87 % of a step has
        arrived at 1 m, so the first months of output lean on ``t_pre``.

    Returns
    -------
    ndarray
        Soil temperature at ``depth``, one bin-average per ``tedges`` bin, in the unit of
        ``surface_temperature``.

    Raises
    ------
    ValueError
        If a time axis is malformed, the series holds NaN, a parameter is not positive,
        or only one of ``kappa`` and ``eta`` is given.

    See Also
    --------
    source_to_endmember : Uses this field, per land cover, as the relaxation target.

    Examples
    --------
    A 10-degree step at the surface, read at 1 m depth: 87 % arrived after a year.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pipetransport.heat import soil_temperature
    >>> tedges = pd.date_range("2025-01-01", periods=366, freq="D")
    >>> t_soil = soil_temperature(
    ...     surface_temperature=np.full(365, 20.0),
    ...     tedges=tedges,
    ...     depth=1.0,
    ...     alpha=0.05,
    ...     t_pre=10.0,
    ... )
    >>> float(t_soil[0].round(2)), float(t_soil[-1].round(1))
    (10.0, 18.7)
    """
    tedges = pd.DatetimeIndex(tedges)
    surface = np.asarray(surface_temperature, dtype=float)
    surface_tedges = tedges if surface_tedges is None else pd.DatetimeIndex(surface_tedges)
    _validate_tedges(surface_tedges, surface, tedges_name="surface_tedges", values_name="surface_temperature")
    _validate_tedges(tedges, np.empty(len(tedges) - 1), tedges_name="tedges", values_name="the output")
    _validate_no_nan(surface, name="surface_temperature")
    _validate_positive(depth, name="depth")
    _validate_positive(alpha, name="alpha")
    if (kappa is None) != (eta is None):
        msg = "kappa and eta displace the surface together; provide both or neither"
        raise ValueError(msg)
    radiation_length = 0.0
    if kappa is not None and eta is not None:
        _validate_positive(kappa, name="kappa")
        if not eta > 0.0:
            msg = "eta must be positive"
            raise ValueError(msg)
        radiation_length = kappa / eta
    pre = float(surface[0]) if t_pre is None else float(t_pre)

    step_times = tedges_to_days(surface_tedges)[:-1]
    steps = np.diff(surface, prepend=pre)
    out_edges = tedges_to_days(tedges, ref=surface_tedges[0])
    # Cumulative response integral at every output edge. The (edges x steps) kernel matrix
    # is built in bounded chunks so a year of hourly data does not materialize N^2 floats.
    cumulative = np.empty(len(out_edges))
    chunk = max(1, 8_388_608 // max(len(step_times), 1))
    for lo in range(0, len(out_edges), chunk):
        lags = out_edges[lo : lo + chunk, None] - step_times[None, :]
        kernel = _step_response_integral(lags, depth=depth, alpha=alpha, radiation_length=radiation_length)
        cumulative[lo : lo + chunk] = kernel @ steps
    return pre + np.diff(cumulative) / np.diff(out_edges)


def segment_heat_rate(
    *,
    network: PipeNetwork,
    kappa: float | pd.Series,
    depth: float | pd.Series = 1.0,
    eta: float | pd.Series | None = None,
    kappa_pipe: float | pd.Series | None = None,
    film_coefficient: float | pd.Series | None = None,
) -> pd.Series:
    """Compute the per-segment heat exchange rate [1/day] from the wall and soil resistances.

    The water column loses heat through the water-side film, the pipe wall and the soil in
    series; dividing the conductance per length by the water's heat capacity per length
    gives the first-order relaxation rate

    ``h = 1 / ((R_film + R_wall + R_soil) * pi * r_i**2)``, with
    ``R_film = 1 / (2 pi r_i * film_coefficient)``,
    ``R_wall = ln(r_o/r_i) / (2 pi kappa_pipe)`` and
    ``R_soil = ln(2 d_eff / r_o) / (2 pi kappa)`` (``d_eff = depth + kappa/eta``).

    ``R_soil`` is the steady buried-pipe resistance -- the fully developed halo, whose
    saturation the mirror image above the surface provides. The rate inherits the
    ``1/(D**2 ln D)`` diameter law: a 100 mm service line relaxes an order of magnitude
    faster than a 400 mm trunk main.

    Parameters
    ----------
    network : PipeNetwork
        Network whose segments carry a ``"diameter"`` column [m] (inner diameter), and a
        ``"wall_thickness"`` column [m] when ``kappa_pipe`` is given.
    kappa : float or pandas.Series
        Soil conductivity over the water heat capacity [m²/day], per segment or shared.
    depth : float or pandas.Series, optional
        Burial depth to the pipe axis [m]. Default 1.0.
    eta : float or pandas.Series or None, optional
        Surface film coefficient [m/day]. ``None`` (default) or ``inf`` is a
        prescribed-temperature surface (no displacement).
    kappa_pipe : float or pandas.Series or None, optional
        Pipe wall conductivity over the water heat capacity [m²/day]. ``None`` (default)
        omits the wall term -- the bare-pipe limit, which also reads the soil resistance from
        ``r_i`` rather than ``r_o``. PE ~0.008, PVC ~0.0035 m²/day. At a fixed SDR the wall
        resistance ``ln(SDR/(SDR - 2)) / (2 pi kappa_pipe)`` does not depend on the diameter
        while ``R_soil`` falls with it, so the wall's share *grows* with the pipe rather than
        shrinking: at ``kappa = 0.025`` and ``depth = 1`` it is 10-17 % of ``R_soil`` for PE
        SDR17 and 18-30 % for PVC SDR21 across 100-400 mm, lowering the rate by 6-10 % and
        13-20 % respectively. Both shares scale with ``kappa / kappa_pipe``, so rescale them
        for your own soil; PVC is never the ~10 % that PE is.
    film_coefficient : float or pandas.Series or None, optional
        Water-side film coefficient ``h_film / (rho c_w)`` [m/day]. ``None`` (default)
        assumes the film is not limiting. Like the mass transfer coefficient of
        :func:`pipetransport.logremoval.segment_decay_rate` it depends on velocity, so pass
        a value representative of the operating range. It is negligible for turbulent trunk
        mains (well under 1 %) but not at the low night-time flows of a service line: fully
        developed laminar flow (``Nu = 3.66``, ``h_film = 3.66 k_water / D``, taking
        ``k_water = 0.6 W/(m K)``) gives ``film_coefficient = 0.454 m/day`` in a 100 mm pipe,
        about 29 % of its soil resistance at ``kappa = 0.025`` and ``depth = 1`` -- 24 % at
        ``kappa = 0.02``, 35 % at ``kappa = 0.03``.

    Returns
    -------
    pandas.Series
        Exchange rate [1/day] indexed by segment name.

    Raises
    ------
    ValueError
        If a required column is missing, a Series misses a segment, a parameter is not
        positive, or the geometry gives ``2 d_eff <= r_o``.

    See Also
    --------
    pipetransport.logremoval.segment_decay_rate : The chlorine analogue of this helper.
    source_to_endmember : Consumes these rates internally.

    Examples
    --------
    Small pipes equilibrate much faster: a 100 mm service line runs a tenfold higher rate
    than the 400 mm trunk main.

    >>> from pipetransport.examples import example_network
    >>> from pipetransport.heat import segment_heat_rate
    >>> network = example_network()
    >>> rates = segment_heat_rate(network=network, kappa=0.025, depth=1.0, eta=0.41)
    >>> float(rates["C-T4"].round(2)), float(rates["Plant-A"].round(2))
    (5.34, 0.53)
    """
    segments = network.segments
    if "diameter" not in segments.columns:
        msg = "the heat exchange rate needs the segment diameter; build the network from length and diameter"
        raise ValueError(msg)

    def per_segment(value: float | pd.Series, name: str) -> npt.NDArray[np.floating]:
        if isinstance(value, pd.Series):
            missing = [seg for seg in segments.index if seg not in value.index]
            if missing:
                msg = f"{name} is missing segment(s): {missing}"
                raise ValueError(msg)
            return value.reindex(segments.index).to_numpy(dtype=float)
        return np.full(len(segments), float(value))

    kappa_seg = per_segment(kappa, "kappa")
    depth_seg = per_segment(depth, "depth")
    _validate_positive(kappa_seg, name="kappa")
    _validate_positive(depth_seg, name="depth")
    eta_seg = None if eta is None else per_segment(eta, "eta")
    if eta_seg is not None and not np.all((eta_seg > 0.0) | np.isposinf(eta_seg)):
        msg = "eta must be positive (inf is a prescribed-temperature surface)"
        raise ValueError(msg)

    r_i = segments["diameter"].to_numpy(dtype=float) / 2.0
    wall_resistance = np.zeros(len(segments))
    r_o = r_i
    if kappa_pipe is not None:
        if "wall_thickness" not in segments.columns:
            msg = "kappa_pipe needs the segment wall thickness; add a 'wall_thickness' column [m]"
            raise ValueError(msg)
        kappa_pipe_seg = per_segment(kappa_pipe, "kappa_pipe")
        _validate_positive(kappa_pipe_seg, name="kappa_pipe")
        thickness = segments["wall_thickness"].to_numpy(dtype=float)
        _validate_positive(thickness, name="wall_thickness")
        r_o = r_i + thickness
        wall_resistance = np.log(r_o / r_i) / (2.0 * np.pi * kappa_pipe_seg)

    d_eff = depth_seg if eta_seg is None else depth_seg + kappa_seg / eta_seg
    if not np.all(2.0 * d_eff > r_o):
        msg = "burial depth must exceed the pipe radius (2 * d_eff > r_o); the line-source geometry needs d >> r"
        raise ValueError(msg)
    film_resistance = np.zeros(len(segments))
    if film_coefficient is not None:
        film_seg = per_segment(film_coefficient, "film_coefficient")
        _validate_positive(film_seg, name="film_coefficient")
        film_resistance = 1.0 / (2.0 * np.pi * r_i * film_seg)
    soil_resistance = np.log(2.0 * d_eff / r_o) / (2.0 * np.pi * kappa_seg)
    total = film_resistance + wall_resistance + soil_resistance
    return pd.Series(1.0 / (total * np.pi * r_i**2), index=segments.index, name="heat_rate")


class _HeatSystem(NamedTuple):
    """Everything the Picard loop reads, built once per call.

    Every per-segment array is indexed by the user's own pipes: one row each, carrying one
    wall-flux history, which is the model's spatial resolution along a pipe.
    """

    nodes: tuple[str, ...]
    n_pad: int
    n_bins: int
    t_inf: npt.NDArray[np.floating]
    dbar: npt.NDArray[np.floating]
    seg_flow: npt.NDArray[np.floating]
    length: npt.NDArray[np.floating]
    internal: NetworkTransfer
    reporting: NetworkTransfer


def _apply(transfer: NetworkTransfer, values: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Apply a banded operator to a padded input series, rows in band layout.

    Parameters
    ----------
    transfer : NetworkTransfer
        Operator whose rows are to be applied.
    values : ndarray
        Input series on the operator's padded input grid.

    Returns
    -------
    ndarray
        One value per operator row and output bin, shape ``(n_rows, n_cout)``.
    """
    columns = np.clip(transfer.col_start[..., None] + np.arange(transfer.band_vals.shape[-1]), 0, len(values) - 1)
    return np.einsum("nkb,nkb->nk", transfer.band_vals, values[columns])


def _pad_paths(chains: list[npt.NDArray[np.intp]]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
    """Stack ragged path chains into the padded index matrix the operator builder takes.

    Parameters
    ----------
    chains : list of ndarray
        Segment rows of each path, source outward; may differ in length and may be empty
        (a row reporting at the source itself).

    Returns
    -------
    paths_idx : ndarray of intp
        Segment row of each path step, shape ``(len(chains), max_depth)``.
    active : ndarray of bool
        Which slots of ``paths_idx`` are real path steps.
    """
    lengths = np.array([chain.size for chain in chains], dtype=np.intp)
    max_depth = int(lengths.max(initial=0))
    active = np.arange(max_depth) < lengths[:, None]
    paths_idx = np.zeros((len(chains), max_depth), dtype=np.intp)
    if max_depth:
        paths_idx[active] = np.concatenate(chains)
    return paths_idx, active


def _build_system(
    *,
    flow: npt.ArrayLike | pd.DataFrame | dict,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    network: PipeNetwork,
    soil: pd.DataFrame,
    surface_temperature: pd.DataFrame,
    surface_tedges: pd.DatetimeIndex | None,
    nodes: list[str] | tuple[str, ...] | None,
    kappa_pipe: float | pd.Series | None,
    film_coefficient: float | pd.Series | None,
    spinup: str | None,
) -> _HeatSystem:
    """Validate the shared inputs and build the operators, targets and kernels once.

    Returns
    -------
    _HeatSystem
        Everything the fixed-point iteration and both public directions read.

    Raises
    ------
    ValueError
        If a time axis is malformed or non-uniform, a required segment or soil column is
        missing, a cover class is unmapped, or a requested node is not part of the network.
    """
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    demand = network.flow_array(flow)
    _validate_tedges(tedges, demand, tedges_name="tedges", values_name="flow")
    _validate_tedges(cout_tedges, np.empty(len(cout_tedges) - 1), tedges_name="cout_tedges", values_name="tout")
    if np.unique(np.diff(tedges.asi8)).size != 1:
        msg = "tedges must be uniformly spaced for the heat pair (the halo memory is a convolution over lag bins)"
        raise ValueError(msg)

    segments = network.segments
    for column in ("length", "diameter", "cover"):
        if column not in segments.columns:
            msg = f"the heat pair needs the segment column {column!r}"
            raise ValueError(msg)
    covers = segments["cover"]
    soil_missing = [c for c in covers.unique() if c not in soil.index]
    if soil_missing:
        msg = f"soil is missing cover class(es): {soil_missing}"
        raise ValueError(msg)
    for column in ("alpha", "kappa", "eta"):
        if column not in soil.columns:
            msg = f"soil needs the column {column!r} (eta=inf is a prescribed-temperature surface)"
            raise ValueError(msg)
    surface_missing = [c for c in covers.unique() if c not in surface_temperature.columns]
    if surface_missing:
        msg = f"surface_temperature is missing cover class(es): {surface_missing}"
        raise ValueError(msg)

    requested = tuple(network.endmembers) if nodes is None else tuple(nodes)
    unknown = [node for node in requested if node not in network.paths]
    if unknown:
        msg = f"unknown node(s): {unknown}; network nodes are {list(network.nodes)}"
        raise ValueError(msg)

    alpha_seg = soil["alpha"].reindex(covers).to_numpy(dtype=float)
    kappa_seg = soil["kappa"].reindex(covers).to_numpy(dtype=float)
    eta_seg = soil["eta"].reindex(covers).to_numpy(dtype=float)
    depth_seg = segments["depth"].to_numpy(dtype=float) if "depth" in segments.columns else np.ones(len(segments))
    kappa_series = pd.Series(kappa_seg, index=segments.index)
    depth_series = pd.Series(depth_seg, index=segments.index)
    eta_series = pd.Series(eta_seg, index=segments.index)
    _validate_positive(alpha_seg, name="soil alpha")

    # Spin-up, exactly as network_transfer resolves it: each endmember's travel time at the
    # leading flow rate, with resolve_spinup dropping the paths it cannot warm-start one by
    # one, so a single stagnant or unreachably deep branch does not suppress the padding of
    # the whole call. Every internal node sits on an endmember path, so those paths bound the
    # internal rows too -- and the candidate list is the endmembers whatever ``nodes`` asks
    # for, so no row's coverage depends on which nodes were requested.
    volume = segments["volume"].to_numpy(dtype=float)
    seg_of = {name: e for e, name in enumerate(segments.index)}
    with np.errstate(divide="ignore"):
        ratio = volume / network.segment_flow(flow=demand)[:, 0]
    end_paths, end_active = _pad_paths([
        np.array([seg_of[name] for name in network.paths[node]], dtype=np.intp) for node in network.endmembers
    ])
    per_path = np.sum(np.where(end_active, ratio[end_paths], 0.0), axis=1)
    tedges_p, demand_p, n_pad = resolve_spinup(spinup, tedges=tedges, flow=demand, warm_start_days=per_path)

    tedges_days = tedges_to_days(tedges_p)
    cout_days = tedges_to_days(cout_tedges, ref=tedges_p[0])
    n_bins = len(tedges_days) - 1
    dt_days = float(tedges_days[1] - tedges_days[0])
    seg_flow = network.segment_flow(flow=demand_p)

    # Undisturbed soil temperature per segment. The field depends on the cover class and
    # the burial depth only, and a network normally holds a handful of distinct pairs
    # against many segments, so it is solved once per pair and broadcast.
    t_inf = np.empty((len(segments), n_bins))
    cover_names = covers.to_numpy()
    pairs = {(str(cover), float(depth)) for cover, depth in zip(cover_names, depth_seg, strict=True)}
    surface_grid = tedges if surface_tedges is None else pd.DatetimeIndex(surface_tedges)
    for cover, depth in pairs:
        eta_cover = float(soil.loc[cover, "eta"])
        rows = (cover_names == cover) & (depth_seg == depth)
        t_inf[rows] = soil_temperature(
            surface_temperature=surface_temperature[cover].to_numpy(dtype=float),
            tedges=tedges_p,
            depth=depth,
            alpha=float(soil.loc[cover, "alpha"]),
            kappa=None if np.isposinf(eta_cover) else float(soil.loc[cover, "kappa"]),
            eta=None if np.isposinf(eta_cover) else eta_cover,
            surface_tedges=surface_grid,
        )

    rate = segment_heat_rate(
        network=network,
        kappa=kappa_series,
        depth=depth_series,
        eta=eta_series,
        kappa_pipe=kappa_pipe,
        film_coefficient=film_coefficient,
    ).to_numpy(dtype=float)
    thickness = segments["wall_thickness"].to_numpy(dtype=float) if kappa_pipe is not None else np.zeros(len(segments))
    r_o = segments["diameter"].to_numpy(dtype=float) / 2.0 + thickness
    d_eff = depth_seg + np.where(np.isposinf(eta_seg), 0.0, kappa_seg / eta_seg)
    dbar = _deficit_kernel(n_bins, dt_days, r_o=r_o, d_eff=d_eff, alpha=alpha_seg, kappa=kappa_seg)

    n_seg = len(segments)

    def chain(node: str) -> npt.NDArray[np.intp]:
        """Segment path from the source to ``node``.

        Returns
        -------
        ndarray of intp
            Segment rows, source outward; empty for the source node itself.
        """
        return np.array([seg_of[name] for name in network.paths[node]], dtype=np.intp)

    # Internal rows, two per segment and both binned on that pipe's own deliveries: the
    # temperature it delivers, and the same water's temperature where it entered -- the same
    # path with the last step replaced by an inert (zero-rate) phantom copy of the pipe,
    # which carries the parcel across without exchanging. Their difference is the per-parcel
    # heat lost in the pipe, attributed to the delivery bin. The phantom copies live as extra
    # segment rows sharing the real hydraulics, so the ordinary machinery builds these rows
    # unchanged.
    entry, delivery = [], []
    for e, name in enumerate(segments.index):
        path = np.concatenate([chain(str(segments.loc[name, "from"])), [e]]).astype(np.intp)
        entry.append(np.concatenate([path[:-1], [n_seg + e]]).astype(np.intp))
        delivery.append(path)
    int_paths, int_active = _pad_paths(entry + delivery)
    rep_paths, rep_active = _pad_paths([chain(node) for node in requested])
    rep_flow = network.node_flow(flow=demand_p, nodes=requested)

    def build(
        paths_idx: npt.NDArray[np.intp],
        active: npt.NDArray[np.bool_],
        node_flow: npt.NDArray[np.floating],
        cout: npt.NDArray[np.floating],
    ) -> NetworkTransfer:
        return paths_transfer(
            tedges_days=tedges_days,
            cout_tedges_days=cout,
            segment_volume=np.tile(volume, 2),
            segment_flow=np.vstack([seg_flow, seg_flow]),
            segment_decay=np.concatenate([rate, np.zeros(n_seg)]),
            node_flow=node_flow,
            paths_idx=paths_idx,
            active=active,
            with_target_terms=True,
        )

    return _HeatSystem(
        nodes=requested,
        n_pad=n_pad,
        n_bins=n_bins,
        t_inf=t_inf,
        dbar=dbar,
        seg_flow=seg_flow,
        length=segments["length"].to_numpy(dtype=float),
        internal=build(int_paths, int_active, np.vstack([seg_flow, seg_flow]), tedges_days),
        reporting=build(rep_paths, rep_active, rep_flow, cout_days),
    )


def _extended(targets: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Pad targets with zero rows for the inert phantom segments of the operators.

    Parameters
    ----------
    targets : ndarray
        Per-segment relaxation targets, shape ``(n_seg, n_bins)``.

    Returns
    -------
    ndarray
        Targets for the ``2 * n_seg`` segment rows the operators are built on.
    """
    return np.vstack([targets, np.zeros_like(targets)])


def _converge_targets(
    system: _HeatSystem,
    tin_padded: npt.NDArray[np.floating],
    *,
    max_sweeps: int,
    atol: float,
    initial: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """Iterate the relaxation targets to their fixed point for one source series.

    Sweep 1 is the undisturbed soil -- the one-way model -- and each further sweep replaces
    the halo with the one implied by the latest flux history. The map contracts by the
    deficit share of the resistance, so the iterate converges geometrically.

    Parameters
    ----------
    system : _HeatSystem
        Prebuilt operators, kernels and undisturbed field.
    tin_padded : ndarray
        Source temperature on the padded input grid.
    max_sweeps : int
        Iteration cap; 1 returns the undisturbed field unchanged.
    atol : float
        Tolerance on the target increment, absolute, in the unit of the temperatures.
    initial : ndarray or None, optional
        Starting targets. ``None`` (default) starts from the undisturbed field, which is
        the one-way model; the reverse direction warm-starts from its previous outer
        iterate instead, which costs a handful of sweeps rather than a full run.

    Returns
    -------
    ndarray
        Per-segment relaxation targets of shape ``(n_seg, n_bins)``.

    Raises
    ------
    RuntimeError
        If the increment is still above the tolerance when the cap is reached.
    """
    if max_sweeps == 1:
        return system.t_inf
    # An absolute tolerance, not a relative one. Normalising the increment by the iterate's
    # own scale loosens the test exactly when the iteration is misbehaving -- a state wrong by
    # 1e20 K passes a relative test on itself -- and it makes the answer depend on whether the
    # caller works in Celsius or in kelvin, which an affine model must not.
    targets = system.t_inf if initial is None else initial
    for _ in range(max_sweeps - 1):
        updated = _update_targets(system, _internal_pass(system, tin_padded, targets))
        increment = float(np.max(np.abs(updated - targets)))
        targets = updated
        if increment <= atol:
            return targets
    msg = f"the two-way fixed point did not converge within max_sweeps={max_sweeps}; raise max_sweeps or atol"
    raise RuntimeError(msg)


def _internal_pass(
    system: _HeatSystem, tin_padded: npt.NDArray[np.floating], targets: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Compute segment entry and delivery temperatures, NaN where the record does not constrain them.

    Parameters
    ----------
    system : _HeatSystem
        Prebuilt operators and kernels.
    tin_padded : ndarray
        Source temperature on the padded input grid.
    targets : ndarray
        Current per-segment relaxation targets.

    Returns
    -------
    ndarray
        Entry temperatures in the first ``n_seg`` rows and delivery temperatures in the
        rest, shape ``(2 * n_seg, n_bins)``.
    """
    t_int = _apply(system.internal, tin_padded) + apply_segment_targets(system.internal, _extended(targets))
    t_int[~system.internal.valid_out] = np.nan
    return t_int


def _update_targets(system: _HeatSystem, t_int: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """One flux-and-halo pass: internal temperatures -> new per-segment targets.

    The wall flux of a segment is its throughflow times the entry-to-delivery temperature
    drop of the water it delivers (the entry reading is the inert-copy row). Attributing
    each parcel's heat loss to its delivery bin keeps the response passive -- a target
    shift can never extract more heat than the parcels carry, since
    ``1 - exp(-h tau) <= h tau`` -- which caps the same-bin loop gain at the deficit
    share of the resistance and makes the iteration contract for any transit; the cost is
    a timing skew of at most one transit in the halo memory. Bins without a defined
    budget (spin-up edge, no throughflow) contribute zero flux: the undisturbed-soil
    assumption applied at bin resolution.

    Parameters
    ----------
    system : _HeatSystem
        Prebuilt operators and kernels.
    t_int : ndarray
        Entry and delivery temperatures from :func:`_internal_pass`.

    Returns
    -------
    ndarray
        Updated per-segment relaxation targets, shape ``(n_seg, n_bins)``.
    """
    n_seg = len(system.length)
    t_entry, t_down = t_int[:n_seg], t_int[n_seg:]
    with np.errstate(invalid="ignore"):
        psi = system.seg_flow * (t_entry - t_down) / system.length[:, None]
    psi = np.where(np.isfinite(psi), psi, 0.0)
    # The soil is undisturbed before the user's record: the warm-start prefix is a
    # fabricated hydraulic history, not a flux history, so it feeds the halo nothing.
    psi[:, : system.n_pad] = 0.0
    dpsi = np.diff(psi, axis=1, prepend=0.0)
    halo = fftconvolve(dpsi, system.dbar, mode="full", axes=1)[:, : system.n_bins]
    return system.t_inf - halo


def source_to_endmember(
    *,
    tin: npt.ArrayLike,
    flow: npt.ArrayLike | pd.DataFrame | dict,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    network: PipeNetwork,
    soil: pd.DataFrame,
    surface_temperature: pd.DataFrame,
    surface_tedges: pd.DatetimeIndex | None = None,
    nodes: list[str] | tuple[str, ...] | None = None,
    kappa_pipe: float | pd.Series | None = None,
    film_coefficient: float | pd.Series | None = None,
    max_sweeps: int = 5000,
    atol: float = 1e-9,
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """Compute the delivered water temperature at each reporting node.

    The delivered temperature is the transport operator's affine reading
    ``W @ tin + b``: every parcel relaxes toward the soil temperature around each pipe it
    crosses, at that pipe's exchange rate. The relaxation target is the undisturbed soil
    temperature at pipe depth (from the land-cover surface forcing) shifted by the halo the
    network's own heat flux has built up -- a fixed point that is found by iterating a
    transport pass, a delivered-water flux pass and one convolution per segment.
    ``max_sweeps=1`` skips the coupling entirely and is exactly the classical one-way
    model (steady buried-pipe resistance, undisturbed soil).

    Parameters
    ----------
    tin : array-like
        Temperature of the produced water leaving the source, constant over each
        ``tedges`` bin. Length ``len(tedges) - 1``. Any temperature unit; all temperature
        inputs share it.
    flow : DataFrame, mapping, or array-like
        Demand at every endmember [m³/day] on the same bins; see
        :func:`pipetransport.transport.source_to_endmember`.
    tedges : pandas.DatetimeIndex
        Time edges of ``tin`` and ``flow``, uniformly spaced (the halo memory is a
        convolution over lag bins).
    cout_tedges : pandas.DatetimeIndex
        Time edges of the output bins; alignment and resolution are free.
    network : PipeNetwork
        Network whose segments carry ``length`` [m], ``diameter`` [m] and ``cover`` (land
        cover class) columns; optionally ``depth`` [m to axis, default 1] and
        ``wall_thickness`` [m, required with ``kappa_pipe``].
    soil : pandas.DataFrame
        One row per cover class, columns ``alpha`` [m²/day], ``kappa`` [m²/day] and
        ``eta`` [m/day] (``inf`` for a prescribed-temperature surface).
    surface_temperature : pandas.DataFrame
        Sol-air temperature per cover class (one column per class), constant over each
        ``surface_tedges`` bin; see :func:`sol_air_temperature`.
    surface_tedges : pandas.DatetimeIndex or None, optional
        Edges of the surface record; defaults to ``tedges``. A record reaching a year or
        more back sharpens the soil state at depth; earlier history is the uniform
        pre-record mean (the first value of each column).
    nodes : list of str or None, optional
        Nodes to report at. Defaults to ``network.endmembers``.
    kappa_pipe : float or pandas.Series or None, optional
        Pipe wall conductivity over the water heat capacity [m²/day]; see
        :func:`segment_heat_rate`. Default None (bare pipe).
    film_coefficient : float or pandas.Series or None, optional
        Water-side film coefficient [m/day]; see :func:`segment_heat_rate`. Default None
        (film not limiting).
    max_sweeps : int, optional
        Iteration cap. 1 is the one-way model. The sweep count is a property of the physics,
        not of the record length: it is set by how much of the steady soil resistance the
        first lag bin still holds, ``Dbar[0] / R_soil``, which rises toward 1 for wide pipes
        on short bins. A few hundred sweeps is typical (286 on the example network at hourly
        bins, unchanged from 30 days of record to a year), but a 400 mm main on hourly bins
        needs about 1200. Exceeding the cap raises rather than returning an unconverged
        answer. Default 5000.
    atol : float, optional
        Convergence tolerance on the relaxation-target increment, absolute and in the unit of
        the temperature inputs. Default 1e-9, several orders below anything a temperature
        measurement resolves. It is absolute rather than relative so that the answer cannot
        depend on whether the caller works in Celsius or in kelvin, and so that an iterate
        that is diverging cannot widen its own convergence test.
    spinup : {"constant"} or None, optional
        Warm-start policy; see :func:`pipetransport.transport.source_to_endmember`.

    Returns
    -------
    numpy.ndarray
        Delivered temperature of shape ``(len(nodes), len(cout_tedges) - 1)``, in the
        unit of ``tin``; the flow-weighted average over each output bin. NaN marks bins
        the record does not constrain.

    Raises
    ------
    ValueError
        If a time axis is malformed or not uniform, an input holds NaN, a required
        segment or soil column is missing, a cover class is unmapped, or a physical
        parameter is out of range.
    RuntimeError
        If the two-way fixed point has not converged within ``max_sweeps``. An unconverged
        iterate is not returned, because it is not an answer.

    See Also
    --------
    endmember_to_source : Reverse direction.
    soil_temperature : The undisturbed field the targets are built from.
    segment_heat_rate : The exchange rates, as a standalone diagnostic.

    Examples
    --------
    Cool plant water warming on its way through warm summer soil:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pipetransport.examples import example_network, example_demand
    >>> from pipetransport.heat import source_to_endmember
    >>>
    >>> network = example_network()
    >>> network.segments["cover"] = [
    ...     "grass",
    ...     "grass",
    ...     "paved",
    ...     "paved",
    ...     "grass",
    ...     "paved",
    ...     "grass",
    ... ]
    >>> soil = pd.DataFrame(
    ...     {"alpha": [0.05, 0.07], "kappa": [0.025, 0.035], "eta": [0.41, 0.41]},
    ...     index=["grass", "paved"],
    ... )
    >>> tedges = pd.date_range("2025-06-01", "2025-06-08", freq="h")
    >>> surface = pd.DataFrame(
    ...     {
    ...         "grass": np.full(len(tedges) - 1, 22.0),
    ...         "paved": np.full(len(tedges) - 1, 30.0),
    ...     },
    ... )
    >>> cout = source_to_endmember(
    ...     tin=np.full(len(tedges) - 1, 8.0),
    ...     flow=example_demand(tedges=tedges, network=network),
    ...     tedges=tedges,
    ...     cout_tedges=tedges,
    ...     network=network,
    ...     soil=soil,
    ...     surface_temperature=surface,
    ... )
    >>> cout.shape
    (4, 168)
    >>> bool(
    ...     np.all(np.diff([8.0, np.nanmean(cout[3]), 30.0]) > 0)
    ... )  # between plant and sol-air
    True
    """
    tin = np.asarray(tin, dtype=float)
    tedges = pd.DatetimeIndex(tedges)
    _validate_tedges(tedges, tin, tedges_name="tedges", values_name="tin")
    _validate_no_nan(tin, name="tin")
    if max_sweeps < 1:
        msg = "max_sweeps must be at least 1 (sweep 1 is the one-way model)"
        raise ValueError(msg)
    system = _build_system(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        network=network,
        soil=soil,
        surface_temperature=surface_temperature,
        surface_tedges=surface_tedges,
        nodes=nodes,
        kappa_pipe=kappa_pipe,
        film_coefficient=film_coefficient,
        spinup=spinup,
    )
    tin_padded = np.concatenate([np.full(system.n_pad, tin[0]), tin])

    targets = _converge_targets(system, tin_padded, max_sweeps=max_sweeps, atol=atol)
    out = _apply(system.reporting, tin_padded) + apply_segment_targets(system.reporting, _extended(targets))
    out[~system.reporting.valid_out] = np.nan
    return out


def endmember_to_source(
    *,
    tout: npt.ArrayLike | pd.DataFrame | dict,
    flow: npt.ArrayLike | pd.DataFrame | dict,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    network: PipeNetwork,
    soil: pd.DataFrame,
    surface_temperature: pd.DataFrame,
    surface_tedges: pd.DatetimeIndex | None = None,
    nodes: list[str] | tuple[str, ...] | None = None,
    kappa_pipe: float | pd.Series | None = None,
    film_coefficient: float | pd.Series | None = None,
    max_sweeps: int = 5000,
    atol: float = 1e-9,
    regularization_strength: float = 1e-10,
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """Reconstruct the produced water temperature from temperatures measured at endmembers.

    The forward model is affine, ``tout = W @ tin + b``, and the bias depends on the
    temperatures only through the fixed point -- so the reverse problem subtracts the
    current soil bias from the observations, deconvolves with the existing banded Tikhonov
    solver, re-evaluates the fluxes with the reconstructed production series, and repeats.
    Measurement gaps are NaN and drop out of the solve exactly as in
    :func:`pipetransport.transport.endmember_to_source`.

    Parameters
    ----------
    tout : DataFrame, mapping, or array-like
        Measured temperature at the reporting nodes, constant over each ``cout_tedges``
        bin; a DataFrame or mapping is keyed by node name. NaN marks a gap.
    flow, tedges, cout_tedges, network, soil, surface_temperature, surface_tedges, nodes, kappa_pipe, film_coefficient, max_sweeps, atol, spinup
        As in :func:`source_to_endmember`.
    regularization_strength : float, optional
        Tikhonov parameter of each banded solve; see
        :func:`pipetransport.utils.solve_inverse_transport_banded`. Default 1e-10.

    Returns
    -------
    numpy.ndarray
        Reconstructed production temperature on ``tedges``, length ``len(tedges) - 1``.
        NaN for bins no measurement constrains.

    Raises
    ------
    ValueError
        As :func:`source_to_endmember`, plus a shape or naming mismatch of ``tout`` and a
        non-positive ``regularization_strength``.
    RuntimeError
        If the fixed point has not converged within ``max_sweeps``. Losing every endmember
        over a window makes the coupled inverse ill-posed, and this is how it shows.

    See Also
    --------
    source_to_endmember : Forward direction.

    Notes
    -----
    The halo is brought to its own fixed point inside every outer step, so the cost is a
    product of two iterations rather than a sum. The reconstruction also leans on a
    fabricated production series over the bins no measurement constrains, and the halo memory
    carries that forward: a measurement gap perturbs the answer after it as well as inside it.

    Examples
    --------
    Round-trip through two endmembers:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pipetransport.examples import example_network, example_demand
    >>> from pipetransport.heat import source_to_endmember, endmember_to_source
    >>>
    >>> network = example_network()
    >>> network.segments["cover"] = "grass"
    >>> soil = pd.DataFrame(
    ...     {"alpha": [0.05], "kappa": [0.025], "eta": [0.41]}, index=["grass"]
    ... )
    >>> tedges = pd.date_range("2025-06-01", "2025-06-15", freq="h")
    >>> demand = example_demand(tedges=tedges, network=network)
    >>> surface = pd.DataFrame({"grass": np.full(len(tedges) - 1, 25.0)})
    >>> hours = np.arange(len(tedges) - 1)
    >>> tin = 10.0 + 2.0 * np.sin(2 * np.pi * hours / 72.0)
    >>> shared = dict(
    ...     flow=demand,
    ...     tedges=tedges,
    ...     cout_tedges=tedges,
    ...     network=network,
    ...     soil=soil,
    ...     surface_temperature=surface,
    ...     nodes=["T1", "T4"],
    ... )
    >>> measured = source_to_endmember(tin=tin, **shared)
    >>> recovered = endmember_to_source(tout=measured, **shared)
    >>> inner = slice(48, -48)
    >>> bool(np.nanmax(np.abs(recovered[inner] - tin[inner])) < 1e-5)
    True
    """
    if max_sweeps < 1:
        msg = "max_sweeps must be at least 1 (sweep 1 is the one-way model)"
        raise ValueError(msg)
    system = _build_system(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        network=network,
        soil=soil,
        surface_temperature=surface_temperature,
        surface_tedges=surface_tedges,
        nodes=nodes,
        kappa_pipe=kappa_pipe,
        film_coefficient=film_coefficient,
        spinup=spinup,
    )
    cout_tedges = pd.DatetimeIndex(cout_tedges)

    named: dict | None = None
    if isinstance(tout, pd.DataFrame):
        named = {str(column): tout[column].to_numpy(dtype=float) for column in tout.columns}
    elif isinstance(tout, dict):
        named = {str(key): value for key, value in tout.items()}
    if named is not None:
        missing = [node for node in system.nodes if node not in named]
        if missing:
            msg = f"tout is missing node(s): {missing}"
            raise ValueError(msg)
        observed = np.stack([np.asarray(named[node], dtype=float) for node in system.nodes])
    else:
        observed = np.atleast_2d(np.asarray(tout, dtype=float))
    if observed.shape[0] != len(system.nodes):
        msg = f"tout must hold one row per reporting node ({len(system.nodes)}), got shape {observed.shape}"
        raise ValueError(msg)
    _validate_tedges(cout_tedges, observed, tedges_name="cout_tedges", values_name="tout")

    n_source = len(pd.DatetimeIndex(tedges)) - 1

    def solve(reporting: NetworkTransfer, targets: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        bias = apply_segment_targets(reporting, _extended(targets))
        band_vals = reporting.band_vals.reshape(-1, reporting.band_vals.shape[-1])
        rhs = np.where(reporting.valid_out.ravel(), (observed - bias).ravel(), np.nan)
        return solve_inverse_transport_banded(
            band_vals=band_vals,
            col_start=reporting.col_start.ravel(),
            observed=rhs,
            n_output=n_source + system.n_pad,
            regularization_strength=regularization_strength,
        )

    def filled(series: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        finite = np.isfinite(series)
        if finite.all():
            return series
        if not finite.any():
            msg = "no output bin constrains the source; nothing to reconstruct"
            raise ValueError(msg)
        index = np.arange(len(series))
        return np.interp(index, index[finite], series[finite])

    # Outer iteration on the production series, with the halo brought to its own fixed
    # point at every step. Alternating single target sweeps with the deconvolution instead
    # does not contract: the solve carries the inverse operator, which undoes the margin the
    # halo iteration has.
    recovered = solve(system.reporting, system.t_inf)
    converged = max_sweeps == 1
    targets = system.t_inf
    for _ in range(max_sweeps - 1):
        targets = _converge_targets(system, filled(recovered), max_sweeps=max_sweeps, atol=atol, initial=targets)
        updated = solve(system.reporting, targets)
        finite = np.isfinite(updated) & np.isfinite(recovered)
        increment = float(np.max(np.abs(updated[finite] - recovered[finite]), initial=0.0))
        recovered = updated
        if increment <= atol:
            converged = True
            break
    if not converged:
        msg = f"the reverse fixed point did not converge within max_sweeps={max_sweeps}; raise max_sweeps or atol"
        raise RuntimeError(msg)
    return recovered[system.n_pad :]
