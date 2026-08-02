"""
Drinking water temperature in the network: two-way heat exchange with the soil.

Water warms toward the soil around the pipe, and the soil warms back. Newton's law of
cooling makes the water temperature relax toward the soil temperature at a rate set by the
pipe's thermal resistance -- first-order decay toward a moving target instead of toward
zero -- so the delivered temperature is an *affine* reading of the same transport operator
the rest of this package uses: ``T_out = W @ T_in + b``, with ``W`` built with the
per-segment exchange rate in the decay-rate slot and ``b`` the soil's contribution
accumulated along each path. Three kernels carry all the soil physics, each exact for
piecewise-constant inputs and two of the three in closed form:

- **Surface to depth**: the undisturbed soil temperature at pipe depth is the superposition
  of Robin-boundary step responses of the half space, driven by the piecewise-constant
  sol-air temperature per land cover. Bin averages use the closed-form time integral of the
  step response, so a surface series that is constant per time bin maps exactly.
- **Wall flux to wall temperature**: the warm halo the pipe builds in the soil is a
  constant-flux cylinder at the pipe wall plus a mirror-image line sink above the surface
  (the image is what makes the halo saturate at the steady buried-pipe resistance). The
  cylinder is the one kernel with no closed form; it is a quadrature, evaluated once per
  distinct Fourier number when the system is built and held to about 1e-13 relative.
  Splitting its response into *steady part + transient deficit* turns the two-way model into
  the one-way model with a memory-shifted target: one short convolution of the wall-flux
  history per segment.
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
flux of a bin is the segment's own enthalpy budget over it, so each parcel's heat is booked
into the bins it actually occupied. Water that stands still exchanges heat for as long as it
stands, and a bin with no throughflow still leaks ``-h (H - V Tb)``; charging that to the
single bin the water finally left in would overstate it by the ratio of the standing time to
the transit, which under intermittent demand runs to several times the driving contrast.

That attribution also makes convergence a property of the model rather than of the
configuration. Every path through the sweep is causal in time and runs upstream to downstream
within a bin, so the iteration matrix is block lower triangular and its eigenvalues are the
same-bin gains ``Dbar[0] V (1 - exp(-h dt)) / (L dt)``, strictly below one for every geometry
and every bin width. The sweep count grows as ``1/(1 - g)`` as the bins narrow -- 58 to 567
sweeps over the configurations in the ``Validity`` notes -- which is the opposite trade from a
flux read off the delivered water: that one grew cheaper on fine bins precisely by mis-timing
the heat of standing water. The whole model stays linear in the produced water temperature
and the surface temperatures, so the reverse problem reuses the existing banded solver.

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

- The pipe term of the halo is the constant-flux *cylinder* response, and it is the one
  quantity in the module that is not closed-form. It is evaluated by quadrature along the
  branch cut of its Laplace transform, once per distinct ``alpha dt / r_o**2`` at build
  time, and holds about 1e-13 relative over every Fourier number reachable here. The image
  keeps its line source, at a cost of the same order as the steady resistance's own
  ``ln(2 d_eff/r_o)`` shape factor already carries: 2e-4 of ``R_soil`` for a 100 mm service
  line buried a metre and 3.5e-3 for a 400 mm main, both growing as the burial approaches
  ``r_o``. What the cylinder buys is the first lag bin, where a line source read at ``r = r_o``
  understates the arrived resistance badly because the heat has not yet diffused past the
  pipe: ``Dbar[0]/R_soil`` is 0.8568 for a 100 mm service line and 0.9323 for a 400 mm main
  on hourly bins, against 0.9418 and 1.0000 for the line source -- the second of which left
  the same-bin loop gain no margin at all. That ratio sets the sweep count and how far the
  fixed point sits from singular, so the swap cuts the sweep count severalfold as well:
  100 against 286 on the example network, 159 against 660 on a 400 mm main.
- **One wall-flux history per pipe.** The soil columns along a pipe are independent and the
  wall flux falls along it like ``exp(-h tau)`` -- by a factor 1.6 over a 2 h transit on a
  100 mm service line -- but the model gives every parcel in a pipe the same soil memory.
  That is the model's spatial resolution along a pipe, and it is a stated assumption rather
  than a parameter. Measured against a reference that keeps one memory per axial cell, it is
  worth 0.08 K on a 100 mm line at a 2 h transit and 0.53 K at 6 h, under 24 h diurnal
  forcing at hourly bins. Refining ``tedges`` does not reduce it; it is a property of the
  transit.

  Declare a pipe as a chain of shorter segments if you need it resolved -- splitting is
  exact for the transport, which is unchanged to round-off by it. The gap closes steeply at
  the first refinement (to 0.03-0.05 K at four pieces) and then flattens onto a floor more
  pieces do not move; issue #32 tracks both removing the assumption and attributing that
  floor.
- The relaxation target is an *effective driving temperature*, not a wall temperature. The
  rate keeps the steady soil resistance, which overstates the resistance while the halo is
  still developing, so the target has to be pushed past the undisturbed soil to reproduce
  the faster early exchange -- measured excursions of several times the driving contrast in
  the bins after a sharp change in the wall flux. The delivered temperature is a weighted
  average of ``tin`` and those targets, so it can leave the range of its own inputs. Only
  ``max_sweeps=1`` is guaranteed inside it. A step in ``tin`` into a *continuously flowing*
  pipe is the mild case: 18 % of the instantaneous plant-to-soil contrast, 1.9 transits after
  the step, back inside the range after nine days, falling to 1-4 % once the pipe is split.
  Intermittent demand is the harder one, and is where the flux attribution earns its keep: a
  line idle 8 h a day now delivers water 7.8 % of the contrast past the soil, and splitting
  the pipe brings it back *inside* the range (-3.8 % at two pieces, -10.3 % at four). Reading
  the flux off the delivered water instead put the same case at 20-28 % and made splitting
  worse rather than better, and on the shapes in issue #24 -- a main flushed 2 h in every 24,
  a line standing ten days -- it reached 8.8 times the contrast, converged and unflagged.
  Those now come back within about a kelvin of the range of their inputs.

  Wide pipes on short bins were once worse than any of this: with the line-source halo a
  400 mm main at a half-hour transit came back spanning -88 to +78 C from water produced at
  8 C into soil at 22 C, and refining ``tedges`` made it worse. The cylinder response closes
  that regime, and no bin width reopens it.
- A bin in which a segment has no throughflow still exchanges heat: the water standing in it
  relaxes toward the soil, and the enthalpy budget books that heat into the bins it happens
  in. Overnight stagnation of a service line reaches ``h tau`` of order 2, so this is a large
  part of what such a pipe does, and it is the case the delivered-water flux mis-timed.
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

import warnings
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import fftconvolve, lfilter
from scipy.special import erfc, erfcx, exp1, j1, y1

from pipetransport._transfer import (
    NetworkTransfer,
    apply_banded,
    apply_segment_targets,
    pad_paths,
    paths_transfer,
    resolve_spinup,
)
from pipetransport._validation import _validate_no_nan, _validate_positive, _validate_tedges
from pipetransport.network import PipeNetwork  # noqa: TC001 -- runtime dependency of the signatures
from pipetransport.utils import solve_inverse_transport_banded, tedges_to_days

# How far ahead of the outer iterate the reverse direction resolves its inner halo fixed
# point. The inner solve is inexact while the reconstruction is still moving and exact once it
# has settled, so the answer is the same to well under ``atol``; at 1e-2 the reverse direction
# spends about a third of the inner sweeps that a fully resolved inner solve does.
_INNER_FORCING = 1e-2
# Anderson window on the reverse outer iterate. Truncated Anderson is truncated GMRES, so
# the window is not a tuning knob with a free choice: too short a memory drops the directions
# the iteration needs and it stalls. A depth of five raises on a 400 mm main at a half-hour
# transit, which ten reconstructs to 1e-9. It buys range, not a guarantee: a pipe past the
# coupling the divergence test names is out of reach at any depth.
# How far a segment's volume may sit from the one its length and diameter imply. Wide enough
# for fittings and for wall-thickness conventions, tight enough to catch a volume that came
# from somewhere else entirely.
_GEOMETRY_TOLERANCE = 0.05
_ANDERSON_DEPTH = 10
# Consecutive growing outer residuals that mark divergence rather than a slow start.
_DIVERGENCE_STEPS = 5
# Consecutive growing outer residuals that mark divergence rather than a slow start.
_DIVERGENCE_STEPS = 5


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


def _cylinder_integral(fo: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """``integral_0^fo Ghat(s) ds`` of the constant-flux cylinder, elementwise and dimensionless.

    A pipe is a cylinder, not a line: the wall flux leaves over a surface of radius ``r_o``,
    and until the heat has diffused well past that radius the line source read at ``r = r_o``
    understates the wall temperature badly. The exact constant-flux cylinder response has no
    closed form, but its Laplace transform does -- ``K0(z) / (2 pi kappa s z K1(z))`` with
    ``z = r_o sqrt(s/alpha)`` -- and wrapping the Bromwich contour around the branch cut
    turns it into a real integral in the dimensionless wavenumber ``b = u r_o``,

    ``Ghat(Fo) = (2/pi**3) integral_0^inf (1 - exp(-Fo b**2)) w(b) db``,
    ``w(b) = 1 / (b**3 (J1(b)**2 + Y1(b)**2))``,

    with ``Ghat`` the wall temperature per unit flux in units of ``1/kappa``: the physical
    response is ``G(t) = Ghat(alpha t / r_o**2) / kappa``. It carries both limits, the one the
    line source misses and the one it reaches -- ``sqrt(Fo/pi)/pi`` as ``Fo -> 0`` (a plane,
    because the wall is locally flat) and ``E1(1/(4 Fo))/(4 pi)`` as ``Fo -> inf`` (the line
    source itself).

    The time integral is what the deficit needs, and it is the same integral with the factor
    ``(x + expm1(-x))/b**2`` in place of ``1 - exp(-x)``, ``x = Fo b**2``. Its tail decays only
    as ``1/b**2``, so the ``(pi/2)/(1 + b**2)`` part of ``w`` -- which carries that whole tail
    -- is peeled off and integrated in closed form, leaving a residue that decays as
    ``1/b**4``. In ``y = ln b`` what is left falls off exponentially at both ends, where the
    trapezoidal rule converges geometrically, so a fixed 360-node grid holds the result to
    about 1e-13 relative over every Fourier number this package can reach.

    Parameters
    ----------
    fo : ndarray
        Fourier number ``alpha t / r_o**2`` [-], any shape, non-negative.

    Returns
    -------
    ndarray
        ``integral_0^fo Ghat(s) ds`` [-], same shape as ``fo``; 0 at ``fo = 0``.
    """
    fo = np.asarray(fo, dtype=float)
    beta = np.exp(np.arange(-24.0, 12.0, 0.1))
    b_sq = np.square(beta)
    residue = 1.0 / (beta * b_sq * (np.square(j1(beta)) + np.square(y1(beta)))) - (np.pi / 2.0) / (1.0 + b_sq)
    weight = residue * beta * (0.1 * 2.0 / np.pi**3)
    # The (fo x beta) exponential matrix is built in bounded chunks, as in soil_temperature,
    # so a year of hourly lag bins does not materialize at once.
    ravelled, quadrature = fo.reshape(-1), np.empty(fo.size)
    chunk = max(1, 8_388_608 // len(beta))
    for lo in range(0, fo.size, chunk):
        x = ravelled[lo : lo + chunk, None] * b_sq
        quadrature[lo : lo + chunk] = ((x + np.expm1(-x)) / b_sq) @ weight
    peeled = fo + 1.0 - erfcx(np.sqrt(fo)) - 2.0 * np.sqrt(fo / np.pi)
    return peeled / (2.0 * np.pi) + quadrature.reshape(fo.shape)


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

    The wall-temperature step response is the constant-flux cylinder at the outer pipe radius
    (:func:`_cylinder_integral`) minus a mirror-image line sink at ``2 d_eff``, which is what
    makes the halo saturate at the steady buried-pipe resistance
    ``R_inf = ln(2 d_eff/r_o)/(2 pi kappa)``. The image keeps its closed form
    ``E1(c_i/t)/(4 pi kappa)``: a cylinder read from ``2 d_eff`` away rather than from its own
    wall is a line to the same order the *steady* resistance above already is, and it costs
    the same order too -- at most 2e-4 of ``R_inf`` for a 100 mm service line buried a metre
    and 3.5e-3 for a 400 mm main, against the 1e-4 and 3.8e-3 that ``ln(2 d_eff/r_o)`` itself
    gives up against ``acosh(d_eff/r_o)``. Both grow as the burial approaches ``r_o``.

    The deficit ``D = R_inf - G`` is what has *not yet arrived*; its bin average over lag bin
    ``m`` comes from the time integral of ``G``, closed-form for the image and quadrature for
    the pipe. Only the dimensionless Fourier number per bin, ``alpha dt / r_o**2``, enters the
    pipe term, so it is evaluated once per distinct value and shared by the segments on it.

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
    c_i = ((2.0 * d_eff) ** 2 / (4.0 * alpha))[:, None]
    r_inf = (np.log(2.0 * d_eff / r_o) / (2.0 * np.pi * kappa))[:, None]
    edge = np.arange(n_bins + 1)[None, :]
    lag = dt_days * edge
    fo_bin, segment_of = np.unique(alpha * dt_days / r_o**2, return_inverse=True)
    cylinder = _cylinder_integral(fo_bin[:, None] * edge)[segment_of]
    cumulative = (
        r_inf * lag
        - (r_o**2 / (kappa * alpha))[:, None] * cylinder
        + _halo_integral(c_i, lag) / (4.0 * np.pi * kappa[:, None])
    )
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
        Bin edges of the surface series; defaults to ``tedges``, and must be uniform with
        the same bin width, which is what makes the superposition a convolution. It is free
        to start and end elsewhere: supply a record starting well before the period of
        interest -- after one year roughly 87 % of a step has arrived at 1 m, so the first
        months of output lean on ``t_pre``.

    Returns
    -------
    ndarray
        Soil temperature at ``depth``, one bin-average per ``tedges`` bin, in the unit of
        ``surface_temperature``.

    Raises
    ------
    ValueError
        If a time axis is malformed or the two do not share one uniform bin width, the
        series holds NaN, a parameter is not positive, or only one of ``kappa`` and ``eta``
        is given.

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
    bin_width = tedges[1] - tedges[0]
    if (
        np.unique(np.diff(tedges.asi8)).size != 1
        or np.unique(np.diff(surface_tedges.asi8)).size != 1
        or surface_tedges[1] - surface_tedges[0] != bin_width
    ):
        msg = (
            "tedges and surface_tedges must share one uniform bin width (the step response depends only on the "
            "lag, which makes the superposition over surface steps a convolution)"
        )
        raise ValueError(msg)
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
    # The response depends on the lag alone, so on one shared bin width the lags of the
    # (output edge, surface step) pairs take ``n_out + n_step`` distinct values rather than
    # their product: the superposition is a convolution over those. A year of hourly forcing
    # costs 17_560 kernel evaluations instead of 77 million, and a vector instead of an N^2
    # matrix that had to be chunked to fit.
    n_out, n_step = len(out_edges) - 1, len(step_times)
    dt_days = bin_width / pd.Timedelta(days=1)
    lag = (out_edges[0] - step_times[0]) + (np.arange(n_out + n_step) - (n_step - 1)) * dt_days
    kernel = _step_response_integral(lag, depth=depth, alpha=alpha, radiation_length=radiation_length)
    # Differenced to a bin-averaged response before the convolution rather than after it, so
    # the transform's round-off is not amplified by the 1/dt of the final differencing.
    response = fftconvolve(np.diff(kernel) / dt_days, steps)[n_step - 1 : n_step - 1 + n_out]
    # The response is causal, and exactly so: an output bin ending at or before the first
    # surface step integrates a kernel that is identically zero over its whole span. The
    # transform blurs that exact zero by its own round-off, so it is restored rather than
    # approximated -- the pre-history is a value the caller supplied, not a computed one.
    return pre + np.where(out_edges[1:] <= step_times[0], 0.0, response)


def segment_heat_rate(
    *,
    network: PipeNetwork,
    kappa: float | pd.Series,
    depth: float | pd.Series = 1.0,
    eta: float | pd.Series = np.inf,
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
        Burial depth to the pipe axis [m]. Default 1.0. It must put the pipe below the
        surface, ``d_eff > r_o``; the exact shape factor ``acosh(d_eff/r_o)`` does not exist
        below that and the ``ln(2 d_eff/r_o)`` used here runs away to a divergent rate rather
        than to an error. That log is the large-``d/r`` limit of the exact factor and
        overstates it by ``1/(4 (d_eff/r_o)**2)``, so accuracy is a matter of how far above
        the guard the pipe is: 0.01 % for a 100 mm service line and 0.4 % for a 400 mm main
        at a metre, 5.3 % at ``d_eff = 2 r_o``, 24 % for a DN1600 there, and 117 % for a
        DN2000 -- leaving the default depth on a transmission main is the mis-entry to watch.
    eta : float or pandas.Series, optional
        Surface film coefficient [m/day]. ``inf`` (default) is a prescribed-temperature
        surface: the radiation length ``kappa / eta`` is then zero and the surface displaces
        the effective depth by nothing.
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
        positive, or the geometry gives ``d_eff <= r_o``.

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
    eta_seg = per_segment(eta, "eta")
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

    d_eff = depth_seg + kappa_seg / eta_seg  # kappa/inf is exactly zero: no branch needed
    if not np.all(d_eff > r_o):
        msg = "burial depth must exceed the pipe radius (d_eff > r_o); the line-source geometry needs d >> r"
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

    ``internal`` holds three rows per segment, all binned on that pipe's own deliveries: the
    delivered temperature, the same reading weighted by ``exp(-h (t_end - t))``, and that
    weighted reading taken at the pipe's entry instead. The first closes the advective half of
    the bin's enthalpy budget and the other two close its storage half; see
    :func:`_update_targets`.
    """

    nodes: tuple[str, ...]
    n_pad: int
    n_bins: int
    dt_days: float
    t_inf: npt.NDArray[np.floating]
    dbar_spectrum: npt.NDArray[np.complexfloating]
    halo_length: int
    seg_flow: npt.NDArray[np.floating]
    length: npt.NDArray[np.floating]
    volume: npt.NDArray[np.floating]
    rho: npt.NDArray[np.floating]
    parent: npt.NDArray[np.intp]
    held_slope: npt.NDArray[np.floating]
    held_offset: npt.NDArray[np.floating]
    h_tau: npt.NDArray[np.floating]
    segment_names: tuple[str, ...]
    internal: NetworkTransfer
    reporting: NetworkTransfer


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
    # Transport reads only the volume, so a network may carry one that its length and diameter
    # do not imply and never notice. Heat reads the pipe three ways and they have to be the
    # same pipe; where they are not, the water's heat capacity per unit length disagrees with
    # the area the exchange rate was built from, and what the user used to see was a
    # convergence failure naming knobs that cannot reconcile a geometry.
    geometric = np.pi / 4.0 * segments["diameter"].to_numpy(dtype=float) ** 2 * segments["length"].to_numpy(dtype=float)
    contradicts = np.abs(segments["volume"].to_numpy(dtype=float) - geometric) > _GEOMETRY_TOLERANCE * geometric
    if contradicts.any():
        msg = (
            f"segment(s) {list(segments.index[contradicts])} carry a 'volume' that their 'length' and "
            f"'diameter' do not imply, by more than {_GEOMETRY_TOLERANCE:.0%}. The heat pair reads the "
            f"pipe three ways -- the exchange rate from the diameter, the wall flux from the length, the "
            f"transit from the volume -- so they must describe one pipe. Drop the volume column to have "
            f"it derived, or reconcile it with pi/4 * diameter**2 * length."
        )
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
    end_paths, end_active = pad_paths([
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
            kappa=float(soil.loc[cover, "kappa"]),
            eta=eta_cover,
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
    d_eff = depth_seg + kappa_seg / eta_seg
    dbar = _deficit_kernel(n_bins, dt_days, r_o=r_o, d_eff=d_eff, alpha=alpha_seg, kappa=kappa_seg)
    # The halo memory convolves this frozen kernel with a new flux history on every sweep, so
    # the kernel is transformed once here, at the length ``scipy.signal.fftconvolve`` would
    # have picked itself -- which makes the per-sweep multiply-and-invert bit-identical to it.
    halo_length = int(next_fast_len(2 * n_bins - 1, real=True))

    n_seg = len(segments)

    def chain(node: str) -> npt.NDArray[np.intp]:
        """Segment path from the source to ``node``.

        Returns
        -------
        ndarray of intp
            Segment rows, source outward; empty for the source node itself.
        """
        return np.array([seg_of[name] for name in network.paths[node]], dtype=np.intp)

    # Internal rows, three per segment and all binned on that pipe's own deliveries. A bin's
    # enthalpy budget needs the water it delivered, and -- because the storage term integrates
    # against the exchange -- the same reading and its inflow counterpart weighted by
    # ``exp(-h (t_end - t))``. The inflow rows run to the pipe's entry node, so a root segment
    # reads its own source series across an empty path.
    entry_chains = [chain(str(segments.loc[name, "from"])) for name in segments.index]
    delivery = [np.concatenate([up, [e]]).astype(np.intp) for e, up in enumerate(entry_chains)]
    int_paths, int_active = pad_paths(delivery + delivery + entry_chains)
    rep_paths, rep_active = pad_paths([chain(node) for node in requested])
    rep_flow = network.node_flow(flow=demand_p, nodes=requested)

    def build(
        paths_idx: npt.NDArray[np.intp],
        active: npt.NDArray[np.bool_],
        node_flow: npt.NDArray[np.floating],
        cout: npt.NDArray[np.floating],
        bin_end_rate: npt.NDArray[np.floating] | None,
    ) -> NetworkTransfer:
        return paths_transfer(
            tedges_days=tedges_days,
            cout_tedges_days=cout,
            segment_volume=volume,
            segment_flow=seg_flow,
            segment_decay=rate,
            node_flow=node_flow,
            paths_idx=paths_idx,
            active=active,
            bin_end_rate=bin_end_rate,
            with_target_terms=True,
        )

    rho = np.exp(-rate * dt_days)
    length = segments["length"].to_numpy(dtype=float)
    running = np.where(seg_flow > 0.0, seg_flow, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a segment that never flows is all-NaN
        running = np.nanmedian(running, axis=1)

    # What the pipes are holding when the record opens. The warm start fills them with water
    # produced at ``tin[0]`` under the leading flow, held long enough to have settled, so each
    # pipe's content is the steady profile ``T(s) = T_inf + (T_entry - T_inf) exp(-h s)`` and
    # its excess over the undisturbed field integrates to ``Q (T_entry - T_inf)(1 - e^-h tau)/h``.
    # The enthalpy budget has to start there, because the readings on the other side of the
    # same balance describe exactly that water: starting from an equilibrated pipe instead
    # differences two different pipes and books a first flux wrong by its own size. Both
    # directions know ``tin[0]``, so the content is carried as its affine coefficients and the
    # sweep evaluates them against whatever series it is reconstructing.
    parent = np.array([up[-1] if up.size else -1 for up in entry_chains], dtype=np.intp)
    with np.errstate(divide="ignore"):
        # A pipe the record opens idle has an infinite transit, so it holds water that has
        # already settled onto the soil and carries no excess: exp(-inf) is the right zero.
        settled = np.exp(-rate * volume / seg_flow[:, 0])
    entry_slope, entry_offset = np.ones(n_seg), np.zeros(n_seg)
    # One pass from the source outward; the loop is over path depth, which is what orders it.
    for e in sorted(range(n_seg), key=lambda seg: entry_chains[seg].size):
        upstream = parent[e]
        if upstream >= 0:
            entry_slope[e] = entry_slope[upstream] * settled[upstream]
            entry_offset[e] = t_inf[upstream, 0] + (entry_offset[upstream] - t_inf[upstream, 0]) * settled[upstream]
    held = seg_flow[:, 0] * (1.0 - settled) / rate
    return _HeatSystem(
        nodes=requested,
        n_pad=n_pad,
        n_bins=n_bins,
        dt_days=dt_days,
        t_inf=t_inf,
        dbar_spectrum=rfft(dbar, n=halo_length, axis=1),
        halo_length=halo_length,
        seg_flow=seg_flow,
        length=length,
        volume=volume,
        rho=rho,
        parent=parent,
        held_slope=held * entry_slope,
        held_offset=held * (entry_offset - t_inf[:, 0]),
        # How far a pipe equilibrates over one transit. It is what decides whether the
        # reverse coupling is invertible at all, so the diagnostics quote it.
        h_tau=rate * volume / np.where(running > 0.0, running, np.nan),
        segment_names=tuple(str(name) for name in segments.index),
        internal=build(
            int_paths, int_active, np.vstack([seg_flow] * 3), tedges_days, np.concatenate([np.zeros(n_seg), rate, rate])
        ),
        reporting=build(rep_paths, rep_active, rep_flow, cout_days, None),
    )


def _converge_targets(
    system: _HeatSystem,
    tin_padded: npt.NDArray[np.floating],
    *,
    max_sweeps: int,
    atol: float,
    initial: npt.NDArray[np.floating] | None = None,
    fabricated: npt.NDArray[np.bool_] | None = None,
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
        iterate instead. That saves only a few per cent of the sweeps until the outer
        iterate is close, which is why the reverse direction also loosens ``atol`` in step
        with its own increment.

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
    # The transport reading of the source series is the same in every sweep; only the bias
    # follows the iterate.
    transported = apply_banded(system.internal, tin_padded)
    for _ in range(max_sweeps - 1):
        updated = _update_targets(system, _internal_pass(system, transported, targets), targets, tin_padded, fabricated)
        increment = float(np.max(np.abs(updated - targets)))
        targets = updated
        if increment <= atol:
            return targets
    msg = f"the two-way fixed point did not converge within max_sweeps={max_sweeps}; raise max_sweeps or atol"
    raise RuntimeError(msg)


def _diverged_message(system: _HeatSystem, previous: float, increment: float, exhausted: int | None = None) -> str:
    """Explain a reverse iteration that will not reach its fixed point.

    Parameters
    ----------
    system : _HeatSystem
        Prebuilt system, read for the segment that couples most strongly.
    previous, increment : float
        The last two outer residuals.
    exhausted : int or None, optional
        The sweep budget, when the loop ran out rather than being cut short by the
        divergence test. Default None.

    Returns
    -------
    str
        Message naming the regime, the segment driving it, and the variant that works.
    """
    worst = int(np.nanargmax(system.h_tau)) if np.isfinite(system.h_tau).any() else 0
    coupling = float(system.h_tau[worst])
    ran_out = f" within max_sweeps={exhausted}" if exhausted is not None else ""
    return (
        f"the reverse two-way fixed point did not converge{ran_out}: the outer residual went "
        f"{previous:.3e} -> {increment:.3e}. The strongest coupling is segment "
        f"{system.segment_names[worst]!r} at h*tau = {coupling:.2f}, and past about 0.7 the "
        f"reverse problem is ill-conditioned rather than merely slow -- water that equilibrates "
        f"over its transit carries little of the produced temperature to the endmember, so no "
        f"cap, tolerance or regularization recovers it. Use max_sweeps=1 for the one-way "
        f"reverse, which stays well-posed, or shorten the segment."
    )


def _internal_pass(
    system: _HeatSystem, transported: npt.NDArray[np.floating], targets: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Read the three internal temperatures, NaN where the record does not constrain them.

    Parameters
    ----------
    system : _HeatSystem
        Prebuilt operators and kernels.
    transported : ndarray
        Reading of the internal operator on the source series, ``apply_banded(system.internal,
        tin_padded)``. It does not depend on the targets, so the sweep loop hoists it out.
    targets : ndarray
        Current per-segment relaxation targets.

    Returns
    -------
    ndarray
        The three readings of every segment stacked, shape ``(3 * n_seg, n_bins)``: the
        delivered temperature, the same reading weighted toward the bin end, and that
        weighted reading taken at the pipe's entry.
    """
    t_int = transported + apply_segment_targets(system.internal, targets)
    t_int[~system.internal.valid_out] = np.nan
    return t_int


def _update_targets(
    system: _HeatSystem,
    t_int: npt.NDArray[np.floating],
    targets: npt.NDArray[np.floating],
    tin_padded: npt.NDArray[np.floating],
    fabricated: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.floating]:
    """One flux-and-halo pass: internal temperatures -> new per-segment targets.

    The wall flux of a bin is the segment's own enthalpy budget over it,
    ``psi = (H[j] - H[j+1] + Q dt (T_in - T_out)) / (L dt)``, with the water content ``H``
    advanced by the exact solution of ``dH/dt = Q (T_in - T_out) - h (H - V Tb)`` over a bin
    of constant flow and target,

        ``H[j+1] = rho H[j] + Q dt (D_in[j] - D_out[j]) + V (1 - rho) Tb[j]``,

    where ``D`` is the inflow or outflow temperature averaged against ``exp(-h (t_end - t))``
    -- the two weighted rows of the internal operator -- and ``rho = exp(-h dt)``. Because a
    parcel's heat release over any interval *is* its temperature drop there, this books each
    parcel's heat into the bins it actually occupied instead of the single bin it was
    delivered in, which is what keeps water that has been standing from charging a whole
    night's exchange to one bin. Content is carried relative to the undisturbed field so the
    difference stays free of cancellation at absolute temperature, and the record telescopes
    exactly: nothing is booked that the water did not carry.

    Stagnation needs no special case -- ``Q = 0`` drops the advective terms and the pipe still
    leaks ``-h (H - V Tb)`` into the halo, which is the whole point: a bin with no throughflow
    books its storage term like any other. What contributes zero flux is the spin-up prefix,
    and a bin whose readings the record does not constrain -- the undisturbed-soil assumption
    applied at bin resolution.

    A bin's own target moves that bin's storage term and the lag-0 deficit brings it straight
    back, so the sweep has a same-bin gain of about ``Dbar[0] V (1 - rho) / (L dt)``, which is
    strictly below one because ``Dbar[0] < R_total`` and ``(1 - exp(-h dt)) / (h dt) < 1``.
    The target also moves that bin's own readings -- a parcel relaxes toward it for the part
    of the bin it spends in the pipe -- which adds ``Dbar[0] Q dt (a - b) / (L dt)`` with
    ``a``, ``b`` the sensitivities of the plain and weighted delivery readings. That term is
    smaller than the first by of order ``h dt / 2`` and is what the measured gains below
    include; it is the reason they are quoted as measured rather than derived.
    Every other path through the map is strictly causal in time and runs strictly from
    upstream to downstream within a bin, so the iteration matrix is block lower triangular and
    its eigenvalues *are* those same-bin gains: the sweep contracts for every geometry and
    every bin width, rather than only where the flux happened to be small. The price is that
    the gain approaches one as the bins narrow, so the sweep count grows as ``1/(1 - g)`` --
    the opposite trade from a flux read off the delivered water, which grew cheaper on fine
    bins by charging standing water's heat to the bin it happened to leave in.

    Parameters
    ----------
    system : _HeatSystem
        Prebuilt operators and kernels.
    t_int : ndarray
        The three internal readings from :func:`_internal_pass`.
    targets : ndarray
        Targets the readings were taken at; the same-bin solve needs them.
    tin_padded : ndarray
        Source temperature on the padded input grid, the inflow of every root segment.
    fabricated : ndarray of bool or None, optional
        Bins whose flux is to be suppressed, treated exactly like a bin the record does not
        constrain. ``None`` (default) suppresses nothing.

    Returns
    -------
    ndarray
        Updated per-segment relaxation targets, shape ``(n_seg, n_bins)``.
    """
    n_seg = len(system.length)
    t_out, d_out, d_in = t_int[:n_seg], t_int[n_seg : 2 * n_seg], t_int[2 * n_seg :]
    # A split leaves temperature unchanged and both flows are constant over a bin, so the
    # water entering a pipe is what its parent delivered; a root segment is fed the source.
    t_in = np.where(system.parent[:, None] >= 0, t_out[system.parent], tin_padded)

    # The warm-start prefix is a fabricated hydraulic history, so it drives neither the wall
    # flux nor the water the pipe is holding when the record opens: the record starts with the
    # pipe in equilibrium with the undisturbed soil. That initial condition belongs to the
    # model rather than to the caller's data, which is what lets the forward and the reverse
    # direction agree about it -- the reverse cannot reconstruct a prefix no measurement
    # constrains, and does not have to.
    usable = np.zeros_like(t_out, dtype=bool)
    usable[:, system.n_pad :] = True
    usable &= ~(np.zeros_like(usable) if fabricated is None else fabricated)
    flowing = usable & (system.seg_flow > 0.0)
    for reading in (t_in, t_out, d_in, d_out):
        flowing &= np.isfinite(reading)
    carried = system.seg_flow * system.dt_days
    advected = np.where(flowing, carried * (t_in - t_out), 0.0)
    storage = system.volume[:, None] * (1.0 - system.rho)[:, None] * (targets - system.t_inf[:, :1])
    forcing = np.where(flowing, carried * (d_in - d_out), 0.0) + np.where(usable, storage, 0.0)

    # The record opens with the pipes holding the warm start's own water; see _build_system.
    # With no warm start there is no prior state to carry, and the leading bins are
    # unconstrained anyway, so the slice is simply absent.
    if system.n_pad:
        forcing[:, system.n_pad - 1] += system.held_slope * tin_padded[0] + system.held_offset

    content = np.zeros_like(forcing)
    for e in range(n_seg):
        content[e] = lfilter([1.0], [1.0, -system.rho[e]], forcing[e])
    psi = (np.concatenate([np.zeros((n_seg, 1)), content[:, :-1]], axis=1) - content + advected) / (
        system.length[:, None] * system.dt_days
    )
    dpsi = np.diff(psi, axis=1, prepend=0.0)
    spectrum = rfft(dpsi, n=system.halo_length, axis=1) * system.dbar_spectrum
    return system.t_inf - irfft(spectrum, n=system.halo_length, axis=1)[:, : system.n_bins]


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
    transport pass, an enthalpy-budget flux pass and one convolution per segment.
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
        on short bins. A hundred or two is typical (100 on the example network at
        hourly bins, unchanged from 30 days of record to a year), and a 400 mm main run at a
        0.5 m/s design velocity on hourly bins needs about 160. Exceeding the cap
        raises rather than returning an unconverged answer. Default 5000.
    atol : float, optional
        Convergence tolerance on the relaxation-target increment, absolute and in the unit of
        the temperature inputs. Default 1e-9, several orders below anything a temperature
        measurement resolves. It is absolute rather than relative so that the answer cannot
        depend on whether the caller works in Celsius or in kelvin, and so that an iterate
        that is diverging cannot widen its own convergence test. The delivered temperature
        settles to a fraction of it -- measured about 0.4 -- and the sweep count grows only as
        its logarithm, so there is little to buy by relaxing it: 1e-7 is the loosest value
        that leaves the package's own tests meaningful, and costs a third of the runtime.
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
    out = apply_banded(system.reporting, tin_padded) + apply_segment_targets(system.reporting, targets)
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
    gap_atol: float = 1e-3,
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
    gap_atol : float, optional
        How far [K] the reconstruction may move when the flux invented over a measurement
        gap is removed before the bin is reported as unconstrained. Default 1e-3, three
        orders below any temperature measurement and six above the no-gap round-trip floor.

    Returns
    -------
    numpy.ndarray
        Reconstructed production temperature on ``tedges``, length ``len(tedges) - 1``.
        NaN for bins no measurement constrains, and for bins whose reconstruction moves by
        more than ``gap_atol`` when the flux invented over such a gap is removed -- so every
        value returned is one the record supports to that tolerance.

    Raises
    ------
    ValueError
        As :func:`source_to_endmember`, plus a shape or naming mismatch of ``tout`` and a
        non-positive ``regularization_strength``.
    RuntimeError
        If the fixed point diverges or has not converged within ``max_sweeps``. Past a
        ``h*tau`` of about 0.7 -- a pipe that equilibrates appreciably over its transit --
        the coupled inverse is ill-conditioned rather than slow, and this is how it shows;
        the message names the segment responsible. ``max_sweeps=1`` is the one-way reverse,
        which stays well-posed.

    See Also
    --------
    source_to_endmember : Forward direction.

    Notes
    -----
    The halo is brought to its own fixed point inside every outer step, so the cost is a
    product of two iterations rather than a sum. The outer step extrapolates over its last few
    iterates rather than simply repeating, which reaches pipes plain repetition cannot -- but
    only so far: the extrapolation is truncated, and past a coupling of roughly
    ``h*tau = 0.7`` nothing reaches the fixed point, which is what the RuntimeError reports.

    The reconstruction leans on a fabricated production series over the bins no measurement
    constrains, and the coupling carries that invention into bins the record *does* constrain
    -- after the gap through the halo memory, and before it because the deconvolution couples
    the whole record. Measured on a single 100 mm pipe with a 72 h outage, that was worth
    0.44 K after the gap and 0.46 K before it, on bins reported as constrained. The answer is
    therefore re-solved once with the invented flux suppressed; because the model is affine
    that difference is the exact imprint of the invention, and the bins it moves by more than
    ``gap_atol`` come back NaN. The one-way reverse needs none of this and is exactly local:
    with a fixed target, the same outage changes nothing outside itself.

    **The record's own end is unconstrained too, and further in than transport coverage
    alone suggests.** Most of the heat the last bins exchange with the soil arrives after the
    record stops, so no measurement in it pins them -- the lead-*out* counterpart of the
    lead-in transient the forward direction has. They lean on flux the model invents past the
    end of the record, so the same re-solve catches them and they come back NaN: on the
    example network at hourly bins that is the last 36 bins of a 240-bin record, and every
    bin still answered is within 7e-4 K. End the record after the period you care about, and
    the answers you get are ones the record supports.

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
    >>> tedges = pd.date_range("2025-06-01", "2025-06-11", freq="h")
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
    >>> inner = slice(36, -96)  # both edges lean on unconstrained bins; see Notes
    >>> residual = float(np.nanmax(np.abs(recovered[inner] - tin[inner])))
    >>> residual < 1e-8  # the Tikhonov pull, O(lambda) once the target preserves constants
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
        bias = apply_segment_targets(reporting, targets)
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
    #
    # The inner fixed point is only asked for the accuracy the outer iterate has itself: a
    # halo resolved to `atol` around a production series still kelvins from its own fixed
    # point is thrown away on the next step. The forcing shrinks with the outer increment, so
    # the last inner solve is the exact one, and the outer loop pays one or two extra steps
    # for it. Seeded at inf, so the first inner solve is a single sweep.
    # The outer map is affine, so plain repetition converges only where its own spectral
    # radius happens to be under one -- and a pipe that equilibrates appreciably over its
    # transit puts it over. Extrapolating over the last few iterates instead (Anderson, which
    # on an affine map is a Krylov method) reaches configurations plain repetition cannot,
    # though with a truncated window it carries no guarantee: past a coupling of roughly
    # h*tau = 0.7 it stops reaching them and the divergence test below is what answers. The
    # residual is
    # measured on the bins the operator covers; that set is a property of the operator, not of
    # the iterate, so it is fixed once and asserted to stay fixed.
    first = solve(system.reporting, system.t_inf)
    covered = np.isfinite(first)

    def converge(fabricated: npt.NDArray[np.bool_] | None) -> npt.NDArray[np.floating]:
        """Drive the outer iteration to its fixed point, optionally suppressing some flux.

        Parameters
        ----------
        fabricated : ndarray of bool or None
            Per-segment, per-bin flux to suppress; see :func:`_update_targets`.

        Returns
        -------
        ndarray
            Reconstructed production series on the padded grid.

        Raises
        ------
        RuntimeError
            If the iteration diverges or exhausts ``max_sweeps``.
        """
        recovered, targets = first, system.t_inf
        increment, previous, best, growing = np.inf, np.inf, np.inf, 0
        history: list[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]] = []
        for _ in range(max_sweeps - 1):
            inner_atol = max(atol, _INNER_FORCING * increment)
            targets = _converge_targets(
                system,
                filled(recovered),
                max_sweeps=max_sweeps,
                atol=inner_atol,
                initial=targets,
                fabricated=fabricated,
            )
            mapped = solve(system.reporting, targets)
            if not np.array_equal(np.isfinite(mapped), covered):
                msg = "the reverse iterate changed which output bins the record covers; this is a bug, please report it"
                raise RuntimeError(msg)
            residual = mapped[covered] - recovered[covered]
            previous, increment = increment, float(np.max(np.abs(residual), initial=0.0))
            if increment <= atol:
                return recovered
            # A residual that keeps growing is not a slow iteration, it is the wrong regime,
            # and no cap or tolerance reaches the answer from there. Catch it while the
            # numbers are still finite: left alone the iterate overflows and the banded solve
            # raises about infs, which says nothing about the cause.
            growing = growing + 1 if increment > previous else 0
            best = min(best, increment)
            if growing >= _DIVERGENCE_STEPS and increment > 100.0 * best:
                raise RuntimeError(_diverged_message(system, previous, increment))
            history.append((recovered[covered], residual))
            del history[: -(_ANDERSON_DEPTH + 1)]
            recovered = mapped
            if len(history) > 1:
                past = np.column_stack([r for _, r in history])
                steps = np.column_stack([x for x, _ in history])
                gamma = np.linalg.lstsq(np.diff(past, axis=1), residual, rcond=None)[0]
                recovered[covered] -= (np.diff(steps, axis=1) + np.diff(past, axis=1)) @ gamma
        raise RuntimeError(_diverged_message(system, previous, increment, exhausted=max_sweeps))

    if max_sweeps == 1:
        return first[system.n_pad :]
    recovered = converge(None)

    # Bins no measurement constrains are filled in above so the flux pass has something to
    # read, and the coupling carries that invention into bins the record does constrain --
    # after the gap through the halo memory, and before it because the deconvolution couples
    # the whole record. Neither reach can be read off the band support. The model is affine,
    # so re-solving without the invented flux gives the exact imprint of it on the answer, and
    # the bins it moves by more than ``gap_atol`` are the ones the record does not support.
    # The one-way reverse needs none of this: with a fixed target it is exactly local.
    if not covered.all():
        reads_invented = apply_banded(system.internal, (~covered).astype(float)) > 0.0
        n_seg = len(system.length)
        honest = converge(reads_invented[:n_seg] | reads_invented[n_seg : 2 * n_seg] | reads_invented[2 * n_seg :])
        with np.errstate(invalid="ignore"):
            recovered = np.where(np.abs(honest - recovered) > gap_atol, np.nan, recovered)
    return recovered[system.n_pad :]
