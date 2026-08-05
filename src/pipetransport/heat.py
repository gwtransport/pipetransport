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
  the one-way model with a memory-shifted target: one convolution of the wall-flux history
  per segment, over the whole record rather than a window. The deficit decays only as
  ``1/t``, so there is no lag at which it can be truncated: cutting it at two months
  discards about 40 % of its mass and costs a seventh of the halo correction.
- **Exchange rate**: the series resistance of the water-side film, the pipe wall and the
  soil, divided by the water column's heat capacity, gives the relaxation rate [1/day] --
  the same ``1/(D**2 ln D)``-flavoured diameter law as chlorine wall decay: service lines
  equilibrate much faster than trunk mains.

The wall flux is carried as its leading axial Legendre modes per pipe -- ``n_modes`` of
them, six by default (issue #32). The soil columns along a pipe are independent and the
memory kernel is the same at every position, so a flux held in a few fixed shapes keeps its
memory in the same shapes: one deficit convolution per mode replaces an axial grid, the
higher modes entering the parcel reading as polynomial components of the target profile, in
the same closed cell forms as the mean. Mode 0 alone is the classical one-history model this
package carried before, and the count is a model order like a mesh rather than a tuning
knob.

The two-way coupling is a fixed point of two vectorized passes (one transport evaluation,
one deficit convolution per segment and mode). Sweep 1 relaxes toward the undisturbed soil
temperature at the steady buried-pipe rate -- exactly the classical one-way model, returned
by ``max_sweeps=1``; every further sweep shifts the targets by the deficit convolutions of
the latest flux-mode histories. Two choices make that iteration usable. The relaxation
rate keeps the *steady* resistance (folding the lag-0 deficit into an early-time resistance
-- the borehole-model move -- lands 5-20 % below the analytic steady buried-pipe law). And
the flux modes of a bin are the segment's own moment budgets over it, so each
parcel's heat is booked into the bins it actually occupied, at the positions it occupied.
Water that stands still exchanges heat for as long as it stands, and a bin with no
throughflow still leaks ``-h (H - V Tb)``; charging that to the single bin the water finally
left in would overstate it by the ratio of the standing time to the transit, which under
intermittent demand runs to several times the driving contrast.

That attribution also makes convergence a property of the model rather than of the
configuration. Every path through the sweep is causal in time and runs upstream to downstream
within a bin, so the iteration matrix is block lower triangular and its same-bin gains
``Dbar[0] V (1 - exp(-h dt)) / (L dt)`` sit strictly below one for every geometry and every
bin width -- for every mode alike, because the projection onto the shapes is idempotent. The
reading sensitivities on top of that slow the plain iteration to a crawl on wide trunks, so
the sweep extrapolates over its last few iterates exactly as the reverse direction does
(Anderson; the map is affine), which holds the count to a few tens of sweeps; the
convergence test reads the plain residual, so the answer is the map's own fixed point. The
one regime past the extrapolation is a pipe flushed several volumes per *bin*, whose modes
finer than the bin width can drive feed back faster than the sweep damps them; the sweep
refuses it by name rather than returning an unconverged answer. Numerically, the moment
recursions restart at chunk anchors from content moments read off the transport operator
itself, so a rounding error in the state cannot ride the advective shear coupling across
the halo memory -- what lets six modes hold round-off accuracy at every coupling strength
and record length. The whole model stays linear in the produced water temperature and the
surface temperatures, so the reverse problem reuses the existing banded solver.

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

Available class and functions:

- :class:`HeatNetwork` - A :class:`~pipetransport.network.PipeNetwork` whose segments carry
  everything constant in time: geometry, land cover, burial depth, wall, film and soil. The
  heat pair reads its physics off this table, so the calls carry only what varies on
  ``tedges``.
- :func:`sol_air_temperature` - The surface forcing per land cover: air temperature plus
  the absorbed solar radiation, less the longwave and latent losses, over the surface film.
- :func:`soil_temperature` - Exact bin-averaged undisturbed soil temperature at depth from
  a piecewise-constant surface series (Robin surface; ``radiation_length=0`` is Dirichlet).
- :func:`segment_heat_rate` - Per-segment exchange rate [1/day] from the wall and soil
  resistances, analogous to :func:`pipetransport.logremoval.segment_decay_rate`.
- :func:`source_to_endmember` - Delivered temperature at the reporting nodes, two-way by
  default; ``max_sweeps=1`` is the one-way model.
- :func:`endmember_to_source` - Reverse: production temperature from delivered
  temperatures, the same fixed point wrapped around the banded deconvolution.

The heat pair requires uniformly spaced ``tedges`` (the halo memory is a convolution) and a
:class:`HeatNetwork`, which validates the segment columns when it is built rather than when
it is solved.

Validity
--------

- The pipe term of the halo is the constant-flux *cylinder* response, and it is the one
  quantity in the module that is not closed-form. It is evaluated by quadrature along the
  branch cut of its Laplace transform, once per distinct ``alpha dt / r_o**2`` at build
  time, and holds about 1e-13 relative over every Fourier number reachable here. The image
  keeps its line source, and the whole conceptual model -- cylinder, line image, saturation at
  ``ln(2 depth/r_o)`` -- has been measured against an independent 2-D solve of the true
  boundary-value problem, which carries no image and is never told a resistance (issue #43).
  Steadily it costs ``(r_o/2 depth)**2/(2 pi kappa)`` with a coefficient measured at 1.00,
  0.995 and 0.98 for ``depth/r_o`` of 21, 5.3 and 2.5: 1.5e-4 of ``R_soil`` for a 100 mm
  service line buried a metre, 3.7e-3 for a 400 mm main, 2.4e-2 at two and a half radii. The
  sign says ``ln(2 depth/r_o)`` sits midway between the two wall conditions, above the
  isothermal ``acosh(depth/r_o)`` by what it falls short of the uniform-flux answer by.
  Transiently the gap is larger, not smaller: the true surface starts cooling the wall before
  the line image does -- the image is read from the axis at ``2 depth`` while the near side of
  the wall sees its own at ``2(depth - r_o)`` -- so while it arrives the model over-credits the
  resistance that has reached the wall, by 2.7 times the steady gap over the geometries
  measured (``depth/r_o`` of 2.5 to 21, the ratio itself growing about as ``ln(2 depth/r_o)``
  while the absolute error shrinks), returning to it only as ``1/t``. All of it grows as the
  burial approaches ``r_o``. Before the surface is felt at all the kernel is exact to 3e-6 of
  ``R_soil``, which is that same solve confirming the cylinder quadrature by a route sharing
  nothing with it. The surface film is *not* approximated: a radiating surface images a source
  as a positive mirror at the true ``2 depth`` plus a tail of sinks at ``2 depth + s`` weighted
  ``2 beta exp(-beta s)``, ``beta = eta/kappa``, and that whole distribution is what the halo
  carries (issue #49). It sums in closed form steadily and reuses the same ``E1`` in time, so
  the cost is one Gauss-Laguerre rule; both classical surfaces fall out of it without a branch,
  ``eta = inf`` giving the Dirichlet sink and ``eta -> 0`` the insulated mirror. What this
  replaced -- a perfect surface displaced down by the radiation length -- understated the
  resistance by 2.0e-4 day/m² under a 20 W/(m² K) surface but by 0.26, about 1 % of ``R_soil``,
  under a 1 W/(m² K) one.
  What the cylinder buys is the first lag bin, where a line source read at ``r = r_o``
  understates the arrived resistance badly because the heat has not yet diffused past the
  pipe: ``Dbar[0]/R_soil`` is 0.8568 for a 100 mm service line and 0.9323 for a 400 mm main
  on hourly bins, against 0.9418 and 1.0000 for the line source -- the second of which left
  the same-bin loop gain no margin at all. That ratio sets the sweep count and how far the
  fixed point sits from singular, so the swap cuts the sweep count severalfold as well:
  100 against 286 on the example network, 159 against 660 on a 400 mm main.
- **Six axial modes per pipe (the default).** The wall flux falls along a pipe like
  ``exp(-h tau)`` -- by a factor 1.6 over a 2 h transit on a 100 mm service line -- and the
  model resolves that profile in its leading Legendre modes. What it truncates is the
  profile past the highest mode, a stated model order with no tuning parameter, and the
  mode ladder measures it. Flowing steadily, two modes already sit at the 0.01 K floor the
  adjudicating comparison's own sub-stepping leaves on shared grids (0.008 K on a 100 mm
  line at a 2 h transit). Under an overnight duty cycle -- the case one history gets badly
  wrong -- the delivered excursion past the soil falls from 26 % of the driving contrast
  at one mode through 8 % at two to under 1 % at the default, the band the Eulerian
  duty-cycle adjudication of issue #32 puts the class truncation at (2 % of contrast for
  six modes against a fully resolved axial grid). Declaring a pipe as series pieces buys
  nothing the modes do not already resolve; it remains exact for the transport.
- The relaxation target is an *effective driving temperature*, not a wall temperature. The
  rate keeps the steady soil resistance, which overstates the resistance while the halo is
  still developing, so the target has to be pushed past the undisturbed soil to reproduce
  the faster early exchange -- measured excursions of several times the driving contrast in
  the bins after a sharp change in the wall flux, and larger per mode as the profile
  sharpens. The delivered temperature is a weighted average of ``tin`` and those targets,
  so it can leave the range of its own inputs; only ``max_sweeps=1`` is guaranteed inside
  it. At the six-mode default the share that reaches the delivered water is small. A step
  in ``tin`` into a *continuously flowing* pipe: 2.4 % of the instantaneous plant-to-soil
  contrast, where the classical one-history model paid 18 %. A line idle 8 h a day, the
  case the flux attribution and the modes carry together: under 1 % of the contrast past
  the soil, from 26 % at one mode and 8 % at two. On the harder shapes of issue #24, a
  line standing ten days delivers within half a kelvin of the range of its inputs, where
  a delivered-water flux attribution once reached 8.8 times the contrast, converged and
  unflagged; and the main flushed 2 h in every 24 -- six pipe volumes through in a single
  bin -- carries its range in the leading modes alone and is refused by name at mode
  counts its bin width cannot drive.

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
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import fftconvolve, lfilter
from scipy.special import erfc, erfcx, exp1, hyperu, j1, roots_laguerre, y1

from pipetransport._transfer import (
    NetworkTransfer,
    _e_table,
    _legendre_monomial,
    _running_start,
    apply_banded,
    apply_content_snapshots,
    apply_segment_targets,
    pad_paths,
    paths_transfer,
    resolve_spinup,
)
from pipetransport._validation import _validate_no_nan, _validate_positive, _validate_tedges
from pipetransport.network import PipeNetwork
from pipetransport.utils import solve_inverse_transport_banded, tedges_to_days

# How far ahead of the outer iterate the reverse direction resolves its inner halo fixed
# point. The inner solve is inexact while the reconstruction is still moving and exact once it
# has settled, so the answer is the same to well under ``atol``; at 1e-2 the reverse direction
# spends about a third of the inner sweeps that a fully resolved inner solve does.
_INNER_FORCING = 1e-2
# How far a segment's volume may sit from the one its length and diameter imply. Wide enough
# for fittings and for wall-thickness conventions, tight enough to catch a volume that came
# from somewhere else entirely.
_GEOMETRY_TOLERANCE = 0.05
# Anderson window on the reverse outer iterate. Truncated Anderson is truncated GMRES, so
# the window is not a tuning knob with a free choice: too short a memory drops the directions
# the iteration needs and it stalls. A depth of five raises on a 400 mm main at a half-hour
# transit, which ten reconstructs to 1e-9. It buys range, not a guarantee: a pipe past the
# regimes the divergence test names is out of reach at any depth.
_ANDERSON_DEPTH = 10
# Consecutive growing outer residuals that mark divergence rather than a slow start.
_DIVERGENCE_STEPS = 5
# Where the reverse two-way fixed point stops being reachable, measured by power iteration on
# the outer affine map for a single observed pipe at constant flow -- the worst case -- in
# issue #40. Two axes put the spectral radius past one independently, and the radius is a
# property of the configuration, not of the iteration: unchanged from 6 to 24 days of record
# and from 30-minute to 2-hour bins at fixed coupling. Coupling: at transits of a few bins or
# more the radius crosses one near h*tau = 0.7 (0.92 at 0.50 and 1.10 at 0.75 on a 100 mm
# main; a 400 mm main crosses nearer 1.0), the excess is broad-band, and no Anderson window
# recovers it. Transit: on the one-history model a segment that emptied in about a bin left
# the deconvolution nearly singular at the fastest alternation the record carries whatever
# the coupling (radius 21 at h*tau = 0.11 for a 100 mm main at a half-bin transit); the
# advected mode kernels tamed that map -- the same pipe now reconstructs to 5e-8 K -- and
# the transit diagnosis remains for configurations that still reach it.
_COUPLING_LIMIT = 0.7
_SHORT_TRANSIT_BINS = 1.5
# Gauss-Laguerre rule for the Robin image tail, ``2 beta int exp(-beta s) f(2 depth + s) ds``.
# The integrand is a diffusion kernel read at a distance that grows with the node, so the rule
# converges faster the stronger the film: at the 20 W/(m² K) of a typical surface eight nodes
# already hold 1e-13 day/m², and the count is set by the weakest film worth serving. At
# 1 W/(m² K) -- a still, insulating cover, where the tail reaches ten times as far -- eight
# nodes hold only 2e-4 and twenty 2e-7, while forty reach 8e-11 against a 320-node rule.
_IMAGE_NODES = roots_laguerre(40)


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
    depth: npt.NDArray[np.floating],
    alpha: npt.NDArray[np.floating],
    kappa: npt.NDArray[np.floating],
    eta: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Bin-averaged transient deficit ``Dbar[m]`` of every segment, shape ``(n_seg, n_bins)``.

    The wall-temperature step response is the constant-flux cylinder at the outer pipe radius
    (:func:`_cylinder_integral`) minus the half space's image of it, which is what makes the
    halo saturate instead of growing without bound.

    A radiating surface does not mirror a source into a single sink. Its exact image is a
    *positive* mirror at the true ``2 depth`` followed by a tail of sinks continuing upward,
    at ``2 depth + s`` with weight ``2 beta exp(-beta s)`` and ``beta = eta/kappa`` the inverse
    radiation length. Both classical limits fall out of that one expression without a branch:
    as ``beta -> inf`` the tail collapses onto the mirror and flips it to the familiar
    Dirichlet sink, and at ``beta = 0`` the bare positive mirror is the insulated surface.

    Steadily the tail sums in closed form, so the halo saturates at

    ``R_inf = (ln(2 depth/r_o) + 2 exp(x) E1(x)) / (2 pi kappa)``,  ``x = 2 depth eta / kappa``

    with ``2 exp(x) E1(x) = 2 hyperu(1, 1, x)``, which is exactly 0 at ``eta = inf``. In time
    the ``s`` integral commutes with the time integral, so each image in the tail contributes
    the same closed form :func:`_halo_integral` the single sink used to, and the tail becomes a
    Gauss-Laguerre rule over ``s``.

    What the conceptual model as a whole costs against the true boundary-value problem is
    measured, not assumed, in ``test_heat.py``'s two-dimensional reference:
    ``(r_o/2 depth)**2/(2 pi kappa)`` at steady state with a coefficient of 1.00 to 0.98 over
    ``depth/r_o`` from 21 down to 2.5, and 2.7 times that transiently while the image arrives.
    Both grow as the burial approaches ``r_o``.

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
    r_o, depth, alpha, kappa, eta : ndarray
        Outer radius [m], burial depth to the axis [m], diffusivity and conductivity ratio
        [m²/day], and surface film coefficient [m/day] of every segment, length ``n_seg``.

    Returns
    -------
    ndarray
        ``Dbar`` [day/m²] per segment and lag bin; ``Dbar[:, 0]`` is ``R_inf - Gbar(dt)``.
    """
    mirror = 2.0 * depth
    r_inf = ((np.log(mirror / r_o) + 2.0 * hyperu(1, 1, mirror * eta / kappa)) / (2.0 * np.pi * kappa))[:, None]
    edge = np.arange(n_bins + 1)[None, :]
    lag = dt_days * edge
    fo_bin, segment_of = np.unique(alpha * dt_days / r_o**2, return_inverse=True)
    cylinder = _cylinder_integral(fo_bin[:, None] * edge)[segment_of]

    # The image tail, ``2 beta int exp(-beta s) I(c(s)) ds``, as a Gauss-Laguerre rule in
    # ``u = beta s``. Accumulated one node at a time rather than broadcast: the full
    # (segment, node, lag) array would be forty times the kernel it collapses to, and a year of
    # hourly lag bins already fills the kernel. ``eta = inf`` puts every node back at the
    # mirror, where the rule sums to ``2 I(c(0))`` and leaves the Dirichlet sink -- so the
    # neutral element needs no branch here either.
    offsets = mirror[:, None] + _IMAGE_NODES[0] / (eta / kappa)[:, None]
    image = -_halo_integral((np.square(mirror) / (4.0 * alpha))[:, None], lag)
    for weight, offset in zip(_IMAGE_NODES[1], offsets.T, strict=True):
        image += 2.0 * weight * _halo_integral((np.square(offset) / (4.0 * alpha))[:, None], lag)

    cumulative = r_inf * lag + image / (4.0 * np.pi * kappa[:, None]) - (r_o**2 / (kappa * alpha))[:, None] * cylinder
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
    radiation_length: float = 0.0,
    t_pre: float | None = None,
) -> npt.NDArray[np.floating]:
    """Compute the bin-averaged undisturbed soil temperature at depth from a surface series.

    Exact superposition of half-space step responses: the piecewise-constant surface
    (sol-air) series steps at each of its bin edges, each step arrives at depth as an
    erfc-family response, and the average over every output bin uses the closed-form time
    integral -- no quadrature, no grid. Before ``tedges[0]`` the surface is uniformly at
    ``t_pre``, so the record opens from the soil state that history has settled into. A
    one-week heatwave arrives at 1 m attenuated to roughly a quarter to a third of its
    surface amplitude, peaking a day or two after it ends; the annual wave arrives at about
    two-thirds amplitude, three to four weeks delayed.

    Parameters
    ----------
    surface_temperature : array-like
        Sol-air temperature per surface bin (see :func:`sol_air_temperature`), length
        ``len(tedges) - 1``.
    tedges : pandas.DatetimeIndex
        Bin edges of both the surface series and the returned soil series, uniformly spaced
        (the superposition over surface steps is a convolution). One value per bin in, one
        per bin out.
    depth : float
        Depth below the surface [m], positive.
    alpha : float
        Soil thermal diffusivity [m²/day], positive.
    radiation_length : float, optional
        ``kappa / eta`` [m], the only combination of the soil conductivity ratio and the
        surface film coefficient the kernel depends on: it sets the Robin surface condition
        by displacing the surface downward by that length. Default 0.0, a
        prescribed-temperature (Dirichlet) surface -- the ``eta = inf`` limit, reached
        without a branch.
    t_pre : float or None, optional
        Uniform surface temperature before the record, the state the first surface step is
        measured against. Defaults to the first surface value -- a record that opens
        settled. The record's own history takes over as it arrives at depth: after one year
        roughly 87 % of a step has reached 1 m, after three weeks about half, so the early
        bins lean on ``t_pre`` in proportion. To serve a longer history, evaluate on the
        longer ``tedges`` and slice the output -- refining and extending the record is the
        caller's job, per the one-grid contract.

    Returns
    -------
    ndarray
        Soil temperature at ``depth``, one bin-average per ``tedges`` bin, in the unit of
        ``surface_temperature``.

    Raises
    ------
    ValueError
        If ``tedges`` is malformed or not uniformly spaced, the series holds NaN, ``depth``
        or ``alpha`` is not positive, or ``radiation_length`` is negative.

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
    _validate_tedges(tedges, surface, tedges_name="tedges", values_name="surface_temperature")
    if np.unique(np.diff(tedges.asi8)).size != 1:
        msg = (
            "tedges must be uniformly spaced (the step response depends only on the lag, "
            "which makes the superposition over surface steps a convolution)"
        )
        raise ValueError(msg)
    _validate_no_nan(surface, name="surface_temperature")
    _validate_positive(depth, name="depth")
    _validate_positive(alpha, name="alpha")
    if not radiation_length >= 0.0:
        msg = "radiation_length must be non-negative (0 is a prescribed-temperature surface)"
        raise ValueError(msg)
    pre = float(surface[0]) if t_pre is None else float(t_pre)

    steps = np.diff(surface, prepend=pre)
    n = len(surface)
    dt_days = (tedges[1] - tedges[0]) / pd.Timedelta(days=1)
    # The response depends on the lag alone, so the (output edge, surface step) pairs take
    # ``2 n`` distinct lag values rather than their product: the superposition is a
    # convolution over those. A year of hourly forcing costs 17_520 kernel evaluations
    # instead of 77 million, and a vector instead of an N^2 matrix that had to be chunked
    # to fit.
    lag = (np.arange(2 * n) - (n - 1)) * dt_days
    kernel = _step_response_integral(lag, depth=depth, alpha=alpha, radiation_length=radiation_length)
    # Differenced to a bin-averaged response before the convolution rather than after it, so
    # the transform's round-off is not amplified by the 1/dt of the final differencing.
    response = fftconvolve(np.diff(kernel) / dt_days, steps)[n - 1 : 2 * n - 1]
    return pre + response


class HeatNetwork(PipeNetwork):
    """A :class:`~pipetransport.network.PipeNetwork` whose segments carry the buried-pipe heat properties.

    The heat pair reads everything that is constant in time off this table -- geometry,
    burial, wall, film and soil -- so its call signatures carry only what varies on
    ``tedges``. Validation happens here, when the network is built, rather than at solve
    time. Soil properties are per segment, not per cover class: two pipes may share a cover
    and still sit in different ground, and ``cover`` is left one job, keying the surface
    forcing.

    Parameters
    ----------
    segments : mapping or pandas.DataFrame
        The pipe segments as :class:`~pipetransport.network.PipeNetwork` takes them, each
        carrying in addition:

        ``length``, ``diameter``
            Required [m] (inner diameter). Heat reads the pipe three ways -- the exchange
            rate from the diameter, the wall flux from the length, the transit from the
            volume -- so a volume-only table is rejected, and a ``volume`` column that its
            length and diameter do not imply is too.
        ``cover``
            Required. Land-cover class; the key into ``surface_temperature``.
        ``alpha``, ``kappa_soil``
            Required. Soil thermal diffusivity and conductivity over the volumetric heat
            capacity of water [m²/day].
        ``depth``
            Burial depth to the pipe axis [m]. Default 1.0. It must put the pipe below the
            surface, ``depth > r_o``; the exact shape factors do not exist below that and the
            ``ln(2 depth/r_o)`` used here runs away to a divergent rate rather than to an
            error. That log is the large-``d/r`` limit of both exact factors, and it sits
            between them: ``1/(4 (depth/r_o)**2)`` above the isothermal-wall
            ``acosh(depth/r_o)`` and, *while* ``depth >> r_o``, the same amount below the
            uniform-flux wall this model actually imposes (measured in ``test_heat.py`` at
            ``depth/r_o`` of 21, 5.3 and 2.5). So accuracy is a matter of how far above the
            guard the pipe is: 0.015 % for a 100 mm service line and 0.4 % for a 400 mm main
            at a metre, 5.3 % at ``depth = 2 r_o``, 24 % for a DN1600 there, and 117 % for a
            DN2000 -- leaving the default depth on a transmission main is the mis-entry to
            watch. Those last figures are against ``acosh``; near the guard the two wall
            conditions part company entirely (as ``depth -> r_o`` the isothermal resistance
            goes to zero while the uniform-flux one stays finite), so against the flux wall
            the same two rows read 13 % and 25 %. Either way the pipe is in the wrong regime.
            The guard reads the *physical* depth: the surface film no longer displaces it, so
            a shallow pipe under a poor surface that once slipped past is now refused.
        ``eta``
            Surface film coefficient [m/day]. Default ``inf``, a prescribed-temperature
            surface, which the exact image reaches without a branch: ``hyperu(1, 1, inf)`` is
            exactly 0, leaving the classical sink at ``2 depth``. Lowering it lengthens the
            tail of images above the surface and raises the soil resistance -- by 4 % at
            5 W/(m² K) and 21 % at 1, against a perfect surface.
        ``wall_thickness``, ``kappa_pipe``
            Pipe wall geometry [m] and conductivity over the water heat capacity [m²/day].
            Defaults 0.0 and ``inf``, which together are exactly the bare pipe: the wall
            resistance vanishes and the soil is read at the inner radius. PE ~0.008, PVC
            ~0.0035 m²/day. At a fixed SDR the wall resistance
            ``ln(SDR/(SDR - 2)) / (2 pi kappa_pipe)`` does not depend on the diameter while
            the soil resistance falls with it, so the wall's share *grows* with the pipe
            rather than shrinking: at ``kappa_soil = 0.025`` and ``depth = 1`` it is 10-17 %
            of the soil resistance for PE SDR17 and 18-30 % for PVC SDR21 across 100-400 mm,
            lowering the rate by 6-10 % and 13-20 % respectively. Both shares scale with
            ``kappa_soil / kappa_pipe``, so rescale them for your own soil; PVC is never the
            ~10 % that PE is.
        ``film_coefficient``
            Water-side film coefficient ``h_film / (rho c_w)`` [m/day]. Default ``inf``, the
            film not limiting. Like the mass transfer coefficient of
            :func:`pipetransport.logremoval.segment_decay_rate` it depends on velocity, so
            give a value representative of the operating range (see issue #46). It is
            negligible for turbulent trunk mains (well under 1 %) but not at the low
            night-time flows of a service line: fully developed laminar flow (``Nu = 3.66``,
            ``h_film = 3.66 k_water / D``, taking ``k_water = 0.6 W/(m K)``) gives
            ``film_coefficient = 0.454 m/day`` in a 100 mm pipe, about 29 % of its soil
            resistance at ``kappa_soil = 0.025`` and ``depth = 1``.
    source : str
        Name of the production node.

    Raises
    ------
    ValueError
        Everything :class:`~pipetransport.network.PipeNetwork` raises, plus: a missing
        required column, a non-positive parameter (``eta``, ``kappa_pipe`` and
        ``film_coefficient`` admit ``inf``; ``wall_thickness`` admits 0), a ``volume`` that
        contradicts the geometry, or a burial that does not clear the outer radius.

    See Also
    --------
    source_to_endmember : Forward direction, which reads this table.
    segment_heat_rate : The exchange rate this table implies, as a standalone diagnostic.
    pipetransport.examples.example_heat_network : A ready-made four-endmember heat network.

    Examples
    --------
    >>> from pipetransport.heat import HeatNetwork, segment_heat_rate
    >>> grass = {"cover": "grass", "alpha": 0.05, "kappa_soil": 0.025, "eta": 0.41}
    >>> paved = {"cover": "paved", "alpha": 0.075, "kappa_soil": 0.035, "eta": 0.41}
    >>> segments = {
    ...     "Plant-A": {
    ...         "from": "Plant",
    ...         "to": "A",
    ...         "length": 2000.0,
    ...         "diameter": 0.40,
    ...         **grass,
    ...     },
    ...     "A-T1": {
    ...         "from": "A",
    ...         "to": "T1",
    ...         "length": 800.0,
    ...         "diameter": 0.15,
    ...         **paved,
    ...     },
    ... }
    >>> network = HeatNetwork(segments=segments, source="Plant")
    >>> float(network.segments.loc["Plant-A", "depth"])  # filled in
    1.0
    >>> round(segment_heat_rate(network=network)["A-T1"], 2)
    3.7
    """

    def __init__(self, *, segments: Mapping[str, Mapping] | pd.DataFrame, source: str) -> None:
        super().__init__(segments=segments, source=source)
        segments = self.segments  # the validated copy the base class stored
        required = ("length", "diameter", "cover", "alpha", "kappa_soil")
        missing = [c for c in required if c not in segments.columns]
        if missing:
            msg = f"HeatNetwork segments need column(s): {missing}"
            raise ValueError(msg)
        # A mapping whose segments do not all carry the same keys arrives with NaN in the
        # gaps, so an omission has to be named as one rather than surfacing as a NaN rate.
        blank = segments[list(required)].isna()
        if blank.to_numpy().any():
            gaps = [f"{name!r} is missing {sorted(blank.columns[row])}" for name, row in blank.iterrows() if row.any()]
            msg = f"every segment needs {list(required)}; segment {'; segment '.join(gaps)}"
            raise ValueError(msg)
        # PipeNetwork accepts a volume-only table and, given a volume, never looks at the
        # geometry; heat reads the pipe three ways, so the geometry has to be there, be
        # positive, and agree with any volume it arrived with. Where it does not, the water's
        # heat capacity per unit length disagrees with the area the exchange rate was built
        # from, and what a user used to see was a convergence failure naming knobs that
        # cannot reconcile a geometry.
        _validate_positive(segments["length"], name="segment length")
        _validate_positive(segments["diameter"], name="segment diameter")
        geometric = (
            np.pi / 4.0 * segments["diameter"].to_numpy(dtype=float) ** 2 * segments["length"].to_numpy(dtype=float)
        )
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
        # The neutral elements of the three resistances, so every downstream read finds one
        # schema and no branch: a zero wall thickness IS the bare pipe, and an infinite
        # conductance contributes exactly no resistance.
        for column, default in (
            ("depth", 1.0),
            ("eta", np.inf),
            ("wall_thickness", 0.0),
            ("kappa_pipe", np.inf),
            ("film_coefficient", np.inf),
        ):
            # Absent for every segment or only for some: either way the gap is the default,
            # so one pipe may carry a wall while its neighbours omit one.
            segments[column] = default if column not in segments.columns else segments[column].fillna(default)
        _validate_positive(segments["alpha"], name="alpha")
        _validate_positive(segments["kappa_soil"], name="kappa_soil")
        _validate_positive(segments["depth"], name="depth")
        for column in ("eta", "kappa_pipe", "film_coefficient"):
            values = segments[column].to_numpy(dtype=float)
            if not np.all((values > 0.0) | np.isposinf(values)):
                msg = f"{column} must be positive (inf is the no-resistance limit)"
                raise ValueError(msg)
        thickness = segments["wall_thickness"].to_numpy(dtype=float)
        if not np.all(thickness >= 0.0):
            msg = "wall_thickness must be non-negative (0 is the bare pipe)"
            raise ValueError(msg)
        r_o = segments["diameter"].to_numpy(dtype=float) / 2.0 + thickness
        if not np.all(segments["depth"].to_numpy(dtype=float) > r_o):
            msg = "burial depth must exceed the outer pipe radius (depth > r_o); the line-source geometry needs d >> r"
            raise ValueError(msg)


def segment_heat_rate(*, network: HeatNetwork) -> dict[str, float]:
    """Compute the per-segment heat exchange rate [1/day] from the wall and soil resistances.

    The water column loses heat through the water-side film, the pipe wall and the soil in
    series; dividing the conductance per length by the water's heat capacity per length
    gives the first-order relaxation rate

    ``h = 1 / ((R_film + R_wall + R_soil) * pi * r_i**2)``, with
    ``R_film = 1 / (2 pi r_i * film_coefficient)``,
    ``R_wall = ln(r_o/r_i) / (2 pi kappa_pipe)`` and
    ``R_soil = (ln(2 depth / r_o) + 2 exp(x) E1(x)) / (2 pi kappa_soil)``,
    ``x = 2 depth eta / kappa_soil``.

    ``R_soil`` is the steady buried-pipe resistance -- the fully developed halo, whose
    saturation the mirror image above the surface provides. The rate inherits the
    ``1/(D**2 ln D)`` diameter law: a 100 mm service line relaxes an order of magnitude
    faster than a 400 mm trunk main.

    Every term reduces at its own neutral element without a branch: a zero ``wall_thickness``
    leaves ``ln(r_o/r_i)`` exactly zero and reads the soil at the inner radius, and an
    infinite ``kappa_pipe``, ``film_coefficient`` or ``eta`` contributes exactly no
    resistance. Those are the defaults :class:`HeatNetwork` fills in.

    Parameters
    ----------
    network : HeatNetwork
        Network whose segments carry the geometry, burial, wall, film and soil columns; see
        :class:`HeatNetwork` for what each means and how to choose it.

    Returns
    -------
    dict of str to float
        Exchange rate [1/day] keyed by segment name.

    Raises
    ------
    TypeError
        If ``network`` is a plain :class:`~pipetransport.network.PipeNetwork`, which carries
        none of the columns this reads.

    See Also
    --------
    pipetransport.logremoval.segment_decay_rate : The chlorine analogue of this helper.
    source_to_endmember : Consumes these rates internally.

    Examples
    --------
    Small pipes equilibrate much faster: a 100 mm service line runs a tenfold higher rate
    than the 400 mm trunk main.

    >>> from pipetransport.examples import example_heat_network
    >>> from pipetransport.heat import segment_heat_rate
    >>> rates = segment_heat_rate(network=example_heat_network())
    >>> round(rates["C-T4"], 2), round(rates["Plant-A"], 2)
    (5.65, 0.49)
    """
    if not isinstance(network, HeatNetwork):
        msg = "segment_heat_rate reads the heat columns of a HeatNetwork; wrap your segment table in one"
        raise TypeError(msg)
    segments = network.segments
    r_i = segments["diameter"].to_numpy(dtype=float) / 2.0
    r_o = r_i + segments["wall_thickness"].to_numpy(dtype=float)
    kappa_soil = segments["kappa_soil"].to_numpy(dtype=float)
    mirror = 2.0 * segments["depth"].to_numpy(dtype=float)
    # ``x / inf`` is exactly 0.0, ``ln(r_i/r_i)`` is exactly 0.0 and ``hyperu(1, 1, inf)`` is
    # exactly 0.0, so the neutral elements need no branch -- and dividing by inf is not a
    # divide error, so no errstate either.
    film = 1.0 / (2.0 * np.pi * r_i * segments["film_coefficient"].to_numpy(dtype=float))
    wall = np.log(r_o / r_i) / (2.0 * np.pi * segments["kappa_pipe"].to_numpy(dtype=float))
    soil = (np.log(mirror / r_o) + 2.0 * hyperu(1, 1, mirror * segments["eta"].to_numpy(dtype=float) / kappa_soil)) / (
        2.0 * np.pi * kappa_soil
    )
    rate = 1.0 / ((film + wall + soil) * np.pi * r_i**2)
    return {str(name): float(value) for name, value in zip(segments.index, rate, strict=True)}


class _HeatSystem(NamedTuple):
    """Everything the Picard loop reads, built once per call.

    Every per-segment array is indexed by the user's own pipes: one row each, carrying the
    ``n_modes`` axial Legendre moments of one wall-flux profile -- the modes that are the
    model's spatial resolution along a pipe.

    ``internal`` holds ``4 n_modes - 1`` rows per segment, all binned on that pipe's own
    deliveries: the delivered temperature, its readings against the time-moment weights
    ``(t_end - t)**p exp(-h (t_end - t))`` for ``p < n_modes`` and against their running
    integrals for ``p < n_modes - 1``, and the same two weighted families taken at the
    pipe's entry. The plain reading closes the advective part of every bin's moment
    budgets, the time-moment family closes the exact bin update of the moment hierarchy,
    and the integrated family closes the bins' time-integrated content, which the flux
    moments read; see :func:`_update_targets`.
    """

    nodes: tuple[str, ...]
    n_pad: int
    n_bins: int
    n_modes: int
    dt_days: float
    t_inf: npt.NDArray[np.floating]
    dbar_spectrum: npt.NDArray[np.complexfloating]
    halo_length: int
    seg_flow: npt.NDArray[np.floating]
    length: npt.NDArray[np.floating]
    volume: npt.NDArray[np.floating]
    rate: npt.NDArray[np.floating]
    rho: npt.NDArray[np.floating]
    theta: npt.NDArray[np.floating]
    nmat: npt.NDArray[np.floating]
    bin_moment: npt.NDArray[np.floating]
    bin_integrated: npt.NDArray[np.floating]
    parent: npt.NDArray[np.intp]
    held_slope: npt.NDArray[np.floating]
    held_offset: npt.NDArray[np.floating]
    h_tau: npt.NDArray[np.floating]
    segment_names: tuple[str, ...]
    internal: NetworkTransfer
    reporting: NetworkTransfer


def _build_system(
    *,
    flow: Mapping[str, npt.ArrayLike],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    network: HeatNetwork,
    surface_temperature: Mapping[str, npt.ArrayLike],
    report_nodes: list[str] | tuple[str, ...] | None,
    spinup: str | None,
    n_modes: int,
) -> _HeatSystem:
    """Validate the shared inputs and build the operators, targets and kernels once.

    Returns
    -------
    _HeatSystem
        Everything the fixed-point iteration and both public directions read.

    Raises
    ------
    TypeError
        If ``network`` is not a :class:`HeatNetwork`.
    ValueError
        If a time axis is malformed or non-uniform, a cover class is unmapped, or a requested
        node is not part of the network.
    """
    if not isinstance(network, HeatNetwork):
        msg = "the heat pair reads the buried-pipe columns of a HeatNetwork; wrap your segment table in one"
        raise TypeError(msg)
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    demand = network.flow_array(flow)
    _validate_tedges(tedges, demand, tedges_name="tedges", values_name="flow")
    _validate_tedges(cout_tedges, np.empty(len(cout_tedges) - 1), tedges_name="cout_tedges", values_name="tout")
    if np.unique(np.diff(tedges.asi8)).size != 1:
        msg = "tedges must be uniformly spaced for the heat pair (the halo memory is a convolution over lag bins)"
        raise ValueError(msg)

    segments = network.segments
    covers = segments["cover"]
    # ``surface_temperature`` is a lookup table, so classes no pipe is buried under are
    # ignored rather than rejected; only the ones the segments ask for have to be there.
    surface_missing = [c for c in covers.unique() if c not in surface_temperature]
    if surface_missing:
        msg = f"surface_temperature is missing cover class(es): {surface_missing}"
        raise ValueError(msg)
    for cover in covers.unique():
        _validate_tedges(
            tedges,
            np.asarray(surface_temperature[cover], dtype=float),
            tedges_name="tedges",
            values_name=f"surface_temperature[{cover!r}]",
        )

    requested = tuple(network.endmembers) if report_nodes is None else tuple(report_nodes)
    unknown = [node for node in requested if node not in network.paths]
    if unknown:
        msg = f"unknown node(s): {unknown}; network nodes are {list(network.nodes)}"
        raise ValueError(msg)

    alpha_seg = segments["alpha"].to_numpy(dtype=float)
    kappa_seg = segments["kappa_soil"].to_numpy(dtype=float)
    eta_seg = segments["eta"].to_numpy(dtype=float)
    depth_seg = segments["depth"].to_numpy(dtype=float)

    # Spin-up, exactly as network_transfer resolves it: each endmember's travel time at the
    # leading flow rate, with resolve_spinup dropping the paths it cannot warm-start one by
    # one, so a single stagnant or unreachably deep branch does not suppress the padding of
    # the whole call. Every internal node sits on an endmember path, so those paths bound the
    # internal rows too -- and the candidate list is the endmembers whatever ``nodes`` asks
    # for, so no row's coverage depends on which nodes were requested.
    volume = segments["volume"].to_numpy(dtype=float)
    seg_of = {name: e for e, name in enumerate(segments.index)}
    with np.errstate(divide="ignore"):
        ratio = volume / network.segment_flow(flow=_running_start(demand)[:, None])[:, 0]
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

    # Undisturbed soil temperature per segment. The field depends on the cover class and the
    # four soil parameters only, and a network normally holds a handful of distinct
    # combinations against many segments, so it is solved once per combination and broadcast.
    t_inf = np.empty((len(segments), n_bins))
    cover_names = covers.to_numpy()
    for cover, depth, alpha, kappa, eta in (
        segments[["cover", "depth", "alpha", "kappa_soil", "eta"]].drop_duplicates().itertuples(index=False)
    ):
        rows = (
            (cover_names == cover)
            & (depth_seg == depth)
            & (alpha_seg == alpha)
            & (kappa_seg == kappa)
            & (eta_seg == eta)
        )
        record = np.asarray(surface_temperature[cover], dtype=float)
        # The warm-start prefix sees the record's opening surface held constant -- the same
        # constant-history policy the spin-up applies to flow and tin.
        t_inf[rows] = soil_temperature(
            surface_temperature=np.concatenate([np.full(n_pad, record[0]), record]),
            tedges=tedges_p,
            depth=float(depth),
            alpha=float(alpha),
            # kappa / inf is exactly 0.0: a prescribed-temperature surface, no branch.
            radiation_length=float(kappa) / float(eta),
        )

    rate = np.array([segment_heat_rate(network=network)[str(name)] for name in segments.index])
    r_o = segments["diameter"].to_numpy(dtype=float) / 2.0 + segments["wall_thickness"].to_numpy(dtype=float)
    dbar = _deficit_kernel(n_bins, dt_days, r_o=r_o, depth=depth_seg, alpha=alpha_seg, kappa=kappa_seg, eta=eta_seg)
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

    # Internal rows, ``4 n_modes - 1`` per segment and all binned on that pipe's own
    # deliveries. Every bin's moment budgets need the water it delivered, the exact bin
    # update of the moment hierarchy reads both faces against the time-moment weights
    # ``(t_end - t)**p exp(-h (t_end - t))``, and the flux moments read the bins'
    # time-integrated content through the running integrals of the same kernels. The
    # inflow rows run to the pipe's entry node, so a root segment reads its own source
    # series across an empty path.
    entry_chains = [chain(str(segments.loc[name, "from"])) for name in segments.index]
    delivery = [np.concatenate([up, [e]]).astype(np.intp) for e, up in enumerate(entry_chains)]
    specs = [(0.0, 0, False)] + [(1.0, p, False) for p in range(n_modes)] + [(1.0, p, True) for p in range(n_modes - 1)]
    entry_specs = specs[1:]
    int_paths, int_active = pad_paths(delivery * len(specs) + entry_chains * len(entry_specs))
    rep_paths, rep_active = pad_paths([chain(node) for node in requested])
    rep_flow = network.node_flow(flow=demand_p, nodes=requested)

    def build(
        paths_idx: npt.NDArray[np.intp],
        active: npt.NDArray[np.bool_],
        node_flow: npt.NDArray[np.floating],
        cout: npt.NDArray[np.floating],
        bin_end_rate: npt.NDArray[np.floating] | None,
        bin_end_power: npt.NDArray[np.integer] | None = None,
        bin_end_integrated: npt.NDArray[np.bool_] | None = None,
        snapshot_rows: npt.NDArray[np.intp] | None = None,
        bin_end_scale: npt.NDArray[np.floating] | None = None,
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
            bin_end_power=bin_end_power,
            bin_end_integrated=bin_end_integrated,
            with_target_terms=True,
            n_target_modes=n_modes,
            snapshot_rows=snapshot_rows,
            bin_end_scale=bin_end_scale,
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
        theta = rate * volume / seg_flow[:, 0]
        settled = np.exp(-theta)
    entry_slope, entry_offset = np.ones(n_seg), np.zeros(n_seg)
    # One pass from the source outward; the loop is over path depth, which is what orders it.
    for e in sorted(range(n_seg), key=lambda seg: entry_chains[seg].size):
        upstream = parent[e]
        if upstream >= 0:
            entry_slope[e] = entry_slope[upstream] * settled[upstream]
            entry_offset[e] = t_inf[upstream, 0] + (entry_offset[upstream] - t_inf[upstream, 0]) * settled[upstream]
    # The settled profile's Legendre moments: the excess ``exp(-theta sigma)`` integrates
    # against the monomials to the E-family, so every mode's opening content is
    # ``V sum_i lam[m, i] E_i(theta)`` times the entry contrast. Exact at ``theta = inf``
    # (an idle opening holds no excess anywhere), where the E-family is exactly zero.
    lam = _legendre_monomial(n_modes)
    held = volume * np.einsum("mi,ie->me", lam, _e_table(n_modes - 1, theta))
    # The advected time moments of one bin: the internal operator's kernels carry
    # ``((t_end - t) q_bin / V)**p``, so the mass of the p-th reading over a bin is
    # ``theta**p dt E_p(h dt)`` -- the ``theta**p`` joins per bin in the update, and this
    # holds the base ``dt E_p``. The running integrals' base is its first difference.
    bin_moment = float(dt_days) * _e_table(n_modes, rate * dt_days)
    nmat = np.zeros((n_modes, n_modes))
    for m in range(n_modes):
        for i in range(m):
            if (m - i) % 2 == 1:
                nmat[m, i] = 2.0 * (2 * i + 1)
    weight_rate = np.concatenate([np.zeros(n_seg) if r == 0.0 else rate for r, _, _ in specs + entry_specs])
    weight_power = np.concatenate([np.full(n_seg, p) for _, p, _ in specs + entry_specs])
    weight_integrated = np.concatenate([np.full(n_seg, g, dtype=bool) for _, _, g in specs + entry_specs])
    return _HeatSystem(
        nodes=requested,
        n_pad=n_pad,
        n_bins=n_bins,
        n_modes=n_modes,
        dt_days=dt_days,
        t_inf=t_inf,
        dbar_spectrum=rfft(dbar, n=halo_length, axis=1),
        halo_length=halo_length,
        seg_flow=seg_flow,
        length=length,
        volume=volume,
        rate=rate,
        rho=rho,
        theta=seg_flow * dt_days / volume[:, None],
        nmat=nmat,
        bin_moment=bin_moment[: n_modes + 1],
        bin_integrated=dt_days * (bin_moment[:n_modes] - bin_moment[1 : n_modes + 1]),
        parent=parent,
        held_slope=held * entry_slope,
        held_offset=held * (entry_offset - t_inf[:, 0]),
        # How far a pipe equilibrates over one transit. It is what decides whether the
        # reverse coupling is invertible at all, so the diagnostics quote it.
        h_tau=rate * volume / np.where(running > 0.0, running, np.nan),
        segment_names=tuple(str(name) for name in segments.index),
        internal=build(
            int_paths,
            int_active,
            np.vstack([seg_flow] * len(specs + entry_specs)),
            tedges_days,
            weight_rate,
            weight_power,
            weight_integrated,
            # The plain delivery rows lead the layout, one per segment: the content
            # snapshots that restart the moment recursions are built on their cells.
            snapshot_rows=np.arange(n_seg, dtype=np.intp),
            bin_end_scale=np.tile(volume, len(specs) + len(entry_specs)),
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
    """Iterate the relaxation target modes to their fixed point for one source series.

    Sweep 1 is the undisturbed soil -- the one-way model -- and each further sweep replaces
    the halo with the one implied by the latest flux-moment histories. The map contracts by
    the deficit share of the resistance, so the iterate converges geometrically.

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
        Starting mode stack of shape ``(n_modes, n_seg, n_bins)``. ``None`` (default)
        starts from the undisturbed field with no axial variation, which is the one-way
        model; the reverse direction warm-starts from its previous outer iterate instead.

    Returns
    -------
    ndarray
        Per-segment target modes of shape ``(n_modes, n_seg, n_bins)``: mode 0 the
        relaxation target, the higher modes its axial Legendre components.

    Raises
    ------
    RuntimeError
        If the increment is still above the tolerance when the cap is reached.
    """
    shape = (system.n_modes, *system.t_inf.shape)
    resting = np.zeros(shape)
    resting[0] = system.t_inf
    if max_sweeps == 1:
        return resting
    # An absolute tolerance, not a relative one. Normalising the increment by the iterate's
    # own scale loosens the test exactly when the iteration is misbehaving -- a state wrong by
    # 1e20 K passes a relative test on itself -- and it makes the answer depend on whether the
    # caller works in Celsius or in kelvin, which an affine model must not.
    state = np.ravel(resting if initial is None else initial)
    # The transport reading of the source series is the same in every sweep; only the bias
    # follows the iterate.
    transported = apply_banded(system.internal, tin_padded)
    # The sweep map is affine and contracts, but its slowest mode crawls when a wide pipe's
    # first lag bin holds most of the soil resistance, so the loop extrapolates over its last
    # few iterates exactly as the reverse direction does (Anderson; on an affine map a Krylov
    # method). The convergence test reads the plain residual, so the fixed point returned is
    # the map's own, not the extrapolation's.
    history: list[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]] = []
    increment, best, growing = np.inf, np.inf, 0
    for _ in range(max_sweeps - 1):
        modes = state.reshape(shape)
        readings = _internal_pass(system, transported, modes)
        updated = _update_targets(system, readings, modes, tin_padded, fabricated)
        mapped = np.ravel(updated)
        residual = mapped - state
        previous, increment = increment, float(np.max(np.abs(residual)))
        if increment <= atol:
            return updated
        # A residual that keeps growing is the wrong regime, not a slow start: a segment
        # flushed several volumes per bin asks the sweep for axial modes the bin width
        # cannot drive, and their reading feedback outruns the same-bin damping. Catch it
        # while the numbers are finite and say which segment and what helps.
        growing = growing + 1 if increment > previous else 0
        best = min(best, increment)
        if (growing >= _DIVERGENCE_STEPS and increment > 100.0 * best) or not np.isfinite(increment):
            raise RuntimeError(_sweep_failure(system, previous, increment))
        history.append((state, residual))
        del history[: -(_ANDERSON_DEPTH + 1)]
        state = mapped
        if len(history) > 1:
            past = np.column_stack([r for _, r in history])
            steps = np.column_stack([x for x, _ in history])
            gamma = np.linalg.lstsq(np.diff(past, axis=1), residual, rcond=None)[0]
            state = mapped - (np.diff(steps, axis=1) + np.diff(past, axis=1)) @ gamma
    raise RuntimeError(_sweep_failure(system, previous, increment, exhausted=max_sweeps))


def _sweep_failure(system: _HeatSystem, previous: float, increment: float, exhausted: int | None = None) -> str:
    """Explain a forward sweep that will not reach its fixed point.

    Parameters
    ----------
    system : _HeatSystem
        Prebuilt system, read for the segment that flushes fastest.
    previous, increment : float
        The last two sweep residuals.
    exhausted : int or None, optional
        The sweep budget, when the loop ran out rather than being cut short by the
        divergence test. Default None.

    Returns
    -------
    str
        Message naming the regime when there is one to name, and the remedy that works.
    """
    ran_out = f" within max_sweeps={exhausted}" if exhausted is not None else ""
    head = (
        f"the two-way fixed point did not converge{ran_out}: the sweep residual went "
        f"{previous:.3e} -> {increment:.3e}. "
    )
    spans = np.nanmax(system.theta, axis=1)
    fastest = int(np.nanargmax(spans)) if np.isfinite(spans).any() else 0
    if spans[fastest] > 1.0:
        return head + (
            f"Segment {system.segment_names[fastest]!r} passes {spans[fastest]:.1f} pipe "
            f"volumes in a single bin at its peak flow, and axial modes finer than the "
            f"bin width can drive feed back through the halo faster than the sweep damps "
            f"them. Refine tedges until every transit spans about a bin or more, or lower "
            f"n_modes for this geometry -- the delivered range of a pipe flushed within a "
            f"bin is carried by the leading modes alone."
        )
    return head + "Raise max_sweeps or atol."


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
        Message naming the regime, the segment driving it, and the remedy that works there.
    """
    ran_out = f" within max_sweeps={exhausted}" if exhausted is not None else ""
    head = (
        f"the reverse two-way fixed point did not converge{ran_out}: the outer residual went "
        f"{previous:.3e} -> {increment:.3e}. "
    )
    worst = int(np.nanargmax(system.h_tau)) if np.isfinite(system.h_tau).any() else 0
    coupling = float(system.h_tau[worst])
    if coupling > _COUPLING_LIMIT:
        return head + (
            f"The strongest coupling is segment {system.segment_names[worst]!r} at "
            f"h*tau = {coupling:.2f}, and past about {_COUPLING_LIMIT} the reverse problem is "
            f"ill-conditioned rather than merely slow -- water that equilibrates over its transit "
            f"carries little of the produced temperature to the endmember, so no cap, tolerance or "
            f"regularization recovers it. Use max_sweeps=1 for the one-way reverse, which stays "
            f"well-posed, or shorten the segment."
        )
    # h*tau = rate * volume / flow and rho = exp(-rate * dt), so the ratio is
    # volume / (flow * dt): how many bins the median flow takes to empty the pipe.
    transit_bins = system.h_tau / -np.log(system.rho)
    shortest = int(np.nanargmin(transit_bins)) if np.isfinite(transit_bins).any() else 0
    span = float(transit_bins[shortest])
    if span < _SHORT_TRANSIT_BINS:
        return head + (
            f"Segment {system.segment_names[shortest]!r} empties in {span:.2f} bins at its median "
            f"flow, and a transit that spans about a bin or less leaves the deconvolution nearly "
            f"singular at the fastest alternation the record carries while the halo coupling still "
            f"feeds that alternation back -- at couplings far below the h*tau boundary. The bin "
            f"width fails here, not the physics: refine tedges until every segment's transit spans "
            f"a few bins, or use max_sweeps=1 for the one-way reverse, which stays well-posed."
        )
    return head + (
        f"The strongest coupling is segment {system.segment_names[worst]!r} at "
        f"h*tau = {coupling:.2f} and the shortest transit is segment "
        f"{system.segment_names[shortest]!r} at {span:.2f} bins, both inside the range where the "
        f"iteration normally converges; this configuration is worth a report. max_sweeps=1 is the "
        f"one-way reverse, which stays well-posed."
    )


def _internal_pass(
    system: _HeatSystem,
    transported: npt.NDArray[np.floating],
    modes: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Read the internal temperatures, NaN where the record does not constrain them.

    Parameters
    ----------
    system : _HeatSystem
        Prebuilt operators and kernels.
    transported : ndarray
        Reading of the internal operator on the source series, ``apply_banded(system.internal,
        tin_padded)``. It does not depend on the targets, so the sweep loop hoists it out.
    modes : ndarray
        Current per-segment target modes, shape ``(n_modes, n_seg, n_bins)``.

    Returns
    -------
    ndarray
        The ``4 n_modes - 1`` readings of every segment stacked, shape
        ``((4 n_modes - 1) n_seg, n_bins)``: the plain delivery, the time-moment and
        integrated-kernel weighted deliveries, and the two weighted families at the
        pipe's entry; see :class:`_HeatSystem`.
    """
    t_int = transported + apply_segment_targets(system.internal, modes)
    t_int[~system.internal.valid_out] = np.nan
    return t_int


def _update_targets(
    system: _HeatSystem,
    t_int: npt.NDArray[np.floating],
    modes: npt.NDArray[np.floating],
    tin_padded: npt.NDArray[np.floating],
    fabricated: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.floating]:
    """One flux-and-halo pass: internal temperatures -> new per-segment target modes.

    The wall-flux profile of a bin is the segment's own budget over it, one Legendre
    moment at a time. The content moments ``y_m = V int P_m(2 sigma - 1) (T - T_ref)
    dsigma`` obey the lower-triangular system

        ``dy/dt = (-h I + (q/V) N) y + f(t)``,
        ``f_m = -q ((T_out - T_ref) - (-1)**m (T_in - T_ref)) + h V n_m ctil_m``,

    with ``N`` the constant Legendre derivative coupling, ``n_m = 1/(2m + 1)`` the shape
    norms and ``ctil`` the target modes (mode 0 relative to the reference). ``N`` is
    nilpotent, so the exact update over a bin of constant flow and modes is

        ``y[j+1] = rho e**(theta N) y[j]
        + sum_p (N**p/p!) (q/V)**p [-q (Wout_p - pi0 Win_p) + h V (n ctil) m_p]``,

    where ``Wface_p`` are the faces read against the time-moment weights
    ``(t_end - t)**p exp(-h (t_end - t))`` -- the moment rows of the internal operator --
    and ``m_p = dt**(p+1) E_p(h dt)`` their masses. The flux moments then telescope,

        ``Psi_m L dt = y_m[j] - y_m[j+1] - q dt ((T_out)_dev - pi0_m (T_in)_dev)
        + (q/V) (N @ int_y)_m``,

    with the bins' time-integrated content assembled the same way from the readings
    against the *running integrals* of the moment kernels -- the integrated rows -- so no
    ``1/(h dt)`` ever divides a small difference: the weak-coupling noise floor the old
    two-moment form paid for reconstructing in-bin means is gone. Because a parcel's heat
    release over any interval *is* its temperature drop there, this books each parcel's
    heat into the bins it actually occupied, at the positions it occupied. The
    flux-moment increments convolve with the per-segment deficit kernel into the next
    target modes, ``ctil_m = -(Dbar * da_m)`` with ``a_m = (2m + 1) Psi_m``: shapes pass
    through the per-column soil memory unchanged.

    Stagnation needs no special case -- ``q = 0`` drops every advective term and each
    moment still leaks ``-h (y_m - V n_m ctil_m)``, the standing profile relaxing toward
    its target shape. A bin whose readings the record does not constrain books the
    model's own opening assumption instead: the water it delivers is settled onto the
    undisturbed soil, so its readings are replaced by the undisturbed field times each
    kernel's weight mass. That keeps the advective replacement of the opening water in
    the budget -- dropping those bins' terms instead mis-books a first-transit flux by
    the size of the standing contrast, and the halo memory's ``1/t`` tail drags that
    error through days of record. What still contributes zero flux is the spin-up prefix,
    and the bins whose flux the reverse direction suppresses to probe its dependence on
    invented data.

    A bin's own mode moves that bin's storage term and the lag-0 deficit brings it
    straight back, so the sweep has a same-bin gain of about
    ``Dbar[0] V (1 - rho) / (L dt)`` -- for every mode alike, because the projection onto
    the shapes is idempotent -- which is strictly below one; every other path through the
    map is causal in time and runs upstream to downstream within a bin, so the iteration
    contracts for every geometry and every bin width. The price is that the gain
    approaches one as the bins narrow, so the sweep count grows as ``1/(1 - g)``.

    Parameters
    ----------
    system : _HeatSystem
        Prebuilt operators and kernels.
    t_int : ndarray
        The internal readings from :func:`_internal_pass`.
    modes : ndarray
        Target modes the readings were taken at, shape ``(n_modes, n_seg, n_bins)``; the
        same-bin storage terms need them.
    tin_padded : ndarray
        Source temperature on the padded input grid, the inflow of every root segment.
    fabricated : ndarray of bool or None, optional
        Bins whose flux is to be suppressed -- the reverse direction's dependence probe.
        ``None`` (default) suppresses nothing.

    Returns
    -------
    ndarray
        Updated per-segment target modes, shape ``(n_modes, n_seg, n_bins)``.

    Raises
    ------
    RuntimeError
        If the internal operator carries no target terms or snapshots, which no public
        entry point can produce.
    """
    n_seg = len(system.length)
    n_modes = system.n_modes
    dt = system.dt_days
    blocks = t_int.reshape(4 * n_modes - 1, n_seg, -1)
    plain_out = blocks[0]
    mom_out, g_out = blocks[1 : 1 + n_modes], blocks[1 + n_modes : 2 * n_modes]
    mom_in, g_in = blocks[2 * n_modes : 3 * n_modes], blocks[3 * n_modes :]

    # The settled substitution: readings the record does not constrain carry the model's
    # own opening assumption -- settled water at the undisturbed soil temperature, whose
    # weighted readings are the field times each kernel's weight mass. The entry faces
    # read the parent's field; a root segment's entry rows are always constrained.
    parent_field = system.t_inf[np.maximum(system.parent, 0)]
    theta_raw = [np.ones_like(system.theta)]
    for _ in range(n_modes):
        theta_raw.append(theta_raw[-1] * system.theta)
    moment_mass = [theta_raw[p] * (system.bin_moment[p][:, None] / dt) for p in range(n_modes)]
    integral_mass = [theta_raw[p] * (system.bin_integrated[p][:, None] / dt) for p in range(n_modes - 1)]
    for block_set, field, masses in (
        ((plain_out,), system.t_inf, [np.ones_like(system.theta)]),
        (mom_out, system.t_inf, moment_mass),
        (g_out, system.t_inf, integral_mass),
        (mom_in, parent_field, moment_mass),
        (g_in, parent_field, integral_mass),
    ):
        for block, mass in zip(block_set, masses, strict=True):
            hole = ~np.isfinite(block)
            if hole.any():
                block[hole] = (field * mass)[hole]

    # A split leaves temperature unchanged and both flows are constant over a bin, so the
    # water entering a pipe is what its parent delivered; a root segment is fed the source.
    t_in = np.where(system.parent[:, None] >= 0, plain_out[system.parent], tin_padded)

    # The warm-start prefix is a fabricated hydraulic history, so it drives neither the
    # wall flux nor the water the pipes hold when the record opens: that initial condition
    # belongs to the model, which is what lets the forward and the reverse direction agree
    # about it. The reverse direction's suppressed bins are excluded the same way.
    usable = np.zeros((n_seg, blocks.shape[-1]), dtype=bool)
    usable[:, system.n_pad :] = True
    usable &= ~(np.zeros_like(usable) if fabricated is None else fabricated)
    flowing = usable & (system.seg_flow > 0.0)

    t_ref = system.t_inf[:, :1]
    vol = system.volume[:, None]
    rate = system.rate[:, None]
    carried = system.seg_flow * dt
    pi0 = (-1.0) ** np.arange(n_modes)
    norm = 1.0 / (2.0 * np.arange(n_modes) + 1.0)
    factorial = np.cumprod(np.concatenate([[1.0], np.arange(1.0, n_modes + 1.0)]))
    npow = [np.eye(n_modes)]
    for _ in range(1, n_modes):
        npow.append(npow[-1] @ system.nmat)
    ctil = modes.copy()
    ctil[0] -= t_ref

    # The per-bin drivers, as mode-vectors per kernel order: the faces in deviations from
    # the reference times the bin width, and the modes' own storage with the kernel's
    # weight mass. The advective parts carry the flow, so a standing bin keeps only the
    # storage leak; NaN outside the usable region is masked before it can propagate.
    def drivers(
        face_out: npt.NDArray[np.floating],
        face_in: npt.NDArray[np.floating],
        masses: list[npt.NDArray[np.floating]],
    ) -> npt.NDArray[np.floating]:
        """Assemble ``B_p[m]`` for one kernel family.

        Parameters
        ----------
        face_out, face_in : ndarray
            Advected-kernel readings of the two faces, stacked per kernel order.
        masses : list of ndarray
            Weight mass of each kernel per segment and bin, over the bin width.

        Returns
        -------
        ndarray
            Drivers of shape ``(n_kernels, n_modes, n_seg, n_bins)``.
        """
        out = np.empty((len(face_out), n_modes, n_seg, blocks.shape[-1]))
        for p in range(len(face_out)):
            fo = np.where(flowing, system.seg_flow * dt * (face_out[p] - t_ref * masses[p]), 0.0)
            fi = np.where(flowing, system.seg_flow * dt * (face_in[p] - t_ref * masses[p]), 0.0)
            storage = np.where(usable, rate * vol * dt * masses[p] * ctil, 0.0)
            for m in range(n_modes):
                out[p, m] = -fo + pi0[m] * fi + norm[m] * storage[m]
        return out

    drive = drivers(mom_out, mom_in, moment_mass)
    drive_int = drivers(g_out, g_in, integral_mass)

    # The exact bin update, mode by mode: the coupling is strictly lower triangular, so
    # each mode reads only already-updated histories and is one forgetting scan. The
    # record opens with the pipes holding the warm start's own settled water, seeded per
    # mode; with no warm start there is no prior state to carry.
    # The content snapshots at the chunk anchors: the recursions restart from them, so
    # a rounding error in the state can never ride the advective shear coupling for more
    # than one chunk -- the polynomial error growth that coupling composes to over the
    # decay memory is what set the old high-mode noise floor. The corrections propagate
    # through plain forgetting factors only, because each mode is corrected before the
    # next one's coupling reads it. Anchors inside the warm-start prefix, or whose
    # in-pipe water the record does not fully cover, keep the free-running state.
    terms = system.internal.target_terms
    if terms is None or terms.snapshots is None:
        msg = "the internal operator must carry target terms and content snapshots; this is a bug, please report it"
        raise RuntimeError(msg)
    snap_abs, snap_unit = apply_content_snapshots(system.internal, modes, tin_padded)
    snapshot = snap_abs - t_ref[None] * snap_unit
    snap_valid = terms.snapshots.anchor_valid & (np.arange(snapshot.shape[-1])[None, :] > system.n_pad)
    chunk_start = terms.chunk_edge[:, 1:].astype(np.intp)
    into_end = terms.rho_into[:, 1:]
    into_start = into_end / system.rho[:, None]

    theta_gated = np.where(flowing, system.theta, 0.0)
    theta_pow = [np.ones_like(theta_gated)]
    for _ in range(1, n_modes):
        theta_pow.append(theta_pow[-1] * theta_gated)
    q_over_v_dt = theta_gated / dt
    y_end = np.zeros((n_modes, n_seg, blocks.shape[-1]))
    y_start = np.zeros_like(y_end)
    for m in range(n_modes):
        forcing = np.zeros((n_seg, blocks.shape[-1]))
        for p in range(n_modes):
            for i in range(n_modes):
                if npow[p][m, i] != 0.0:
                    forcing += (npow[p][m, i] / factorial[p]) * drive[p, i]
        for p in range(1, m + 1):
            for i in range(n_modes):
                if npow[p][m, i] != 0.0:
                    forcing += system.rho[:, None] * (npow[p][m, i] / factorial[p]) * theta_pow[p] * y_start[i]
        if system.n_pad:
            forcing[:, system.n_pad - 1] += system.held_slope[m] * tin_padded[0] + system.held_offset[m]
        free = np.empty((n_seg, blocks.shape[-1]))
        for e in range(n_seg):
            free[e] = lfilter([1.0], [1.0, -system.rho[e]], forcing[e])
        free_edge = np.concatenate([np.zeros((n_seg, 1)), free], axis=1)
        delta = np.where(snap_valid, free_edge - snapshot[m], 0.0)
        delta_start = np.take_along_axis(delta, chunk_start, axis=1)
        y_end[m] = free - into_end * delta_start
        y_start[m] = free_edge[:, :-1] - into_start * delta_start

    # The bins' time-integrated content, from the same drivers against the running
    # integrals of the kernels: every term bounded, no small-denominator assembly.
    int_y = np.zeros_like(y_end)
    gated_mass = [np.where(flowing, dt * moment_mass[p], 0.0) if p else dt * moment_mass[0] for p in range(n_modes)]
    for m in range(n_modes):
        for p in range(n_modes - 1):
            for i in range(n_modes):
                if npow[p][m, i] != 0.0:
                    int_y[m] += (npow[p][m, i] / factorial[p]) * (gated_mass[p] * y_start[i] + drive_int[p, i])

    # The flux moments telescope; their increments convolve with the deficit kernel into
    # the next target modes. The prefix sets the state the record opens from and is not
    # part of the record's flux history: the step onto that state would otherwise register
    # as heat drawn from the soil before the record starts.
    dev_out = np.where(flowing, carried * (plain_out - t_ref), 0.0)
    dev_in = np.where(flowing, carried * (t_in - t_ref), 0.0)
    flux = np.empty_like(y_end)
    for m in range(n_modes):
        budget = y_start[m] - y_end[m] - (dev_out - pi0[m] * dev_in)
        for i in range(n_modes):
            if system.nmat[m, i] != 0.0:
                budget += system.nmat[m, i] * q_over_v_dt * int_y[i]
        flux[m] = (2 * m + 1) * budget / (system.length[:, None] * dt)
        flux[m, :, : system.n_pad] = 0.0
    increments = np.diff(flux, axis=2, prepend=0.0)
    spectrum = rfft(increments, n=system.halo_length, axis=2) * system.dbar_spectrum
    shift = -irfft(spectrum, n=system.halo_length, axis=2)[:, :, : system.n_bins]
    shift[0] += system.t_inf
    return shift


def source_to_endmember(
    *,
    tin: npt.ArrayLike,
    flow: Mapping[str, npt.ArrayLike],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    network: HeatNetwork,
    surface_temperature: Mapping[str, npt.ArrayLike],
    report_nodes: list[str] | tuple[str, ...] | None = None,
    n_modes: int = 6,
    max_sweeps: int = 5000,
    atol: float = 1e-7,
    spinup: str | None = "constant",
) -> dict[str, npt.NDArray[np.floating]]:
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
    flow : mapping
        Demand at every endmember [m³/day], keyed by endmember name, on the same ``tedges``
        bins; see :func:`pipetransport.transport.source_to_endmember`.
    tedges : pandas.DatetimeIndex
        Time edges of ``tin`` and ``flow``, uniformly spaced (the halo memory is a
        convolution over lag bins).
    cout_tedges : pandas.DatetimeIndex
        Time edges of the output bins; alignment and resolution are free.
    network : HeatNetwork
        Network carrying everything that is constant in time -- geometry, land cover, burial
        depth, wall, film and soil -- one row per segment; see :class:`HeatNetwork`.
    surface_temperature : mapping
        Sol-air temperature keyed by cover class: one bin-constant array per class on the
        ``tedges`` bins; see :func:`sol_air_temperature`. Classes no segment is buried under
        are ignored, so a full land-cover catalogue may be passed as is.
    report_nodes : list of str or None, optional
        Nodes to report at, in output row order. Any node of the network is allowed.
        Defaults to ``network.endmembers``.
    n_modes : int, optional
        Axial Legendre modes of each pipe's wall-flux profile -- the model's spatial
        resolution along a pipe, a model order like a mesh rather than a tuning knob.
        1 is the classical one-history model, whose truncation under intermittent
        demand reaches a quarter of the driving contrast; 6 (default) resolves an
        overnight-stagnation profile to under 1 % of it. The bin width bounds what the
        count can buy: a segment flushed more than about a pipe volume per bin cannot
        drive modes finer than the bin resolves, and asking for them is refused by name
        with the two remedies that work -- fewer modes for that geometry, or finer
        ``tedges``. Runtime and memory grow with the count -- see ``max_sweeps`` for the
        measured costs -- so a study that only needs the classical behaviour can ask
        for less. Default 6.
    max_sweeps : int, optional
        Iteration cap. 1 is the one-way model. The sweep count is a property of the physics,
        not of the record length: it is set by how much of the steady soil resistance the
        first lag bin still holds, ``Dbar[0] / R_soil``, which rises toward 1 for wide pipes
        on short bins. A hundred or two is typical. Exceeding the cap
        raises rather than returning an unconverged answer, and a configuration whose sweep
        genuinely diverges -- a pipe flushed several volumes per bin at a mode count its
        bin width cannot drive -- is refused by name well before the cap. Default 5000.

        The working set is the operator's stored cell factors, and it scales with the
        record and roughly with the square of the mode count rather than with the sweeps:
        measured on the example network at hourly bins, the peak is about 120 MiB per day
        of record at the six-mode default (a month solves in a minute within 4 GiB) and
        about 40 MiB per day at two modes. A year of hourly data wants either the two-mode
        model or a machine with tens of GiB.
    atol : float, optional
        Convergence tolerance on the relaxation-target increment, absolute and in the unit of
        the temperature inputs. Default 1e-7, several orders below anything a temperature
        measurement resolves and below the model's own axial truncation by more. It is
        absolute rather than relative so that the answer cannot depend on whether the caller
        works in Celsius or in kelvin, and so that an iterate that is diverging cannot widen
        its own convergence test. The delivered temperature settles to a fraction of it --
        measured about 0.4 -- and the sweep count grows only as its logarithm. The floor is
        the sweep map's own round-off: evaluating six Legendre modes of an effective-target
        profile that swings by tens of kelvin leaves a few 1e-8 of noise on segments whose
        transit is shorter than a bin, so tolerances below that are reachable only on
        well-resolved geometries or fewer modes.
    spinup : {"constant"} or None, optional
        Warm-start policy for the water the pipes hold when the record opens; see
        :func:`pipetransport.transport.source_to_endmember` for the mechanics. Default
        ``"constant"``.

        The heat pair has three memories on three clocks, and this parameter serves only
        the fastest:

        * **The water in the pipes** (hours to days -- one transit). ``"constant"`` pads
          the record internally with the leading flow and ``tin[0]``, so the record opens
          with every pipe holding settled water. Automatic; no data needed.
        * **The halo the network has built in the soil around itself** (days to weeks --
          the deficit kernel decays over ``d_eff**2 / alpha``, about three weeks for a
          pipe at a metre in typical soil). The model assumes it is absent: the record
          opens onto undisturbed soil, as if every pipe were laid that day. See
          :ref:`assumption-soil-columns` for measured consequences.
        * **The undisturbed soil state at depth** (weeks to seasons). Surface history
          from before ``tedges[0]`` is the uniform first value of each
          ``surface_temperature`` series.

        The second and third are the caller's to warm up, and one recipe serves both:
        prepend about three weeks -- ``d_eff**2 / alpha`` -- of realistic history to
        ``tedges`` (the measured surface record, a typical demand pattern, a production
        temperature near the record's opening value) and leave ``cout_tedges`` on the
        period of interest; the output grid is free, so nothing needs discarding. Three
        weeks lets the network build the bulk of its halo and delivers about half of any
        recent surface swing to a metre's depth (a year would deliver 87 %); the seasonal
        baseline older than the lead-in enters as the first surface value, so open the
        lead-in where the record is representative rather than at an extreme. The cost is
        the operator, about 1.3 MiB and the proportional build time per extra day at
        hourly bins on the example network.

    Returns
    -------
    dict of str to ndarray
        Delivered temperature keyed by reporting node, in ``report_nodes`` order, each a
        series of ``len(cout_tedges) - 1`` values in the unit of ``tin``: the flow-weighted
        average over each output bin. NaN marks bins the record does not constrain. The
        mapping is what :func:`endmember_to_source` takes as ``tout``, so a round trip
        composes verbatim.

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
    :ref:`concept-relaxation` : How the halo turns the one-way model into this one.
    :ref:`assumption-uniform-wall-flux` : The axial mode truncation, and what it costs.
    :ref:`assumption-effective-target` : Why the delivered temperature can leave its own hull.

    Examples
    --------
    Cool plant water warming on its way through warm summer soil:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pipetransport.examples import example_heat_network, example_demand
    >>> from pipetransport.heat import source_to_endmember
    >>>
    >>> network = example_heat_network()
    >>> tedges = pd.date_range("2025-06-01", "2025-06-05", freq="h")
    >>> surface = {
    ...     "grass": np.full(len(tedges) - 1, 22.0),
    ...     "paved": np.full(len(tedges) - 1, 30.0),
    ... }
    >>> cout = source_to_endmember(
    ...     tin=np.full(len(tedges) - 1, 8.0),
    ...     flow=example_demand(tedges=tedges, network=network),
    ...     tedges=tedges,
    ...     cout_tedges=tedges,
    ...     network=network,
    ...     surface_temperature=surface,
    ... )
    >>> list(cout)
    ['T1', 'T2', 'T3', 'T4']
    >>> bool(
    ...     np.all(np.diff([8.0, np.nanmean(cout["T4"]), 30.0]) > 0)
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
    if n_modes < 1:
        msg = "n_modes must be at least 1 (mode 0 is the position-uniform flux history)"
        raise ValueError(msg)
    system = _build_system(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        network=network,
        surface_temperature=surface_temperature,
        report_nodes=report_nodes,
        spinup=spinup,
        n_modes=n_modes,
    )
    tin_padded = np.concatenate([np.full(system.n_pad, tin[0]), tin])

    modes = _converge_targets(system, tin_padded, max_sweeps=max_sweeps, atol=atol)
    out = apply_banded(system.reporting, tin_padded) + apply_segment_targets(system.reporting, modes)
    out[~system.reporting.valid_out] = np.nan
    return dict(zip(system.nodes, out, strict=True))


def endmember_to_source(
    *,
    tout: Mapping[str, npt.ArrayLike],
    flow: Mapping[str, npt.ArrayLike],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    network: HeatNetwork,
    surface_temperature: Mapping[str, npt.ArrayLike],
    n_modes: int = 6,
    max_sweeps: int = 5000,
    atol: float = 1e-7,
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
    tout : mapping
        Measured temperature, keyed by the node it was measured at: one bin-constant array
        per node on the ``cout_tedges`` bins. The keys *are* the observation set -- pass
        only the nodes that were sampled. NaN inside an array marks a gap.
    flow, tedges, cout_tedges, network, surface_temperature, n_modes, max_sweeps, atol, spinup
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
        more than ``gap_atol`` when the flux invented over such a gap is removed. That is a
        statement about *dependence* on invented data rather than a bound on accuracy: a bin
        the record determines poorly for some other reason can be insensitive to that
        particular removal and still be returned.

    Raises
    ------
    ValueError
        As :func:`source_to_endmember`, plus a shape or naming mismatch of ``tout`` and a
        non-positive ``regularization_strength``.
    RuntimeError
        If the fixed point diverges or has not converged within ``max_sweeps``. Past a
        ``h*tau`` of about 0.7 -- a pipe that equilibrates appreciably over its transit --
        the coupled inverse is ill-conditioned rather than slow, and this is how it shows;
        the message names the segment responsible. A transit that spans about a bin or less
        fails the same way at any coupling, and there the message points at the remedy that
        works: ``tedges`` fine enough for every transit to span a few bins. ``max_sweeps=1``
        is the one-way reverse, which stays well-posed.

    See Also
    --------
    source_to_endmember : Forward direction.

    Notes
    -----
    The lead-in recipe of :func:`source_to_endmember` applies unchanged, and is cheaper to
    satisfy here: ``tout`` is measured, so the observations usually reach back before the
    period of interest anyway.

    The halo is brought to its own fixed point inside every outer step, so the cost is a
    product of two iterations rather than a sum. The outer step extrapolates over its last few
    iterates rather than simply repeating, which reaches pipes plain repetition cannot -- but
    only so far: the extrapolation is truncated, and two regimes sit beyond it, measured by
    power iteration on the outer map for a single observed pipe at constant flow -- the worst
    case; a network observing several nodes under varying demand can reach further. The
    spectral radius is a property of the configuration, not of the iteration: it is unchanged
    from 6 to 24 days of record and from 30-minute to 2-hour bins at fixed coupling, so a
    longer record or a deeper Anderson window buys nothing past it. **Coupling**: at transits
    of a few bins or more the radius crosses one near ``h*tau = 0.7`` (measured 0.92 at 0.50
    and 1.10 at 0.75 on a 100 mm main; a 400 mm main crosses nearer 1.0), the excess is
    broad-band, and nothing reaches the fixed point -- a 40 mm service line at a 1 h transit
    already sits at ``h*tau = 1.12``, which is why the two-way reverse is unavailable on fast
    service lines however short their transit sounds. **Transit**: on the one-history
    model a segment that emptied in about a bin was nearly singular at the fastest
    alternation the record carries whatever its coupling -- radius 21 at ``h*tau = 0.11``
    for a 100 mm main at a half-bin transit. The advected mode kernels changed that outer
    map: the same pipe now reconstructs to 5e-8 K at the six-mode default, and finer bins
    remain the sharper choice wherever a transit spans less than a bin. The regime
    diagnosis stays: a configuration that does diverge with a sub-bin transit is still
    told to refine ``tedges``, which is still the remedy that works there. The
    RuntimeError names whichever regime applies.

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
    >>> from pipetransport.examples import example_heat_network, example_demand
    >>> from pipetransport.heat import (
    ...     HeatNetwork,
    ...     source_to_endmember,
    ...     endmember_to_source,
    ... )
    >>>
    >>> # One cover for every pipe, and a light dry soil: it keeps every segment's h*tau
    >>> # inside the coupling boundary the Notes describe, which is what makes the two-way
    >>> # reverse well-posed on this network.
    >>> flat = example_heat_network().segments.assign(cover="grass", kappa_soil=0.015)
    >>> network = HeatNetwork(segments=flat.drop(columns="volume"), source="Plant")
    >>> tedges = pd.date_range("2025-06-01", "2025-06-09", freq="h")
    >>> demand = example_demand(tedges=tedges, network=network)
    >>> surface = {"grass": np.full(len(tedges) - 1, 25.0)}
    >>> hours = np.arange(len(tedges) - 1)
    >>> tin = 10.0 + 2.0 * np.sin(2 * np.pi * hours / 72.0)
    >>> shared = dict(
    ...     flow=demand,
    ...     tedges=tedges,
    ...     cout_tedges=tedges,
    ...     network=network,
    ...     surface_temperature=surface,
    ...     n_modes=2,  # the deconvolution does not depend on the mode count
    ... )
    >>> measured = source_to_endmember(tin=tin, report_nodes=["T1", "T4"], **shared)
    >>> recovered = endmember_to_source(tout=measured, **shared)
    >>> inner = slice(36, -96)  # both edges lean on unconstrained bins; see Notes
    >>> residual = float(np.nanmax(np.abs(recovered[inner] - tin[inner])))
    >>> residual < 1e-6  # a fraction of atol, down to the Tikhonov pull at tighter ones
    True
    """
    if max_sweeps < 1:
        msg = "max_sweeps must be at least 1 (sweep 1 is the one-way model)"
        raise ValueError(msg)
    # The keys of ``tout`` are the observation set: every one must be a node, and their order
    # in the solve is the network's own, so the answer cannot depend on how the caller
    # happened to build the mapping.
    unknown = [node for node in tout if node not in network.paths]
    if unknown:
        msg = f"unknown node(s) in tout: {unknown}; network nodes are {list(network.nodes)}"
        raise ValueError(msg)
    if not tout:
        msg = "tout must hold at least one observed node"
        raise ValueError(msg)
    if n_modes < 1:
        msg = "n_modes must be at least 1 (mode 0 is the position-uniform flux history)"
        raise ValueError(msg)
    system = _build_system(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        network=network,
        surface_temperature=surface_temperature,
        report_nodes=tuple(node for node in network.nodes if node in tout),
        spinup=spinup,
        n_modes=n_modes,
    )
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    observed = np.stack([np.asarray(tout[node], dtype=float) for node in system.nodes])
    _validate_tedges(cout_tedges, observed, tedges_name="cout_tedges", values_name="tout")

    n_source = len(pd.DatetimeIndex(tedges)) - 1

    def solve(
        reporting: NetworkTransfer,
        modes: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        bias = apply_segment_targets(reporting, modes)
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
    # transit puts it over, as does one that empties in about a bin. Extrapolating over the
    # last few iterates instead (Anderson, which on an affine map is a Krylov method) reaches
    # configurations plain repetition cannot, though with a truncated window it carries no
    # guarantee: past the regimes _diverged_message names it stops reaching them and the
    # divergence test below is what answers. The
    # residual is
    # measured on the bins the operator covers; that set is a property of the operator, not of
    # the iterate, so it is fixed once and asserted to stay fixed.
    resting = np.zeros((n_modes, *system.t_inf.shape))
    resting[0] = system.t_inf
    first = solve(system.reporting, resting)
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
        recovered, state = first, resting
        increment, previous, best, growing = np.inf, np.inf, np.inf, 0
        history: list[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]] = []
        for _ in range(max_sweeps - 1):
            inner_atol = max(atol, _INNER_FORCING * increment)
            state = _converge_targets(
                system,
                filled(recovered),
                max_sweeps=max_sweeps,
                atol=inner_atol,
                initial=state,
                fabricated=fabricated,
            )
            mapped = solve(system.reporting, state)
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
        honest = converge(reads_invented.reshape(4 * n_modes - 1, n_seg, -1).any(axis=0))
        with np.errstate(invalid="ignore"):
            recovered = np.where(np.abs(honest - recovered) > gap_atol, np.nan, recovered)
    return recovered[system.n_pad :]
