.. _concepts-heat:

How the heat model works
========================

One idea on top of :doc:`concepts`: a parcel that decays toward a moving target rather than
toward zero, and the soil memory that target carries.

.. _concept-relaxation:

Relaxation toward a moving target
---------------------------------

Temperature does not decay --- it *relaxes*. A parcel in a buried pipe loses heat through the
water-side film, the pipe wall and the soil in series, toward the temperature of its surroundings
rather than toward zero:

.. math::
   :label: relaxation

   \frac{\mathrm{d}T}{\mathrm{d}u} = -h_e \left( T - T_{b,e}(u) \right) ,
   \qquad
   h_e = \frac{1}{\left( R_\text{film} + R_\text{wall} + R_\text{soil} \right)\, \pi r_i^2} ,

with :math:`R_\text{soil} = \ln(2 d / r_o) / (2 \pi \kappa)` the steady buried-pipe resistance. The
rate inherits a :math:`1/(D^2 \ln D)` diameter law, so the same asymmetry that governs residual
(:ref:`concept-first-order-decay`) governs temperature: a 100 mm service line equilibrates with its
soil an order of magnitude faster than a 400 mm trunk main.

The single change --- a target that is not zero --- makes the delivered value **affine** rather than
linear in the produced value:

.. math::
   :label: affine

   T_\text{delivered} = W\, T_\text{produced} + b .

:math:`W` is the very same operator, built with :math:`h_e` in the decay slot; :math:`b` collects the
soil's contribution along the path. Because :math:`T_{b,e}` is piecewise constant on the input bins
and every arrival time is linear on a refined cell, :math:`b` has a closed form on each cell --- the
same integral of an exponential the surviving fraction uses. Nothing about the transport is
approximated to obtain it, and setting every :math:`h_e = 0` gives :math:`b = 0` exactly.

The halo the network builds
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The obvious target is the undisturbed soil temperature at pipe depth, :math:`T_\infty`, propagated
down from the land-cover surface forcing. That is the classical *one-way* model, and it is wrong in
a specific, one-signed way: a pipe that has been losing heat for weeks is not surrounded by
undisturbed soil but by a warm (or cold) **halo** of its own making, which reduces the temperature
difference driving further exchange.

The step response of that halo splits into a steady part and a transient deficit,
:math:`G(t) = R_\text{soil} - D(t)`, and it is the deficit that carries the whole memory. Absorbing
the steady part into :math:`h_e` leaves the one-way model with a *shifted* target,

.. math::

   T_{b,e}[n] = T_\infty[n] - \sum_{j \le n} \Delta\psi_e[j]\, \bar{D}[n-j] ,

one convolution of the segment's own wall-flux history per segment. The two-way answer is the fixed
point of alternating a transport pass, a flux pass and this convolution --- and its *first* iterate,
with an undisturbed halo, is exactly the one-way model. ``max_sweeps=1`` returns it, so the classical
model is not a separate code path but the first step of the same iteration.

The memory is long. Because the deficit decays like :math:`3/\tau` in days, the model's assumption
that the soil starts undisturbed costs kelvins at the beginning of a record and takes a season to
pay off:

.. code:: python

   import numpy as np
   import pandas as pd

   from pipetransport.heat import HeatNetwork, segment_heat_rate, source_to_endmember

   segments = {
       "Plant-T1": {
           "from": "Plant", "to": "T1", "length": 1000.0, "diameter": 0.1,
           "cover": "grass", "alpha": 0.05, "kappa_soil": 0.025, "eta": 0.41,
       }
   }
   network = HeatNetwork(segments=segments, source="Plant")

   tedges = pd.date_range("2025-01-01", periods=120 * 24 + 1, freq="h")
   n_bins = len(tedges) - 1
   transit_days = 2.0 / 24.0
   flow = {"T1": np.full(n_bins, float(network.segments.loc["Plant-T1", "volume"]) / transit_days)}
   shared = dict(
       tin=np.full(n_bins, 8.0), flow=flow, tedges=tedges, cout_tedges=tedges,
       network=network,
       surface_temperature={"grass": np.full(n_bins, 20.0)},
   )

   two_way = source_to_endmember(**shared)["T1"]
   one_way = source_to_endmember(**shared, max_sweeps=1)["T1"]

   # A fully developed halo is the analytic steady buried-pipe law, which is what one-way assumes.
   rate = segment_heat_rate(network=network)["Plant-T1"]
   steady = 20.0 + (8.0 - 20.0) * np.exp(-rate * transit_days)
   np.testing.assert_allclose(one_way[-1], steady, rtol=1e-9)

   for day in (1, 7, 30, 119):
       correction = two_way[day * 24] - one_way[day * 24]
       print(f"day {day:3d}: halo correction {correction:+.2f} K")

Cold water into warm soil takes up 2.42 K more than the one-way model allows on the first day,
0.83 K after a week and 0.08 K after four months --- the soil around the pipe cooling toward the
state the one-way model assumed from the outset. Which way the correction points is set by the sign
of the flux, so a network whose production temperature crosses the soil temperature during a record
sees the stored heat come *back out*. It can also carry the delivered water briefly *outside* the
range of the produced water and the soil, which is not a bug --- see
:ref:`assumption-effective-target`.
