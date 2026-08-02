.. _concepts:

Core concepts
=============

``pipetransport`` answers one question in several forms: *which water is at this tap right now, and
where has it been?* Delivered quality, water age, chlorine residual and the reverse reconstruction
of the produced quality are all read off a single linear operator that maps the produced-water
series onto a reporting node. This page builds that operator from the physics and defines the
vocabulary the API docstrings refer back to.

Nothing here needs a hydraulic solver. The only hydraulic input is the demand at the delivery
points, which is what a distribution utility actually meters; every internal pipe flow follows from
mass conservation.

.. contents:: Contents
   :local:
   :depth: 2

One source, one path
--------------------

The networks in scope are **rooted trees**: one production point at the root, pipes that split at
junctions and never merge, and demand drawn at the leaves --- the *endmembers*. See
:ref:`assumption-tree-topology` for what that excludes.

Three consequences carry the whole model.

**A split does not change concentration.** At a junction a single incoming stream of concentration
:math:`c_\text{in}` divides over outgoing branches:

.. math::

   Q_\text{in}\, c_\text{in} = \sum_j Q_j\, c_j,
   \qquad \sum_j Q_j = Q_\text{in}.

Mass balance alone does not force :math:`c_j = c_\text{in}`; what forces it is that every branch
draws from the same, cross-sectionally uniform slug of water, which is exactly what plug flow
asserts (:ref:`assumption-plug-flow`). Given it, the balance is satisfied identically and **no
mixing operator appears anywhere in the network**. A merge would be a different story: the node
would carry a flow-weighted blend of two transport histories, with weights set by the hydraulics
rather than by the demand.

**Every node is fed by exactly one path.** :attr:`~pipetransport.network.PipeNetwork.paths` maps a
node to the ordered tuple of segments between it and the source. The water quality at that node is
therefore the produced quality, delayed --- and, for a reactive compound, attenuated --- along that
one path. One boundary signal ``cin`` describes the entire network.

**The flow follows from the demand.** Pipes are full, rigid and store nothing beyond their fixed
water volume (:ref:`assumption-no-storage`), so the throughflow of segment :math:`e` at any instant
is the summed demand of the endmembers :math:`\mathcal{E}(e)` below it,

.. math::

   Q_e(t) = \sum_{m \in \mathcal{E}(e)} q_m(t),
   \qquad
   Q_0(t) = \sum_{\text{all } m} q_m(t),

with :math:`Q_0` the production at the source. That is
:meth:`~pipetransport.network.PipeNetwork.segment_flow`, and it is recomputed at every time step:
nothing in the model requires the demands to move in proportion to one another.

.. code:: python

   import numpy as np

   from pipetransport.examples import example_network

   network = example_network()
   demand = np.array([[240.0], [360.0], [120.0], [80.0]])  # T1, T2, T3, T4 [m3/day]

   # A segment carries the summed demand below it; the source carries the total.
   print(dict(zip(network.segments.index, network.segment_flow(flow=demand).ravel())))
   assert np.isclose(network.node_flow(flow=demand, nodes=["Plant"])[0, 0], demand.sum())

.. _concept-arrival-map:

The arrival map
---------------

Inside one pipe
~~~~~~~~~~~~~~~

A pipe is a plug-flow reactor of fixed water volume :math:`V_e`. Write its cumulative throughflow
volume as

.. math::

   C_e(t) = \int_{t_0}^{t} Q_e(t')\, \mathrm{d}t' .

A parcel that enters the pipe at time :math:`s` leaves it at the time :math:`a_e(s)` at which it has
displaced exactly one pipe volume:

.. math::
   :label: displacement

   C_e\bigl(a_e(s)\bigr) - C_e(s) = V_e .

This is the **displacement condition**. It says nothing about velocity or pressure --- only that the
water ahead of a parcel has to leave before the parcel does, and there is exactly :math:`V_e` of it.
The segment travel time is :math:`\tau_e(s) = a_e(s) - s`, and it collapses to the familiar
:math:`V_e / Q_e` **only** when the flow happens to be constant over the whole crossing. Under a
diurnal demand it is not: a parcel that enters at the evening peak crosses partly at the peak and
partly on the falling limb, and no single instantaneous flow rate reproduces its travel time.

A retardation factor :math:`R \geq 1` (reversible exchange with the pipe wall or biofilm) simply
replaces :math:`V_e` by :math:`R\,V_e` in :eq:`displacement`: the compound has to displace more
volume than the water does, so it arrives later.

Along a path
~~~~~~~~~~~~

The arrival time at a node is the composition of the per-pipe maps along its path
:math:`(e_1, \dots, e_M)`, ordered from the source outward:

.. math::

   A = a_{e_M} \circ \cdots \circ a_{e_1},
   \qquad
   \tau(s) = A(s) - s .

Because the demand is piecewise constant on ``tedges``, every :math:`C_e` is piecewise linear, so
:math:`a_e` and :math:`a_e^{-1}` are exact linear interpolations and :math:`A` is piecewise linear in
:math:`s`. Its breakpoints are *not* only at ``tedges``: a parcel also kinks when a flow change
overtakes it *inside* a pipe it is still travelling through. The package therefore refines the
source-time grid with the preimage of the bin edges taken at every node along the path, which makes
every arrival time and every travel time exactly linear on each refined cell --- no quadrature error
anywhere.

Zero flow is admissible and needs no special case: :math:`C_e` plateaus, and a parcel in a stagnant
pipe simply waits until flow resumes. Negative flow is not admissible (:ref:`assumption-forward-flow`).

The snippet below implements :eq:`displacement` in a couple of lines of NumPy for a single pipe
under a strongly varying demand, and checks it against the package: a pulse leaves in one input bin
and must arrive within the interval the map sends that bin to, carrying its water volume with it.

.. code:: python

   import numpy as np
   import pandas as pd

   from pipetransport.network import PipeNetwork
   from pipetransport.transport import source_to_endmember
   from pipetransport.utils import cumulative_flow_volume

   segments = pd.DataFrame({"from": ["Plant"], "to": ["T1"], "volume": [100.0]}, index=["Plant-T1"])
   network = PipeNetwork(segments=segments, source="Plant")

   tedges = pd.date_range("2025-06-01", periods=73, freq="h")
   days = np.arange(73) / 24.0
   demand = 200.0 + 120.0 * np.cos(2 * np.pi * (np.arange(72) - 8) / 24.0)  # m3/day
   cumulative = cumulative_flow_volume(demand, np.diff(days))  # m3 at every bin edge

   # Displacement condition: a parcel entering at edge i leaves once one pipe volume has passed.
   volume = 100.0
   arrival = np.interp(cumulative + volume, cumulative, days, right=np.nan)

   # A one-hour pulse leaving in bin 30 arrives inside [arrival[30], arrival[31]].
   cin = np.zeros(72)
   cin[30] = 1.0
   cout = source_to_endmember(
       cin=cin, flow=demand[None, :], tedges=tedges, cout_tedges=tedges, network=network
   )[0]
   expected = np.arange(int(np.floor(arrival[30] * 24)), int(np.ceil(arrival[31] * 24)))
   np.testing.assert_array_equal(np.flatnonzero(cout > 0.0), expected)

   # ... and it carries exactly the water volume it left in (mass conservation).
   np.testing.assert_allclose(np.sum(cout * demand / 24.0), cumulative[31] - cumulative[30])
   print(f"pulse leaves at {days[30] * 24:.0f} h, arrives in [{arrival[30] * 24:.2f}, {arrival[31] * 24:.2f}] h")

.. _concept-label-coordinate:

The label coordinate
--------------------

Transport in this package is not formulated on a time axis but on the **cumulative throughflow
volume at the reporting node**,

.. math::

   u = N(t) = \int_{t_0}^{t} q_n(t')\, \mathrm{d}t' ,

with :math:`q_n` the throughflow past node :math:`n` --- the summed demand below it, from
:meth:`~pipetransport.network.PipeNetwork.node_flow`. A parcel delivered at time :math:`t` is
*labelled* :math:`u = N(t)`, and it keeps that label from the moment it leaves the source, because
the label counts the water delivered at that node ahead of it and nothing overtakes anything in plug
flow. Water diverted into a sibling branch never passes the node and is simply not counted, which is
why each node has its own label axis.

Two properties follow, and they are what make the operator exact rather than approximate.

**A uniform average in the label is a flow-weighted average in time.** Since
:math:`\mathrm{d}u = q_n\, \mathrm{d}t`,

.. math::

   \frac{1}{u_{j+1} - u_j} \int_{u_j}^{u_{j+1}} X \, \mathrm{d}u
   \;=\;
   \frac{\int_{T_j}^{T_{j+1}} X(t)\, q_n(t)\, \mathrm{d}t}{\int_{T_j}^{T_{j+1}} q_n(t)\, \mathrm{d}t},
   \qquad u_j = N(T_j).

The left-hand side is a plain, unweighted integral; the right-hand side is the flow-weighted bin
average --- the physically correct one, since what a customer receives over an hour is the
volume-weighted mixture, not the time-weighted one. There is no separate weighting step in the code:
choosing the right coordinate *is* the weighting.

**The operator is an interval overlap.** Output bin :math:`j` occupies the label interval
:math:`[u_j, u_{j+1}]`. The water that left the source during input bin :math:`l` occupies
:math:`[g_l, g_{l+1}]` with :math:`g_l = N\bigl(A(t_l)\bigr)`. Both edge sequences are known exactly,
so for a conservative tracer

.. math::

   c_n(T_j) = \sum_l W_{jl}\, c_\text{in}(t_l),
   \qquad
   W_{jl} = \frac{\bigl| [u_j, u_{j+1}] \cap [g_l, g_{l+1}] \bigr|}{u_{j+1} - u_j}.

The rows are non-negative and sum to :math:`1` whenever the record covers the bin, which is mass
conservation. ``cout_tedges`` is free in alignment and resolution because it only enters through
:math:`u_j`: coarse output bins are wider intervals on the same axis, never a re-binning of a result.

Consequently, averaging a fine result up to a coarse grid is *not* a plain mean --- it is a
throughflow-weighted mean, and the package reproduces it exactly:

.. code:: python

   import numpy as np
   import pandas as pd

   from pipetransport.examples import example_demand, example_network
   from pipetransport.transport import source_to_endmember

   network = example_network()
   tedges = pd.date_range("2025-06-01", "2025-06-05", freq="h")
   demand = example_demand(tedges=tedges, network=network)
   cin = 1.0 + 0.3 * np.sin(2 * np.pi * np.arange(len(tedges) - 1) / 17.0)

   kwargs = {"cin": cin, "flow": demand, "tedges": tedges, "network": network, "nodes": ["T4"]}
   hourly = source_to_endmember(cout_tedges=tedges, **kwargs)[0]
   six_hourly = source_to_endmember(cout_tedges=tedges[::6], **kwargs)[0]

   # Weight of each hourly bin = the water T4 draws in it = the width of its label interval.
   volume = (network.node_flow(flow=demand, nodes=["T4"])[0] / 24.0).reshape(-1, 6)
   weighted = (hourly.reshape(-1, 6) * volume).sum(axis=1) / volume.sum(axis=1)
   np.testing.assert_allclose(weighted, six_hourly, rtol=0.0, atol=1e-14)

   plain = hourly.reshape(-1, 6).mean(axis=1)  # the wrong average
   print(f"unweighted mean is off by up to {np.nanmax(np.abs(plain - six_hourly)):.3f}")

.. _concept-effective-volume:

Effective volume: what a path really costs
------------------------------------------

The composed map handles arbitrary demand, but there is one case with a closed form worth knowing,
because it is the mental model most practitioners already carry --- and because it is the case in
which the usual intuition about shared mains is wrong.

Suppose every segment carries a **constant fraction** of the production,
:math:`Q_e(t) = f_e\, Q_0(t)`. (This happens whenever the endmember demands are constant, and more
generally whenever they are all proportional to one another, however the total varies.) Then
:math:`C_e = f_e\, C_0`, and the displacement condition :eq:`displacement` for segment :math:`e`
becomes a condition on the *source* cumulative volume:

.. math::

   f_e\, C_0\bigl(a_e(s)\bigr) - f_e\, C_0(s) = V_e
   \quad\Longleftrightarrow\quad
   C_0\bigl(a_e(s)\bigr) = C_0(s) + \frac{V_e}{f_e} .

Every segment now displaces in the same currency, so composing the path telescopes:

.. math::
   :label: effective-volume

   C_0\bigl(A(s)\bigr) = C_0(s) + V_\text{eff},
   \qquad
   V_\text{eff} = \sum_{e \in \text{path}} \frac{V_e}{f_e} .

The whole path behaves as **one** pipe of volume :math:`V_\text{eff}` carrying the full production,
and for a constant production rate the travel time is simply
:math:`\tau = V_\text{eff} / Q_0`. It is the closed form the general machinery is verified against,
and the snippet below reproduces it to machine precision.

Why the trunk main counts in full
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tempting intuition is that a trunk main feeding four districts is "shared", so each district
should only wait for its quarter of it. Equation :eq:`effective-volume` says otherwise: the trunk
leaving the plant has :math:`f = 1`, so it contributes :math:`V/1 = V` --- its **full** volume --- to
every path below it.

The sub-tube picture shows why. Slice the trunk lengthwise into parallel sub-tubes, one per
endmember below it, each taking the same share of the cross-section everywhere as it takes of the
trunk's flow: :math:`\phi_m` for endmember :math:`m`. That sub-tube holds :math:`\phi_m V` of water and
carries :math:`\phi_m Q_0` of flow, so under plug flow all sub-tubes advance at the same velocity ---
the slicing is consistent --- and each takes

.. math::

   \frac{\phi_m V}{\phi_m Q_0} = \frac{V}{Q_0}

to cross. The share cancels. A district does not wait for its own slice of the trunk; it waits for
the whole trunk, because its slice is proportionally thinner. Sharing a main buys capacity, not
speed.

The same equation makes the opposite case just as stark. A peripheral branch with :math:`f = 0.1`
contributes :math:`10\,V_e`: the same metre of pipe costs ten times as much waiting, because only a
tenth of the production passes through it. That is why the *oldest* water in a network is almost
always at the end of a long, thin, low-demand branch, and why total pipe volume is a poor predictor
of age.

.. code:: python

   import numpy as np
   import pandas as pd

   from pipetransport.examples import example_network
   from pipetransport.residence_time import full

   network = example_network()
   tedges = pd.date_range("2025-06-01", periods=49, freq="h")
   demand = np.array([[240.0], [360.0], [120.0], [80.0]]) * np.ones(48)  # constant -> f is fixed
   production = demand[:, 0].sum()

   fraction = network.segment_flow(flow=demand)[:, 0] / production
   volume = network.segments["volume"].to_numpy()
   row_of = {name: i for i, name in enumerate(network.segments.index)}

   age = full(flow=demand, tedges=tedges, network=network)  # days
   for i, node in enumerate(network.endmembers):
       rows = [row_of[segment] for segment in network.paths[node]]
       effective = np.sum(volume[rows] / fraction[rows])
       np.testing.assert_allclose(age[i, -1], effective / production, rtol=1e-12)
       print(f"{node}: path holds {volume[rows].sum():6.1f} m3, effective {effective:6.1f} m3, "
             f"age {age[i, -1] * 24:5.2f} h")

   # Mass conservation ties the two together: flow-weighting the effective volumes over the
   # endmembers returns the total network water volume, so the mean age of all delivered water
   # is the network turnover time, V_total / Q_0.
   share = demand[:, 0] / production
   np.testing.assert_allclose(share @ age[:, -1], network.segments["volume"].sum() / production)

In the example network T4's path holds *less* water than T1's (330 m³ against 371 m³) yet its
effective volume is far larger (683 m³ against 440 m³), and its water is delivered at 20.5 h instead
of 13.2 h.

``pipetransport`` never assumes a constant split --- it recomputes the flow at every time step ---
but it reproduces :eq:`effective-volume` exactly when the assumption happens to hold.

.. _concept-water-age:

Water age
---------

**Water age** is the time between leaving the source and being delivered: the travel time
:math:`\tau` of the arrival map. It is the most-used water quality indicator in a distribution
network, because it is the contact time of every reaction the water undergoes --- disinfectant
decay, disinfection by-product formation, nitrification, biofilm growth, temperature equilibration
with the ground --- and because it is one of the few things a utility can actually change, through
network design, valving and flushing.

:func:`pipetransport.residence_time.full` reports it in two directions, which answer different
questions:

- ``direction="endmember_to_source"`` (default): *how old is the water arriving now?* Averaged over
  each ``cout_tedges`` bin with the node's throughflow as weight, i.e. uniformly in the label
  (:ref:`concept-label-coordinate`).
- ``direction="source_to_endmember"``: *how long until the water produced now arrives?* Averaged
  over each ``tedges`` bin, weighted by the volume of that bin's production that is actually
  destined for this node --- so production that mostly serves other branches contributes little.

Any node may be a reporting node, not just an endmember; a junction reports the age of the water
passing through it, and the source reports zero.

Why the age swings over the day
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pipe holds a fixed volume of water, so its crossing time is the time needed to *displace* that
volume. Halve the flow and you double the crossing time. Demand in a drinking water network follows
a strong diurnal cycle, so age is never a single number: it rises whenever the pipes feeding a tap
slow down --- in the small hours for a residential branch --- and falls at the demand peak, and the
swing can be a large fraction of the mean.

The swing is not the same everywhere, and the reason is :eq:`effective-volume`. The shared trunk
main sees only the *aggregate* production, which is smooth --- individual peaks fall at different
hours and largely cancel. A peripheral branch sees its own endmember's demand, which is not smooth
at all. So a tap far out on a low-demand branch has both the oldest water and by far the most
variable age:

.. code:: python

   import numpy as np
   import pandas as pd

   from pipetransport.examples import example_demand, example_network
   from pipetransport.residence_time import full

   network = example_network()
   tedges = pd.date_range("2025-06-01", "2025-06-05", freq="h")
   demand = example_demand(tedges=tedges, network=network)  # deliberately non-proportional

   production = demand.sum(axis=1).to_numpy()
   print(f"production swings {production.min():.0f}-{production.max():.0f} m3/day")
   print(f"T4's own demand swings {demand['T4'].min():.0f}-{demand['T4'].max():.0f} m3/day")

   age_hours = full(flow=demand, tedges=tedges, network=network) * 24.0
   for node, series in zip(network.endmembers, age_hours):
       print(f"{node}: age {np.nanmin(series):5.2f}-{np.nanmax(series):5.2f} h")

In the example network the production varies by about :math:`\pm 3\,\%` while T4's own demand varies
by :math:`\pm 70\,\%`; T4's age swings from 18.0 h to 25.2 h, whereas T1, on a branch carrying three
times the flow, stays between 13.1 h and 14.4 h. A model built on fixed flow fractions cannot
produce that behaviour at all, because in such a model the age of every tap moves in lockstep with
the production.

.. _concept-first-order-decay:

First-order decay along a path
------------------------------

Disinfectant residual and pathogen inactivation credit are conventionally modelled as first order
(:ref:`assumption-first-order-decay`): a parcel retains :math:`e^{-k \tau}` of what it started with
after a contact time :math:`\tau` at rate :math:`k`. In a network the rate is **not** the same in
every pipe, so the exponent is accumulated segment by segment along the path:

.. math::

   \frac{c_\text{delivered}}{c_\text{produced}} = e^{-\varphi},
   \qquad
   \varphi = \sum_{e \in \text{path}} k_e\, \tau_e(s) ,

which for a constant flow becomes :math:`\varphi = \sum_e k_e V_e / Q_e`. The exponent is carried
alongside the arrival map, and because it is linear in the label on each refined cell, its bin
average integrates in closed form. Setting every :math:`k_e = 0` returns the conservative operator
exactly, so ``decay_rate=0.0`` is not a special case in the code.

Why thin pipes eat residual
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pipe consumes disinfectant both in the bulk water and at the wall, and the two combine into one
apparent first-order rate (Rossman, Clark and Grayman, 1994):

.. math::

   k = k_b + \frac{k_w\, k_f}{R_h\,(k_w + k_f)},
   \qquad R_h = \frac{D}{4},

with :math:`k_b` [1/day] the bulk rate, :math:`k_w` [m/day] the wall rate constant, :math:`k_f`
[m/day] the bulk-to-wall mass transfer coefficient and :math:`R_h` the hydraulic radius of the full
pipe. The wall term scales with the surface-to-volume ratio :math:`1/R_h = 4/D`: **halve the
diameter and the wall contribution doubles**. In the fast-mass-transfer limit
(:math:`k_f \to \infty`, the default of :func:`~pipetransport.logremoval.segment_decay_rate`) it is
simply :math:`k = k_b + 4 k_w / D`.

The two effects compound at the periphery. A thin service line has the largest surface-to-volume
ratio *and*, carrying the least flow, the longest contact time per metre (:eq:`effective-volume`).
That is why residual collapses at the ends of a network rather than degrading gradually, and why the
transport operator carries a per-segment rate rather than one rate for the whole network:

.. code:: python

   import numpy as np
   import pandas as pd

   from pipetransport.examples import example_network
   from pipetransport.logremoval import segment_decay_rate
   from pipetransport.transport import source_to_endmember

   network = example_network()
   tedges = pd.date_range("2025-06-01", periods=49, freq="h")
   demand = np.array([[240.0], [360.0], [120.0], [80.0]]) * np.ones(48)

   # k = 0.3 + 4 * 0.02 / D: 0.5 /day in the 400 mm trunk, 1.1 /day in the 100 mm branch.
   decay = segment_decay_rate(network=network, bulk_decay_rate=0.3, wall_decay_rate=0.02)
   residual = source_to_endmember(
       cin=np.ones(48), flow=demand, tedges=tedges, cout_tedges=tedges,
       network=network, decay_rate=decay,
   )

   # Constant flow, so the exponent has the closed form sum(k_e V_e / Q_e).
   flow = network.segment_flow(flow=demand)[:, 0]
   volume = network.segments["volume"].to_numpy()
   row_of = {name: i for i, name in enumerate(network.segments.index)}
   for i, node in enumerate(network.endmembers):
       rows = [row_of[segment] for segment in network.paths[node]]
       phi = np.sum(decay.to_numpy()[rows] * volume[rows] / flow[rows])
       np.testing.assert_allclose(residual[i, -1], np.exp(-phi), rtol=1e-13)
       print(f"{node}: residual {residual[i, -1]:.3f} of what left the plant")

Age is a diagnostic, not a shortcut
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is tempting to compute the residual as :math:`e^{-k\,\bar{\tau}}` from the mean age
:math:`\bar\tau`. That is not the same number. An output bin gathers a *band* of ages, and the
average of an exponential exceeds the exponential of the average (Jensen's inequality), so
:math:`\overline{e^{-\varphi}} \geq e^{-\bar\varphi}` always. In the example network at
:math:`k = 0.6` /day the gap reaches about :math:`10^{-4}` relative on hourly output bins and about
:math:`10^{-3}` on 12-hour bins --- small, because a single bin gathers a narrow band of ages, but it
grows with the width of that band and with :math:`k`. Pass ``decay_rate`` to
:func:`~pipetransport.transport.source_to_endmember` for the exact residual; use
:func:`~pipetransport.residence_time.full` for the age itself.

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

   from pipetransport.heat import segment_heat_rate, source_to_endmember
   from pipetransport.network import PipeNetwork

   segments = pd.DataFrame(
       {"from": ["Plant"], "to": ["T1"], "length": [1000.0], "diameter": [0.1], "cover": ["grass"]},
       index=["Plant-T1"],
   )
   network = PipeNetwork(segments=segments, source="Plant")
   soil = pd.DataFrame({"alpha": [0.05], "kappa": [0.025], "eta": [0.41]}, index=["grass"])

   tedges = pd.date_range("2025-01-01", periods=120 * 24 + 1, freq="h")
   n_bins = len(tedges) - 1
   transit_days = 2.0 / 24.0
   flow = np.full((1, n_bins), float(network.segments.loc["Plant-T1", "volume"]) / transit_days)
   shared = dict(
       tin=np.full(n_bins, 8.0), flow=flow, tedges=tedges, cout_tedges=tedges,
       network=network, soil=soil,
       surface_temperature=pd.DataFrame({"grass": np.full(n_bins, 20.0)}),
   )

   two_way = source_to_endmember(**shared)
   one_way = source_to_endmember(**shared, max_sweeps=1)

   # A fully developed halo is the analytic steady buried-pipe law, which is what one-way assumes.
   rate = float(segment_heat_rate(network=network, kappa=0.025, eta=0.41)["Plant-T1"])
   steady = 20.0 + (8.0 - 20.0) * np.exp(-rate * transit_days)
   np.testing.assert_allclose(one_way[0, -1], steady, rtol=1e-9)

   for day in (1, 7, 30, 119):
       correction = two_way[0, day * 24] - one_way[0, day * 24]
       print(f"day {day:3d}: halo correction {correction:+.2f} K")

Cold water into warm soil takes up 2.42 K more than the one-way model allows on the first day,
0.83 K after a week and 0.08 K after four months --- the soil around the pipe cooling toward the
state the one-way model assumed from the outset. Which way the correction points is set by the sign
of the flux, so a network whose production temperature crosses the soil temperature during a record
sees the stored heat come *back out*. It can also carry the delivered water briefly *outside* the
range of the produced water and the soil, which is not a bug --- see
:ref:`assumption-effective-target`.

.. _concept-coverage:

What the record does and does not constrain
-------------------------------------------

The operator is exact, but only where the data reaches. ``NaN`` in a result is never a numerical
failure; it marks an output bin the record genuinely does not determine. There are three causes:

**Spin-up.** The water delivered in the first bins left the source before the record began. With
``spinup=None`` those bins are ``NaN``. The default, ``spinup="constant"``, extends the record
backwards --- holding every endmember demand, and the produced quality, at its first observed value
--- for the longest source-to-node travel time, so the early bins carry a value that rests on an
explicit, stated assumption rather than on data. Warm-started bins are not observations; treat the
first travel time of the record with suspicion either way.

**The end of the record.** An output bin extending past the last flow edge cannot be closed and is
``NaN``.

**No throughflow.** A node that draws no water during an output bin delivers nothing to average, and
the bin is ``NaN``. This is the correct answer, not a gap to be filled: a closed tap has no
flow-weighted concentration. It is also why a long stagnation shows up as a block of ``NaN``
followed by markedly older water.

In the reverse direction (:func:`~pipetransport.transport.endmember_to_source`) the same logic runs
the other way: source bins that no measurement constrains come back ``NaN``, and ``NaN`` in the
measured ``cout`` simply drops that observation out of the solve --- which is how a sparse grab
sampling campaign is expressed. Because each sampling point constrains a different, demand-dependent
window of the production history, and the windows move as the demand pattern shifts, several
sampling points together pin down more than the sum of what each does alone.

.. code:: python

   import numpy as np
   import pandas as pd

   from pipetransport.network import PipeNetwork
   from pipetransport.transport import source_to_endmember

   segments = pd.DataFrame(
       {"from": ["Plant", "A", "A"], "to": ["A", "T1", "T2"], "volume": [300.0, 40.0, 60.0]},
       index=["Plant-A", "A-T1", "A-T2"],
   )
   network = PipeNetwork(segments=segments, source="Plant")
   tedges = pd.date_range("2025-06-01", periods=97, freq="h")

   demand = np.stack([np.full(96, 400.0), np.full(96, 200.0)])
   demand[1, 40:48] = 0.0  # T2 draws nothing for eight hours

   cout = source_to_endmember(
       cin=np.sin(np.arange(96) / 5.0) + 2.0, flow=demand, tedges=tedges,
       cout_tedges=tedges, network=network,
   )
   np.testing.assert_array_equal(np.flatnonzero(np.isnan(cout[1])), np.arange(40, 48))
   assert not np.isnan(cout[0]).any()  # T1 keeps drawing, so T1 keeps reporting

References
----------

Rossman, L. A., Clark, R. M., & Grayman, W. M. (1994). Modeling chlorine residuals in drinking-water
distribution systems. *Journal of Environmental Engineering*, 120(4), 803-820.
