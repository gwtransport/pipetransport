.. _assumptions:

Assumptions
===========

``pipetransport`` buys its exactness and its speed with a small number of structural assumptions.
They are not hidden: every one of them is checked, or explicitly not checkable, and each costs
something specific when it is violated. This page states each assumption, says what breaks, and says
what you can do about it.

The short version: the package models **one source feeding a tree, always forward, always full,
plug flow inside every pipe, geometry known and constant**. Loops, a second production plant,
storage tanks and reservoirs, and flow reversal are outside its scope --- not approximated, not
supported. Demand is assumed to be known where it is drawn.

.. contents:: Contents
   :local:
   :depth: 1

.. _assumption-tree-topology:

1. Single source, tree topology
-------------------------------

**Assumption.** The network is a rooted tree: one production point at the root, every other node fed
by exactly one segment, and flow that splits at junctions but never merges. Equivalently, there is
exactly one path from the source to any node.

**Why it matters.** This is the structural property the whole model rests on. Because a split leaves
concentration unchanged (:ref:`concepts`), a single produced-quality series describes every node in
the network, and the transport to a node is a pure delay plus decay along its one path. No mixing
operator appears anywhere, and no hydraulic solve is needed --- segment flows follow from mass
conservation on the endmember demand.

**What breaks if it is violated.**

- *Looped or gridded networks.* A node fed from two directions delivers a flow-weighted blend of two
  transport histories. The blend fractions are set by head losses, not by demand, so they cannot be
  derived from the metered draw-off at all: the loop needs a momentum/head solve. Worse, the split
  ratio in a loop moves with demand and can reverse (see :ref:`assumption-forward-flow`).
- *A second treatment plant.* The delivered water is a mixture of two source signals, and the mixing
  zone between the two supply areas migrates over the day. One ``cin`` cannot represent it.

**What you can do.**

- Model the branched part. Most distribution systems are looped in the transmission core and
  strictly branched at the periphery, and it is the periphery that accumulates the age and loses the
  residual. Take a node inside the core whose supply direction is stable, treat it as the source,
  and give it the quality measured or modelled there.
- If you know the blend fractions of two supplies at a boundary node (from a hydraulic model or from
  a conservative tracer), model each supply path separately and blend the results: a flow-weighted
  average of concentrations, or :func:`pipetransport.logremoval.parallel_mean` for disinfection
  credit, which correctly lets the least-treated stream dominate.
- Use a full hydraulic and water quality solver (an EPANET-class model) when the loop hydraulics
  themselves are the question.

**How it is checked.** Enforced. :class:`~pipetransport.network.PipeNetwork` raises on a node with
two feeds ("merging flows are not supported"), on a source that has an incoming segment, and on any
node not reachable from the source (a disconnected branch or a cycle).

.. _assumption-plug-flow:

2. Plug flow inside every pipe
------------------------------

**Assumption.** Water moves through a pipe as a plug: the concentration is uniform over the cross
section, no parcel overtakes another, and the profile is translated along the pipe without
longitudinal mixing.

**Why it matters.** Plug flow is what makes the displacement condition
:math:`C_e(a_e(s)) - C_e(s) = V_e` the complete description of a pipe, and it is also what makes a
split concentration-preserving: each outgoing branch draws from a cross-sectionally uniform slug.
The model therefore produces **no numerical dispersion at all** --- there is no spatial grid --- and
the only spreading in a result comes from the network itself: different paths and a varying flow
that stretches and compresses the signal.

**What breaks if it is violated.** Real pipes disperse longitudinally, and the error is not
symmetric across the network.

- *Turbulent trunk and district mains* --- the usual regime at
  :math:`\mathrm{Re} = v D / \nu \gtrsim 4000` --- mix radially within seconds, and the residual
  axial spreading is modest. A rough estimate with :math:`E \approx 10\, a\, u_*` (pipe radius
  :math:`a`, shear velocity :math:`u_* \approx v \sqrt{f/8}`) gives, for a 300 mm main at 0.5 m/s
  over 2 km, a front smeared over roughly 1 % of the travel distance. Plug flow is a good
  approximation here.
- *Laminar service lines* --- small diameters at low, intermittent velocity --- are the honest
  weakness. The velocity profile is parabolic: centre-line water moves at twice the mean velocity
  while water at the wall barely moves. Molecular diffusion is what eventually averages that profile
  out (the Taylor--Aris regime), but it needs a time of order :math:`a^2/D_m`, roughly 40 hours for a
  25 mm line, far longer than water usually spends there. So the profile is *not* averaged out: the
  leading edge of a front can arrive in about half the plug-flow travel time, and the tail runs long.

  **Plug flow is therefore optimistic exactly where the water is oldest and the residual lowest.**
  For a contamination trace, the arrival time the model reports at a tap is the arrival of the bulk;
  first detectable arrival can be substantially earlier.
- *Long stagnation.* After hours without flow, buoyancy- and temperature-driven mixing in a dead end
  smears the plug even without any throughflow.

**What you can do.**

- Check the regime for the pipes that matter: :math:`\mathrm{Re} = vD/\nu` with
  :math:`\nu \approx 1.3 \times 10^{-6}` m²/s for water at 10 °C and
  :math:`v = 4 Q_e / (\pi D^2)` from :meth:`~pipetransport.network.PipeNetwork.segment_flow`
  (converted to m³/s). Trust sharp-front timing where the flow is turbulent; treat it as a
  bulk-arrival estimate where it is not.
- Decide what you want from a service line before you model it. Its volume adds only a small delay,
  but its wall reaction can dominate the residual loss, so keeping it as a segment is usually right
  --- just do not read a sharp front at its outlet as sharp. If only the arrival time matters, report
  at the main feeding it and treat the last stretch as a smeared delay.
- Read the delivered series as what it is: a flow-weighted bin average, not an instantaneous grab
  sample. Coarser ``cout_tedges`` bins are less sensitive to the smearing the model omits.
- Do not reach for ``retardation_factor`` here. It models a *reversible exchange with the wall* that
  delays a compound relative to the water; it multiplies the pipe volume and shifts the arrival. It
  does not spread anything.

The check itself is three lines; the unit conversion is the part that bites, since the package works
in m³/day and the Reynolds number in m³/s.

.. code:: python

   import numpy as np
   import pandas as pd

   from pipetransport.examples import example_demand, example_network

   network = example_network()
   tedges = pd.date_range("2025-06-01", periods=25, freq="h")
   demand = example_demand(tedges=tedges, network=network)

   flow = network.segment_flow(flow=demand) / 86400.0  # m3/day -> m3/s
   diameter = network.segments["diameter"].to_numpy()[:, None]
   velocity = 4.0 * flow / (np.pi * diameter**2)  # m/s
   reynolds = velocity * diameter / 1.3e-6  # water at 10 C

   for name, row in zip(network.segments.index, reynolds, strict=True):
       flag = "" if row.min() > 4000.0 else "   <- leaves the turbulent regime"
       print(f"{name}: Re {row.min():6.0f} - {row.max():6.0f}{flag}")

   # Every main stays turbulent around the clock; the thin, low-demand branch does not.
   row_of = {name: i for i, name in enumerate(network.segments.index)}
   assert reynolds[row_of["Plant-A"]].min() > 4000.0
   assert reynolds[row_of["C-T4"]].min() < 4000.0

In the example network the 100 mm branch to T4 falls to :math:`\mathrm{Re} \approx 2.8 \times 10^3`
at the afternoon trough of its industrial demand --- the transitional regime, where the plug-flow
front is least trustworthy --- and that is precisely the branch carrying the oldest water and the
lowest residual.

**How it is checked.** Not checkable from the inputs the package receives --- it is a statement
about the flow regime, and the package never sees a velocity. Compare a measured breakthrough (a
tracer, or a chlorine or temperature step) against the modelled one: a symmetric smear around the
modelled arrival is dispersion, whereas a systematic shift is a volume error
(:ref:`assumption-known-geometry`).

.. _assumption-no-storage:

3. No storage, no leakage: demand metered at the endmembers
-----------------------------------------------------------

**Assumption.** Pipes are always full and rigid, the water is incompressible, and the network holds
water nowhere else --- no tanks, towers, service reservoirs or air vessels --- and loses none. Every
draw-off is metered at an endmember of the model. Consequently the throughflow of a segment equals
the summed demand of the endmembers below it at every instant, and the production equals the total
demand.

**Why it matters.** This is what removes the hydraulic solver. Given the demand at the leaves, every
internal flow is fixed by mass conservation, at every time step, with no calibration and no
assumption of proportionality between demands.

**What breaks if it is violated.**

- *A service reservoir or water tower.* A tank that fills at night and empties at peak decouples the
  flow upstream of it from the demand downstream, so the mass-conservation accounting is simply
  wrong on both sides. A tank is also a mixing vessel, not a plug: it blends water of very different
  ages, and its outflow quality depends on its internal mixing regime, not on a travel time.
- *Leakage and unmetered demand.* Real losses run at roughly 5-25 % of production. A leak is a
  demand that the model does not know about: every segment upstream of it is modelled as carrying
  too little water, so travel times come out **too long** and residuals too low.
- *A draw-off part-way along a pipe.* A segment in the model carries one flow over its whole length.
  A significant tapping midway means the upstream half moves faster than the downstream half.
- *Compressibility and pipe elasticity* are negligible for transport. They matter for pressure
  transients, not for the volume displaced over minutes to hours.

**What you can do.**

- Cut the model at the tank. Run one :class:`~pipetransport.network.PipeNetwork` from the plant to
  the tank inlet, apply your own mixing model for the tank (a continuously stirred tank is the usual
  first approximation, and a compartment model if you have one), then run a second network with the
  tank as its source and the mixed quality as ``cin``.
- Represent leakage and unmetered demand as extra endmembers, placed where you believe they occur ---
  a leaking trunk main becomes a demand at the node below it. If only a system-wide loss rate is
  known, distributing it in proportion to the metered demand is a much better model than ignoring
  it.
- Split a pipe with a significant intermediate draw-off into two segments with a node between them,
  and put the draw-off at that node. The tree grows by one leaf; nothing else changes.

**How it is checked.** Partly. Negative demand is rejected, and the flow array must cover every
endmember. The physical premise is checkable against your own data: compare the metered production
series with the summed endmember demand. A persistent gap is leakage or unmetered demand, and its
size is the size of your travel-time bias.

.. _assumption-forward-flow:

4. Forward flow only
--------------------

**Assumption.** Flow is non-negative everywhere and at all times, running from the source toward the
endmembers. Zero flow is allowed; reversal is not.

**Why it matters.** Non-negative flow makes every cumulative volume :math:`C_e` non-decreasing, so
the displacement condition has a unique solution, parcels keep their order, and the label coordinate
(:ref:`concept-label-coordinate`) is monotone. Stagnation is handled exactly rather than specially:
a parcel in a pipe that stops simply waits, and a node that draws no water during an output bin
reports ``NaN`` because there is no water to average.

**What breaks if it is violated.** Reversal genuinely happens in real systems --- a valve operation,
a fire flow, a tank switching from filling to emptying, a demand shift that flips the split in a
looped core. Under reversal, water re-enters a pipe it has already left, carrying downstream history
back upstream. The arrival map is then not invertible, the "one path" claim of
:ref:`assumption-tree-topology` fails, and mixing occurs at what used to be a split. The package
cannot represent that, and does not pretend to: a negative demand raises.

**What you can do.**

- Split the record at the reversal events and model each forward-flow period separately, seeding the
  next period through its spin-up. Accept that the first travel time after a reversal is wrong ---
  it is exactly the interval in which the water re-mixed.
- Aggregate to a time step over which the net flow is positive. This is defensible only when the
  reversed volume is small compared with the pipe volume; otherwise it hides a real mixing event
  behind a smooth number.
- If demands come from a hydraulic model, inspect the sign of the modelled pipe flows before
  converting them into endmember demands.
- Remember that a long stagnation, while handled exactly here, is physically the regime where plug
  flow degrades fastest (:ref:`assumption-plug-flow`) and where wall reaction has the most time to
  act.

**How it is checked.** Enforced. :meth:`~pipetransport.network.PipeNetwork.flow_array` raises on any
negative demand ("reverse flow not supported"), and on ``NaN``.

.. _assumption-known-geometry:

5. Known, time-constant geometry
--------------------------------

**Assumption.** Every segment has a known water volume that does not change over time, supplied
either directly as a ``volume`` column or computed from the inner diameter and length as
:math:`V = \tfrac{\pi}{4} D^2 L`. The diameter additionally sets the wall-reaction term in
:func:`~pipetransport.logremoval.segment_decay_rate`.

**Why it matters.** Volume is the *only* geometric quantity transport depends on: length and
diameter enter the arrival map exclusively through :math:`\tfrac{\pi}{4}D^2L`. The dependence is
linear, and so is the error: at a given flow, a 10 % error in a segment volume is a 10 % error in
that segment's travel time, and shifts that segment's contribution to the decay exponent of every
path through it by the same 10 %.

**What breaks if it is violated.**

- *As-built versus recorded diameters.* A pipe recorded as 150 mm that was laid as 125 mm holds 31 %
  less water.
- *Tuberculation and lining.* Corrosion products and cement lining reduce the bore. A 10 % loss of
  bore is a 19 % loss of volume --- and it raises the surface-to-volume ratio, so wall decay speeds
  up at the same time as travel time drops.
- *Lumped segments.* When a whole district's mains are collapsed into one equivalent segment, its
  volume is a fitted quantity, not a measured one, and its single diameter is a fiction for the
  purpose of wall decay.
- *Operational changes.* A closed or throttled valve, a rerouted main, or a seasonal network
  reconfiguration changes the path itself, not just its volume. The model's geometry is constant
  over the whole record.

**What you can do.**

- Calibrate the effective volume rather than trusting the drawings. The composed arrival map is
  linear in the segment volumes, so a measured breakthrough (tracer, chlorine step, or the natural
  temperature signal) pins the path volume down directly: a modelled front arriving 10 % early means
  the path volume is about 10 % low.
- Pass ``volume`` directly whenever you have a better estimate than :math:`\tfrac{\pi}{4}D^2L` ---
  but note that :func:`~pipetransport.logremoval.segment_decay_rate` then has no diameter to work
  with and will refuse a wall reaction. Supply both columns when you want wall decay.
- Sanity-check the total. ``repr`` of a :class:`~pipetransport.network.PipeNetwork` prints the total
  water volume; divided by the mean production it is the network turnover time, which under a
  constant split is exactly the demand-weighted mean age of all delivered water
  (:ref:`concept-effective-volume`). If that number is implausible for your system, the geometry is
  wrong before any transport result can be right.
- Re-build the network for each operating configuration if the network is reconfigured seasonally,
  and model each period on its own record.

**How it is checked.** Partly. Volumes, lengths and diameters must be strictly positive, and the
volume column is derived once at construction. Whether those numbers describe your pipes is outside
the package's reach.

.. _assumption-first-order-decay:

6. First-order kinetics at a time-constant rate
-----------------------------------------------

**Assumption.** Reaction is first order in the transported quantity, with a rate :math:`k_e` that is
constant in time for each segment (it may differ freely between segments). The decayed quantity does
not feed back on the flow, and there is no reaction between species.

**Why it matters.** First-order kinetics is what lets the whole path collapse into a single
dimensionless exponent :math:`\varphi = \sum_e k_e \tau_e` carried alongside the arrival map, and
keeps the operator linear --- which is what makes the reverse direction a solvable linear inverse
problem at all.

**What breaks if it is violated.** Bulk chlorine decay is often better described as first order in
chlorine *and* in the reactant it consumes, giving an apparent rate that falls as the water ages;
fitting a single :math:`k_b` over a long path then overestimates the loss at the far end. The wall
mass transfer coefficient :math:`k_f` depends on velocity and therefore on time, so a fixed
:math:`k_e` is a representative value, not an exact one. Disinfection by-product *formation* is not
first-order removal at all. Nothing here models two species reacting with each other.

**What you can do.** Fit :math:`k_b` over the age range you actually care about rather than over the
whole record; take :math:`k_f \to \infty` (the default of
:func:`~pipetransport.logremoval.segment_decay_rate`) to get the wall-controlled upper bound on the
wall term, and pass a representative finite :math:`k_f` if you want a mass-transfer-limited estimate;
bracket the answer by running the fast and slow ends of your rate estimates. For a conservative
tracer, leave ``decay_rate`` at its default of 0.0 --- the conservative operator is the exact zero
limit of the decaying one, not a separate code path.

**How it is checked.** Partly. Negative rates are rejected, and a ``Series`` of rates must cover
every segment. The kinetic form itself is your modelling choice.

Units
-----

The package does not enforce units and does not convert them. Internally everything is days, metres,
cubic metres, cubic metres per day and inverse days; concentrations are carried in whatever unit
``cin`` uses, since the operator is linear and unit-agnostic. Feeding it demand in m³/hour against
volumes in m³, say, silently returns travel times 24 times too long --- no error, no warning. That
responsibility is yours.
