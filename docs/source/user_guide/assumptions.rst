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

.. _assumption-soil-columns:

Heat exchange with the soil
---------------------------

These assumptions apply only to :mod:`pipetransport.heat`; the rest of the package is unaffected by
them. They sit on top of everything above --- the heat model is the same transport operator read as
an affine map, so tree topology, plug flow and known demand are all still required.

**Assumption.** The soil around a segment behaves as a set of independent radial columns around a
*constant-flux cylinder* at the pipe wall, with a mirror-image line sink above the ground surface;
the soil is homogeneous and time-constant per segment; the pipe wall is a memoryless series
resistance; and the water is well mixed across the pipe section.

**Why it matters.** Independence of the columns is what makes the wall temperature a local quantity
and reduces the halo to one convolution per segment. It is well founded: axial conduction reaches
about a metre in a month, against axial variation scales of hundreds of metres. The mirror image is
what makes the halo saturate at the steady buried-pipe resistance instead of growing without bound.
The wall's own thermal response time is minutes, far below any sensible bin width, so it carries no
memory of its own.

That the source is a cylinder rather than a line matters most in the *first* lag bin, which is the
one the same-bin coupling leans on. Below a Fourier number :math:`\alpha \Delta t / r_o^2` of roughly
5 --- which is every configuration this package is for --- a line source read at :math:`r = r_o` has
barely begun to respond, because it puts the whole wall flux on a line of zero radius and then samples
the field where the pipe surface actually is. The cylinder response is the only one of the module's
three kernels without a closed form: it is evaluated by quadrature along the branch cut of its Laplace
transform, once per distinct :math:`\alpha \Delta t / r_o^2` when the system is built, to about
:math:`10^{-13}` relative. So the module's kernels are exact for piecewise-constant inputs, but one of
them is exact only to a stated tolerance rather than in closed form. The image keeps its line source:
read from :math:`2 d_\text{eff}` away rather than from the pipe's own wall, a cylinder is a line to
the same order the *steady* resistance already is, and it costs the same order --- at most
:math:`2\times10^{-4}` of :math:`R_\text{soil}` for a 100 mm service line buried a metre and
:math:`3.5\times10^{-3}` for a 400 mm main, against the :math:`1\times10^{-4}` and
:math:`3.8\times10^{-3}` that :math:`\ln(2 d_\text{eff}/r_o)` itself gives up against
:math:`\operatorname{acosh}(d_\text{eff}/r_o)`. Both grow as the burial approaches :math:`r_o`.

**What breaks if it is violated.** Two mains sharing a trench warm each other's soil, which this
model does not represent --- each segment sees only its own halo, so the delivered temperature of
both is underestimated in summer. Freeze--thaw and seasonal moisture change the soil properties
within a run, which the model holds fixed. A real pipe is not a perfect cylinder of constant flux
either: the flux around the circumference is higher on the side facing the surface, which this model
averages away, and it falls along the pipe, which :ref:`assumption-uniform-wall-flux` covers
separately.

**What the cylinder kernel is worth, in numbers.** The first lag bin holds a fraction
:math:`\bar{D}[0]/R_\text{soil}` of the steady soil resistance as *not yet arrived*, and that fraction
is what sets the same-bin loop gain. With the cylinder it is 0.8568 for a 100 mm service line and
0.9323 for a 400 mm main on hourly bins; with a line source it would be 0.9418 and 1.0000. The 1.0000
is the one that mattered: with no margin at all the fixed point is near-singular, and a 400 mm main at
a half-hour transit came back spanning :math:`-88` to :math:`+78` °C from water produced at 8 °C into
soil at 22 °C --- converged, without a warning, at default settings --- while refining ``tedges`` made
it worse rather than better. The cylinder closes that regime: the same pipe delivers 8.2--9.0 °C
against a one-way 8.15 °C, and no bin width reopens it. It also cuts the sweep count about fourfold,
100 against 286 on the example network.

**The pre-history matters more than any of that.** The model starts with an undisturbed halo, so a
pipe that has in reality been running for years is modelled as one switched on at the first bin,
meeting soil that accepts heat almost without resistance. On a 100 mm service line with a sustained
12 K difference the delivered temperature is several kelvin too close to the soil at the start of the
record; a week of lead-in brings that under 1 K, a month under 0.4 K and a season under 0.1 K.
Supply lead-in and leave ``cout_tedges`` on the period you care about; the output grid is free, so
nothing needs discarding.

**What you can do.** Start ``tedges`` about three weeks --- :math:`d_\text{eff}^2/\alpha` --- before
the period you care about, with realistic history: the measured surface record, a typical demand
pattern, and a production temperature near the record's opening value. That one lead-in serves both
slow memories: it lets the network build the halo the model otherwise starts without (the measured
decay above), and it delivers about half of any recent surface swing to a metre's depth --- the
seasonal baseline older than the lead-in enters as the first value of each surface series, so open
the lead-in where the record is representative rather than at an extreme. Use bin widths that resolve
the forcing you care about --- the halo memory reads the flux history only through bin averages. Keep
parallel mains out, or merge them into one equivalent segment.

**How it is checked.** Partly, and mostly when the :class:`~pipetransport.heat.HeatNetwork` is
built rather than when it is solved: geometry and soil parameters must be positive, the burial depth
must clear the outer pipe radius, and any ``volume`` column must agree with the length and diameter.
At solve time every segment's cover class must appear in the surface record; ``tedges`` must be uniformly spaced, since the halo memory is a convolution. The
adequacy of the lead-in, the bin width and the trench spacing is your modelling choice.

.. _assumption-uniform-wall-flux:

How far along a pipe the wall flux is resolved
----------------------------------------------

**Assumption.** The wall flux is uniform along a pipe, and each pipe keeps one soil memory. That is
the model's spatial resolution along a pipe; there is no parameter, and nothing is subdivided
internally.

**Why it matters.** Because the soil columns are independent, the flux is a *local* quantity, and it
falls along a pipe like :math:`e^{-h_e \tau}` --- by a factor 1.6 over a 2 h transit on a 100 mm
service line, and 3.8 over 6 h. Charging the whole pipe with one average flux history therefore gives
every parcel in it the same soil memory. Under 24 h diurnal forcing at hourly bins that is worth
about 0.5--0.9 K on a 100 mm line. It is not a bin-resolution question that a finer ``tedges`` would
answer: it is a *spatial* one.

**What you can do.** Declare the pipe as a chain of shorter segments. That is exact in the transport
it does not touch --- :math:`k` series pieces of volume :math:`V/k` at the same flow compose to the
same arrival map, and their exchange exponents add to the whole pipe's, so ``W``, the residence times
and the coverage mask are unchanged to round-off and only the soil memory is refined.

**Check that it settles when you do.** Under continuous flow it does: on a 2 km / 100 mm line the
per-doubling steps fall about fourfold (0.21, 0.058, 0.014, 0.0034 K). Under intermittent demand it
does **not**. On the same pipe drawing 11 h a day the steps are 2.39, 3.14, 2.61, 2.31 K, and deeper
stagnation diverges outright --- at 2 h a day, 1 against 16 pieces differ by 818 K and the call still
returns without raising. The mechanism is that :math:`\psi = Q (T_\text{in} - T_\text{out}) / L`
saturates at the full parcel-to-wall contrast after a stagnation however short the piece is, so
refining amplifies the flux fed to the halo instead of resolving it. Refine only where the branch
flows continuously, and read a moving answer as a warning rather than as progress.

**How it is checked.** Not checked --- there is no parameter to validate. That one flux history per
pipe is an approximation, and that refining it converges under continuous flow, are pinned by tests
against an independently written reference which keeps one soil memory per axial cell.


.. _assumption-effective-target:

The relaxation target is an effective driving temperature
----------------------------------------------------------

**Assumption.** The water relaxes toward :math:`T_b = T_\infty - \text{memory}` at the *steady*
exchange rate. :math:`T_b` is a bookkeeping quantity, not the temperature of anything.

**Why it matters.** The rate carries the fully developed soil resistance, which is too large while
the halo is still building. To reproduce the faster early exchange the target has to be driven past
the undisturbed soil, away from the water --- by several times the driving contrast in the bins just
after a sharp change in the wall flux. That is the correct behaviour of the split into *steady part
plus transient deficit*, not an artefact.

**What breaks if it is violated.** Nothing internally, but it breaks an expectation: because the
delivered temperature is a genuine weighted average of the produced water and those targets, **it can
fall outside the range of its own inputs**, and there is no general bound on by how much.

A step in the produced temperature into a *continuously flowing* pipe is the mild case: 18 % of the
instantaneous plant-to-soil contrast --- 4.3 K on a 100 mm line after a 24 K step --- peaking 1.9
transits after the step and back inside the range after about nine days. Splitting the pipe
(:ref:`assumption-uniform-wall-flux`) into a chain of shorter segments cuts that to 1--4 %.

Intermittent demand used to be far worse, and is the case the wall-flux attribution was changed for.
Booking each parcel's heat over the bins it actually occupied rather than over the single bin it was
delivered in, a 100 mm line idle 8 hours a day delivers water 7.8 % of the contrast past the soil,
and resolving the pipe now *helps*: the excursion falls monotonically and is back inside the range at
two pieces (--3.8 %) and four (--10.3 %). Reading the flux off the delivered water instead put the
same case at 20--28 % and made refining worse rather than better, and on sharper duty cycles --- a
main flushed two hours in every 24, a line standing ten days --- it reached 8.8 times the contrast,
converged and unflagged. Those shapes now come back within about a kelvin of the range of their
inputs.

The one-way model (``max_sweeps=1``) has a fixed target and is still the only variant guaranteed
inside the range of its inputs.

**What you can do.** Treat any delivered temperature outside the hull of your inputs as carrying this
overshoot rather than as a physical prediction, and compare against ``max_sweeps=1`` when you need a
bound that cannot leave the hull. Be most suspicious on branches with a strong duty cycle, where the
effect is largest --- though refining the pipe now settles it rather than aggravating it.

**How it is checked.** Not at runtime --- the invariant a runtime check would assert is false, and a
guard that fires on correct physics is worse than none. Its *size* is pinned by a test.

Stagnation
----------

**Assumption.** None any more: a segment with no throughflow exchanges heat with the soil like any
other, and this section records what used to be assumed instead.

**Why it matters.** The wall flux of a bin is the segment's own enthalpy budget over it, so a bin
that delivers nothing still reports the heat the water standing in it gave up ---
:math:`-h (H - V T_b)`, the same term that drives a flowing bin. An overnight stagnation of a service
line reaches :math:`h\tau \approx 2`, nearly full equilibration, so this is most of what such a pipe
does with its day.

**What used to break.** The flux was read off the water a pipe *delivered* and charged to the bin it
was delivered in. A bin that delivered nothing therefore reported nothing, and a whole idle night's
exchange landed on the few bins of the next morning's flush --- overstated by the ratio of the
standing time to the transit. On a line idle 8 hours a day that drove the delivered temperature
20--28 % of the plant-to-soil contrast past the soil and *grew* when the pipe was refined; on sharper
duty cycles it reached several times the contrast, converged and unflagged. The corresponding
excursion is now 7.8 %, and refining settles it.

**What you can do.** Nothing special. The one case still worth attention is the very first bins of a
record, where the model assumes the pipe starts in equilibrium with undisturbed soil rather than
carrying a history it cannot know.

**How it is checked.** By the only test in the suite with zero flow anywhere: it pins the excursion
under a duty cycle and asserts that refining the pipe reduces it monotonically and brings the answer
back inside the range of its inputs.

Units
-----

The package does not enforce units and does not convert them. Internally everything is days, metres,
cubic metres, cubic metres per day and inverse days; concentrations are carried in whatever unit
``cin`` uses, since the operator is linear and unit-agnostic. Feeding it demand in m³/hour against
volumes in m³, say, silently returns travel times 24 times too long --- no error, no warning. That
responsibility is yours.
