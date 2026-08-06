.. _assumptions-heat:

What the heat model assumes
===========================

The assumptions below apply only to :mod:`pipetransport.heat`. Everything the rest of the
package assumes is in :doc:`assumptions`, and none of it is affected by these.

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
them is exact only to a stated tolerance rather than in closed form.

The image keeps its line source, and what that whole conceptual choice costs --- cylinder at the wall,
line image at :math:`2 d_\text{eff}`, saturating at :math:`\ln(2 d_\text{eff}/r_o)` --- is measured
rather than bounded. An independent two-dimensional solve of the true problem in the cross-sectional
plane, which carries no image at all and is never told a resistance, puts the steady uniform-flux wall
temperature *above* the package's by :math:`(r_o/2 d_\text{eff})^2/(2\pi\kappa)`, the measured
coefficient being 1.00, 0.995 and 0.98 at :math:`d_\text{eff}/r_o` of 21, 5.3 and 2.5 --- so
:math:`1.5\times10^{-4}` of :math:`R_\text{soil}` for a 100 mm service line buried a metre,
:math:`3.7\times10^{-3}` for a 400 mm main and :math:`2.4\times10^{-2}` at two and a half radii. The
sign is worth reading: to leading order :math:`\ln(2 d_\text{eff}/r_o)` sits midway between the two
wall conditions, exceeding the isothermal :math:`\operatorname{acosh}(d_\text{eff}/r_o)` by what it
falls short of the uniform-flux answer by --- only to leading order, since the quartic terms differ.
Transiently the gap is *larger* rather than smaller. The true surface starts cooling the wall before
the line image does --- the image is read from the axis at :math:`2 d_\text{eff}`, while the near side
of the wall sees its own at :math:`2(d_\text{eff} - r_o)` --- so while it arrives the model credits the
wall with more resistance than has reached it, by 2.7 times the steady gap over the geometries measured
(:math:`d_\text{eff}/r_o` of 2.5 to 21; the ratio itself keeps growing roughly as
:math:`\ln(2 d_\text{eff}/r_o)`, though the absolute error shrinks). It returns to the steady gap only
as :math:`1/t`. All of it grows as the burial approaches :math:`r_o`.

The surface film is folded into the same picture by displacing the surface downward by the radiation
length :math:`\kappa/\eta` and treating it as perfect --- the effective depth
:math:`d_\text{eff} = d + \kappa/\eta`. A genuine Robin surface is not one image but a distribution of
them, and summing that distribution in closed form gives
:math:`2\pi\kappa R = \ln(2 d/r_o) + 2 e^{x} E_1(x)` with :math:`x = 2 d \eta/\kappa`. Measured against
it --- and independently against the two-dimensional solve, which agrees with the closed form to its
own discretisation floor --- the displacement captures 99.95 % of what the film does, and always
errs low: :math:`2.0\times10^{-4}` day/m² at :math:`h_s = 20` W/(m² K), which is
:math:`8.5\times10^{-6}` of :math:`R_\text{soil}` and so more than an order below the cylinder-image
gap it is combined with. It grows as the film weakens.

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

**Assumption.** The wall flux along a pipe is carried in its leading axial Legendre modes ---
``n_modes`` of them, six by default --- and each mode keeps its own soil memory. The count is the
model's spatial resolution along a pipe, a model order like a mesh; nothing is subdivided
internally, and one mode is the classical uniform-flux model.

**Why it matters.** Because the soil columns are independent, the flux is a *local* quantity, and it
falls along a pipe like :math:`e^{-h_e \tau}` --- by a factor 1.6 over a 2 h transit on a 100 mm
service line, and 3.8 over 6 h. Charging the whole pipe with one average flux history gives every
parcel in it the same soil memory: under 24 h diurnal forcing at hourly bins that is worth about
0.07 K over a 2 h transit and 0.5 K over 6 h on a 100 mm line, and under an overnight duty cycle it
delivers water a quarter of the water--soil contrast past the soil. It is not a bin-resolution
question that a finer ``tedges`` would answer: it is a *spatial* one, and the modes answer it.

**What the modes are worth, in numbers.** On the duty-cycled 100 mm line --- standing 8 h a night,
the shape the uniform flux gets most wrong --- the delivered excursion past the soil falls from
26 % of the contrast at one mode through 8 % at two to under 1 % at the six-mode default. Measured
against an Eulerian reference that keeps a full axial grid, the six-mode class truncation on that
duty cycle is about 2 % of the contrast, and steadily flowing pipes sit at the comparison's own
0.01 K discretisation floor from two modes on. Runtime grows roughly linearly with the count.

**What splitting still is.** Declaring the pipe as a chain of shorter segments remains exact in the
transport --- :math:`k` series pieces of volume :math:`V/k` at the same flow compose to the same
arrival map, so ``W``, the residence times and the coverage mask are unchanged to round-off --- but
it buys the soil memory nothing the modes do not already resolve: on the duty cycle above the
six-mode answer moves by under a percent of the contrast when the pipe is split in four.

**The one resolution limit left.** A pipe flushed *several volumes in a single bin* --- the
issue-#24 main pushing six volumes through per hourly bin --- cannot drive axial modes finer than
the bin width, and asking for them turns the sweep's feedback into amplification. The sweep refuses
that configuration by name; its delivered range is carried by the leading modes alone, so lowering
``n_modes`` for such a geometry, or refining ``tedges`` until the transit spans a bin, are both
exact answers rather than concessions.

**How it is checked.** Not checked --- there is no parameter to validate. That one flux history per
pipe is an approximation, and that refining it converges, are pinned by tests against an
independently written reference which keeps one soil memory per axial cell; on a shared time grid
the two collapse onto each other, so the residual between them is the comparison's discretisation
rather than unattributed physics (issue #32).


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

A step in the produced temperature into a *continuously flowing* pipe is the mild case: 2.4 % of the
instantaneous plant-to-soil contrast at the six-mode default --- 0.6 K on a 100 mm line after a 24 K
step --- where the classical one-history model paid 18 %.

Intermittent demand used to be far worse, and is the case the wall-flux attribution and the axial
modes carry together. Booking each parcel's heat over the bins it actually occupied, at the
positions it occupied, a 100 mm line idle 8 hours a day delivers water 26 % of the contrast past the
soil at one mode, 8 % at two, and under 1 % at the six-mode default
(:ref:`assumption-uniform-wall-flux`). Reading the flux off the delivered water instead put the same
case at 20--28 % and made refining worse rather than better, and on sharper duty cycles --- a main
flushed two hours in every 24, a line standing ten days --- it reached 8.8 times the contrast,
converged and unflagged. The standing line now comes back within half a kelvin of the range of its
inputs, and the flushed main --- six pipe volumes through in a single hourly bin --- carries its
range in the leading modes alone and is refused by name at mode counts its bin width cannot drive.

The one-way model (``max_sweeps=1``) has a fixed target and is still the only variant guaranteed
inside the range of its inputs.

**What you can do.** Treat any delivered temperature outside the hull of your inputs as carrying this
overshoot rather than as a physical prediction, and compare against ``max_sweeps=1`` when you need a
bound that cannot leave the hull. Be most suspicious on branches with a strong duty cycle, where the
effect is largest --- though raising the axial modes now settles it rather than aggravating it.

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
excursion is now 26 % of the contrast at one axial mode, 8 % at two and under 1 % at the six-mode
default, and raising the modes settles it.

**What you can do.** Nothing special. The one case still worth attention is the very first bins of a
record, where the model assumes the pipe starts in equilibrium with undisturbed soil rather than
carrying a history it cannot know.

**How it is checked.** By the only test in the suite with zero flow anywhere: it pins the excursion
under a duty cycle at each rung of the mode ladder and asserts that raising the modes brings the
answer back inside the range of its inputs.
