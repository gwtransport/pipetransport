.. _modules:

Which module do I need
======================

``pipetransport`` is small on purpose. There is one physical model --- the source-to-node transfer
operator derived in :ref:`concepts` --- and the public modules are different questions asked of it.
Pick the module by the question, not by the quantity.

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - Module
     - The question it answers
     - Start with
   * - :mod:`pipetransport.network`
     - *What is my network, and how much water does each pipe carry at each time step?*
     - :class:`~pipetransport.network.PipeNetwork`,
       :meth:`~pipetransport.network.PipeNetwork.segment_flow`,
       :meth:`~pipetransport.network.PipeNetwork.node_flow`
   * - :mod:`pipetransport.transport`
     - *What quality arrives at each delivery point? And, the other way round, what must have left
       the plant?*
     - :func:`~pipetransport.transport.source_to_endmember`,
       :func:`~pipetransport.transport.endmember_to_source`
   * - :mod:`pipetransport.residence_time`
     - *How old is the water delivered now --- or how long until the water produced now arrives?*
     - :func:`~pipetransport.residence_time.full`
   * - :mod:`pipetransport.logremoval`
     - *What decay rate does each pipe deserve, and how much disinfectant or pathogen credit is
       left?*
     - :func:`~pipetransport.logremoval.segment_decay_rate`,
       :func:`~pipetransport.logremoval.residence_time_to_log_removal`,
       :func:`~pipetransport.logremoval.parallel_mean`
   * - :mod:`pipetransport.heat`
     - *How warm is the water at the tap, given the weather and what the pipes are buried in?*
     - :func:`~pipetransport.heat.source_to_endmember`,
       :func:`~pipetransport.heat.segment_heat_rate`,
       :func:`~pipetransport.heat.soil_temperature`
   * - :mod:`pipetransport.examples`
     - *Give me a realistic network and demand pattern to try this on.*
     - :func:`~pipetransport.examples.example_network`,
       :func:`~pipetransport.examples.example_demand`
   * - :mod:`pipetransport.plot`
     - *Show me the topology, the flow split over the day, or a per-endmember series.*
     - :func:`~pipetransport.plot.network`,
       :func:`~pipetransport.plot.flow_allocation`,
       :func:`~pipetransport.plot.endmember_series`
   * - :mod:`pipetransport.utils`
     - *Build the time axis, the cumulative volume axis, step-plot coordinates, or solve a banded
       inverse problem.*
     - :func:`~pipetransport.utils.compute_time_edges`,
       :func:`~pipetransport.utils.cumulative_flow_volume`,
       :func:`~pipetransport.utils.step_plot_coords`,
       :func:`~pipetransport.utils.solve_inverse_transport_banded`

The modelling calls all take the same three inputs --- a
:class:`~pipetransport.network.PipeNetwork`, the demand at every endmember, and a ``tedges``
bin-edge axis --- and all follow the same conventions: values are constant over each bin, output is
a flow-weighted bin average (:ref:`concept-label-coordinate`), and ``NaN`` marks a bin the record
does not constrain (:ref:`concept-coverage`).

The route through the package
-----------------------------

**1. Describe the network.** :class:`~pipetransport.network.PipeNetwork` takes a table of segments
with a ``from`` and ``to`` node and either a ``volume`` or a ``length`` and ``diameter``, plus the
name of the source. It validates that the result is a tree rooted at the source
(:ref:`assumption-tree-topology`) and derives the endmembers and the source-to-node paths. Give it
the demand at the endmembers and it returns every internal flow by mass conservation --- no
hydraulic solve, no proportionality assumption.

**2. Ask the forward question.** :func:`~pipetransport.transport.source_to_endmember` maps a
produced-quality series onto any set of reporting nodes. Add ``decay_rate`` and the same call is a
chlorine residual model; add ``retardation_factor`` and it is a compound that exchanges reversibly
with the pipe wall. Reporting nodes are free: an endmember is the usual choice, but a junction
reports the quality passing through it.

**3. Or the reverse question.** :func:`~pipetransport.transport.endmember_to_source` reconstructs
the produced quality from measurements at one or more nodes, by inverting the very same operator as
one regularized banded least-squares problem. ``NaN`` in the measurements marks unsampled bins, so a
sparse grab-sample campaign needs no special handling, and several sampling points reinforce one
another because each constrains a different, demand-dependent window of the production history.

**4. Read the age.** :func:`~pipetransport.residence_time.full` returns the travel times underlying
the same operator, in either direction (:ref:`concept-water-age`). Use it as a diagnostic --- for
siting monitoring points, judging flushing, or explaining a residual profile --- not as a shortcut
to the residual itself, which :func:`~pipetransport.transport.source_to_endmember` computes exactly.

**5. Convert to the currency you report in.** :mod:`pipetransport.logremoval` holds the conversions
between a rate constant :math:`k` [1/day], a log10 rate :math:`\mu` [log10/day] and log removal, and
the two rules that matter in a network: the diameter-dependent per-segment rate from bulk and wall
reaction (:func:`~pipetransport.logremoval.segment_decay_rate`), and the blending rule for streams
that mix (:func:`~pipetransport.logremoval.parallel_mean`), in which concentrations mix and log
removals do not.

**6. Draw it.** :mod:`pipetransport.plot` takes the package's own outputs directly. Each function
draws into an existing Axes when given one, so the figures compose into multi-panel layouts.

The operator itself lives in the private module ``pipetransport._transfer``. It has no public API;
its module docstring is the implementation-level companion to :ref:`concepts`.

A complete run
--------------

Forward transport, water age and chlorine residual for the example network, in one pass:

.. code:: python

   import numpy as np
   import pandas as pd

   from pipetransport.examples import example_demand, example_network
   from pipetransport.logremoval import (
       decay_rate_to_log10_decay_rate,
       residence_time_to_log_removal,
       segment_decay_rate,
   )
   from pipetransport.residence_time import full
   from pipetransport.transport import source_to_endmember

   network = example_network()
   tedges = pd.date_range("2025-06-01", "2025-06-03", freq="h")
   demand = example_demand(tedges=tedges, network=network)
   print(network)

   # Chlorine dosed to 1 mg/L, decaying in the bulk and at the pipe wall.
   decay = segment_decay_rate(network=network, bulk_decay_rate=0.3, wall_decay_rate=0.02)
   residual = source_to_endmember(
       cin=np.ones(len(tedges) - 1), flow=demand, tedges=tedges,
       cout_tedges=tedges, network=network, decay_rate=decay,
   )
   age = full(flow=demand, tedges=tedges, network=network)  # days

   # Log removal at the bulk rate alone: a diagnostic reading of the age, not the residual, which
   # carries the wall term too and comes from the transport call above.
   mu = decay_rate_to_log10_decay_rate(0.3)
   credit = residence_time_to_log_removal(residence_times=age, log10_decay_rate=mu)
   for node, res, hours, log_removal in zip(network.endmembers, residual, age * 24.0, credit):
       print(
           f"{node}: age {np.nanmean(hours):5.2f} h, residual {np.nanmin(res):.3f} mg/L, "
           f"bulk credit {np.nanmean(log_removal):.3f} log"
       )

   # Every row of the operator sums to 1 without decay: a constant input is delivered unchanged.
   conservative = source_to_endmember(
       cin=np.ones(len(tedges) - 1), flow=demand, tedges=tedges,
       cout_tedges=tedges, network=network,
   )
   np.testing.assert_allclose(conservative, 1.0, rtol=0.0, atol=1e-12)

Where to read next
------------------

- :ref:`concepts` --- the arrival map, the label coordinate, effective volume, water age and decay.
- :ref:`assumptions` --- what the model assumes, what breaks, and what to do about it.
