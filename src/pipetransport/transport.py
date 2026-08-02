"""
Water quality transport from the production point to the endmembers, and back.

Water leaves the source with a measured quality, travels along a unique path of pipes, and is
delivered at an endmember. Both public functions are two readings of the same linear operator
``W``, whose row ``j`` holds the fractions of each source bin that make up output bin ``j``
after first-order decay: :func:`source_to_endmember` applies it, :func:`endmember_to_source`
inverts it. The operator is exact for demands that vary arbitrarily and independently over
time -- the flow split at every junction is recomputed per time step, never assumed constant.
The construction lives in the private ``pipetransport._transfer`` module.

Both directions take the endmember demand as ``flow`` and derive every internal segment flow
from mass conservation, so the only hydraulic input is what a distribution utility actually
meters. Output is reported on ``cout_tedges``, which may differ in alignment and resolution
from the input grid, as the flow-weighted average over each output bin.

Available functions:

- :func:`source_to_endmember` - Forward transport: given the produced water quality, the
  network geometry, and the demand at every endmember, return the delivered quality at each
  reporting node on ``cout_tedges``. ``decay_rate`` applies first-order decay per segment,
  which turns the same call into a chlorine-residual model. Output bins that the record does
  not fully constrain (spin-up, a bin extending past the flow record, or a bin during which
  the node draws no water) are NaN.

- :func:`endmember_to_source` - Reverse direction: reconstruct the produced water quality on
  ``tedges`` from quality measured at one or more endmembers. The per-node operators are
  stacked into one banded least-squares problem and deconvolved with Tikhonov regularization,
  so measurements at several endmembers -- each constraining a different, flow-dependent
  window of the production history -- reinforce one another. NaN entries in ``cout`` mark
  measurement gaps and drop out of the solve; source bins that nothing constrains come back
  NaN.

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from pipetransport._transfer import apply_banded, network_transfer
from pipetransport._validation import _validate_no_nan, _validate_tedges
from pipetransport.network import PipeNetwork  # noqa: TC001 -- runtime dependency of the signatures
from pipetransport.utils import solve_inverse_transport_banded


def source_to_endmember(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike | pd.DataFrame | dict,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    network: PipeNetwork,
    nodes: list[str] | tuple[str, ...] | None = None,
    decay_rate: float | pd.Series = 0.0,
    retardation_factor: float = 1.0,
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """Compute the delivered water quality at each reporting node from the produced quality.

    Parameters
    ----------
    cin : array-like
        Quality of the produced water leaving the source (concentration, temperature, or any
        conservative-plus-first-order-decay quantity), constant over each interval
        ``[tedges[i], tedges[i+1])``. Length ``len(tedges) - 1``.
    flow : DataFrame, mapping, or array-like
        Demand at every endmember [m³/day] on the same bins. A DataFrame or mapping is keyed
        by endmember name; an array must have shape ``(n_endmembers, len(cin))`` ordered as
        ``network.endmembers``.
    tedges : pandas.DatetimeIndex
        Time edges of the ``cin`` and ``flow`` bins. Length ``len(cin) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges of the output bins. Length ``n_output + 1``; alignment and resolution are
        free.
    network : PipeNetwork
        The distribution network.
    nodes : list of str or None, optional
        Nodes to report at, in output row order. Any node of the network is allowed -- a
        junction reports the quality passing through it. Defaults to ``network.endmembers``.
    decay_rate : float or pandas.Series, optional
        First-order decay rate [1/day]: a scalar shared by every segment, or a Series indexed
        by segment name for a per-pipe rate (see
        :func:`pipetransport.logremoval.segment_decay_rate`). Default 0.0, a conservative
        tracer.
    retardation_factor : float, optional
        Multiplier on every segment volume, ``>= 1``. Values above 1 model a compound that
        exchanges reversibly with the pipe wall and therefore travels slower than the water.
        Default 1.0.
    spinup : {"constant"} or None, optional
        ``"constant"`` (default) warm-starts the record by extending it backwards at the
        first observed demand and quality, so the earliest output bins carry a value instead
        of NaN. ``None`` keeps strict validity: any output bin fed even partly from before
        the record is NaN.

    Returns
    -------
    numpy.ndarray
        Delivered quality of shape ``(len(nodes), len(cout_tedges) - 1)``, in the units of
        ``cin``. Each value is the flow-weighted average over its output bin. NaN marks bins
        the record does not constrain: spin-up under ``spinup=None``, a bin extending past
        the end of the flow record, or a bin during which the node draws no water.

    Raises
    ------
    ValueError
        If a time axis is not strictly increasing or has the wrong length, if ``cin`` or
        ``flow`` holds NaN, if ``flow`` is negative, if ``retardation_factor < 1``, if a
        decay rate is negative, or if a requested node is not part of the network.

    See Also
    --------
    endmember_to_source : Reverse direction (deconvolution).
    pipetransport.residence_time.full : Water age behind these same travel times.
    pipetransport.logremoval.segment_decay_rate : Per-pipe chlorine decay from bulk and wall reaction.
    :ref:`concept-label-coordinate` : Why the output average is exactly flow-weighted.

    Notes
    -----
    ``decay_rate`` and ``retardation_factor`` combine as
    ``exp(-decay_rate * retardation_factor * t_water)``, with ``t_water`` the residence time of
    the water: the rate applies over the whole retarded transit, so the adsorbed and dissolved
    phases decay alike. This matches :mod:`gwtransport`, which feeds a retarded residence time
    to its log-removal, and is the convention of radioactive decay (Bear and Cheng, 2010,
    eq. 7.4.7). A compound that degrades only while dissolved decays as
    ``exp(-decay_rate * t_water)`` instead -- the retardation cancels, since the compound is
    adsorbed ``1 - 1 / retardation_factor`` of the time but travels ``retardation_factor``
    times longer -- which is what passing ``decay_rate / retardation_factor`` reproduces.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pipetransport.examples import example_network, example_demand
    >>> from pipetransport.transport import source_to_endmember
    >>>
    >>> network = example_network()
    >>> tedges = pd.date_range("2025-06-01", "2025-06-08", freq="h")
    >>> demand = example_demand(tedges=tedges, network=network)
    >>>
    >>> cin = np.zeros(len(tedges) - 1)
    >>> cin[24:27] = 1.0  # a three-hour pulse leaving the plant
    >>> cout = source_to_endmember(
    ...     cin=cin, flow=demand, tedges=tedges, cout_tedges=tedges, network=network
    ... )
    >>> cout.shape
    (4, 168)
    >>> bool(np.nanmax(cout[0]) > 0.9)  # the pulse arrives at T1 nearly undiluted
    True

    Adding first-order decay turns the same call into a chlorine-residual model:

    >>> residual = source_to_endmember(
    ...     cin=np.ones(len(tedges) - 1),
    ...     flow=demand,
    ...     tedges=tedges,
    ...     cout_tedges=tedges,
    ...     network=network,
    ...     decay_rate=0.5,
    ... )
    >>> bool(np.all(residual[:, -1] < 1.0))
    True
    """
    tedges = pd.DatetimeIndex(tedges)
    cin = np.asarray(cin, dtype=float)
    _validate_tedges(tedges, cin, tedges_name="tedges", values_name="cin")
    _validate_no_nan(cin, name="cin")

    _, transfer, n_pad = network_transfer(
        network=network,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        nodes=nodes,
        decay_rate=decay_rate,
        retardation_factor=retardation_factor,
        spinup=spinup,
    )
    # The warm start extends the record backwards at the first observed quality.
    cin = np.concatenate([np.full(n_pad, cin[0]), cin])

    out = apply_banded(transfer, cin)
    out[~transfer.valid_out] = np.nan
    return out


def endmember_to_source(
    *,
    cout: npt.ArrayLike | pd.DataFrame | dict,
    flow: npt.ArrayLike | pd.DataFrame | dict,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    network: PipeNetwork,
    nodes: list[str] | tuple[str, ...] | None = None,
    decay_rate: float | pd.Series = 0.0,
    retardation_factor: float = 1.0,
    regularization_strength: float = 1e-10,
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """Reconstruct the produced water quality from quality measured at the endmembers.

    Inverts the forward model by stacking the per-node operators of
    :func:`source_to_endmember` into one banded least-squares problem and solving it with
    Tikhonov regularization toward the transpose-and-normalize reference. Measurements at
    several endmembers reinforce one another: each constrains a different window of the
    production history, and the windows move as the demand pattern shifts.

    Parameters
    ----------
    cout : DataFrame, mapping, or array-like
        Measured quality at the reporting nodes, constant over each ``cout_tedges`` bin. A
        DataFrame or mapping is keyed by node name; an array must have shape
        ``(len(nodes), len(cout_tedges) - 1)``. NaN marks a measurement gap and drops that
        bin out of the solve, so a sparse sampling campaign is expressed by leaving the
        unsampled bins NaN.
    flow : DataFrame, mapping, or array-like
        Demand at every endmember [m³/day] on the ``tedges`` bins; see
        :func:`source_to_endmember`.
    tedges : pandas.DatetimeIndex
        Time edges of the ``flow`` bins and of the reconstructed output. Length
        ``len(flow) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges of the ``cout`` bins.
    network : PipeNetwork
        The distribution network.
    nodes : list of str or None, optional
        Nodes the rows of ``cout`` refer to. Defaults to ``network.endmembers``. Pass the
        measured subset when only some endmembers are sampled.
    decay_rate : float or pandas.Series, optional
        First-order decay rate [1/day]; see :func:`source_to_endmember`. Default 0.0.
    retardation_factor : float, optional
        Multiplier on every segment volume, ``>= 1``. Default 1.0.
    regularization_strength : float, optional
        Tikhonov parameter λ, strictly positive. Larger values trust the smooth reference
        more; smaller values trust the data more. A good starting value for noisy data is
        ``(noise_std / signal_amplitude)**2``; the default 1e-10 preserves machine precision
        on noiseless input. Default 1e-10.
    spinup : {"constant"} or None, optional
        Warm-start policy for building the forward operator; see
        :func:`source_to_endmember`. The warm-start prefix is solved for internally and
        dropped before returning, so the output stays aligned with ``tedges``.

    Returns
    -------
    numpy.ndarray
        Reconstructed quality of the produced water on ``tedges``, length
        ``len(tedges) - 1``, in the units of ``cout``. NaN for source bins that no
        measurement constrains.

    Raises
    ------
    ValueError
        If a time axis is not strictly increasing or has the wrong length, if ``cout`` has
        the wrong shape or misses a requested node, if ``flow`` holds NaN or negative values,
        if ``retardation_factor < 1``, if a decay rate is negative, if
        ``regularization_strength`` is not positive, or if a requested node is not part of
        the network.

    See Also
    --------
    source_to_endmember : Forward direction.
    pipetransport.utils.solve_inverse_transport_banded : The banded Tikhonov solver used here.

    Examples
    --------
    Round-trip: transport a source signal to the endmembers, then recover it from two of them.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pipetransport.examples import example_network, example_demand
    >>> from pipetransport.transport import source_to_endmember, endmember_to_source
    >>>
    >>> network = example_network()
    >>> tedges = pd.date_range("2025-06-01", "2025-06-15", freq="h")
    >>> demand = example_demand(tedges=tedges, network=network)
    >>> hours = np.arange(len(tedges) - 1)
    >>> cin = 2.0 + np.sin(2 * np.pi * hours / 48.0)
    >>>
    >>> measured = source_to_endmember(
    ...     cin=cin,
    ...     flow=demand,
    ...     tedges=tedges,
    ...     cout_tedges=tedges,
    ...     network=network,
    ...     nodes=["T1", "T4"],
    ... )
    >>> recovered = endmember_to_source(
    ...     cout=measured,
    ...     flow=demand,
    ...     tedges=tedges,
    ...     cout_tedges=tedges,
    ...     network=network,
    ...     nodes=["T1", "T4"],
    ... )
    >>> inner = slice(48, -48)
    >>> bool(np.nanmax(np.abs(recovered[inner] - cin[inner])) < 1e-6)
    True
    """
    report_nodes, transfer, n_pad = network_transfer(
        network=network,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        nodes=nodes,
        decay_rate=decay_rate,
        retardation_factor=retardation_factor,
        spinup=spinup,
    )
    cout_tedges = pd.DatetimeIndex(cout_tedges)

    named: dict | None = None
    if isinstance(cout, pd.DataFrame):
        named = {str(column): cout[column].to_numpy(dtype=float) for column in cout.columns}
    elif isinstance(cout, dict):
        named = {str(key): value for key, value in cout.items()}
    if named is not None:
        missing = [node for node in report_nodes if node not in named]
        if missing:
            msg = f"cout is missing node(s): {missing}"
            raise ValueError(msg)
        observed = np.stack([np.asarray(named[node], dtype=float) for node in report_nodes])
    else:
        observed = np.atleast_2d(np.asarray(cout, dtype=float))
    if observed.shape[0] != len(report_nodes):
        msg = f"cout must hold one row per reporting node ({len(report_nodes)}), got shape {observed.shape}"
        raise ValueError(msg)
    _validate_tedges(cout_tedges, observed, tedges_name="cout_tedges", values_name="cout")

    # Stack the per-node operators into one banded system. The bands already share one
    # width, so stacking is a plain reshape; a node whose output bin the record does not
    # constrain contributes no equation.
    band_vals = transfer.band_vals.reshape(-1, transfer.band_vals.shape[-1])
    col_start = transfer.col_start.ravel()
    rhs = np.where(transfer.valid_out.ravel(), observed.ravel(), np.nan)

    n_source = len(pd.DatetimeIndex(tedges)) - 1
    recovered = solve_inverse_transport_banded(
        band_vals=band_vals,
        col_start=col_start,
        observed=rhs,
        n_output=n_source + n_pad,
        regularization_strength=regularization_strength,
    )
    # Drop the warm-start prefix so the output aligns with the user-provided tedges.
    return recovered[n_pad:]
