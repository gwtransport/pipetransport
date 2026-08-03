"""
Figures for distribution network water quality.

Three views cover what the package produces. :func:`network` draws the topology, so the reader
can see which endmember sits behind which pipe. :func:`flow_allocation` shows how the
production splits over the endmembers through the day -- the quantity that makes this package
necessary, because a fixed-fraction model cannot represent it. :func:`endmember_series` draws
any per-endmember timeseries the package returns: delivered quality, chlorine residual, or
water age.

Encoding rules
--------------

Series identity is categorical: each endmember keeps one hue from a fixed, colorblind-checked
order and never a cycled one, so a chart with fewer series does not repaint the survivors.
Beyond eight endmembers the extras are drawn in a muted neutral rather than in invented hues.
Magnitude along a pipe is sequential: one blue hue, light to dark, with a colorbar. Two of the
categorical hues sit below 3:1 against the chart surface, so every series carries a legend and
-- up to four series -- a direct label as well; identity is never colour alone.

Every function draws into an existing Axes when one is passed, so the figures compose into
multi-panel layouts, and returns the Axes it drew into.

Available functions:

- :func:`network` - Node-link schematic of the tree. Line width follows pipe diameter, line
  colour follows a per-segment quantity (segment water volume by default, or anything else
  indexed by segment name, such as mean throughflow), and the endmembers carry the same hues
  the timeseries figures use.

- :func:`flow_allocation` - Stacked area of endmember demand against time. The top of the
  stack is the source production; the changing shape of the stack is the time-varying flow
  split.

- :func:`endmember_series` - Step plot of one value per endmember per time bin, with a legend
  and, for four series or fewer, direct labels at each series' maximum.

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from pipetransport.utils import step_plot_coords

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from pipetransport.network import PipeNetwork

# Categorical hues in fixed slot order, colourblind-validated as a set against the light chart
# surface. Series nine and beyond fall back to the neutral rather than to an invented hue.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
# One-hue sequential ramp, light to dark, for magnitude along a pipe.
SEQUENTIAL_COLORS = ("#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b")
_INK, _INK_SOFT, _MUTED, _SURFACE = "#0b0b0b", "#52514e", "#a3a29a", "#fcfcfb"
# Above this many series direct labels overlap more than they clarify; the legend carries on.
_MAX_DIRECT_LABELS = 4


def series_colors(n: int) -> list[str]:
    """Return ``n`` categorical colours in fixed slot order.

    Parameters
    ----------
    n : int
        Number of series.

    Returns
    -------
    list of str
        Hex colours: the fixed palette slots for the first eight series, then a neutral grey
        for any beyond. Hues are never cycled, so slot ``i`` always means the same series.

    Examples
    --------
    >>> from pipetransport.plot import series_colors
    >>> series_colors(2)
    ['#2a78d6', '#eb6834']
    """
    return [SERIES_COLORS[i] if i < len(SERIES_COLORS) else _MUTED for i in range(n)]


def _tree_layout(network: PipeNetwork) -> dict[str, tuple[float, float]]:
    """Place every node: depth to the right, endmembers spread evenly down the vertical axis.

    Parameters
    ----------
    network : PipeNetwork
        Network to lay out.

    Returns
    -------
    dict
        Node name to ``(x, y)``. Internal nodes sit at the mean height of their children,
        which is computed by walking the breadth-first node order backwards.
    """
    children: dict[str, list[str]] = {}
    for upstream, downstream in zip(network.segments["from"], network.segments["to"], strict=True):
        children.setdefault(upstream, []).append(downstream)
    offset = (len(network.endmembers) - 1) / 2.0
    height = {name: float(i) - offset for i, name in enumerate(network.endmembers)}
    for node in reversed(network.nodes):
        if node in children:
            height[node] = float(np.mean([height[child] for child in children[node]]))
    return {node: (float(len(network.paths[node])), -height[node]) for node in network.nodes}


def network(
    *,
    network: PipeNetwork,
    values: pd.Series | None = None,
    value_label: str = "segment water volume [m³]",
    node_positions: dict[str, tuple[float, float]] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draw the network as a node-link schematic.

    Parameters
    ----------
    network : PipeNetwork
        Network to draw.
    values : pandas.Series or None, optional
        Per-segment quantity driving the line colour, indexed by segment name. Defaults to the
        segment water volume. Pass e.g.
        ``pd.Series(network.segment_flow(flow=demand).mean(axis=1), index=network.segments.index)``
        to colour by mean throughflow instead.
    value_label : str, optional
        Colorbar label describing ``values``.
    node_positions : dict or None, optional
        Node name to ``(x, y)``. Defaults to a layered tree layout with the source on the left.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into. A new figure is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn into.

    Raises
    ------
    ValueError
        If ``values`` misses a segment of the network.

    See Also
    --------
    flow_allocation : How the production splits over the endmembers.
    pipetransport.network.PipeNetwork.segment_flow : Produces a per-segment quantity to colour by.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from pipetransport.examples import example_network
    >>> from pipetransport.plot import network as plot_network
    >>> ax = plot_network(network=example_network())
    >>> ax.get_title()
    'Distribution network'
    """
    magnitude = network.segments["volume"] if values is None else values
    missing = [name for name in network.segments.index if name not in magnitude.index]
    if missing:
        msg = f"values is missing segment(s): {missing}"
        raise ValueError(msg)
    magnitude = magnitude.reindex(network.segments.index).astype(float)

    ax = plt.subplots(figsize=(10.5, 5.0))[1] if ax is None else ax
    positions = _tree_layout(network) if node_positions is None else node_positions
    ramp = LinearSegmentedColormap.from_list("pipetransport_sequential", list(SEQUENTIAL_COLORS))
    span = Normalize(vmin=float(magnitude.min()), vmax=float(magnitude.max()))
    diameter = network.segments["diameter"] if "diameter" in network.segments.columns else None

    for i, (name, segment) in enumerate(network.segments.iterrows()):
        x0, y0 = positions[segment["from"]]
        x1, y1 = positions[segment["to"]]
        # Width carries geometry, colour carries the quantity: two channels, no redundancy.
        width = 2.0 + 16.0 * float(diameter.iloc[i] / diameter.max()) if diameter is not None else 5.0
        ax.plot(
            [x0, x1], [y0, y1], color=ramp(span(magnitude.iloc[i])), linewidth=width, solid_capstyle="round", zorder=1
        )
        ax.annotate(
            str(name),
            xy=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=_INK_SOFT,
        )

    endmember_color = dict(zip(network.endmembers, series_colors(len(network.endmembers)), strict=True))
    for node, (x, y) in positions.items():
        is_endmember = node in endmember_color
        ax.plot(
            x,
            y,
            marker="o",
            markersize=11,
            color=endmember_color.get(node, _SURFACE),
            markeredgecolor=_INK,
            markeredgewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            node,
            xy=(x, y),
            xytext=(14, 0) if is_endmember else (0, -15),
            textcoords="offset points",
            ha="left" if is_endmember else "center",
            va="center" if is_endmember else "top",
            fontsize=9.5,
            fontweight="bold" if is_endmember else "normal",
            color=_INK,
        )

    ax.set_title("Distribution network", fontsize=12, color=_INK)
    ax.margins(x=0.12, y=0.22)
    ax.set_axis_off()
    ax.figure.colorbar(plt.cm.ScalarMappable(norm=span, cmap=ramp), ax=ax, label=value_label, fraction=0.03, pad=0.02)
    return ax


def flow_allocation(
    *,
    network: PipeNetwork,
    flow: npt.ArrayLike | pd.DataFrame | dict,
    tedges: pd.DatetimeIndex,
    ax: Axes | None = None,
) -> Axes:
    """Draw the endmember demand as a stacked area, whose top edge is the source production.

    The shape of the stack over the day *is* the time-varying flow split: if every endmember
    moved in proportion, each band would keep a constant share of the total.

    Parameters
    ----------
    network : PipeNetwork
        Network whose endmembers label the bands.
    flow : DataFrame, mapping, or array-like
        Demand at every endmember [m³/day] on the ``tedges`` bins.
    tedges : pandas.DatetimeIndex
        Time bin edges, ``n + 1`` edges for ``n`` bins.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into. A new figure is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn into.

    See Also
    --------
    network : The topology behind the split.
    endmember_series : Per-endmember quality, residual or age over the same axis.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import pandas as pd
    >>> from pipetransport.examples import example_network, example_demand
    >>> from pipetransport.plot import flow_allocation
    >>> net = example_network()
    >>> tedges = pd.date_range("2025-06-01", periods=49, freq="h")
    >>> ax = flow_allocation(
    ...     network=net, flow=example_demand(tedges=tedges, network=net), tedges=tedges
    ... )
    >>> ax.get_ylabel()
    'demand [m³/day]'
    """
    tedges = pd.DatetimeIndex(tedges)
    demand = network.flow_array(flow)
    ax = plt.subplots(figsize=(10.5, 4.0))[1] if ax is None else ax
    colors = series_colors(len(network.endmembers))

    lower = np.zeros(2 * demand.shape[1])
    for i, (name, color) in enumerate(zip(network.endmembers, colors, strict=True)):
        x, band = step_plot_coords(tedges, demand[i])
        upper = lower + band
        ax.fill_between(x, lower, upper, color=color, linewidth=0, label=name, zorder=2)
        # A 2 px surface-coloured seam keeps adjacent bands from reading as one shape.
        ax.plot(x, upper, color=_SURFACE, linewidth=2.0, zorder=3)
        lower = upper

    ax.set_ylabel("demand [m³/day]", color=_INK)
    ax.set_title("Endmember demand stacks up to the source production", fontsize=12, color=_INK)
    ax.set_xlim(tedges[0], tedges[-1])
    ax.set_ylim(0.0, float(lower.max()) * 1.05)
    ax.grid(visible=True, axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(colors), 6), frameon=False, fontsize=9)
    return ax


def endmember_series(
    *,
    values: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    labels: list[str] | tuple[str, ...],
    ylabel: str = "",
    title: str = "",
    ax: Axes | None = None,
) -> Axes:
    """Draw one bin-constant timeseries per endmember as a step plot.

    Takes any ``(n_series, n_bins)`` result of the package: delivered quality from
    :func:`pipetransport.transport.source_to_endmember`, chlorine residual from the same call
    with a decay rate, or water age from :func:`pipetransport.residence_time.full`.

    Parameters
    ----------
    values : array-like
        Bin-constant values of shape ``(n_series, len(tedges) - 1)``. NaN leaves a gap.
    tedges : pandas.DatetimeIndex
        Time bin edges matching the last axis of ``values``.
    labels : list of str
        Series names, one per row, in the same order.
    ylabel : str, optional
        Y-axis label, including units.
    title : str, optional
        Axes title.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into. A new figure is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn into.

    Raises
    ------
    ValueError
        If ``values`` has the wrong shape for ``tedges`` or ``labels``.

    See Also
    --------
    flow_allocation : The demand pattern driving these series.
    pipetransport.transport.source_to_endmember : Produces delivered quality and residual.
    pipetransport.residence_time.full : Produces water age.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import pandas as pd
    >>> from pipetransport.examples import example_network, example_demand
    >>> from pipetransport.residence_time import full
    >>> from pipetransport.plot import endmember_series
    >>> net = example_network()
    >>> tedges = pd.date_range("2025-06-01", periods=97, freq="h")
    >>> age = full(
    ...     flow=example_demand(tedges=tedges, network=net), tedges=tedges, network=net
    ... )
    >>> ax = endmember_series(
    ...     values=age * 24.0,
    ...     tedges=tedges,
    ...     labels=list(net.endmembers),
    ...     ylabel="water age [h]",
    ... )
    >>> len(ax.get_lines())
    4
    """
    tedges = pd.DatetimeIndex(tedges)
    values = np.atleast_2d(np.asarray(values, dtype=float))
    if values.shape != (len(labels), len(tedges) - 1):
        msg = f"values must have shape ({len(labels)}, {len(tedges) - 1}), got {values.shape}"
        raise ValueError(msg)

    ax = plt.subplots(figsize=(10.5, 4.0))[1] if ax is None else ax
    colors = series_colors(len(labels))
    for row, name, color in zip(values, labels, colors, strict=True):
        ax.plot(*step_plot_coords(tedges, row), color=color, linewidth=2.0, label=name, zorder=2)
        # Two palette hues sit below 3:1 on the chart surface, so identity never rests on
        # colour alone: up to four series also carry a direct label at their peak.
        if len(labels) <= _MAX_DIRECT_LABELS and np.any(np.isfinite(row)):
            peak = int(np.nanargmax(row))
            ax.annotate(
                name,
                xy=(tedges[peak], row[peak]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
                fontweight="bold",
                color=color,
                zorder=4,
            )

    ax.set_ylabel(ylabel, color=_INK)
    if title:
        ax.set_title(title, fontsize=12, color=_INK)
    ax.set_xlim(tedges[0], tedges[-1])
    ax.margins(y=0.15)
    ax.grid(visible=True, axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(labels), 6), frameon=False, fontsize=9)
    return ax
