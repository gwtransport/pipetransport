"""Tests for pipetransport.plot: the network schematic, the flow allocation stack and the series plot.

Figures are checked on what they encode, not on whether they run: the number and geometry of the
artists drawn, the colour slot every series lands in, the labels, the guarded error paths, and --
for the stacked area -- that the top of the stack really is the source production computed by
:meth:`PipeNetwork.node_flow`.
"""

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from pipetransport import plot
from pipetransport.examples import example_demand
from pipetransport.plot import _MAX_DIRECT_LABELS, _MUTED, _SURFACE, SEQUENTIAL_COLORS, SERIES_COLORS
from pipetransport.utils import step_plot_coords

# Headless: every figure below is inspected through its artists, never rendered to a screen.
mpl.use("Agg")

# The layered tree layout of the example network: x is the path depth, endmembers are spread one
# unit apart down the vertical axis and an internal node sits at the mean height of its children.
EXAMPLE_POSITIONS = {
    "Plant": (0.0, 0.0),
    "A": (1.0, 0.0),
    "B": (2.0, 1.0),
    "C": (2.0, -1.0),
    "T1": (3.0, 1.5),
    "T2": (3.0, 0.5),
    "T3": (3.0, -0.5),
    "T4": (3.0, -1.5),
}


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure after each test so the suite does not leak Agg canvases."""
    yield
    plt.close("all")


def _ramp(fraction):
    """Colour of the sequential ramp at ``fraction`` in [0, 1], as an RGBA tuple."""
    return LinearSegmentedColormap.from_list("pipetransport_sequential", list(SEQUENTIAL_COLORS))(fraction)


# ============================================================================
# series_colors: categorical identity is a fixed slot, never a cycle
# ============================================================================


def test_series_colors_returns_the_palette_in_slot_order():
    assert plot.series_colors(len(SERIES_COLORS)) == list(SERIES_COLORS)
    assert len(set(SERIES_COLORS)) == len(SERIES_COLORS)
    # Anchor the first slots to the documented hues so the palette cannot drift silently.
    assert plot.series_colors(2) == ["#2a78d6", "#eb6834"]
    assert len(SERIES_COLORS) == 8


@pytest.mark.parametrize("n", range(1, len(SERIES_COLORS) + 1))
def test_series_colors_slot_i_does_not_move_as_series_are_added(n):
    # Dropping or adding a series must not repaint the survivors, so slot i is the same hue for
    # every n that reaches it.
    assert plot.series_colors(n) == list(SERIES_COLORS[:n])
    full = plot.series_colors(len(SERIES_COLORS))
    assert all(plot.series_colors(n)[i] == full[i] for i in range(n))


def test_series_colors_falls_back_to_the_neutral_past_eight():
    colors = plot.series_colors(11)
    assert colors[: len(SERIES_COLORS)] == list(SERIES_COLORS)
    assert colors[len(SERIES_COLORS) :] == [_MUTED] * 3
    assert _MUTED not in SERIES_COLORS


# ============================================================================
# plot.network
# ============================================================================


def test_network_draws_one_line_per_segment_and_one_marker_per_node(network):
    ax = plot.network(network=network)

    lines = ax.get_lines()
    assert len(lines) == len(network.segments) + len(network.nodes) == 15
    for line, (name, segment) in zip(lines[: len(network.segments)], network.segments.iterrows(), strict=True):
        x0, y0 = EXAMPLE_POSITIONS[segment["from"]]
        x1, y1 = EXAMPLE_POSITIONS[segment["to"]]
        np.testing.assert_allclose(line.get_xdata(), [x0, x1], rtol=0.0, atol=0.0, err_msg=name)
        np.testing.assert_allclose(line.get_ydata(), [y0, y1], rtol=0.0, atol=1e-15, err_msg=name)
    for line, node in zip(lines[len(network.segments) :], network.nodes, strict=True):
        np.testing.assert_allclose(
            (line.get_xdata()[0], line.get_ydata()[0]), EXAMPLE_POSITIONS[node], rtol=0.0, atol=1e-15
        )

    # Every segment and every node is named, so the schematic is readable without a legend.
    assert [text.get_text() for text in ax.texts] == list(network.segments.index) + list(network.nodes)
    assert ax.get_title() == "Distribution network"
    assert ax.axison is False
    assert ax.figure.axes[1].get_ylabel() == "segment water volume [m³]"


def test_network_line_width_follows_pipe_diameter(network):
    ax = plot.network(network=network)

    diameter = network.segments["diameter"].to_numpy(float)
    widths = [line.get_linewidth() for line in ax.get_lines()[: len(network.segments)]]
    np.testing.assert_allclose(widths, 2.0 + 16.0 * diameter / diameter.max(), rtol=1e-13)
    # The 400 mm trunk is the widest stroke and the 100 mm branch to T4 the thinnest.
    assert widths.index(max(widths)) == 0
    assert widths.index(min(widths)) == len(network.segments) - 1


def test_network_colour_spans_the_value_range(network):
    ax = plot.network(network=network)

    volume = network.segments["volume"].to_numpy(float)
    colors = [line.get_color() for line in ax.get_lines()[: len(network.segments)]]
    np.testing.assert_allclose(colors[int(volume.argmax())], _ramp(1.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(colors[int(volume.argmin())], _ramp(0.0), rtol=0.0, atol=0.0)
    # Colour is a monotone read-out of the value: ordering the segments by volume orders the ramp
    # positions the same way.
    span = (volume - volume.min()) / (volume.max() - volume.min())
    np.testing.assert_allclose(colors, [_ramp(f) for f in span], rtol=0.0, atol=0.0)


def test_network_values_override_recolours_and_relabels(network, hourly_tedges):
    demand = example_demand(tedges=hourly_tedges, network=network)
    mean_flow = dict(zip(network.segments.index, network.segment_flow(flow=demand).mean(axis=1), strict=True))
    ax = plot.network(network=network, values=mean_flow, value_label="mean throughflow [m³/day]")

    colors = [line.get_color() for line in ax.get_lines()[: len(network.segments)]]
    values = np.array([mean_flow[name] for name in network.segments.index])
    span = (values - values.min()) / (values.max() - values.min())
    np.testing.assert_allclose(colors, [_ramp(f) for f in span], rtol=0.0, atol=0.0)
    # The trunk carries the whole production and C-T4 a tenth of it, so they take the ramp ends.
    assert max(mean_flow, key=mean_flow.get) == "Plant-A"
    assert min(mean_flow, key=mean_flow.get) == "C-T4"
    assert ax.figure.axes[1].get_ylabel() == "mean throughflow [m³/day]"


def test_network_raises_when_values_misses_a_segment(network):
    partial = dict.fromkeys(network.segments.index[:-1], 1.0)
    with pytest.raises(ValueError, match=r"values is missing segment\(s\): \['C-T4'\]"):
        plot.network(network=network, values=partial)


def test_network_endmember_markers_carry_the_series_hues(network):
    ax = plot.network(network=network)

    node_lines = dict(zip(network.nodes, ax.get_lines()[len(network.segments) :], strict=True))
    for name, color in zip(network.endmembers, plot.series_colors(len(network.endmembers)), strict=True):
        assert node_lines[name].get_color() == color
    for name in set(network.nodes) - set(network.endmembers):
        assert node_lines[name].get_color() == _SURFACE


def test_network_draws_into_the_given_ax_and_returns_it(network, two_branch):
    fig, (left, right) = plt.subplots(1, 2)
    returned = plot.network(network=network, ax=left)
    assert returned is left
    assert len(left.get_lines()) == len(network.segments) + len(network.nodes)
    assert len(right.get_lines()) == 0

    plot.network(network=two_branch, ax=right)
    assert len(right.get_lines()) == len(two_branch.segments) + len(two_branch.nodes)
    # Two panels plus one colorbar each; nothing landed on a stray new figure.
    assert len(fig.axes) == 4
    assert plt.get_fignums() == [fig.number]


def test_network_without_a_diameter_column_uses_a_constant_width(two_branch):
    ax = plot.network(network=two_branch)

    assert "diameter" not in two_branch.segments.columns
    widths = [line.get_linewidth() for line in ax.get_lines()[: len(two_branch.segments)]]
    np.testing.assert_allclose(widths, 5.0, rtol=0.0, atol=0.0)
    # Colour still carries the volume, which is the only quantity available.
    colors = [line.get_color() for line in ax.get_lines()[: len(two_branch.segments)]]
    np.testing.assert_allclose(colors[0], _ramp(1.0), rtol=0.0, atol=0.0)


# ============================================================================
# plot.flow_allocation
# ============================================================================


def test_flow_allocation_stack_top_is_the_source_production(network, short_tedges):
    demand = example_demand(tedges=short_tedges, network=network)
    ax = plot.flow_allocation(network=network, flow=demand, tedges=short_tedges)

    cumulative = np.cumsum(network.flow_array(demand), axis=0)
    seams = ax.get_lines()
    assert len(seams) == len(network.endmembers)
    for seam, band_top in zip(seams, cumulative, strict=True):
        expected_x, expected_y = step_plot_coords(short_tedges, band_top)
        np.testing.assert_array_equal(seam.get_xdata(), expected_x)
        np.testing.assert_allclose(seam.get_ydata(), expected_y, rtol=1e-13)
        assert seam.get_color() == _SURFACE

    # The top seam is the production past the source, independently computed by node_flow.
    production = network.node_flow(flow=demand, nodes=[network.source])[0]
    np.testing.assert_allclose(seams[-1].get_ydata(), step_plot_coords(short_tedges, production)[1], rtol=1e-13)
    np.testing.assert_allclose(ax.get_ylim(), (0.0, production.max() * 1.05), rtol=1e-13)


def test_flow_allocation_band_extents_match_the_demand(network, short_tedges):
    demand = example_demand(tedges=short_tedges, network=network)
    ax = plot.flow_allocation(network=network, flow=demand, tedges=short_tedges)

    cumulative = np.cumsum(network.flow_array(demand), axis=0)
    lower = np.vstack([np.zeros(cumulative.shape[1]), cumulative[:-1]])
    assert len(ax.collections) == len(network.endmembers)
    for i, collection in enumerate(ax.collections):
        # Every vertex of the filled polygon lies on one of the two step curves bounding the band,
        # and the lower one never exceeds the upper, so the extremes pin the band down.
        vertices = collection.get_paths()[0].vertices[:, 1]
        np.testing.assert_allclose(vertices.max(), cumulative[i].max(), rtol=1e-13)
        np.testing.assert_allclose(vertices.min(), lower[i].min(), rtol=1e-13)
        assert to_rgba(collection.get_facecolor()[0]) == to_rgba(SERIES_COLORS[i])


def test_flow_allocation_labels_axes_and_legend(network, short_tedges):
    demand = example_demand(tedges=short_tedges, network=network)
    ax = plot.flow_allocation(network=network, flow=demand, tedges=short_tedges)

    assert ax.get_ylabel() == "demand [m³/day]"
    assert ax.get_title() == "Endmember demand stacks up to the source production"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == list(network.endmembers)
    np.testing.assert_allclose(ax.get_xlim(), mdates.date2num([short_tedges[0], short_tedges[-1]]), rtol=1e-13)


def test_flow_allocation_draws_into_the_given_ax(network, short_tedges):
    demand = example_demand(tedges=short_tedges, network=network)
    fig, ax = plt.subplots()
    returned = plot.flow_allocation(network=network, flow=demand, tedges=short_tedges, ax=ax)

    assert returned is ax
    assert plt.get_fignums() == [fig.number]
    # Insertion order is not row order: the keys decide, so a shuffled mapping draws the same.
    shuffled = {name: demand[name] for name in reversed(list(demand))}
    reference = plot.flow_allocation(network=network, flow=shuffled, tedges=short_tedges)
    for drawn, expected in zip(ax.get_lines(), reference.get_lines(), strict=True):
        np.testing.assert_allclose(drawn.get_ydata(), expected.get_ydata(), rtol=0.0, atol=0.0)


# ============================================================================
# plot.endmember_series
# ============================================================================


def test_endmember_series_draws_one_step_curve_per_row(network, short_tedges):
    rows = [np.linspace(1.0, 4.0, len(short_tedges) - 1) * (i + 1) for i in range(4)]
    values = dict(zip(network.endmembers, rows, strict=True))
    ax = plot.endmember_series(values=values, tedges=short_tedges, ylabel="water age [h]", title="Age")

    lines = ax.get_lines()
    assert len(lines) == 4
    for line, row, color in zip(lines, rows, plot.series_colors(4), strict=True):
        expected_x, expected_y = step_plot_coords(short_tedges, row)
        np.testing.assert_array_equal(line.get_xdata(), expected_x)
        np.testing.assert_allclose(line.get_ydata(), expected_y, rtol=0.0, atol=0.0)
        assert line.get_color() == color
    assert ax.get_ylabel() == "water age [h]"
    assert ax.get_title() == "Age"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == list(network.endmembers)


def test_endmember_series_accepts_a_single_series(short_tedges):
    row = np.linspace(0.0, 1.0, len(short_tedges) - 1)
    ax = plot.endmember_series(values={"T1": row}, tedges=short_tedges)

    assert len(ax.get_lines()) == 1
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), np.repeat(row, 2), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("values", "wrong"),
    [
        pytest.param({"a": np.zeros(3), "b": np.zeros(4)}, r"\['a'\]", id="one series short"),
        pytest.param({"a": np.zeros(5), "b": np.zeros(5)}, r"\['a', 'b'\]", id="every series long"),
        pytest.param({"a": np.zeros((2, 4))}, r"\['a'\]", id="two-dimensional"),
    ],
)
def test_endmember_series_raises_on_a_length_mismatch(values, wrong):
    tedges = pd.date_range("2025-06-01", periods=5, freq="h")
    with pytest.raises(ValueError, match="every series in values must have length 4; wrong: " + wrong):
        plot.endmember_series(values=values, tedges=tedges)


def test_endmember_series_labels_the_peak_of_each_series_up_to_four():
    tedges = pd.date_range("2025-06-01", periods=5, freq="h")
    rows = np.array([[1.0, 5.0, 2.0, 0.0], [3.0, 1.0, 1.0, 9.0], [4.0, 2.0, 7.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
    labels = ["a", "b", "c", "d"]
    ax = plot.endmember_series(values=dict(zip(labels, rows, strict=True)), tedges=tedges)

    assert len(labels) == _MAX_DIRECT_LABELS
    assert [text.get_text() for text in ax.texts] == labels
    for text, row, color in zip(ax.texts, rows, plot.series_colors(4), strict=True):
        peak = int(np.argmax(row))
        # The label sits on the step curve, at the left edge of the peak bin where the step starts.
        assert text.xy[0] == tedges[peak]
        assert text.xy[1] == row[peak] == row.max()
        assert text.get_color() == color


def test_endmember_series_drops_the_direct_labels_beyond_four():
    tedges = pd.date_range("2025-06-01", periods=5, freq="h")
    values = dict(zip("abcde", np.arange(20.0).reshape(5, 4), strict=True))
    ax = plot.endmember_series(values=values, tedges=tedges)

    assert len(values) == _MAX_DIRECT_LABELS + 1
    assert len(ax.texts) == 0
    # The legend carries identity on its own once the direct labels would start overlapping.
    assert [text.get_text() for text in ax.get_legend().get_texts()] == list("abcde")
    assert len(ax.get_lines()) == 5


def test_endmember_series_skips_the_label_of_an_all_nan_row():
    # NaN marks output bins the record does not constrain; a series that is entirely unconstrained
    # has no peak to point at, and must not take the whole figure down with it.
    tedges = pd.date_range("2025-06-01", periods=5, freq="h")
    values = {"blind": np.full(4, np.nan), "seen": np.array([1.0, 3.0, np.nan, 2.0])}
    ax = plot.endmember_series(values=values, tedges=tedges)

    assert [text.get_text() for text in ax.texts] == ["seen"]
    assert ax.texts[0].xy == (tedges[1], 3.0)
    assert np.all(np.isnan(ax.get_lines()[0].get_ydata()))
    np.testing.assert_array_equal(ax.get_lines()[1].get_ydata(), np.repeat([1.0, 3.0, np.nan, 2.0], 2))


def test_endmember_series_draws_into_the_given_ax_and_returns_it(short_tedges):
    fig, (top, bottom) = plt.subplots(2, 1)
    values = {"a": np.zeros(len(short_tedges) - 1), "b": np.zeros(len(short_tedges) - 1)}
    returned = plot.endmember_series(values=values, tedges=short_tedges, ax=bottom)

    assert returned is bottom
    assert len(bottom.get_lines()) == 2
    assert len(top.get_lines()) == 0
    assert plt.get_fignums() == [fig.number]
