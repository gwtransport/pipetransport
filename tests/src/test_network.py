"""
Tests for :mod:`pipetransport.network`.

Three groups: the geometry and topology derived at construction time (volume, node order,
endmembers, paths), the validation gate that rejects everything that is not a tree of positive
pipes, and the flow accounting, which is pure mass conservation and is therefore checked
against hand-written sums of the endmember demand rather than against the package's own
matrices.
"""

import re

import numpy as np
import pandas as pd
import pytest

from pipetransport.network import PipeNetwork

# ============================================================================
# Geometry and topology
# ============================================================================


def test_volume_from_length_and_diameter(network):
    # V = pi/4 * D^2 * L, evaluated in the same order as the package, so equality is bitwise.
    length = network.segments["length"].to_numpy(dtype=float)
    diameter = network.segments["diameter"].to_numpy(dtype=float)
    expected = np.pi / 4.0 * diameter**2 * length
    np.testing.assert_allclose(network.segments["volume"].to_numpy(dtype=float), expected, rtol=0.0, atol=0.0)

    # Spot-check two segments against literals derived by hand from the example geometry.
    np.testing.assert_allclose(network.segments.loc["Plant-A", "volume"], np.pi / 4.0 * 0.40**2 * 2000.0, rtol=1e-15)
    np.testing.assert_allclose(network.segments.loc["C-T4", "volume"], np.pi / 4.0 * 0.10**2 * 2500.0, rtol=1e-15)


def test_explicit_volume_column_wins_over_geometry():
    # A 'volume' column is taken as-is: the (inconsistent) length/diameter are not used.
    segments = pd.DataFrame(
        {
            "from": ["Plant", "A"],
            "to": ["A", "T1"],
            "length": [2000.0, 800.0],
            "diameter": [0.40, 0.15],
            "volume": [7.0, 11.0],
        },
        index=["Plant-A", "A-T1"],
    )
    net = PipeNetwork(segments=segments, source="Plant")
    np.testing.assert_allclose(net.segments["volume"].to_numpy(dtype=float), [7.0, 11.0], rtol=0.0, atol=0.0)
    # ... and the geometry columns survive untouched next to it.
    np.testing.assert_allclose(net.segments["length"].to_numpy(dtype=float), [2000.0, 800.0], rtol=0.0, atol=0.0)


def test_extra_columns_survive_and_input_is_not_mutated():
    segments = pd.DataFrame(
        {
            "from": ["Plant", "A"],
            "to": ["A", "T1"],
            "length": [2000.0, 800.0],
            "diameter": [0.40, 0.15],
            "material": ["PVC", "cast iron"],
            "decay_rate": [0.1, 0.4],
        },
        index=["Plant-A", "A-T1"],
    )
    net = PipeNetwork(segments=segments, source="Plant")

    assert list(net.segments["material"]) == ["PVC", "cast iron"]
    np.testing.assert_allclose(net.segments["decay_rate"].to_numpy(dtype=float), [0.1, 0.4], rtol=0.0, atol=0.0)
    # Row order and index are preserved, so segment_flow rows line up with the input table.
    assert list(net.segments.index) == ["Plant-A", "A-T1"]
    # The derived volume is appended; the caller's frame is a separate object and keeps none of it.
    assert "volume" in net.segments.columns
    assert "volume" not in segments.columns


def test_nodes_are_breadth_first_with_source_first(network, two_branch, single_pipe):
    assert network.nodes == ("Plant", "A", "B", "C", "T1", "T2", "T3", "T4")
    assert two_branch.nodes == ("Plant", "A", "T1", "T2")
    assert single_pipe.nodes == ("Plant", "T1")

    for net in (network, two_branch, single_pipe):
        assert net.nodes[0] == net.source
        # Every node of the graph appears exactly once.
        assert set(net.nodes) == set(net.segments["from"]) | set(net.segments["to"])
        assert len(set(net.nodes)) == len(net.nodes)
        # Breadth-first: depth never decreases along the order, so a parent always precedes its child.
        depth = [len(net.paths[node]) for node in net.nodes]
        assert depth == sorted(depth)
        position = {node: i for i, node in enumerate(net.nodes)}
        for node in net.nodes[1:]:
            parent = net.segments.loc[net.paths[node][-1], "from"]
            assert position[parent] < position[node]


def test_endmembers_are_exactly_the_leaves(network, two_branch, single_pipe):
    assert network.endmembers == ("T1", "T2", "T3", "T4")
    assert two_branch.endmembers == ("T1", "T2")
    assert single_pipe.endmembers == ("T1",)

    for net in (network, two_branch, single_pipe):
        upstream_nodes = set(net.segments["from"])
        assert set(net.endmembers) == set(net.nodes) - upstream_nodes


def test_paths_run_from_the_source_outward(network, single_pipe):
    assert network.paths == {
        "Plant": (),
        "A": ("Plant-A",),
        "B": ("Plant-A", "A-B"),
        "C": ("Plant-A", "A-C"),
        "T1": ("Plant-A", "A-B", "B-T1"),
        "T2": ("Plant-A", "A-B", "B-T2"),
        "T3": ("Plant-A", "A-C", "C-T3"),
        "T4": ("Plant-A", "A-C", "C-T4"),
    }
    assert network.paths[network.source] == ()
    assert single_pipe.paths == {"Plant": (), "T1": ("Plant-T1",)}

    # Each path is a connected chain that starts at the source and ends at the node itself.
    for node, path in network.paths.items():
        walk = network.source
        for name in path:
            assert network.segments.loc[name, "from"] == walk
            walk = network.segments.loc[name, "to"]
        assert walk == node


def test_repr_names_source_and_counts_segments_endmembers_volume(network, single_pipe):
    assert repr(network) == "PipeNetwork(source='Plant', segments=7, endmembers=4, volume=473.2 m3)"
    assert repr(single_pipe) == "PipeNetwork(source='Plant', segments=1, endmembers=1, volume=100.0 m3)"


# ============================================================================
# Validation
# ============================================================================


def _table(index, frm, to, **columns):
    """Build a segment table from parallel lists.

    Parameters
    ----------
    index : list of str
        Segment names.
    frm, to : list of str
        Upstream and downstream node of each segment.
    **columns : list
        Further columns, e.g. ``volume`` or ``length``/``diameter``.

    Returns
    -------
    pandas.DataFrame
        Segment table ready for :class:`PipeNetwork`.
    """
    return pd.DataFrame({"from": frm, "to": to, **columns}, index=index)


@pytest.mark.parametrize(
    ("segments", "source", "message"),
    [
        pytest.param(
            pd.DataFrame({"from": [], "to": [], "volume": []}),
            "Plant",
            "segments must hold at least one pipe segment",
            id="empty",
        ),
        pytest.param(
            pd.DataFrame({"from": ["Plant"], "volume": [10.0]}, index=["a"]),
            "Plant",
            re.escape("segments is missing required column(s): ['to']"),
            id="missing-to",
        ),
        pytest.param(
            pd.DataFrame({"to": ["A"], "volume": [10.0]}, index=["a"]),
            "Plant",
            re.escape("segments is missing required column(s): ['from']"),
            id="missing-from",
        ),
        pytest.param(
            _table(["a", "a"], ["Plant", "Plant"], ["A", "B"], volume=[10.0, 20.0]),
            "Plant",
            re.escape("segments index must be unique; duplicated segment name(s): ['a']"),
            id="duplicate-name",
        ),
        pytest.param(
            _table(["a"], ["Plant"], ["A"], volume=[0.0]),
            "Plant",
            "segment volume must be positive",
            id="zero-volume",
        ),
        pytest.param(
            _table(["a"], ["Plant"], ["A"], volume=[-1.0]),
            "Plant",
            "segment volume must be positive",
            id="negative-volume",
        ),
        pytest.param(
            _table(["a"], ["Plant"], ["A"], volume=[np.nan]),
            "Plant",
            "segment volume must be positive",
            id="nan-volume",
        ),
        pytest.param(
            _table(["a"], ["Plant"], ["A"], length=[0.0], diameter=[0.3]),
            "Plant",
            "segment length must be positive",
            id="zero-length",
        ),
        pytest.param(
            _table(["a"], ["Plant"], ["A"], length=[np.nan], diameter=[0.3]),
            "Plant",
            "segment length must be positive",
            id="nan-length",
        ),
        pytest.param(
            _table(["a"], ["Plant"], ["A"], length=[100.0], diameter=[-0.3]),
            "Plant",
            "segment diameter must be positive",
            id="negative-diameter",
        ),
        pytest.param(
            _table(["a"], ["Plant"], ["A"], length=[100.0], diameter=[np.nan]),
            "Plant",
            "segment diameter must be positive",
            id="nan-diameter",
        ),
        pytest.param(
            _table(["a"], ["Plant"], ["A"], length=[100.0]),
            "Plant",
            "segments must hold either a 'volume' column or both 'length' and 'diameter' columns",
            id="no-geometry",
        ),
        pytest.param(
            _table(["a", "b"], ["Plant", "A"], ["A", "A"], volume=[10.0, 20.0]),
            "Plant",
            re.escape("a segment cannot start and end at the same node; offending segment(s): ['b']"),
            id="self-loop",
        ),
        pytest.param(
            _table(["Plant-A", "B-A"], ["Plant", "B"], ["A", "A"], volume=[10.0, 20.0]),
            "Plant",
            re.escape("node 'A' is fed by more than one segment ('Plant-A' and 'B-A'); merging flows are not"),
            id="merge",
        ),
        pytest.param(
            _table(["s", "p"], ["Src", "Plant"], ["Plant", "A"], volume=[10.0, 20.0]),
            "Plant",
            re.escape("source 'Plant' is fed by segment 's'; the source must be the tree root"),
            id="source-has-feed",
        ),
        pytest.param(
            _table(["a"], ["Plant"], ["A"], volume=[10.0]),
            "Nope",
            re.escape("source 'Nope' is not the upstream node of any segment"),
            id="source-not-a-node",
        ),
        pytest.param(
            _table(["a"], ["Plant"], ["A"], volume=[10.0]),
            "A",
            re.escape("source 'A' is fed by segment 'a'; the source must be the tree root"),
            id="source-is-a-leaf",
        ),
        pytest.param(
            _table(["a", "b"], ["Plant", "X"], ["A", "Y"], volume=[10.0, 20.0]),
            "Plant",
            re.escape("node(s) not reachable from source 'Plant': ['Y'] (disconnected branch or cycle)"),
            id="disconnected-branch",
        ),
        pytest.param(
            _table(["a", "b", "c"], ["Plant", "X", "Y"], ["A", "Y", "X"], volume=[10.0, 20.0, 30.0]),
            "Plant",
            re.escape("node(s) not reachable from source 'Plant': ['X', 'Y'] (disconnected branch or cycle)"),
            id="cycle",
        ),
    ],
)
def test_invalid_network_raises(segments, source, message):
    with pytest.raises(ValueError, match=message):
        PipeNetwork(segments=segments, source=source)


# ============================================================================
# Flow accounting
# ============================================================================


def test_segment_flow_equals_summed_demand_below_each_segment(network, short_tedges, diurnal_demand):
    # Deliberately non-proportional demand, so a fixed-fraction shortcut could not reproduce this.
    demand = diurnal_demand(network, short_tedges)
    t1, t2, t3, t4 = (demand[name] for name in ("T1", "T2", "T3", "T4"))
    expected = np.stack([
        t1 + t2 + t3 + t4,  # Plant-A carries the whole production
        t1 + t2,  # A-B feeds the T1/T2 district
        t3 + t4,  # A-C feeds the T3/T4 district
        t1,
        t2,
        t3,
        t4,
    ])
    # Sums of at most four terms in a different association order than np.stack above: the
    # disagreement is a single rounding of the running sum, hence the 1e-15 bound.
    np.testing.assert_allclose(network.segment_flow(flow=demand), expected, rtol=1e-15)


def test_segment_flow_two_branch(two_branch, short_tedges, constant_demand):
    demand = constant_demand(two_branch, short_tedges, means=[130.0, 45.0])
    n_bins = len(short_tedges) - 1
    expected = np.stack([
        np.full(n_bins, 175.0),  # Plant-A
        np.full(n_bins, 130.0),  # A-T1
        np.full(n_bins, 45.0),  # A-T2
    ])
    np.testing.assert_allclose(two_branch.segment_flow(flow=demand), expected, rtol=0.0, atol=0.0)


def test_node_flow_at_source_endmember_and_junction(network, short_tedges, diurnal_demand):
    demand = diurnal_demand(network, short_tedges)
    node_flow = network.node_flow(flow=demand)
    segment_flow = network.segment_flow(flow=demand)
    row_of_node = {node: i for i, node in enumerate(network.nodes)}
    row_of_segment = {name: i for i, name in enumerate(network.segments.index)}

    # Source: total production.
    np.testing.assert_allclose(node_flow[row_of_node["Plant"]], sum(demand.values()), rtol=1e-15)
    # Endmember: its own demand, exactly.
    for name in network.endmembers:
        np.testing.assert_allclose(node_flow[row_of_node[name]], demand[name], rtol=0.0, atol=0.0)
    # Junction: the flow of the segment feeding it. Precision floor 1e-15: node_flow and
    # segment_flow are matmuls of differently shaped selector matrices, and BLAS picks a
    # different summation order per shape, which costs ~1 ULP on the 2- and 4-term sums.
    for node in ("A", "B", "C"):
        feeding = network.paths[node][-1]
        np.testing.assert_allclose(node_flow[row_of_node[node]], segment_flow[row_of_segment[feeding]], rtol=1e-15)


def test_node_flow_conserves_mass_at_every_junction(network, short_tedges, diurnal_demand):
    # What flows past a node equals what flows out of it through its children.
    demand = diurnal_demand(network, short_tedges)
    node_flow = network.node_flow(flow=demand)
    segment_flow = network.segment_flow(flow=demand)
    row_of_node = {node: i for i, node in enumerate(network.nodes)}
    for node in network.nodes:
        outgoing = network.segments.index[network.segments["from"] == node]
        if len(outgoing) == 0:
            continue
        rows = [network.segments.index.get_loc(name) for name in outgoing]
        np.testing.assert_allclose(node_flow[row_of_node[node]], segment_flow[rows].sum(axis=0), rtol=1e-15)


def test_node_flow_honours_requested_order(network, short_tedges, diurnal_demand):
    demand = diurnal_demand(network, short_tedges)
    default = network.node_flow(flow=demand)
    requested = ["T4", "Plant", "B", "T4"]
    picked = network.node_flow(flow=demand, nodes=requested)
    row_of_node = {node: i for i, node in enumerate(network.nodes)}
    # Precision floor 1e-15: selecting 4 of the 8 rows changes the BLAS blocking, so the 4-term
    # source sum can land 1 ULP away from the same sum computed in the full 8-row matmul.
    np.testing.assert_allclose(picked, default[[row_of_node[node] for node in requested]], rtol=1e-15)


def test_node_flow_rejects_unknown_nodes(network, short_tedges, diurnal_demand):
    demand = diurnal_demand(network, short_tedges)
    with pytest.raises(ValueError, match=re.escape("unknown node(s): ['Zed']")):
        network.node_flow(flow=demand, nodes=["Plant", "Zed"])


def test_effective_volume_of_a_path_under_a_constant_split(
    two_branch, short_tedges, constant_demand, analytic_travel_time
):
    # With constant demand every segment carries a fixed fraction f of the production, so the
    # path collapses to sum(V_i / f_i) of source throughflow. The trunk (f = 1) contributes its
    # full 300 m3 to both paths; the branches are inflated by 1/f.
    demand = constant_demand(two_branch, short_tedges, means=[100.0, 200.0])
    production = 300.0
    # T1: 300 / 1 + 40 / (100/300) = 420 m3 -> 1.4 d.  T2: 300 / 1 + 60 / (200/300) = 390 m3 -> 1.3 d.
    np.testing.assert_allclose(analytic_travel_time(two_branch, demand, "T1"), 420.0 / production, rtol=1e-15)
    np.testing.assert_allclose(analytic_travel_time(two_branch, demand, "T2"), 390.0 / production, rtol=1e-15)


# ============================================================================
# flow_array coercion
# ============================================================================


def test_flow_array_orders_by_endmember_not_by_insertion(network):
    """The key names the row; the order they were written in cannot reach the answer."""
    values = {"T1": [240.0, 250.0], "T2": [360.0, 350.0], "T3": [120.0, 130.0], "T4": [80.0, 70.0]}
    shuffled = {name: values[name] for name in ("T3", "T1", "T4", "T2")}
    expected = np.array([values[name] for name in network.endmembers])
    np.testing.assert_allclose(network.flow_array(values), expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(network.flow_array(shuffled), expected, rtol=0.0, atol=0.0)


def test_flow_array_passes_its_own_output_through_unchanged(network):
    """The coerced array is accepted back, which is what lets internal callers coerce once."""
    values = {"T1": [240.0, 250.0], "T2": [360.0, 350.0], "T3": [120.0, 130.0], "T4": [80.0, 70.0]}
    coerced = network.flow_array(values)
    np.testing.assert_array_equal(network.flow_array(coerced), coerced)


def test_flow_array_rejects_missing_endmember(network):
    values = {"T1": [240.0], "T3": [120.0], "T4": [80.0]}
    with pytest.raises(ValueError, match=re.escape("flow is missing endmember(s): ['T2']")):
        network.flow_array(values)


def test_flow_array_rejects_a_key_that_is_not_an_endmember(network):
    """An extra key is a typo or a modelling error, and either way the demand it carries is
    not part of the network -- silently dropping it would hide both."""
    values = {"T1": [240.0], "T2": [360.0], "T3": [120.0], "T4": [80.0], "T5": [10.0]}
    with pytest.raises(ValueError, match=re.escape("flow holds key(s) that are not endmembers: ['T5']")):
        network.flow_array(values)
    with pytest.raises(ValueError, match=re.escape("['B']")):
        network.flow_array({**{k: v for k, v in values.items() if k != "T5"}, "B": [1.0]})


@pytest.mark.parametrize(
    "flow",
    [
        pytest.param(np.ones((2, 4)), id="transposed"),
        pytest.param(np.ones(4), id="one-dimensional"),
        pytest.param(np.ones((4, 3, 2)), id="three-dimensional"),
        pytest.param(np.ones((5, 3)), id="too-many-rows"),
    ],
)
def test_flow_array_rejects_wrong_shape(network, flow):
    message = re.escape("flow must be a mapping keyed by endmember, or the (4, n_bins) array")
    with pytest.raises(ValueError, match=message):
        network.flow_array(flow)


def test_flow_array_rejects_nan_and_negative(network):
    bad = np.array([[240.0, 250.0], [360.0, np.nan], [120.0, 130.0], [80.0, 70.0]])
    with pytest.raises(ValueError, match="flow contains NaN values"):
        network.flow_array(bad)
    with pytest.raises(ValueError, match="flow contains NaN values"):
        network.flow_array({"T1": [1.0], "T2": [np.nan], "T3": [1.0], "T4": [1.0]})

    negative = np.array([[240.0, 250.0], [360.0, -1e-12], [120.0, 130.0], [80.0, 70.0]])
    with pytest.raises(ValueError, match="flow must be non-negative"):
        network.flow_array(negative)
    with pytest.raises(ValueError, match="flow must be non-negative"):
        network.flow_array({"T1": [1.0], "T2": [-1.0], "T3": [1.0], "T4": [1.0]})
    # +inf passes every "< 0" test but is not a flow either.
    with pytest.raises(ValueError, match="flow must be non-negative"):
        network.flow_array({"T1": [1.0], "T2": [np.inf], "T3": [1.0], "T4": [1.0]})


def test_zero_demand_is_allowed_and_propagates(two_branch, short_tedges, constant_demand):
    # A closed tap is a legitimate boundary condition: the branch stops, the trunk keeps the rest.
    demand = constant_demand(two_branch, short_tedges, means=[0.0, 45.0])
    n_bins = len(short_tedges) - 1
    expected = np.stack([np.full(n_bins, 45.0), np.zeros(n_bins), np.full(n_bins, 45.0)])
    np.testing.assert_allclose(two_branch.segment_flow(flow=demand), expected, rtol=0.0, atol=0.0)
