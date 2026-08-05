"""
Topology, geometry and flow accounting of a branched distribution network.

A :class:`PipeNetwork` is a rooted tree of pipe segments fed by a single production point.
Water leaves the source, splits at junctions, and is drawn at the leaves -- the *endmembers*.
Because a split leaves concentration unchanged, every endmember is reached by exactly one
path and the water quality along that path is a pure function of the source signal. That is
the structural property the transport model rests on; see :ref:`assumption-tree-topology`.

Geometry is time-constant. Each segment holds a water volume ``V = pi/4 * D**2 * L`` derived
from its inner diameter and length, or supplied directly as a ``volume`` column.

Flow is time-varying and is specified where it is metered: at the endmembers. Water is
incompressible and the network stores no water beyond what the pipes hold, so the flow in an
internal segment is the sum of the demands downstream of it, and the production at the source
is the sum of all demands. Nothing in the model requires those demands to move in proportion;
:meth:`PipeNetwork.segment_flow` recomputes the split at every time step.

Available class:

- :class:`PipeNetwork` - Validated rooted tree built from a segment table. Exposes the node
  and endmember sets, the source-to-node segment paths, the per-segment water volumes, and
  the two flow-accounting operators :meth:`~PipeNetwork.segment_flow` (throughflow per pipe)
  and :meth:`~PipeNetwork.node_flow` (throughflow past a node).

This file is part of pipetransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/pipetransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from pipetransport._validation import _validate_no_nan, _validate_non_negative, _validate_positive


class PipeNetwork:
    """A single-source, branched (tree) pipe network with time-constant geometry.

    Parameters
    ----------
    segments : pandas.DataFrame
        One row per pipe segment, indexed by a unique segment name. Required columns
        ``"from"`` and ``"to"`` name the upstream and downstream node of the segment. The
        water volume comes either from a ``"volume"`` column [m³] or from ``"length"`` [m]
        and ``"diameter"`` [m] (inner diameter), as ``pi/4 * diameter**2 * length``. Any
        further columns are carried through unchanged.
    source : str
        Name of the production node. Must be a node of the graph and must have no
        incoming segment.

    Attributes
    ----------
    segments : pandas.DataFrame
        Copy of the input table with a guaranteed ``"volume"`` column, in input row order.
    source : str
        Name of the production node.
    nodes : tuple of str
        Every node, in breadth-first order from the source (so the source comes first and a
        node always follows its parent).
    endmembers : tuple of str
        Nodes with no outgoing segment -- the leaves, where demand is drawn.
    paths : dict of str to tuple of str
        Segment names on the source-to-node path, ordered from the source outward. The
        source maps to an empty tuple.

    Raises
    ------
    ValueError
        If the segment table is empty or malformed, if the geometry is not positive, if
        ``source`` is not a node or has an incoming segment, or if the graph is not a tree
        rooted at ``source`` (a node with two feeds, a disconnected node, or a cycle).

    See Also
    --------
    pipetransport.transport.source_to_endmember : Forward transport across this network.
    pipetransport.examples.example_network : A ready-made four-endmember network.
    :ref:`assumption-tree-topology` : Why merges are excluded.

    Examples
    --------
    >>> import pandas as pd
    >>> from pipetransport.network import PipeNetwork
    >>> segments = pd.DataFrame(
    ...     {
    ...         "from": ["Plant", "A", "A"],
    ...         "to": ["A", "T1", "T2"],
    ...         "length": [2000.0, 800.0, 400.0],
    ...         "diameter": [0.40, 0.15, 0.20],
    ...     },
    ...     index=["Plant-A", "A-T1", "A-T2"],
    ... )
    >>> network = PipeNetwork(segments=segments, source="Plant")
    >>> network.endmembers
    ('T1', 'T2')
    >>> network.paths["T1"]
    ('Plant-A', 'A-T1')
    >>> float(network.segments.loc["Plant-A", "volume"].round(1))
    251.3
    """

    def __init__(self, *, segments: pd.DataFrame, source: str) -> None:
        segments = pd.DataFrame(segments).copy()
        if segments.empty:
            msg = "segments must hold at least one pipe segment"
            raise ValueError(msg)
        missing = {"from", "to"} - set(segments.columns)
        if missing:
            msg = f"segments is missing required column(s): {sorted(missing)}"
            raise ValueError(msg)
        if not segments.index.is_unique:
            duplicated = sorted(set(segments.index[segments.index.duplicated()]))
            msg = f"segments index must be unique; duplicated segment name(s): {duplicated}"
            raise ValueError(msg)

        if "volume" in segments.columns:
            _validate_positive(segments["volume"], name="segment volume")
        else:
            geometry = {"length", "diameter"} - set(segments.columns)
            if geometry:
                msg = "segments must hold either a 'volume' column or both 'length' and 'diameter' columns"
                raise ValueError(msg)
            _validate_positive(segments["length"], name="segment length")
            _validate_positive(segments["diameter"], name="segment diameter")
            segments["volume"] = (
                np.pi / 4.0 * segments["diameter"].to_numpy(float) ** 2 * segments["length"].to_numpy(float)
            )

        self.segments = segments
        self.source = source

        upstream = segments["from"].to_numpy()
        downstream = segments["to"].to_numpy()
        if np.any(upstream == downstream):
            offenders = sorted(set(segments.index[upstream == downstream]))
            msg = f"a segment cannot start and end at the same node; offending segment(s): {offenders}"
            raise ValueError(msg)

        # Exactly one feed per node is what makes the tree a tree: a second feed would blend two
        # transport histories, which a single-source model cannot represent.
        parent_segment: dict[str, str] = {}
        for name, node in zip(segments.index, downstream, strict=True):
            if node in parent_segment:
                msg = (
                    f"node {node!r} is fed by more than one segment "
                    f"({parent_segment[node]!r} and {name!r}); merging flows are not supported"
                )
                raise ValueError(msg)
            parent_segment[node] = name
        if source in parent_segment:
            msg = f"source {source!r} is fed by segment {parent_segment[source]!r}; the source must be the tree root"
            raise ValueError(msg)

        children: dict[str, list[tuple[str, str]]] = {}
        for name, up, down in zip(segments.index, upstream, downstream, strict=True):
            children.setdefault(up, []).append((name, down))
        if source not in children:
            msg = f"source {source!r} is not the upstream node of any segment"
            raise ValueError(msg)

        # Breadth-first walk from the source. It reaches every node exactly once in a tree, so a
        # node left unvisited is disconnected or sits on a cycle.
        nodes: list[str] = [source]
        paths: dict[str, tuple[str, ...]] = {source: ()}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for name, child in children.get(node, []):
                nodes.append(child)
                paths[child] = (*paths[node], name)
                queue.append(child)
        unreachable = sorted(set(parent_segment) - set(nodes))
        if unreachable:
            msg = f"node(s) not reachable from source {source!r}: {unreachable} (disconnected branch or cycle)"
            raise ValueError(msg)

        self.nodes = tuple(nodes)
        self.paths = paths
        self.endmembers = tuple(node for node in nodes if node not in children)

        # Endmember membership of every node's subtree, as a boolean matrix. Reading it off the
        # paths avoids a second graph traversal: endmember e sits below node n exactly when n's
        # path is a prefix of e's path.
        endmember_paths = [paths[e] for e in self.endmembers]
        self._below_node = np.array([
            [path[: len(paths[node])] == paths[node] for path in endmember_paths] for node in self.nodes
        ])
        # A segment carries whatever its downstream node carries.
        node_row = {node: i for i, node in enumerate(self.nodes)}
        self._below_segment = self._below_node[[node_row[node] for node in downstream]]

    def __repr__(self) -> str:
        """Return a one-line summary naming the source and counting segments and endmembers.

        Returns
        -------
        str
            Summary of the network size, e.g.
            ``PipeNetwork(source='Plant', segments=7, endmembers=4, volume=1113.9 m3)``.
        """
        return (
            f"PipeNetwork(source={self.source!r}, segments={len(self.segments)}, "
            f"endmembers={len(self.endmembers)}, volume={self.segments['volume'].sum():.1f} m3)"
        )

    def flow_array(self, flow: Mapping[str, npt.ArrayLike] | npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Coerce endmember demand into a ``(n_endmembers, n_bins)`` array in :attr:`endmembers` order.

        Parameters
        ----------
        flow : mapping
            Demand at every endmember [m³/day], keyed by endmember name: one bin-constant
            array of length ``n_bins`` per endmember. The keys must be exactly
            :attr:`endmembers` -- a missing one leaves the network underdetermined and an
            unknown one is a typo, so both raise. (An already-coerced
            ``(n_endmembers, n_bins)`` array passes through unchanged, which is the
            idempotence internal callers rely on; the mapping is the input form.)

        Returns
        -------
        ndarray
            Demand of shape ``(n_endmembers, n_bins)``, ordered as :attr:`endmembers`.

        Raises
        ------
        ValueError
            If the mapping misses an endmember or holds a key that is not one, if an
            already-coerced array has the wrong shape, or if any demand is NaN or negative.

        Examples
        --------
        >>> from pipetransport.examples import example_network
        >>> network = example_network()
        >>> demand = {
        ...     "T1": [240.0, 250.0],
        ...     "T2": [360.0, 350.0],
        ...     "T3": [120.0, 130.0],
        ...     "T4": [80.0, 70.0],
        ... }
        >>> network.flow_array(demand).shape
        (4, 2)
        """
        if isinstance(flow, Mapping):
            named = {str(key): value for key, value in flow.items()}
            missing = [e for e in self.endmembers if e not in named]
            if missing:
                msg = f"flow is missing endmember(s): {missing}"
                raise ValueError(msg)
            unknown = sorted(set(named) - set(self.endmembers))
            if unknown:
                msg = f"flow holds key(s) that are not endmembers: {unknown}; endmembers are {list(self.endmembers)}"
                raise ValueError(msg)
            array = np.stack([np.asarray(named[e], dtype=float) for e in self.endmembers])
        else:
            # The pass-through for an array this method already coerced; not a public input form.
            array = np.asarray(flow, dtype=float)
            if array.ndim != 2 or array.shape[0] != len(self.endmembers):  # noqa: PLR2004
                msg = (
                    f"flow must be a mapping keyed by endmember, or the "
                    f"({len(self.endmembers)}, n_bins) array a previous flow_array call produced; "
                    f"got shape {array.shape}"
                )
                raise ValueError(msg)
        _validate_no_nan(array, name="flow")
        _validate_non_negative(array, name="flow", message="flow must be non-negative (reverse flow not supported)")
        return array

    def segment_flow(self, *, flow: Mapping[str, npt.ArrayLike] | npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Throughflow of every segment, from mass conservation on the endmember demand.

        The pipes store no water beyond their fixed volume, so a segment carries the summed
        demand of the endmembers below it at every instant. No proportionality between
        demands is assumed: the split is recomputed per time bin.

        Parameters
        ----------
        flow : mapping
            Demand at every endmember [m³/day], keyed by endmember name; see
            :meth:`flow_array`.

        Returns
        -------
        ndarray
            Segment throughflow [m³/day] of shape ``(n_segments, n_bins)``, in
            :attr:`segments` row order.

        See Also
        --------
        node_flow : Throughflow past a node rather than through a segment.

        Examples
        --------
        >>> from pipetransport.examples import example_network
        >>> network = example_network()
        >>> demand = {"T1": [240.0], "T2": [360.0], "T3": [120.0], "T4": [80.0]}
        >>> network.segment_flow(flow=demand).ravel()
        array([800., 600., 200., 240., 360., 120.,  80.])
        """
        return self._below_segment @ self.flow_array(flow)

    def node_flow(
        self,
        *,
        flow: Mapping[str, npt.ArrayLike] | npt.NDArray[np.floating],
        nodes: list[str] | tuple[str, ...] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Throughflow past one or more nodes, from mass conservation on the endmember demand.

        The throughflow past a node is the summed demand of the endmembers below it: the
        production rate at the source, the demand itself at an endmember, and the flow of the
        feeding segment anywhere in between. It is the weight with which a node's outgoing
        water quality is averaged over a time bin.

        Parameters
        ----------
        flow : mapping
            Demand at every endmember [m³/day], keyed by endmember name; see
            :meth:`flow_array`.
        nodes : list of str or None, optional
            Nodes to report, in the requested order. Defaults to :attr:`nodes`.

        Returns
        -------
        ndarray
            Node throughflow [m³/day] of shape ``(len(nodes), n_bins)``.

        Raises
        ------
        ValueError
            If a requested name is not a node of the network.

        See Also
        --------
        segment_flow : Throughflow through a segment rather than past a node.

        Examples
        --------
        >>> from pipetransport.examples import example_network
        >>> network = example_network()
        >>> demand = {"T1": [240.0], "T2": [360.0], "T3": [120.0], "T4": [80.0]}
        >>> network.node_flow(flow=demand, nodes=["Plant", "B", "T4"]).ravel()
        array([800., 600.,  80.])
        """
        node_row = {node: i for i, node in enumerate(self.nodes)}
        requested = self.nodes if nodes is None else tuple(nodes)
        unknown = [node for node in requested if node not in node_row]
        if unknown:
            msg = f"unknown node(s): {unknown}; network nodes are {list(self.nodes)}"
            raise ValueError(msg)
        return self._below_node[[node_row[node] for node in requested]] @ self.flow_array(flow)
