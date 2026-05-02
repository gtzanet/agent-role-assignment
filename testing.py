from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd

from causality.task_interference_analyzer import compute_task_interaction_weight_matrix
from causality.causal_discovery import (
    _node_type_to_tier,
    _normalize_node_metadata,
    infer_causal_graph_correlation,
    infer_causal_graph_lingam,
    infer_causal_graph_notears,
)
from causality.task_interference_analyzer import TaskInterferenceAnalyzer, TaskInterferenceGraph
from environment import CausalGraph, Node, NodeType

try:
    import lingam  # noqa: F401

    HAS_LINGAM = True
except ImportError:
    HAS_LINGAM = False


class TestEnvironmentModule(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = CausalGraph()
        self.n_input = Node("input_a", NodeType.INPUT, decision_space_size=4)
        self.n_mid = Node("mid", NodeType.INTERMEDIARY)
        self.n_kpi = Node("kpi_x", NodeType.KPI)
        for node in (self.n_input, self.n_mid, self.n_kpi):
            self.graph.add_node(node)

    def test_node_repr_contains_fields(self) -> None:
        text = repr(self.n_input)
        self.assertIn("input_a", text)
        self.assertIn("input", text)
        self.assertIn("size=4", text)

    def test_add_edge_requires_existing_nodes(self) -> None:
        with self.assertRaises(ValueError):
            self.graph.add_edge("missing", "kpi_x")

        with self.assertRaises(ValueError):
            self.graph.add_edge("input_a", "missing")

    def test_get_inputs_and_kpis(self) -> None:
        self.graph.add_edge("input_a", "mid")
        self.graph.add_edge("mid", "kpi_x")

        inputs = self.graph.get_inputs()
        kpis = self.graph.get_kpis()

        self.assertEqual([n.name for n in inputs], ["input_a"])
        self.assertEqual([n.name for n in kpis], ["kpi_x"])

    def test_get_reachable_kpis(self) -> None:
        self.graph.add_edge("input_a", "mid")
        self.graph.add_edge("mid", "kpi_x")

        reachable = self.graph.get_reachable_kpis("input_a")
        self.assertEqual(reachable, {"kpi_x"})

    def test_get_reachable_kpis_unknown_node(self) -> None:
        with self.assertRaises(ValueError):
            self.graph.get_reachable_kpis("unknown")


class TestBuildTigModule(unittest.TestCase):
    def test_rejects_invalid_dimensions(self) -> None:
        delta = np.array([1.0, 2.0])
        omega = np.array([1.0, 1.0])
        A = np.array([2.0, 2.0])

        with self.assertRaises(ValueError):
            compute_task_interaction_weight_matrix(delta, omega, A, C_max=4.0, rho=0.1)

    def test_rejects_omega_shape_mismatch(self) -> None:
        delta = np.ones((2, 3), dtype=float)
        omega = np.ones(2, dtype=float)
        A = np.ones(2, dtype=float)

        with self.assertRaises(ValueError):
            compute_task_interaction_weight_matrix(delta, omega, A, C_max=4.0, rho=0.1)

    def test_rejects_action_space_shape_mismatch(self) -> None:
        delta = np.ones((3, 2), dtype=float)
        omega = np.ones(2, dtype=float)
        A = np.ones(2, dtype=float)

        with self.assertRaises(ValueError):
            compute_task_interaction_weight_matrix(delta, omega, A, C_max=4.0, rho=0.1)

    def test_output_is_symmetric_has_zero_diagonal_and_finite(self) -> None:
        rng = np.random.default_rng(123)

        # Randomized checks over many input shapes to approximate broad coverage.
        for _ in range(100):
            n_tasks = int(rng.integers(2, 8))
            n_kpis = int(rng.integers(1, 8))

            delta = rng.uniform(0.0, 1.0, size=(n_tasks, n_kpis))
            omega = rng.uniform(0.1, 3.0, size=n_kpis)
            A = rng.uniform(1.0, 10.0, size=n_tasks)
            c_max = float(rng.uniform(1.0, 60.0))
            rho = float(rng.uniform(0.01, 0.8))

            W = compute_task_interaction_weight_matrix(delta, omega, A, C_max=c_max, rho=rho)

            self.assertEqual(W.shape, (n_tasks, n_tasks))
            self.assertTrue(np.isfinite(W).all())
            self.assertTrue(np.allclose(W, W.T, atol=1e-10))
            self.assertTrue(np.allclose(np.diag(W), 0.0, atol=1e-12))
            self.assertTrue((W <= 1.0 + 1e-9).all())
            self.assertTrue((W >= -1e-9).all())


class TestCausalDiscoveryHelpers(unittest.TestCase):
    def test_node_type_to_tier_all_values(self) -> None:
        self.assertEqual(_node_type_to_tier(NodeType.INPUT), 0)
        self.assertEqual(_node_type_to_tier(NodeType.INTERMEDIARY), 1)
        self.assertEqual(_node_type_to_tier(NodeType.KPI), 2)

    def test_node_type_to_tier_invalid(self) -> None:
        with self.assertRaises(ValueError):
            _node_type_to_tier("invalid")

    def test_normalize_metadata_success_and_defaults(self) -> None:
        cols = ["a", "b"]
        node_types = {"a": NodeType.INPUT, "b": NodeType.KPI}
        metadata = _normalize_node_metadata(cols, node_types)

        self.assertEqual(metadata["a"], (NodeType.INPUT, 1))
        self.assertEqual(metadata["b"], (NodeType.KPI, 1))

    def test_normalize_metadata_with_decision_space(self) -> None:
        cols = ["a", "b"]
        node_types = {"a": NodeType.INPUT, "b": NodeType.KPI}
        dss = {"a": 7}
        metadata = _normalize_node_metadata(cols, node_types, dss)

        self.assertEqual(metadata["a"], (NodeType.INPUT, 7))
        self.assertEqual(metadata["b"], (NodeType.KPI, 1))

    def test_normalize_metadata_missing_node_type(self) -> None:
        cols = ["a", "b"]
        node_types = {"a": NodeType.INPUT}

        with self.assertRaises(ValueError):
            _normalize_node_metadata(cols, node_types)


class TestCausalDiscoveryAlgorithms(unittest.TestCase):
    def _write_df(self, tmpdir: Path, name: str, df: pd.DataFrame) -> Path:
        path = tmpdir / name
        df.to_csv(path, index=False)
        return path

    def test_correlation_graph_respects_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            x = np.linspace(0.0, 10.0, 500)
            y = 2.0 * x + 1.0
            z = 3.0 * y - 2.0
            df = pd.DataFrame({"x": x, "y": y, "z": z})
            path = self._write_df(tmpdir, "corr.csv", df)

            node_types = {
                "x": NodeType.INPUT,
                "y": NodeType.INTERMEDIARY,
                "z": NodeType.KPI,
            }
            dss = {"x": 4}

            graph = infer_causal_graph_correlation(str(path), node_types=node_types, threshold=0.1, decision_space_sizes=dss)

            self.assertCountEqual(graph.nodes.keys(), ["x", "y", "z"])
            self.assertIn(("x", "y"), graph.graph.edges())
            self.assertIn(("x", "z"), graph.graph.edges())
            self.assertIn(("y", "z"), graph.graph.edges())
            self.assertNotIn(("z", "x"), graph.graph.edges())
            self.assertEqual(graph.nodes["x"].decision_space_size, 4)

    def test_lingam_dependency_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
            path = self._write_df(tmpdir, "dep.csv", df)
            node_types = {"a": NodeType.INPUT, "b": NodeType.KPI}

            if HAS_LINGAM:
                graph = infer_causal_graph_lingam(str(path), node_types=node_types, threshold=0.1)
                self.assertCountEqual(graph.nodes.keys(), ["a", "b"])
            else:
                with self.assertRaises(ImportError):
                    infer_causal_graph_lingam(str(path), node_types=node_types, threshold=0.1)

    @unittest.skipUnless(HAS_LINGAM, "lingam is not installed")
    def test_lingam_raises_when_no_numeric_columns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            df = pd.DataFrame({"a": ["u", "v", "w"], "b": ["x", "y", "z"]})
            path = self._write_df(tmpdir, "non_numeric.csv", df)

            node_types = {"a": NodeType.INPUT, "b": NodeType.KPI}
            with self.assertRaises(RuntimeError):
                infer_causal_graph_lingam(str(path), node_types=node_types, threshold=0.1)

    @unittest.skipUnless(HAS_LINGAM, "lingam is not installed")
    def test_lingam_raises_when_insufficient_non_constant_columns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})
            path = self._write_df(tmpdir, "constant.csv", df)

            node_types = {"a": NodeType.INPUT, "b": NodeType.KPI}
            with self.assertRaises(RuntimeError):
                infer_causal_graph_lingam(str(path), node_types=node_types, threshold=0.1)

    @unittest.skipUnless(HAS_LINGAM, "lingam is not installed")
    def test_lingam_raises_when_samples_not_greater_than_features(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            df = pd.DataFrame(
                {
                    "a": [1.0, 2.0],
                    "b": [2.0, 3.0],
                    "c": [3.0, 4.0],
                }
            )
            path = self._write_df(tmpdir, "small.csv", df)

            node_types = {
                "a": NodeType.INPUT,
                "b": NodeType.INTERMEDIARY,
                "c": NodeType.KPI,
            }
            with self.assertRaises(RuntimeError):
                infer_causal_graph_lingam(str(path), node_types=node_types, threshold=0.01)

    @unittest.skipUnless(HAS_LINGAM, "lingam is not installed")
    def test_lingam_smoke_test_valid_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            rng = np.random.default_rng(7)
            x = rng.normal(0.0, 1.0, size=600)
            y = 1.8 * x + rng.normal(0.0, 0.05, size=600)
            z = 2.3 * y + rng.normal(0.0, 0.05, size=600)
            df = pd.DataFrame({"x": x, "y": y, "z": z})
            path = self._write_df(tmpdir, "lingam.csv", df)

            node_types = {
                "x": NodeType.INPUT,
                "y": NodeType.INTERMEDIARY,
                "z": NodeType.KPI,
            }
            graph = infer_causal_graph_lingam(str(path), node_types=node_types, threshold=0.05)

            self.assertCountEqual(graph.nodes.keys(), ["x", "y", "z"])
            for src, dst in graph.graph.edges():
                self.assertNotEqual(src, dst)

    def test_notears_with_mocked_backend_filters_edges_and_removes_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            df = pd.DataFrame(
                {
                    "x": [0, 1, 2, 3, 4, 5],
                    "y": [1, 2, 4, 8, 16, 32],
                    "z": [2, 4, 8, 16, 32, 64],
                }
            )
            path = self._write_df(tmpdir, "notears.csv", df)

            node_types = {
                "x": NodeType.INPUT,
                "y": NodeType.INTERMEDIARY,
                "z": NodeType.KPI,
            }

            class DummyStructureModel(nx.DiGraph):
                def remove_edges_below_threshold(self, threshold: float) -> None:
                    for u, v, data in list(self.edges(data=True)):
                        if float(data.get("weight", 0.0)) < threshold:
                            self.remove_edge(u, v)

            def fake_from_pandas(_df, tabu_edges=None):
                self.assertIsNotNone(tabu_edges)
                # Ensure tier constraints are translated to tabu edges.
                self.assertIn(("y", "x"), tabu_edges)
                self.assertIn(("z", "y"), tabu_edges)
                self.assertIn(("z", "x"), tabu_edges)

                g = DummyStructureModel()
                g.add_nodes_from(["x", "y", "z"])
                g.add_edge("x", "y", weight=0.9)
                g.add_edge("y", "z", weight=0.8)
                g.add_edge("z", "x", weight=0.2)  # Creates a cycle; should be removed.
                g.add_edge("x", "z", weight=0.05)  # Below threshold; should be removed.
                return g

            with mock.patch("causal_discovery.from_pandas", side_effect=fake_from_pandas):
                graph = infer_causal_graph_notears(
                    str(path),
                    node_types=node_types,
                    threshold=0.1,
                    decision_space_sizes={"x": 4},
                )

            self.assertCountEqual(graph.nodes.keys(), ["x", "y", "z"])
            self.assertEqual(graph.nodes["x"].decision_space_size, 4)
            self.assertIn(("x", "y"), graph.graph.edges())
            self.assertIn(("y", "z"), graph.graph.edges())
            self.assertNotIn(("z", "x"), graph.graph.edges())
            self.assertNotIn(("x", "z"), graph.graph.edges())
            self.assertTrue(nx.is_directed_acyclic_graph(graph.graph))


class TestTaskInterferenceAnalyzer(unittest.TestCase):
    def test_generate_tig_reachability_mode(self) -> None:
        graph = CausalGraph()
        for node in (
            Node("x", NodeType.INPUT, decision_space_size=4),
            Node("y", NodeType.INTERMEDIARY),
            Node("z", NodeType.KPI),
        ):
            graph.add_node(node)
        graph.add_edge("x", "y")
        graph.add_edge("y", "z")

        analyzer = TaskInterferenceAnalyzer(
            graph,
            {
                "inputs": ["x"],
                "intermediates": ["y"],
                "outputs": ["z"],
            },
        )

        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0, 4.0], "z": [2.0, 3.0, 4.0, 5.0]})
            tig = analyzer.generate_tig(df, output_dir=tmpdir, delta_mode="reachability", save_plot=False)

            self.assertIsInstance(tig, TaskInterferenceGraph)
            self.assertEqual(tig.tasks, ["x"])
            self.assertEqual(tig.kpis, ["z"])
            self.assertEqual(tig.delta.shape, (1, 1))
            self.assertEqual(tig.w.shape, (1, 1))
            self.assertTrue((tmpdir / "tig_delta.csv").exists())
            self.assertTrue((tmpdir / "tig_W.csv").exists())
            self.assertTrue((tmpdir / "tig_summary.json").exists())
            self.assertFalse((tmpdir / "tig_plot.png").exists())

    def test_partition_tig_greedy_modularity(self) -> None:
        analyzer = TaskInterferenceAnalyzer(CausalGraph(), {"inputs": [], "intermediates": [], "outputs": []})
        tasks = ["t0", "t1", "t2", "t3"]
        w = np.array(
            [
                [0.0, 0.9, 0.0, 0.0],
                [0.9, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.8],
                [0.0, 0.0, 0.8, 0.0],
            ],
            dtype=float,
        )

        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            tig = TaskInterferenceGraph(
                tasks=tasks,
                kpis=[],
                delta=np.zeros((len(tasks), 0), dtype=float),
                omega=np.ones((0,), dtype=float),
                action_space=np.ones((len(tasks),), dtype=float),
                w=w,
            )
            result = analyzer.partition_tig(tig=tig, n_agents=2, algorithm="greedy_modularity", output_dir=tmpdir)

            partitions = result["partitions"]
            sizes = sorted(len(tasks_in_partition) for tasks_in_partition in partitions.values())
            self.assertEqual(sizes, [2, 2])
            self.assertCountEqual([task for values in partitions.values() for task in values], tasks)
            self.assertTrue((tmpdir / "partitions.json").exists())
            self.assertTrue((tmpdir / "partition_assignments.csv").exists())
            self.assertTrue((tmpdir / "partition_summary.json").exists())
            self.assertTrue((tmpdir / "agent_assignments.yaml").exists())


def load_suite() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentModule))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildTigModule))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalDiscoveryHelpers))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalDiscoveryAlgorithms))
    suite.addTests(loader.loadTestsFromTestCase(TestTaskInterferenceAnalyzer))
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
