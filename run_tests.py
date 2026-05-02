#!/usr/bin/env python3
"""Run function-level test coverage for the core workflow modules."""

from __future__ import annotations

import argparse
import sys
import time
import unittest

import testing


class _ResultCollector(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self._started: dict[unittest.case.TestCase, float] = {}
        self.timings: list[tuple[str, str, float]] = []

    def startTest(self, test: unittest.case.TestCase) -> None:
        self._started[test] = time.perf_counter()
        super().startTest(test)

    def _record(self, test: unittest.case.TestCase, status: str) -> None:
        started = self._started.pop(test, None)
        elapsed = time.perf_counter() - started if started is not None else 0.0
        self.timings.append((status, test.id(), elapsed))

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        self._record(test, "PASS")
        super().addSuccess(test)

    def addFailure(self, test: unittest.case.TestCase, err) -> None:
        self._record(test, "FAIL")
        super().addFailure(test, err)

    def addError(self, test: unittest.case.TestCase, err) -> None:
        self._record(test, "ERROR")
        super().addError(test, err)


class _ResultFactory(unittest.TextTestRunner):
    resultclass = _ResultCollector


def _flatten_suite(suite: unittest.TestSuite) -> list[unittest.case.TestCase]:
    tests: list[unittest.case.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            tests.extend(_flatten_suite(item))
        else:
            tests.append(item)
    return tests


def _filter_suite_by_name(suite: unittest.TestSuite, substring: str) -> unittest.TestSuite:
    selected = [test for test in _flatten_suite(suite) if substring.lower() in test.id().lower()]
    return unittest.TestSuite(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run unit tests for core functions in task_interference_analyzer.py, environment.py, and causal_discovery.py."
        )
    )
    parser.add_argument(
        "--match",
        default=None,
        help="Run only tests whose full id contains this substring (case-insensitive).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use verbose unittest output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    suite = testing.load_suite()
    if args.match:
        suite = _filter_suite_by_name(suite, args.match)

    total_discovered = suite.countTestCases()
    if total_discovered == 0:
        print("No tests selected. Adjust --match and try again.")
        return 2

    print("Function-Level Test Run")
    print(f"Selected tests: {total_discovered}")
    if args.match:
        print(f"Filter: {args.match}")
    print()

    verbosity = 2 if args.verbose else 1
    runner = _ResultFactory(verbosity=verbosity)

    started = time.perf_counter()
    result: _ResultCollector = runner.run(suite)  # type: ignore[assignment]
    elapsed = time.perf_counter() - started

    print("\nPer-test timing")
    for status, test_id, test_elapsed in sorted(result.timings, key=lambda item: item[2], reverse=True):
        print(f"[{status}] {test_elapsed:7.3f}s  {test_id}")

    n_fail = len(result.failures)
    n_err = len(result.errors)
    n_pass = result.testsRun - n_fail - n_err

    print("\nSummary")
    print(f"Passed: {n_pass}")
    print(f"Failed: {n_fail}")
    print(f"Errors: {n_err}")
    print(f"Total:  {result.testsRun}")
    print(f"Elapsed: {elapsed:.2f}s")

    if result.wasSuccessful():
        print("Result: PASS")
        return 0

    print("Result: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
