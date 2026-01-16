"""
Test all examples by auto-discovery and execution.

This module discovers all example Python files and runs them as tests.
Examples are executed as subprocesses to ensure isolation.

Usage:
    pytest tests/test_examples.py -v
    pytest tests/test_examples.py -v -k "mlp"  # run only mlp-related examples
"""
import subprocess
import sys
from pathlib import Path

import pytest


# Get the example directory
EXAMPLE_DIR = Path(__file__).parent.parent / "example"


def discover_examples():
    """Discover all example Python files."""
    examples = []

    for py_file in EXAMPLE_DIR.rglob("*.py"):
        # Skip __init__.py files
        if py_file.name == "__init__.py":
            continue

        # Create a readable test ID from the path
        relative = py_file.relative_to(EXAMPLE_DIR)
        test_id = str(relative).replace("/", "_").replace(".py", "")

        examples.append(pytest.param(py_file, id=test_id))

    return examples


@pytest.mark.parametrize("example_path", discover_examples())
def test_example(example_path: Path):
    """Run an example file and verify it completes without error."""
    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        timeout=120,  # 2 minute timeout per example
    )

    # Print output for debugging
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, f"Example failed with return code {result.returncode}"
