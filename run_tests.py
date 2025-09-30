#!/usr/bin/env python3
"""
Test runner script for the Meal AI Agent project.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(test_category=None, verbose=False):
    """
    Run tests based on category.
    
    Args:
        test_category: Category of tests to run (agents, services, integration, workflow, debug, all)
        verbose: Whether to run with verbose output
    """
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    # Add test path based on category
    if test_category == "agents":
        cmd.append("tests/test_agents/")
    elif test_category == "services":
        cmd.append("tests/test_services/")
    elif test_category == "integration":
        cmd.append("tests/integration/")
    elif test_category == "workflow":
        cmd.append("tests/workflow/")
    elif test_category == "debug":
        cmd.append("tests/debug/")
    elif test_category == "all":
        cmd.append("tests/")
    else:
        # Default: run all tests except debug
        cmd.extend(["tests/test_agents/", "tests/test_services/", "tests/integration/", "tests/workflow/"])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed with exit code {e.returncode}")
        return e.returncode

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run tests for Meal AI Agent")
    parser.add_argument(
        "category",
        nargs="?",
        choices=["agents", "services", "integration", "workflow", "debug", "all"],
        default=None,
        help="Test category to run (default: all except debug)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run with verbose output"
    )
    
    args = parser.parse_args()
    
    print("🧪 Meal AI Agent Test Runner")
    print("=" * 40)
    
    exit_code = run_tests(args.category, args.verbose)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
