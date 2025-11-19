"""Run all examples sequentially.

This script runs all test cases and summarizes results.
"""

import sys
import os

# Ensure proper imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import example modules
from example_gaussian_bump import main as run_gaussian
from example_multiple_inclusions import main as run_multiple
from example_step_function import main as run_step


def main():
    """Run all examples and collect results."""
    print("\n" + "="*70)
    print("RUNNING ALL MRE INVERSE PROBLEM EXAMPLES")
    print("="*70 + "\n")
    
    results = {}
    
    # Example 1: Gaussian bump
    print("\n" + "-"*70)
    print("Running Example 1: Gaussian Bump")
    print("-"*70)
    try:
        results['gaussian'] = run_gaussian()
    except Exception as e:
        print(f"❌ Example 1 failed with error: {e}")
        results['gaussian'] = False
    
    # Example 2: Multiple inclusions
    print("\n" + "-"*70)
    print("Running Example 2: Multiple Inclusions")
    print("-"*70)
    try:
        results['multiple'] = run_multiple()
    except Exception as e:
        print(f"❌ Example 2 failed with error: {e}")
        results['multiple'] = False
    
    # Example 3: Step function
    print("\n" + "-"*70)
    print("Running Example 3: Step Function")
    print("-"*70)
    try:
        results['step'] = run_step()
    except Exception as e:
        print(f"❌ Example 3 failed with error: {e}")
        results['step'] = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"  Gaussian Bump:        {'✅ PASSED' if results['gaussian'] else '❌ FAILED'}")
    print(f"  Multiple Inclusions:  {'✅ PASSED' if results['multiple'] else '❌ FAILED'}")
    print(f"  Step Function:        {'✅ PASSED' if results['step'] else '❌ FAILED'}")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
