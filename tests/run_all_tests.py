"""
Comprehensive test runner for LLM-JEPA Symbolic Regression.

Runs all tests and generates a detailed report.
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_tests():
    """Run all tests and generate report."""
    
    print("=" * 80)
    print("LLM-JEPA Symbolic Regression - Comprehensive Test Suite")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print("=" * 80)
    print()
    
    # Test directories
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    # Run pytest with verbose output
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",           # Verbose output
        "--tb=short",   # Short traceback format
        "-ra",          # Show extra test summary info
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=project_root)
    
    print()
    print("=" * 80)
    if result.returncode == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ TESTS FAILED (exit code: {result.returncode})")
    print("=" * 80)
    
    return result.returncode


def run_inference_effectiveness_test():
    """Run specific test for inference effectiveness."""
    print()
    print("=" * 80)
    print("Running Inference Effectiveness Test")
    print("=" * 80)
    
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.tokenizer import TOKEN2IDX
    from inference.beam_search import skeletonize, fit_and_score
    
    test_cases = [
        ("Linear: y = 2.5*x + 3", 
         ['c1', 'x1', '*', 'c2', '+'],
         np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32),
         lambda X: (2.5 * X.flatten() + 3.0).astype(np.float32)),
        
        # Note: Quadratic uses 'sq' operator which is x^2, so structure is c1*sq(x1) = c1*x1^2
        ("Quadratic: y = 0.5*x^2",
         ['c1', 'x1', 'sq', '*'],
         np.linspace(0, 5, 100).reshape(-1, 1).astype(np.float32),  # Only positive to avoid complex numbers
         lambda X: (0.5 * X.flatten()**2).astype(np.float32)),
        
        ("Sinusoidal: y = 2*sin(3*x)",
         ['c1', 'c2', 'x1', '*', 'sin', '*'],
         np.linspace(0, 2*np.pi, 200).reshape(-1, 1).astype(np.float32),
         lambda X: (2.0 * np.sin(3.0 * X.flatten())).astype(np.float32)),
    ]
    
    results = []
    for name, tokens, X, y_fn in test_cases:
        y = y_fn(X)
        ids = [TOKEN2IDX[t] for t in tokens]
        skel = skeletonize(ids)
        
        if skel is None:
            print(f"  ❌ {name}: Skeletonize failed")
            results.append(False)
            continue
            
        result = fit_and_score(skel, tokens, X, y, ['x1'], max_iter=15)
        
        if result['r2'] > 0.9:
            status = "✅"
        elif result['r2'] > 0.5:
            status = "⚠️"
        else:
            status = "❌"
            
        print(f"  {status} {name}: R² = {result['r2']:.4f}")
        print(f"      Skeleton: {skel}")
        print(f"      Fitted: {result['expression']}")
        
        results.append(result['r2'] > 0.5)
    
    print()
    passed = sum(results)
    total = len(results)
    print(f"Inference Effectiveness: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == '__main__':
    # Run pytest tests
    pytest_result = run_tests()
    
    # Run inference effectiveness test
    inf_result = run_inference_effectiveness_test()
    
    # Final summary
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Pytest Tests: {'PASSED' if pytest_result == 0 else 'FAILED'}")
    print(f"Inference Effectiveness: {'PASSED' if inf_result else 'FAILED'}")
    print("=" * 80)
    
    if pytest_result == 0 and inf_result:
        print("\n✅ ALL TESTS PASSED - Pipeline is ready!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Please review the output above")
        sys.exit(1)
