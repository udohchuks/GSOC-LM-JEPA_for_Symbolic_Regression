"""
Test BFGS fitting effectiveness with increased iterations.

Tests various function types to verify the improved optimization.
"""
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.tokenizer import TOKEN2IDX
from inference.beam_search import skeletonize, fit_and_score

print("=" * 70)
print("BFGS FITTING EFFECTIVENESS TEST (max_iter=100, n_restarts=5)")
print("=" * 70)

test_cases = [
    # (name, tokens, X, y_fn, expected_r2_threshold)
    ("Linear: y = 2.5*x + 3", 
     ['c1', 'x1', '*', 'c2', '+'],
     np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32),
     lambda X: (2.5 * X.flatten() + 3.0).astype(np.float32),
     0.99),
    
    ("Linear: y = -1.5*x + 7", 
     ['c1', 'x1', '*', 'c2', '+'],
     np.linspace(-5, 5, 100).reshape(-1, 1).astype(np.float32),
     lambda X: (-1.5 * X.flatten() + 7.0).astype(np.float32),
     0.99),
    
    ("Quadratic: y = 0.5*x^2",
     ['c1', 'x1', 'sq', '*'],
     np.linspace(0, 5, 100).reshape(-1, 1).astype(np.float32),
     lambda X: (0.5 * X.flatten()**2).astype(np.float32),
     0.95),
    
    ("Quadratic: y = 2*x^2 + 3*x + 1",
     ['c1', 'x1', 'sq', '*', 'c2', 'x1', '*', '+', 'c3', '+'],
     np.linspace(-3, 3, 150).reshape(-1, 1).astype(np.float32),
     lambda X: (2.0 * X.flatten()**2 + 3.0 * X.flatten() + 1.0).astype(np.float32),
     0.95),
    
    ("Exponential: y = 2*exp(0.5*x)",
     ['c1', 'c2', 'x1', '*', 'exp', '*'],
     np.linspace(0, 3, 100).reshape(-1, 1).astype(np.float32),
     lambda X: (2.0 * np.exp(0.5 * X.flatten())).astype(np.float32),
     0.90),
    
    ("Sinusoidal: y = 2*sin(3*x)",
     ['c1', 'c2', 'x1', '*', 'sin', '*'],
     np.linspace(0, 2*np.pi, 200).reshape(-1, 1).astype(np.float32),
     lambda X: (2.0 * np.sin(3.0 * X.flatten())).astype(np.float32),
     0.80),  # Lower threshold for sinusoidal
    
    ("Sinusoidal: y = 1.5*cos(2*x)",
     ['c1', 'c2', 'x1', '*', 'cos', '*'],
     np.linspace(0, 2*np.pi, 200).reshape(-1, 1).astype(np.float32),
     lambda X: (1.5 * np.cos(2.0 * X.flatten())).astype(np.float32),
     0.80),
    
    ("Multivariate: y = 2*x1 + 3*x2 + 1",
     ['c1', 'x1', '*', 'c2', 'x2', '*', '+', 'c3', '+'],
     np.random.randn(200, 2).astype(np.float32),
     lambda X: (2.0 * X[:, 0] + 3.0 * X[:, 1] + 1.0).astype(np.float32),
     0.95),
    
    ("Multivariate: y = x1*x2 + 2",
     ['c1', 'x1', 'x2', '*', '*', 'c2', '+'],
     np.random.randn(200, 2).astype(np.float32),
     lambda X: (X[:, 0] * X[:, 1] + 2.0).astype(np.float32),
     0.90),
    
    ("Logarithmic: y = 2*ln(x) + 1",
     ['c1', 'x1', 'log', '*', 'c2', '+'],
     np.linspace(0.1, 10, 150).reshape(-1, 1).astype(np.float32),
     lambda X: (2.0 * np.log(X.flatten()) + 1.0).astype(np.float32),
     0.95),
]

results = []
for name, tokens, X, y_fn, threshold in test_cases:
    y = y_fn(X)
    ids = [TOKEN2IDX[t] for t in tokens]
    skel = skeletonize(ids)
    
    if skel is None:
        print(f"  ❌ {name}: Skeletonize failed")
        results.append((name, None, 0.0, threshold, False))
        continue
        
    result = fit_and_score(skel, tokens, X, y, ['x1'] if X.shape[1] == 1 else ['x1', 'x2'], 
                           max_iter=100, n_restarts=5)
    
    r2 = result['r2']
    passed = r2 > threshold
    
    if passed:
        status = "✅"
    elif r2 > 0.5:
        status = "⚠️"
    else:
        status = "❌"
        
    print(f"  {status} {name}")
    print(f"      R² = {r2:.4f} (threshold: {threshold})")
    print(f"      Skeleton: {skel}")
    print(f"      Fitted: {result['expression']}")
    
    results.append((name, result['expression'], r2, threshold, passed))

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

passed = sum(1 for _, _, _, _, p in results if p)
total = len(results)

print(f"Passed: {passed}/{total}")
print()

for name, expr, r2, threshold, passed in results:
    status = "✅" if passed else "❌"
    print(f"  {status} {name}: R²={r2:.4f}")

print()
if passed == total:
    print("🎉 ALL TESTS PASSED!")
elif passed >= total * 0.8:
    print(f"✓ GOOD: {passed}/{total} tests passed")
else:
    print(f"⚠️  NEEDS IMPROVEMENT: {passed}/{total} tests passed")

print("=" * 70)
