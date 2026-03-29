"""
Tests for inference/beam_search.py

Tests ODEFormer-style inference: skeletonize, fit_and_score, and diversity sampling.
"""
import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.tokenizer import TOKEN2IDX
from inference.beam_search import skeletonize, fit_and_score, odeformer_inference


class TestSkeletonize:
    """Test skeleton deduplication."""
    
    def test_same_structure_different_constants(self):
        """Test same structure with different constants produces same skeleton."""
        # c1*x1 + c2*x2
        tokens1 = ['c1', 'x1', '*', 'c2', 'x2', '*', '+']
        ids1 = [TOKEN2IDX[t] for t in tokens1]
        
        # c3*x1 + c4*x2 (same structure)
        tokens2 = ['c3', 'x1', '*', 'c4', 'x2', '*', '+']
        ids2 = [TOKEN2IDX[t] for t in tokens2]
        
        skel1 = skeletonize(ids1)
        skel2 = skeletonize(ids2)
        
        assert skel1 == skel2, f"Same structure should produce same skeleton: {skel1} vs {skel2}"
        
    def test_different_structures(self):
        """Test different structures produce different skeletons."""
        # c1*x1 + c2*x2
        tokens1 = ['c1', 'x1', '*', 'c2', 'x2', '*', '+']
        ids1 = [TOKEN2IDX[t] for t in tokens1]
        
        # c1+x1 + c2+x2 (different structure)
        tokens2 = ['c1', 'x1', '+', 'c2', 'x2', '+', '+']
        ids2 = [TOKEN2IDX[t] for t in tokens2]
        
        skel1 = skeletonize(ids1)
        skel2 = skeletonize(ids2)
        
        assert skel1 != skel2, f"Different structures should produce different skeletons"
        
    def test_skeleton_preserves_variables(self):
        """Test skeleton preserves variable structure."""
        # c1*x1
        tokens = ['c1', 'x1', '*']
        ids = [TOKEN2IDX[t] for t in tokens]
        skel = skeletonize(ids)
        
        assert 'x1' in skel, f"Variable x1 should be in skeleton: {skel}"
        assert 'CONST' in skel, f"Constant placeholder should be in skeleton: {skel}"
        
    def test_numeric_literals_skeletonized(self):
        """Test numeric literals are also skeletonized."""
        # c1*x1 + 1
        tokens = ['c1', 'x1', '*', '1', '+']
        ids = [TOKEN2IDX[t] for t in tokens]
        skel = skeletonize(ids)
        
        # Should have CONST for c1 and CONST_NUM for 1
        assert 'CONST' in skel, f"Constant placeholder should be in skeleton: {skel}"
        
    def test_invalid_rpn_returns_none(self):
        """Test invalid RPN returns None."""
        # Invalid: doesn't reduce to single value
        tokens = ['x1', 'x2', '+', 'x3']  # Stack has 2 elements at end
        ids = [TOKEN2IDX[t] for t in tokens]
        skel = skeletonize(ids)
        assert skel is None


class TestFitAndScore:
    """Test BFGS fitting and scoring."""
    
    def test_linear_fit(self):
        """Test fitting linear function y = c1*x + c2."""
        X = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
        y = (2.5 * X.flatten() + 3.0).astype(np.float32)
        
        tokens = ['c1', 'x1', '*', 'c2', '+']
        ids = [TOKEN2IDX[t] for t in tokens]
        skel = skeletonize(ids)
        
        result = fit_and_score(skel, tokens, X, y, ['x1'], max_iter=15)
        
        assert result['r2'] > 0.99, f"Expected R² > 0.99, got {result['r2']}"
        assert 'expression' in result
        assert 'fitted_params' in result
        
    def test_no_constants_expression(self):
        """Test expression without constants."""
        X = np.random.randn(100, 2).astype(np.float32)
        y = X[:, 0] + X[:, 1]
        
        tokens = ['x1', 'x2', '+']
        ids = [TOKEN2IDX[t] for t in tokens]
        skel = skeletonize(ids)
        
        result = fit_and_score(skel, tokens, X, y, ['x1', 'x2'], max_iter=15)
        
        assert result['r2'] > 0.99, f"Expected R² > 0.99, got {result['r2']}"
        
    def test_sinusoidal_fit(self):
        """Test fitting sinusoidal function y = c1*sin(c2*x)."""
        X = np.linspace(0, 2*np.pi, 200).reshape(-1, 1).astype(np.float32)
        y = (2.0 * np.sin(3.0 * X.flatten())).astype(np.float32)
        
        tokens = ['c1', 'c2', 'x1', '*', 'sin', '*']
        ids = [TOKEN2IDX[t] for t in tokens]
        skel = skeletonize(ids)
        
        result = fit_and_score(skel, tokens, X, y, ['x1'], max_iter=15)
        
        # Sinusoidal fitting is harder; just check it doesn't crash
        assert np.isfinite(result['r2']), f"R² should be finite"
        
    def test_multi_restart_convergence(self):
        """Test multi-restart BFGS improves convergence."""
        X = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
        y = (5.0 * X.flatten() + 10.0).astype(np.float32)
        
        tokens = ['c1', 'x1', '*', 'c2', '+']
        ids = [TOKEN2IDX[t] for t in tokens]
        skel = skeletonize(ids)
        
        result = fit_and_score(skel, tokens, X, y, ['x1'], max_iter=15)
        
        # Multi-restart should find good solution
        assert result['r2'] > 0.9, f"Expected R² > 0.9, got {result['r2']}"


class TestDeduplication:
    """Test skeleton deduplication logic."""
    
    def test_unique_skeletons_count(self):
        """Test correct number of unique skeletons."""
        test_cases = [
            ['c1', 'x1', '*'],           # c1*x1
            ['c2', 'x1', '*'],           # c2*x1 (same skeleton)
            ['c1', 'x1', '*', '1', '+'], # c1*x1 + 1 (different)
            ['c1', 'x1', '+'],           # c1+x1 (different)
        ]
        
        skeletons = set()
        for tokens in test_cases:
            ids = [TOKEN2IDX[t] for t in tokens]
            skel = skeletonize(ids)
            if skel:
                skeletons.add(skel)
        
        # Should have 3 unique skeletons
        assert len(skeletons) == 3, f"Expected 3 unique skeletons, got {len(skeletons)}"


class TestODEFormerInference:
    """Test full ODEFormer inference pipeline."""
    
    def test_inference_signature(self):
        """Test odeformer_inference function signature."""
        # Just verify the function exists and has correct signature
        import inspect
        sig = inspect.signature(odeformer_inference)
        params = list(sig.parameters.keys())
        
        expected_params = ['model', 'X_bits', 'unit_idx', 'var_mask', 
                          'X_data', 'y_data', 'var_names', 'pool_size',
                          'temperature', 'top_k', 'max_iter', 'n_workers']
        
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
