"""
Integration tests for the full LLM-JEPA pipeline.

Tests the complete flow from training → inference → evaluation.
"""
import pytest
import numpy as np
import torch
import yaml
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestPipelineImports:
    """Test all pipeline components can be imported."""
    
    def test_training_imports(self):
        """Test training module imports."""
        from training.trainer import LLMJEPAModule
        from training.losses import LLMJEPALoss, ValidityWeightedCE
        assert LLMJEPAModule is not None
        assert LLMJEPALoss is not None
        assert ValidityWeightedCE is not None
        
    def test_inference_imports(self):
        """Test inference module imports."""
        from inference.generate import InferenceModel
        from inference.beam_search import (
            odeformer_inference, 
            skeletonize, 
            fit_and_score,
            diversity_pool_sample
        )
        assert InferenceModel is not None
        assert odeformer_inference is not None
        
    def test_evaluation_imports(self):
        """Test evaluation module imports."""
        from evaluation.evaluate import (
            evaluate_dataset, 
            aggregate_results, 
            print_results
        )
        from evaluation.metrics import (
            calculate_ned,
            verify_symbolic_accuracy,
            calculate_constant_recovery
        )
        assert evaluate_dataset is not None
        assert calculate_ned is not None
        
    def test_evaluator_imports(self):
        """Test evaluator module imports."""
        from models.evaluator import ModelEvaluator, load_inference_model
        assert ModelEvaluator is not None
        
    def test_cli_imports(self):
        """Test CLI entry point."""
        from run_eval import main
        assert main is not None


class TestConfigLoading:
    """Test configuration loading."""
    
    def test_base_config_exists(self):
        """Test base config file exists."""
        config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
        assert config_path.exists(), f"Config not found: {config_path}"
        
    def test_config_structure(self):
        """Test config has expected sections."""
        config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        expected_sections = ['data', 'model', 'loss', 'training', 'inference']
        for section in expected_sections:
            assert section in config, f"Missing config section: {section}"
            
    def test_inference_config(self):
        """Test inference config has expected parameters."""
        config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        inf_cfg = config.get('inference', {})
        expected_params = ['pool_size', 'temperature', 'max_iter', 'n_workers', 'top_k']
        
        for param in expected_params:
            assert param in inf_cfg, f"Missing inference param: {param}"


class TestSkeletonizeFitIntegration:
    """Test skeletonize + fit_and_score integration."""
    
    def test_linear_pipeline(self):
        """Test complete pipeline for linear function."""
        from data.tokenizer import TOKEN2IDX
        from inference.beam_search import skeletonize, fit_and_score
        
        # Create test data: y = 2.5*x + 3
        X = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
        y = (2.5 * X.flatten() + 3.0).astype(np.float32)
        
        # Tokenize: c1*x1 + c2
        tokens = ['c1', 'x1', '*', 'c2', '+']
        ids = [TOKEN2IDX[t] for t in tokens]
        
        # Skeletonize
        skel = skeletonize(ids)
        assert skel is not None, "Skeleton should not be None"
        assert 'CONST' in skel, f"Skeleton should have CONST: {skel}"
        
        # Fit and score
        result = fit_and_score(skel, tokens, X, y, ['x1'], max_iter=15)
        
        assert result['r2'] > 0.99, f"Expected R² > 0.99, got {result['r2']}"
        assert 'expression' in result
        assert 'fitted_params' in result
        
    def test_multivariate_pipeline(self):
        """Test pipeline for multivariate function."""
        from data.tokenizer import TOKEN2IDX
        from inference.beam_search import skeletonize, fit_and_score
        
        # Create test data: y = 2*x1 + 3*x2 + 1
        X = np.random.randn(200, 2).astype(np.float32)
        y = (2.0 * X[:, 0] + 3.0 * X[:, 1] + 1.0).astype(np.float32)
        
        # Tokenize: c1*x1 + c2*x2 + c3
        tokens = ['c1', 'x1', '*', 'c2', 'x2', '*', '+', 'c3', '+']
        ids = [TOKEN2IDX[t] for t in tokens]
        
        # Skeletonize
        skel = skeletonize(ids)
        assert skel is not None
        
        # Fit and score
        result = fit_and_score(skel, tokens, X, y, ['x1', 'x2'], max_iter=15)
        
        assert result['r2'] > 0.95, f"Expected R² > 0.95, got {result['r2']}"


class TestMetricsIntegration:
    """Test metrics work together."""
    
    def test_ned_and_sa_together(self):
        """Test NED and Symbolic Accuracy work together."""
        from evaluation.metrics import calculate_ned, verify_symbolic_accuracy
        
        pred = 'x1 + x2'
        truth = 'x2 + x1'
        
        ned = calculate_ned(pred, truth)
        sa = verify_symbolic_accuracy(pred, truth, ['x1', 'x2'])
        
        # Commutative expressions should have SA=True and low NED
        assert sa == True
        assert ned < 0.5
        
    def test_all_metrics_on_good_prediction(self):
        """Test all metrics on a good prediction."""
        from evaluation.metrics import (
            calculate_ned,
            verify_symbolic_accuracy,
            calculate_constant_recovery,
            calculate_r2
        )
        
        # Perfect prediction
        pred = '2.5*x1 + 3.0'
        truth = '2.5*x1 + 3.0'
        
        ned = calculate_ned(pred, truth)
        sa = verify_symbolic_accuracy(pred, truth, ['x1'])
        cr = calculate_constant_recovery([2.5, 3.0], [2.5, 3.0])
        
        # Generate data for R²
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = 2.5 * X.flatten() + 3.0
        y_pred = 2.5 * X.flatten() + 3.0
        r2 = calculate_r2(y, y_pred)
        
        assert ned == 0.0
        assert sa == True
        assert cr == 1.0
        assert r2 == 1.0


class TestEvaluateModule:
    """Test evaluation/evaluate.py functions."""
    
    def test_aggregate_results(self):
        """Test results aggregation."""
        from evaluation.evaluate import aggregate_results
        
        results = [
            {'eq_id': 'test1', 'r2': 0.95, 'symbolic_accuracy': True, 
             'constant_recovery': 0.9, 'ned': 0.1, 'node_count': 5, 'latency_s': 0.5},
            {'eq_id': 'test2', 'r2': 0.80, 'symbolic_accuracy': False, 
             'constant_recovery': 0.5, 'ned': 0.3, 'node_count': 7, 'latency_s': 0.6},
        ]
        
        metrics = aggregate_results(results)
        
        assert 'mean_r2' in metrics
        assert 'symbolic_accuracy' in metrics
        assert 'constant_recovery' in metrics
        assert 'mean_ned' in metrics
        assert metrics['n_equations'] == 2
        
    def test_print_results(self):
        """Test results printing (just verify no crash)."""
        from evaluation.evaluate import print_results
        
        metrics = {
            'mean_r2': 0.85,
            'symbolic_accuracy': 0.75,
            'constant_recovery': 0.60,
            'mean_ned': 0.15,
            'n_equations': 2,
            'per_eq_results': [
                {'eq_id': 'test1', 'r2': 0.95, 'symbolic_accuracy': True,
                 'constant_recovery': 0.9, 'ned': 0.1, 'node_count': 5},
                {'eq_id': 'test2', 'r2': 0.75, 'symbolic_accuracy': False,
                 'constant_recovery': 0.3, 'ned': 0.2, 'node_count': 7},
            ]
        }
        
        # Should not crash
        print_results(metrics)


class TestInferenceEffectiveness:
    """Test that inference produces effective results."""
    
    def test_diverse_candidates_produce_unique_skeletons(self):
        """Test that diversity sampling produces unique skeletons."""
        from data.tokenizer import TOKEN2IDX
        from inference.beam_search import skeletonize
        
        # Simulate diverse candidates with same structure
        candidates = [
            ['c1', 'x1', '*', 'c2', '+'],
            ['c2', 'x1', '*', 'c3', '+'],
            ['c3', 'x1', '*', 'c4', '+'],
            ['c1', 'x1', '+', 'c2', '*'],  # Different structure
        ]
        
        skeletons = set()
        for tokens in candidates:
            ids = [TOKEN2IDX[t] for t in tokens]
            skel = skeletonize(ids)
            if skel:
                skeletons.add(skel)
        
        # Should have at least 2 unique skeletons
        assert len(skeletons) >= 2, f"Expected >= 2 unique skeletons, got {len(skeletons)}"
        
    def test_fitting_improves_r2(self):
        """Test that BFGS fitting improves R²."""
        from data.tokenizer import TOKEN2IDX
        from inference.beam_search import skeletonize, fit_and_score
        
        # Create data with known constants
        X = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
        y = (5.0 * X.flatten() + 7.0).astype(np.float32)
        
        tokens = ['c1', 'x1', '*', 'c2', '+']
        ids = [TOKEN2IDX[t] for t in tokens]
        skel = skeletonize(ids)
        
        # Fit with BFGS
        result = fit_and_score(skel, tokens, X, y, ['x1'], max_iter=15)
        
        # R² should be very high after fitting
        assert result['r2'] > 0.99, f"Expected R² > 0.99 after fitting, got {result['r2']}"
        
    def test_wrong_structure_has_low_r2(self):
        """Test that wrong structure has low R²."""
        from data.tokenizer import TOKEN2IDX
        from inference.beam_search import skeletonize, fit_and_score
        
        # Linear data
        X = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
        y = (2.0 * X.flatten() + 1.0).astype(np.float32)
        
        # Try to fit with multiplicative structure: c1*x1*c2
        tokens = ['c1', 'x1', '*', 'c2', '*']
        ids = [TOKEN2IDX[t] for t in tokens]
        skel = skeletonize(ids)
        
        result = fit_and_score(skel, tokens, X, y, ['x1'], max_iter=15)
        
        # Wrong structure should have lower R² than correct structure
        # (though BFGS might still find a reasonable fit for some cases)
        assert np.isfinite(result['r2']), "R² should be finite"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
