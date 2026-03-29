"""
Pytest configuration for LLM-JEPA tests.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


@pytest.fixture(scope="session")
def test_data():
    """Provide test data for all tests."""
    import numpy as np
    
    # Linear test data: y = 2.5*x + 3
    X_linear = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
    y_linear = (2.5 * X_linear.flatten() + 3.0).astype(np.float32)
    
    # Sinusoidal test data: y = 2*sin(3*x)
    X_sin = np.linspace(0, 2*np.pi, 200).reshape(-1, 1).astype(np.float32)
    y_sin = (2.0 * np.sin(3.0 * X_sin.flatten())).astype(np.float32)
    
    # Multivariate test data: y = 2*x1 + 3*x2 + 1
    X_multi = np.random.randn(200, 2).astype(np.float32)
    y_multi = (2.0 * X_multi[:, 0] + 3.0 * X_multi[:, 1] + 1.0).astype(np.float32)
    
    return {
        'linear': (X_linear, y_linear),
        'sinusoidal': (X_sin, y_sin),
        'multivariate': (X_multi, y_multi),
    }
