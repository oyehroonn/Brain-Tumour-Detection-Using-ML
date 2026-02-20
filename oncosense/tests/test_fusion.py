"""
Tests for EGU-Fusion module.
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.fusion import EGUFusion
from src.models.uncertainty import predictive_entropy


@pytest.fixture
def sample_data():
    """Create sample prediction data."""
    batch_size = 4
    num_classes = 4
    num_models = 3
    
    # Create probabilities from different "models"
    probs_list = [
        F.softmax(torch.randn(batch_size, num_classes), dim=1)
        for _ in range(num_models)
    ]
    
    # Create uncertainties
    uncertainties = torch.rand(batch_size, num_models)
    
    return probs_list, uncertainties


def test_egu_fusion_init():
    """Test EGUFusion initialization."""
    fusion = EGUFusion(alpha=1.0)
    
    assert fusion.alpha == 1.0
    assert fusion.max_prob_threshold == 0.6
    assert fusion.entropy_threshold == 1.2


def test_compute_weights(sample_data):
    """Test uncertainty-based weight computation."""
    probs_list, uncertainties = sample_data
    fusion = EGUFusion(alpha=1.0)
    
    weights = fusion.compute_weights(uncertainties)
    
    # Weights should sum to 1 for each sample
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)
    
    # Lower uncertainty should get higher weight
    # Create a case where model 0 has lowest uncertainty
    test_unc = torch.tensor([[0.1, 0.5, 0.9]])
    test_weights = fusion.compute_weights(test_unc)
    
    assert test_weights[0, 0] > test_weights[0, 1] > test_weights[0, 2]


def test_fuse(sample_data):
    """Test probability fusion."""
    probs_list, uncertainties = sample_data
    fusion = EGUFusion(alpha=1.0)
    
    fused_probs, weights = fusion.fuse(probs_list, uncertainties)
    
    # Fused probs should sum to 1
    assert torch.allclose(fused_probs.sum(dim=1), torch.ones(4), atol=1e-5)
    
    # Shape check
    assert fused_probs.shape == (4, 4)


def test_should_abstain(sample_data):
    """Test abstention logic."""
    fusion = EGUFusion(
        max_prob_threshold=0.6,
        entropy_threshold=1.2,
        abstention_mode="or"
    )
    
    # High confidence - should not abstain
    high_conf = torch.tensor([[0.9, 0.05, 0.03, 0.02]])
    assert not fusion.should_abstain(high_conf).item()
    
    # Low confidence - should abstain
    low_conf = torch.tensor([[0.3, 0.3, 0.2, 0.2]])
    assert fusion.should_abstain(low_conf).item()


def test_forward(sample_data):
    """Test full forward pass."""
    probs_list, uncertainties = sample_data
    fusion = EGUFusion(alpha=1.0)
    
    result = fusion(probs_list, uncertainties)
    
    assert "fused_probs" in result
    assert "predictions" in result
    assert "max_prob" in result
    assert "entropy" in result
    assert "weights" in result
    assert "abstain" in result
    
    assert result["fused_probs"].shape == (4, 4)
    assert result["predictions"].shape == (4,)
    assert result["abstain"].shape == (4,)


def test_alpha_effect():
    """Test that alpha parameter affects weight distribution."""
    batch_size = 1
    num_models = 3
    
    # Create uncertainties with clear differences
    uncertainties = torch.tensor([[0.1, 0.5, 0.9]])
    
    fusion_low_alpha = EGUFusion(alpha=0.5)
    fusion_high_alpha = EGUFusion(alpha=5.0)
    
    weights_low = fusion_low_alpha.compute_weights(uncertainties)
    weights_high = fusion_high_alpha.compute_weights(uncertainties)
    
    # High alpha should give more extreme weights (more concentration on best model)
    assert weights_high[0, 0] > weights_low[0, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
