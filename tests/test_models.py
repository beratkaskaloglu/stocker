"""
tests/test_models.py
Model forward pass şekil testleri.
"""
import pytest
import torch


@pytest.fixture
def dummy_input():
    """batch=4, seq_len=60, feature_dim=211"""
    return torch.randn(4, 60, 211)


@pytest.fixture
def dummy_gasf_input():
    """batch=4, channels=3, H=64, W=64"""
    return torch.randn(4, 3, 64, 64)


class TestLSTMAttention:
    def test_output_shapes(self, dummy_input):
        from core.models.lstm_attention import LSTMAttentionModel
        model = LSTMAttentionModel()
        # TODO: output = model(dummy_input)
        # assert output["direction_logits"].shape == (4, 3)
        # assert output["price"].shape == (4, 1)
        # assert output["confidence"].shape == (4, 1)
        pytest.skip("Not implemented yet")


class TestResNetGASF:
    def test_output_shapes(self, dummy_gasf_input):
        from core.models.resnet_gasf import ResNetGASFModel
        model = ResNetGASFModel()
        # TODO: output = model(dummy_gasf_input)
        # assert output["direction_logits"].shape == (4, 3)
        pytest.skip("Not implemented yet")
