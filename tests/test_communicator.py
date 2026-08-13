import torch
import pytest
from colonyml.communicator import NodeCommunicator


def test_simulate_allreduce_basic():
    gradients = [
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([5.0, 6.0, 7.0, 8.0]),
        torch.tensor([9.0, 10.0, 11.0, 12.0])
    ]
    result = NodeCommunicator.simulate_allreduce(gradients)
    expected = torch.tensor([5.0, 6.0, 7.0, 8.0])
    assert torch.allclose(result, expected)


def test_simulate_allreduce_two_nodes():
    gradients = [
        torch.tensor([2.0, 4.0]),
        torch.tensor([6.0, 8.0])
    ]
    result = NodeCommunicator.simulate_allreduce(gradients)
    expected = torch.tensor([4.0, 6.0])
    assert torch.allclose(result, expected)


def test_simulate_allreduce_single_node():
    gradients = [torch.tensor([1.0, 2.0, 3.0])]
    result = NodeCommunicator.simulate_allreduce(gradients)
    assert torch.allclose(result, gradients[0])


def test_simulate_allreduce_preserves_shape():
    gradients = [torch.randn(32, 64) for _ in range(4)]
    result = NodeCommunicator.simulate_allreduce(gradients)
    assert result.shape == gradients[0].shape


def test_simulate_allreduce_large_tensor():
    gradients = [torch.randn(1000, 1000) for _ in range(3)]
    result = NodeCommunicator.simulate_allreduce(gradients)
    expected = torch.stack(gradients).mean(dim=0)
    assert torch.allclose(result, expected, atol=1e-5)


def test_simulate_allreduce_empty_raises():
    with pytest.raises((ValueError, Exception)):
        NodeCommunicator.simulate_allreduce([])


def test_simulate_allreduce_identical_gradients():
    g = torch.tensor([3.0, 6.0, 9.0])
    gradients = [g.clone() for _ in range(5)]
    result = NodeCommunicator.simulate_allreduce(gradients)
    assert torch.allclose(result, g)


def test_simulate_allreduce_zeros():
    gradients = [torch.zeros(100) for _ in range(3)]
    result = NodeCommunicator.simulate_allreduce(gradients)
    assert torch.allclose(result, torch.zeros(100))