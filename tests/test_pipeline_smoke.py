"""Smoke test end-to-end su dati sintetici piccoli."""
import torch
from tsgnn.model.tsgnn import TSGNN
from tsgnn.training.loss import TSGNNLoss
from tsgnn.training.trainer import create_synthetic_training_data

def test_forward_pass_smoke():
    """Il modello deve completare un forward pass senza errori."""
    N, E, d, K = 10, 20, 2, 3
    edge_index = torch.randint(0, N, (2, E))
    model = TSGNN(
        num_nodes=N, num_edges=E, stalk_dim=d, input_dim=N, 
        esm_dim=32, conditioning_dim=16, edge_index=edge_index, 
        num_diffusion_steps=1
    )
    node_seq  = torch.randn(K, N, N)
    allele_emb = torch.randn(32)
    preds, maps, laps = model(node_seq, allele_emb)
    assert len(preds) == K
    assert preds[0].shape == (N, N)

def test_backward_pass_smoke():
    """I gradienti devono fluire attraverso tutto il modello."""
    N, E, d, K = 8, 15, 2, 3
    edge_index = torch.randint(0, N, (2, E))
    model = TSGNN(
        num_nodes=N, num_edges=E, stalk_dim=d, input_dim=N, 
        esm_dim=32, conditioning_dim=16, edge_index=edge_index, 
        num_diffusion_steps=1
    )
    node_seq   = torch.randn(K, N, N)
    allele_emb = torch.randn(32, requires_grad=True)
    preds, _, _ = model(node_seq, allele_emb)
    loss = sum(p.sum() for p in preds)
    loss.backward()
    assert allele_emb.grad is not None, "Gradiente non fluisce all'allele embedding"
