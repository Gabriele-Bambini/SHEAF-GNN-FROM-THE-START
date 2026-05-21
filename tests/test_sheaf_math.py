"""
Test 8.1 — Matematica del Laplaciano Sheaf.

Verifica:
- L_F è positive semi-definite (PSD)
- Implementazione vettorizzata == loop
- Per d=1 e mappe identità, L_F si riduce al Laplaciano del grafo
- Diffusione non esplode per d grande
"""
import torch
import pytest


# ── helpers ────────────────────────────────────────────────────────────────

def _make_edge_index(edges):
    """Lista di tuple (src, tgt) → (2, E) tensor."""
    src, tgt = zip(*edges)
    return torch.tensor([list(src), list(tgt)], dtype=torch.long)


def _import_layers(N, E, d, edge_index, laplacian_eps: float = 0.0):
    """Build loop+vectorized layers. Default eps=0 so theoretical tests match exactly."""
    from tsgnn.model.sheaf import SheafDiffusionLayer
    from tsgnn.model.sheaf_vectorized import VectorizedSheafDiffusion
    loop = SheafDiffusionLayer(N, E, d, edge_index, laplacian_eps=laplacian_eps)
    vec  = VectorizedSheafDiffusion(N, E, d, edge_index, laplacian_eps=laplacian_eps)
    return loop, vec


# ── Test 1: PSD ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [1, 2, 4])
def test_laplacian_psd(d):
    """L_F deve avere tutti gli autovalori >= -epsilon (PSD)."""
    N, E = 5, 6
    edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(0,2)]
    edge_index = _make_edge_index(edges)
    _, vec = _import_layers(N, E, d, edge_index)

    torch.manual_seed(0)
    maps = torch.randn(E, 2, d, d)
    L = vec.compute_connection_laplacian_vectorized(maps)

    eigvals = torch.linalg.eigvalsh(L)
    min_eig = eigvals.min().item()
    assert min_eig >= -1e-4, (
        f"L_F non PSD per d={d}: autovalore minimo = {min_eig:.6f}"
    )


# ── Test 2: vectorized == loop ───────────────────────────────────────────────

@pytest.mark.parametrize("d", [1, 2, 3])
def test_vectorized_matches_loop(d):
    """Implementazione vettorizzata deve matchare il loop per ogni d."""
    torch.manual_seed(42)
    N, E = 6, 8
    edge_index = torch.randint(0, N, (2, E))
    loop_layer, vec_layer = _import_layers(N, E, d, edge_index)

    maps = torch.randn(E, 2, d, d)
    L_loop = loop_layer.compute_connection_laplacian(maps)
    L_vec  = vec_layer.compute_connection_laplacian_vectorized(maps)

    assert torch.allclose(L_loop, L_vec, atol=1e-5), (
        f"Mismatch loop vs vectorized per d={d}. "
        f"Max diff: {(L_loop - L_vec).abs().max():.2e}"
    )


# ── Test 3: riduzione a graph Laplacian ─────────────────────────────────────

def test_laplacian_reduces_to_graph_laplacian_d1():
    """For d=1 and identity restriction maps on a triangle, verify sheaf structure.

    Mathematical note: the sheaf Laplacian uses the formula
        L_F[v,v] = Σ_e F_{v,e}^T F_{v,e}     (diagonal blocks)
        L_F[u,v] = -F_{u,e}^T F_{v,e}          (off-diagonal blocks)

    For d=1, F = scalar. With 6 directed edges (each undirected edge split into
    two directed edges), each node v has degree 4 in the directed graph
    (2 out + 2 in), so L_F[v,v] = 4 and L_F[u,v] = -2 for adjacent u,v.
    This equals 2 * L_graph for the underlying undirected graph.

    We verify this factor-2 relationship explicitly.
    """
    # Triangle: every undirected edge represented as two directed edges
    edges = [(0,1),(1,0),(1,2),(2,1),(2,0),(0,2)]
    N, E, d = 3, 6, 1
    edge_index = _make_edge_index(edges)
    _, vec = _import_layers(N, E, d, edge_index)

    # Identity maps (scalar 1.0 for every directed arc)
    maps = torch.ones(E, 2, 1, 1)
    L_F = vec.compute_connection_laplacian_vectorized(maps)

    # Underlying undirected graph Laplacian (triangle: each node has degree 2)
    A = torch.zeros(N, N)
    for s, t in edges:
        A[s, t] = 1.0
    D = torch.diag(A.sum(dim=1))
    L_graph = D - A

    # With bidirectional directed edges, L_F = 2 * L_graph
    assert torch.allclose(L_F, 2.0 * L_graph, atol=1e-5), (
        f"Expected L_F = 2 * L_graph for bidirectional triangle with identity maps.\n"
        f"L_F:\n{L_F}\n2*L_graph:\n{2.0 * L_graph}"
    )



# ── Test 4: diffusione stabile ───────────────────────────────────────────────

@pytest.mark.parametrize("d", [2, 4, 8])
def test_diffusion_numerical_stability(d):
    """x dopo diffusione non deve esplodere (no NaN, no Inf, norma ragionevole)."""
    torch.manual_seed(7)
    N, E = 10, 15
    edge_index = torch.randint(0, N, (2, E))
    _, vec = _import_layers(N, E, d, edge_index)

    maps = torch.randn(E, 2, d, d) * 0.1   # piccole mappe per stabilità
    L_F  = vec.compute_connection_laplacian_vectorized(maps)
    x    = torch.randn(N, d)

    x_out = vec.diffuse(x, L_F)

    assert not torch.isnan(x_out).any(), "NaN dopo diffusione"
    assert not torch.isinf(x_out).any(), "Inf dopo diffusione"
    assert x_out.norm().item() < 1e4,    f"Norma esplosiva: {x_out.norm():.2e}"


# ── Test 5: simmetria del Laplaciano ────────────────────────────────────────

def test_laplacian_symmetry():
    """L_F deve essere simmetrico (L = L^T)."""
    torch.manual_seed(1)
    N, E, d = 5, 7, 2
    edge_index = torch.randint(0, N, (2, E))
    _, vec = _import_layers(N, E, d, edge_index)

    maps = torch.randn(E, 2, d, d)
    L = vec.compute_connection_laplacian_vectorized(maps)

    assert torch.allclose(L, L.T, atol=1e-5), (
        f"L_F non simmetrico. Max asimmetria: {(L - L.T).abs().max():.2e}"
    )
