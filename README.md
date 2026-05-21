# Temporal Sheaf Neural Networks (TS-GNN)

Sheaf neural networks for temporal biological systems and gene regulatory networks.

## Quick Start

### Install
```bash
pip install -e .
```

### Run HPC Job
```bash
cd hpc/
./submit_smoke.sbatch  # Smoke test
./submit_real_hpo_smoke.sbatch  # Full validation
```

## Key Architecture

- **L0**: MLP baseline
- **L1**: GCN backbone
- **L2**: Static sheaf (Cayley maps)
- **L3**: Sheaf with Cayley orthogonal transforms
- **L4**: ESM-informed sheaf
- **L5**: Temporal ODE sheaf

Final validated: **A1+A2+B2** (A3/B1 dropped post-ablation)

## Datasets

- **TP53**: Synthetic + real tumor mutants (BRCA/CRC)
- **Ovarian**: Gene expression + clinical outcome prediction
- **Cross-dataset transfer**: BRCA→Ovarian validation

## Recent Results

HPC ablation job 482916 (A100 GPU):
- Baseline: r_δ = +0.3523
- A1 (orthogonal maps): neutral
- A2 (implicit diffusion): +0.20 boost
- A3 (predict_delta): **dropped** (-0.43 penalty)
- B1 (nodal FiLM): **dropped** (-0.40 penalty)
- B2 (neural ODE): neutral, kept

## Project Structure

```
src/
  sheaf_ts/       # Core sheaf neural network
  tsgnn/          # Temporal GNN + biology integrations
hpc/              # SLURM job submission + monitoring
configs/          # YAML training configs
tests/            # Unit + integration tests
scripts/          # Analysis & evaluation utilities
```

## References

See `docs/ABLATION_REPORT.md` for full HPC validation results.
