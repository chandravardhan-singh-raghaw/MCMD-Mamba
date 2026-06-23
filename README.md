# MCMD-Mamba

Multi-Channel Multi-Direction Mamba for fundus image classification.

## Structure

- `configs/` — Training and model configs (default.yaml, model.yaml)
- `src/mcmd_mamba/` — Main package (config, models, training, utils)
- `tests/` — Tests for stage1, stage2, stage3, and E2E pipeline

## Quick Start

```bash
# Install
pip install -e .

# Run tests
PYTHONPATH=src python tests/test_stage1.py
PYTHONPATH=src python tests/test_stage2_exhaustive.py
PYTHONPATH=src python tests/test_stage3_exhaustive.py
PYTHONPATH=src python tests/test_stage1_stage2_stage3_e2e.py
```

## License

See [LICENSE](LICENSE).
