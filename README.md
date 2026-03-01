## PyTorch CLRS Algorithmic Reasoning Benchmark

This repository is an **unofficial PyTorch port** of the
[CLRS Algorithmic Reasoning Benchmark](https://github.com/google-deepmind/clrs)
from DeepMind. The original implementation is in JAX; this project re‑implements
the baselines and evaluation pipeline in PyTorch while preserving:

- **The same set of 30 CLRS algorithms** (sorting, searching, dynamic programming,
  graph, string, and geometry routines).
- **The same supervision signals** (inputs, hints, outputs) and evaluation
  metrics.
- **A processor–encoder–decoder architecture**, but implemented in pure PyTorch.

The code and documentation in this repository are derived from and remain
compatible with the original CLRS design, but this project is **not affiliated
with or endorsed by DeepMind**.

---

### Main entry point: `examples/run.py`

**The main way to use this repo is the training script** `clrs_pytorch.examples.run`.
It trains the PyTorch CLRS baseline on one or more algorithms with synthetic
data (no TensorFlow required by default).

**From the command line:**

```bash
# Install the package first
pip install -e .

# Train on BFS (short run; uses synthetic data)
python -m clrs_pytorch.examples.run \
  --algorithms=bfs \
  --batch_size=16 \
  --train_steps=1000 \
  --eval_every=50 \
  --test_lengths=16
```

- **`--algorithms`**: Comma-separated list, e.g. `bfs`, `dfs`, `naive_string_matcher`, `quicksort`. See the 30 algorithm names in the repo.
- **`--train_lengths`**: Problem sizes used for training (default `4,7,11,13,16`). Use `-1` only if you have installed the optional dataset extras and want the official CLRS30 benchmark data.
- **`--test_lengths`**: Problem sizes for the final test evaluation. Use **`16`** (or any positive size) to run test on synthetic data without needing TensorFlow. Omit or use `-1` only if you installed `.[dataset]` and want the official test set.
- **`--checkpoint_path`**: Where to save the best model (default `artifacts/checkpoints/checkpoint.pth`).
- **`--performance_path`**: Where to save validation/test scores (default `artifacts/metrics/performance.json`).

Checkpoints and metrics are written under `artifacts/`; the script creates the directories if they don’t exist.

---

### Google Colab: copy‑paste script

Run the following in a Colab cell to clone the repo, install, and train on BFS (short run). No GPU required for this small example.

```python
# Clone and install (run once)
!git clone https://github.com/YOUR_USERNAME/clrs_pytorch.git
%cd clrs_pytorch
!pip install -e .

# Train on BFS for 100 steps (synthetic data; no TFDS)
!python -m clrs_pytorch.examples.run \
  --algorithms=bfs \
  --batch_size=8 \
  --train_steps=100 \
  --eval_every=25 \
  --test_lengths=16
```

Replace `YOUR_USERNAME/clrs_pytorch` with your fork or the actual repo URL. To use the official CLRS30 dataset in Colab, first run `!pip install -e ".[dataset]"` and then use `--train_lengths=-1` and `--test_lengths=-1` (and set `--dataset_path` if needed).

---

### Installation

- **Core (recommended for most users):**

```bash
pip install -e .
```

This is enough to run `clrs_pytorch.examples.run` with **synthetic data** (no TensorFlow).

- **Optional – official CLRS30 dataset (TensorFlow + TFDS):**

If you want to load the pre-generated CLRS30 benchmark data:

```bash
pip install -e ".[dataset]"
```

Then you can use `--train_lengths=-1` and `--test_lengths=-1` in `run.py` to use that data. See section “Using the official CLRS30 benchmark dataset” below.

- **Optional – development (tests):**

```bash
pip install -e ".[dev]"
pytest -q
```

---

### Quick start (programmatic)

#### Using samplers directly (no TFDS)

You can also drive training yourself with the public API:

```python
import clrs_pytorch

sampler, spec = clrs_pytorch.build_sampler(
    name='bfs',
    seed=42,
    num_samples=1_000,
    length=16,
)

def iterate_sampler(batch_size: int):
    while True:
        yield sampler.next(batch_size)

train_iter = iterate_sampler(batch_size=32)
feedback = next(train_iter)  # Use with clrs_pytorch.models.BaselineModel, etc.
```

#### Using the official CLRS30 benchmark dataset

After `pip install -e ".[dataset]"`:

```python
import clrs_pytorch

train_ds, num_samples, spec = clrs_pytorch.create_dataset(
    folder='/tmp/CLRS30',
    algorithm='bfs',
    split='train',
    batch_size=32
)
for feedback in train_ds.as_numpy_iterator():
    # train step
    ...
```

Dataset is downloaded from GCP on first use. URL and folder name: `clrs_pytorch.get_dataset_gcp_url()`, `clrs_pytorch.get_clrs_folder()`.

### Dependencies

The **core** `requirements/requirements.txt` are intentionally minimal and
PyTorch‑centric:

- `torch` – main deep learning backend
- `jax`, `jaxlib`, `opt_einsum` – lightweight use in some chunking utilities
- `absl-py`, `attrs`, `numpy`, `toolz`, `six`, `requests` – configuration,
  array handling, and utilities

Optional extras:

- `.[dataset]` – TensorFlow + TFDS support to reproduce the official CLRS30
  trajectories and benchmarking protocol.
- `.[dev]` – `pytest` and related tooling for running the full test suite.

### Tests

All unit tests live alongside the code under `clrs_pytorch/_src` and
`clrs_pytorch/examples`. From a clean checkout:

```bash
pip install -e ".[dev]"
pytest -q
```

At the time of writing, **151 tests pass** covering:

- all 30 algorithms’ PyTorch implementations,
- samplers and chunking,
- encoders/decoders/processors,
- loss and evaluation utilities, and
- the example training loop.

### Relationship to the original CLRS repo

This project is a PyTorch re‑implementation of the
[DeepMind CLRS benchmark](https://github.com/google-deepmind/clrs). Most
interfaces (algorithm names, specs, feedback structure) are intentionally kept
compatible so that:

- results can be compared directly between the JAX and PyTorch versions, and
- pretrained weights can be ported or initialized consistently where desired.

If you use this repository in academic work, please **cite the original CLRS
papers**:

```latex
@article{deepmind2022clrs,
  title   = {The CLRS Algorithmic Reasoning Benchmark},
  author  = {Petar Veli\v{c}kovi\'{c} and Adri\`{a} Puigdom\`{e}nech Badia and
             David Budden and Razvan Pascanu and Andrea Banino and Misha Dashevskiy and
             Raia Hadsell and Charles Blundell},
  journal = {arXiv preprint arXiv:2205.15659},
  year    = {2022}
}

@article{deepmind2024clrstext,
  title   = {The CLRS-Text Algorithmic Reasoning Language Benchmark},
  author  = {Larisa Markeeva and Sean McLeish and Borja Ibarz and Wilfried Bounsi
             and Olga Kozlova and Alex Vitvitskyi and Charles Blundell and
             Tom Goldstein and Avi Schwarzschild and Petar Veli\v{c}kovi\'{c}},
  journal = {arXiv preprint arXiv:2406.04229},
  year    = {2024}
}
```

### License

This repository is distributed under the **Apache 2.0** license; see `LICENSE`
for details. Much of the design and some code structure are adapted from the
original CLRS JAX implementation by DeepMind, also licensed under Apache 2.0
([upstream repository](https://github.com/google-deepmind/clrs)).