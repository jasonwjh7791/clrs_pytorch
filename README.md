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

### Installation

- **Core library (PyTorch models, samplers, losses, evaluation):**

```bash
pip install -e .
```

This installs the `clrs_pytorch` package with only the dependencies needed
to:

- build PyTorch CLRS models,
- sample synthetic training data directly in PyTorch/JAX‑style, and
- run the unit tests.

- **Optional dataset support (TF/TFDS CLRS30 benchmark):**

The official CLRS30 dataset is distributed via TensorFlow Datasets. To enable
loading and chunking those TFDS trajectories, install the extra dataset
dependencies:

```bash
pip install -e ".[dataset]"
```

This enables:

- `clrs_pytorch.create_dataset(...)`
- `clrs_pytorch.create_chunked_dataset(...)`
- automatic downloading/unpacking of `CLRS30_v*.tar.gz` via
  `clrs_pytorch.get_dataset_gcp_url()` / `get_clrs_folder()`.

- **Developer extras (tests, linting, etc.):**

```bash
pip install -e ".[dev]"
pytest -q
```

### Quick start

#### Using samplers directly (no TFDS)

For most research on neural algorithmic reasoning, you do **not** need
TensorFlow – you can generate synthetic data on the fly using the built‑in
samplers:

```python
import numpy as np
import torch
import clrs_pytorch

# Build a sampler for BFS on graphs of length 16.
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

feedback = next(train_iter)          # CLRS feedback object
model = clrs_pytorch.Model(...)      # or use clrs_pytorch.models.BaselineModel
```

Here `feedback.features.inputs`, `feedback.features.hints`, and
`feedback.outputs` match the CLRS spec; the PyTorch losses and evaluation code
in `clrs_pytorch._src.losses` and `clrs_pytorch._src.evaluation` operate
directly on these objects.

#### Using the official CLRS30 benchmark dataset

To load the **official CLRS30 dataset** (pre-generated trajectories matching the
DeepMind paper's exact train/val/test splits), first install the dataset extras:

```bash
pip install -e ".[dataset]"
```

Then you can load it:

```python
import clrs_pytorch

# Load the official BFS training set
train_ds, num_samples, spec = clrs_pytorch.create_dataset(
    folder='/tmp/CLRS30',  # or any path where you want the dataset
    algorithm='bfs',
    split='train',
    batch_size=32
)

# Iterate over batches (as numpy arrays, compatible with PyTorch)
for feedback in train_ds.as_numpy_iterator():
    # feedback has the same structure as sampler output
    # feedback.features.inputs, feedback.features.hints, feedback.outputs
    loss = model(feedback, algo_idx=0)
    # ... training step ...
```

The dataset will be automatically downloaded from Google Cloud Storage on first
use. You can also manually download it:

```python
import requests
import shutil
import os

url = clrs_pytorch.get_dataset_gcp_url()  # Returns the GCP URL
folder = clrs_pytorch.get_clrs_folder()   # Returns 'CLRS30_v1.0.0'
# Download and extract manually if needed
```

**Note**: If you try to use `create_dataset()` or `create_chunked_dataset()`
without installing `.[dataset]`, you'll get a clear error message telling you
to install the dataset extras.

#### Using the training script

The high‑level training loop for CLRS baselines lives in
`clrs_pytorch.examples.run`. You can run it with:

```bash
python -m clrs_pytorch.examples.run \
  --algorithms=naive_string_matcher \
  --batch_size=16 \
  --train_steps=1000
```

This will:

- build PyTorch baseline models for the requested algorithms,
- construct train/val/test samplers (either synthetic or TFDS‑backed, depending
  on the `train_lengths`/`test_lengths` flags),
- train with hint supervision, and
- write checkpoints and JSON metrics under `artifacts/`.

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