<div align="center">
  <h1>RocketStack: Level-Aware Deep Recursive Ensemble Learning Architecture</h1>
  <p>Official reference implementation
  <p>
    <a href="https://doi.org/10.1016/j.eswa.2026.133521">Research article</a>
    ·
    <a href="https://caghangir-1.github.io/RocketStack-Ensemble-Architecture-ai/">Documentation and API</a>
    ·
    <a href="https://github.com/caghangir-1/RocketStack-Ensemble-Architecture-ai">Source code</a>
  </p>
</div>

## 📚 Citation
If you intend to use this architecture or codebase in your research, please cite the following publication:

Demirel, Ç. (2026). RocketStack: Level-aware deep recursive ensemble learning architecture. *Expert Systems with Applications*, Article 133521. https://doi.org/10.1016/j.eswa.2026.133521

The repository provides the canonical, versioned implementation of the
architecture. The project website provides its public documentation and API
reference.

## System overview

RocketStack extends horizontal ensemble diversity into a vertically recursive
stacking process. Its objective is to increase representational depth while
controlling computational growth and premature overfitting. At each recursive
level, out-of-fold predictions are reintegrated with the current feature
representation, periodic feature compression manages dimensionality, and
stochastic model pruning narrows the active learner pool.

The framework supports binary and multiclass classification and provides both
trainable, scikit-learn-like models and the complete exploratory routines used
in the research implementation.

### Architecture highlights

- **Recursive stacking:** OOF probability features are propagated through
  successive meta-learning levels.
- **Level-aware pruning:** The blurrization heuristic introduces controlled
  score perturbation and removes weak learners between levels.
- **Intermediate feature processing:** Stochastic Feature Elimination,
  autoencoder reduction, and attention-based selection are supported.
- **Diverse learner pool:** The implementation integrates scikit-learn,
  XGBoost, LightGBM, and CatBoost estimators.
- **Research and trainable APIs:** Reproducibility-oriented exploration
  routines coexist with reusable `fit`, `predict`, `predict_proba`, and `score`
  workflows.

## Installation

RocketStack supports Python 3.10–3.13.

After the first PyPI release:

```bash
python -m pip install rocketstack_ensemble_ai
```

The current repository version can be installed directly from GitHub:

```bash
python -m pip install "git+https://github.com/caghangir-1/RocketStack-Ensemble-Architecture-ai.git"
```

To install a local checkout:

```bash
python -m pip install .
```

The complete model pool uses NumPy, scikit-learn, XGBoost, LightGBM, CatBoost,
Optuna, and TensorFlow. These dependencies are installed automatically.
scikit-learn is currently constrained below version 1.8 because the research
implementation uses the legacy `AdaBoostClassifier` `algorithm` parameter.

## Quick start

### Binary classification

```python
from rocketstack_ensemble_ai import ascentTheRocket_binary_model

model = ascentTheRocket_binary_model(
    level=3,
    return_model="best",
    iffeatselection=False,
    blur_strength="light",
    n_jobs=-1,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
```

### Multiclass classification

```python
from rocketstack_ensemble_ai import ascentTheRocket_multiclass_model

model = ascentTheRocket_multiclass_model(
    level=3,
    return_model="best",
    iffeatselection_or_not=False,
    n_jobs=-1,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Complete class-based API

```python
from rocketstack_ensemble_ai import RocketStack

rocket = RocketStack()
model = rocket.ascentTheRocket_binary_model(level=3, return_model="best")
model.fit(X_train, y_train)
```

The original research workflows remain available through
`AscentTheRocket_binary_massive_exploration` and
`AscentTheRocket_multiclass_massive_exploration`.

## Documentation

The complete documentation and API reference are available at:

**[RocketStack Official Documentation](https://caghangir-1.github.io/RocketStack-Ensemble-Architecture-ai/)**

The GitHub Pages website is the presentation and documentation layer. This
repository is the canonical source-code, issue-tracking, and release layer.

## Versioning and development status

The installable API begins at version `0.1.0` while it undergoes broader public
validation before a stable `1.0.0` release. RocketStack follows semantic
versioning, and the installed version is available as
`rocketstack_ensemble_ai.__version__`.

Development checks can be run with:

```bash
python -m pytest
python -m build
python -m twine check dist/*
```

## License

Copyright © 2026 Çağatay Demirel. RocketStack is distributed under the GNU
General Public License v3.0 only (`GPL-3.0-only`).
