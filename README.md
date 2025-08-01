# GLM-HMM Modeling of IBL Mouse Behavior

This repository adapts Zoe Ashwood’s GLM-HMM model for state-dependent behavior to work on **any IBL mouse**. It enables model fitting, evaluation, and visualization of behavioral state dynamics for IBL animals (pooling sessions), resulting in a probability of engagement per trial.

Original modeling framework:\
[https://github.com/zashwood/glm-hmm](https://github.com/zashwood/glm-hmm)

---

## Key Features

- Adapted to work with **any IBL animal**, both biased and unbiased paradigms
- Computed for BWM data through the [`brainwidemap`](https://github.com/int-brain-lab/brainwidemap) library
- Posterior state inference and GLM-HMM parameter fitting
- Utilities for:
  - Animal-wise model fitting
  - Visualizing generative weights and state transitions
  - Pooling and analyzing empirical and expected dwell times

---

## Installation

You must install **Zoe Ashwood’s fork of the **``** library**, as the standard version lacks support for GLM-HMMs.

```bash
git clone https://github.com/zashwood/ssm.git
cd ssm
pip install -e .
```

Other required packages (install via pip or conda):

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- ibllib
- brainwidemap

---

## Example Usage

To fit a 2-state GLM-HMM model to an individual IBL mouse, e.g. 'NYU-11', (p(state 1) = probability to be engaged in that trial):

```python
model_single_mouse('NYU-11', run_description='K_2')
plot_model_params('NYU-11', run_description='K_2')
```

To process all IBL BWM animals:

```python
do_for_all(run_description='K_2')
```

## Data File: `merged_behavioral_and_states.pqt`

This file contains trial-by-trial data from all BWM animals, merged with GLM-HMM model outputs, stored in **Parquet** format.

### Column Descriptions

| Column             | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `animal`           | Mouse identifier (e.g., "NYU-11")                                           |
| `eid`              | Experiment ID (unique per session)                                          |
| `contrastLeft`     | Contrast level of the left visual stimulus on a given trial (NaN if none)   |
| `contrastRight`    | Contrast level of the right visual stimulus on a given trial (NaN if none)  |
| `rewarded`         | Trial outcome: `1` if rewarded, `-1` if not rewarded                        |
| `probabilityLeft`  | Block-level bias: probability that the left side is correct on that trial   |
| `p_state1`         | Posterior probability that the mouse was in latent state 1 (engaged) in this trial    |
| `p_state2`         | Posterior probability of latent state 2 (disengaged)                                    |
| `signed_contrast`  | Net stimulus contrast: `contrastRight - contrastLeft`, zero-centered        |

`p_state1` is the engagement probability per trial.

---

## Acknowledgments

Modeling framework based on:

> Ashwood et al. (2022). *Mice alternate between discrete strategies during perceptual decision-making*. Nature Neuroscience.\
> GitHub: [https://github.com/zashwood/glm-hmm](https://github.com/zashwood/glm-hmm)

Data from the [International Brain Laboratory](https://www.internationalbrainlab.com/).\
Adaptation by Michael Schartner.

---

## License

This project inherits licensing and reuse rights from the original authors where applicable. Check `LICENSE` for more info.
