# GLM-HMM Modeling of IBL Mouse Behavior

This repository adapts Zoe Ashwood’s GLM-HMM model for state-dependent behavior to work on **any IBL mouse** using the BrainWideMap (BWM) dataset. It enables model fitting, evaluation, and visualization of behavioral state dynamics for IBL animal (pooling sessions), resulting in a probability of engagement per trial.

Original modeling framework:\
➡️ [https://github.com/zashwood/glm-hmm](https://github.com/zashwood/glm-hmm)

---

## Key Features

- Adapted to work with **all IBL animals**, both biased and unbiased paradigms
- Compatible with BWM data through the [`brainwidemap`](https://github.com/int-brain-lab/brainwidemap) library
- Posterior state inference and GLM-HMM parameter fitting
- Utilities for:
  - Animal-wise model fitting
  - Visualizing generative weights and state transitions
  - Pooling and analyzing emirical and expected dwell times and engagement probability

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

To process all animals:

```python
do_for_all(run_description='K_2')
```

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
