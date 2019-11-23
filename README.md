# Class Imbalance in WW Polarization
Treating the measurement of the same-sign W polarization fraction as a class imbalance problem. This is one of the two cases studies in my paper [arXiv:1905.00339](https://arxiv.org/abs/1905.00339).

## Motivation
A couple of papers, [arXiv:1510.01691](https://arxiv.org/abs/1510.01691), and more recently [arXiv:1812.07591](https://arxiv.org/abs/1812.07591), have used deep learning to determine the polarization fraction, _W<sub>L</sub> W<sub>L</sub> / &Sigma;<sub>i, j</sub> W<sub>i</sub> W<sub>j</sub>_, in same-sign _WW_ scattering. 

In this reaction two protons (_p_) collide at the Large Hadron Collider and produce two jets (_j_), collimated sprays of hadronic particles, and two _W_ bosons with the same electric charge. This process is interesting as a probe of the unitarization (probability conservation) mechanism in the Standard Model (SM) of particle physics.

The polarization fraction is predicted to be small in the SM, ~7%. Thus there is an imbalance of events where both _W_'s are longitudinally polarized vs. when one or none is longitudinally polarized. This motivates trying to treat this as a class imbalance problem, something which neither of the above papers do.

## Setup
Clone repository
```
git clone https://github.com/christopher-w-murphy/Class-Imbalance-in-WW-Polarization
```
Install requistes if necessary. Running
```
pip install -r env/minimal.txt
```
from the main directory of this repo will install `pandas`, `scikit-learn`, `imbalanced-learn`, `lightgbm`, and , `tensorflow`. The full pip freeze in available in the `env` folder as is my Jupyter environment.
