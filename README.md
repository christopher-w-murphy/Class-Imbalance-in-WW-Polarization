# Class-Imbalance-in-WW-Polarization
Treating the measurement of the same-sign W polarization fraction as a class imbalance problem

## Motivation
A couple of papers, [arXiv:1510.01691](https://arxiv.org/abs/1510.01691), and more recently [arXiv:1812.07591](https://arxiv.org/abs/1812.07591), have used deep learning to determine the polarization fraction, _W<sub>L</sub> W<sub>L</sub> / &Sigma;<sub>i, j</sub> W<sub>i</sub> W<sub>j</sub>_, in same-sign _WW_ scattering. 

In this reaction two protons (_p_) collide at the Large Hadron Collider and produce two jets (_j_), collimated sprays of hadronic particles, and two _W_ bosons with the same electric charge. This process is interesting as a probe of the unitarization (probability conservation) mechanism in the Standard Model (SM) of particle physics.

The polarization fraction is predicted to be small in the SM, ~5%. Thus there is an imbalance of events where both _W_'s are longitudinally polarized vs. when one or none is longitudinally polarized. This motivates trying to treat this as a class imbalance problem, something which neither of the above papers do.

## Setup
Clone repository
```
git clone https://github.com/christopher-w-murphy/Class-Imbalance-in-WW-Polarization
```

## Requisites
See the `env` folder for full details. The only thing to watch out for is Imbalanced Learn is not fully compatible with Keras v2.0 or lower. I'm using Keras v2.3.1 

## Analysis
See the [notebook](https://github.com/christopher-w-murphy/Class-Imbalance-in-WW-Polarization/blob/master/notebook/comparison.ipynb) for full details.

### Comparison
First is a comparison between this work and arXiv:1812.07591, which establishes some baseline models. I am aiming to show that I can approximately reproduce the results of 1812.07591 given a different simulated dataset. Below are the predicted _LL_ fractions from this work.

![Predicted LL Fraction](https://github.com/christopher-w-murphy/Class-Imbalance-in-WW-Polarization/blob/master/static/predicted_LL_fraction.png)

### Different Metrics
Next the baseline models are evaluated using metrics that are better suited for class imbalance problems. For example, below are the precision-recall curves for these models

![Precision-Recall Curves](https://github.com/christopher-w-murphy/Class-Imbalance-in-WW-Polarization/blob/master/static/PR_curve.png)

### Different Models
Lastly, different models, aimed specifically at class inbalance, are trained to see if there is an improvement in performance. These include:
- Class Weights
- Focal Loss
- Balanced Random Forest
- Balanced Batch Generator

Note: LaTeX equations are not properly visualized in Jupyter Notebook on Github. One option is to paste the link to the notebook into http://nbviewer.jupyter.org.

## Citation
If you use this code please cite:
```
@article{Murphy:2019utt,
      author         = "Murphy, Christopher W.",
      title          = "{Class Imbalance Techniques for High Energy Physics}",
      year           = "2019",
      eprint         = "1905.00339",
      archivePrefix  = "arXiv",
      primaryClass   = "hep-ph",
      SLACcitation   = "%%CITATION = ARXIV:1905.00339;%%"
}
```
