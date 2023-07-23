[![Python](https://img.shields.io/pypi/pyversions/bojaxns.svg)](https://badge.fury.io/py/bojaxns)
[![PyPI](https://badge.fury.io/py/bojaxns.svg)](https://badge.fury.io/py/bojaxns)

Main
Status: ![Workflow name](https://github.com/JoshuaAlbert/bojaxns/actions/workflows/unittests.yml/badge.svg?branch=main)

Develop
Status: ![Workflow name](https://github.com/JoshuaAlbert/bojaxns/actions/workflows/unittests.yml/badge.svg?branch=develop)

## Mission: _To make advanced Bayesian Optimisation easy._

# What is it?

Bojaxns is:

1) a Bayesian Optimisation package for easily performing advanced non-myopic Bayesian optimisation.
2) using [JAXNS](https://github.com/JoshuaAlbert/jaxns) under the hood to marginalise over multiple models.
3) using multi-step lookahead to plan out your next step.
4) available for academic use and non-commercial use (without permission) read the license.

# Documentation

For examples, check out the [documentation](https://bojax.readthedocs.io/) (still in progress).

# Install

**Notes:**

1. Bojaxns requires >= Python 3.9.
2. It is always highly recommended to use a unique virtual environment for each project.
   To use `miniconda`, have it installed, and run

```bash
# To create a new env, if necessary
conda create -n bojaxns_py python=3.11
conda activate bojaxns_py
```

## For end users

Install directly from PyPi,

```bash
pip install bojaxns
```

## For development

Clone repo `git clone https://www.github.com/JoshuaAlbert/bojaxns.git`, and install:

```bash
cd bojaxns
pip install -r requirements.txt
pip install -r requirements-tests.txt
pip install .
```
# Change Log

20 July, 2023 -- Bojaxns 1.0.0 released
