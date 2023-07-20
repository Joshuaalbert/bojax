[![Python](https://img.shields.io/pypi/pyversions/bojax.svg)](https://badge.fury.io/py/bojax)
[![PyPI](https://badge.fury.io/py/bojax.svg)](https://badge.fury.io/py/bojax)

Main
Status: ![Workflow name](https://github.com/JoshuaAlbert/bojax/actions/workflows/unittests.yml/badge.svg?branch=main)

Develop
Status: ![Workflow name](https://github.com/JoshuaAlbert/bojax/actions/workflows/unittests.yml/badge.svg?branch=develop)

## Mission: _To make advanced Bayesian Optimisation easy._

# What is it?

Bojax is:

1) a Bayesian Optimisation package for easily performing advanced non-myopic Bayesian optimisation.
2) using [JAXNS](https://github.com/JoshuaAlbert/jaxns) under the hood to marginalise over multiple models.
3) using multi-step lookahead to plan out your next step.
4) available for academic use and non-commercial use (without permission) read the license.

# Documentation

We don't have documentation yet!

# Install

**Notes:**

1. Bojax requires >= Python 3.9.
2. It is always highly recommended to use a unique virtual environment for each project.
   To use `miniconda`, have it installed, and run

```bash
# To create a new env, if necessary
conda create -n bojax_py python=3.11
conda activate bojax_py
```

## For end users

Install directly from PyPi,

```bash
pip install bojax
```

## For development

Clone repo `git clone https://www.github.com/JoshuaAlbert/bojax.git`, and install:

```bash
cd jaxns
pip install -r requirements.txt
pip install -r requirements-tests.txt
pip install .
```
# Change Log

20 July, 2023 -- Bojax 1.0.0 released
