from abc import abstractmethod
from typing import NamedTuple, Dict

import chex
from jax import numpy as jnp, tree_map, vmap
from jax.random import PRNGKey
from jaxns import resample, PriorModelType


class AbstractAcquisition:
    """
    A class that represents any acquisition function. All acquisition functions take a point in the U-domain
    and returns a metric that gives a proxy as to how valuable it would be to try that point.
    All acquisition values only make sense relatively.
    """

    @abstractmethod
    def __call__(self, u_star: jnp.ndarray):
        ...


def _assert_rank(rank: int, **kwargs):
    for name, t in kwargs.items():
        if len(t.shape) != rank:
            raise ValueError(f"{name} shoue be rank {rank} got {t.shape}.")


def _assert_same_leading_dim(*args):
    n = set()
    for arg in args:
        n.add(arg.shape[0])
    if len(n) > 1:
        raise ValueError(f"Got mismatched leading dimensions: {n}")


class ConditionalPredictive:

    @abstractmethod
    def _ndims(self):
        ...

    @property
    def ndims(self):
        return self._ndims()

    @abstractmethod
    def posterior(self):
        ...

    @abstractmethod
    def marginal_likelihood(self):
        ...

    @abstractmethod
    def __call__(self, U_star: jnp.ndarray, cov: bool = False):
        ...


class MarginalisationData(NamedTuple):
    samples: Dict[str, chex.Array]
    log_dp_mean: chex.Array


class ConditionalPredictiveFactory:

    @abstractmethod
    def ndims(self):
        ...

    @abstractmethod
    def build_prior_model(self) -> PriorModelType:
        ...

    @abstractmethod
    def __call__(self, **samples) -> ConditionalPredictive:
        ...


class AcquisitionFactory:

    @abstractmethod
    def __call__(self, **sample) -> AbstractAcquisition:
        ...


class MarginalisedAcquisitionFunction(AbstractAcquisition):
    """
    Class that represents a marginalisation of an acquisition function over samples.
    """

    def __init__(self, key: PRNGKey, ns_results: MarginalisationData, acquisition_factory: AcquisitionFactory, S: int):
        self._acquisition_factory = acquisition_factory
        self._key = key
        self._ns_results = ns_results
        self._S = int(S)

    def __call__(self, u_star: jnp.ndarray):
        def _eval(**sample):
            acquisition = self._acquisition_factory(**sample)
            return acquisition(u_star=u_star)

        samples = resample(self._key, self._ns_results.samples, self._ns_results.log_dp_mean,
                           S=self._S, replace=True)
        marginalised = tree_map(lambda marg: jnp.nanmean(marg, axis=0), vmap(_eval)(**samples))
        return marginalised
        #
        # return marginalise_static(
        #     key=self._key,
        #     samples=self._ns_results.samples,
        #     log_weights=self._ns_results.log_dp_mean,
        #     ESS=int(self._ns_results.ESS),
        #     fun=_eval
        # )


class MarginalisedConditionalPredictive(ConditionalPredictive):
    """
    Class that represents a marginalisation of an acquisition function over samples.
    """

    def __init__(self, key: PRNGKey, ns_results: MarginalisationData,
                 conditional_predictive_factory: ConditionalPredictiveFactory,
                 S: int):
        self._conditional_predictive_factory = conditional_predictive_factory
        self._key = key
        self._ns_results = ns_results
        self._S = int(S)

    def _ndims(self):
        return self._conditional_predictive_factory.ndims()

    def posterior(self):
        def _eval(**sample):
            conditional_predictive = self._conditional_predictive_factory(**sample)
            return conditional_predictive.posterior()

        samples = resample(self._key, self._ns_results.samples, self._ns_results.log_dp_mean,
                           S=self._S, replace=True)
        marginalised = tree_map(lambda marg: jnp.nanmean(marg, axis=0), vmap(_eval)(**samples))
        return marginalised

    def marginal_likelihood(self):
        def _eval(**sample):
            conditional_predictive = self._conditional_predictive_factory(**sample)
            return conditional_predictive.marginal_likelihood()

        samples = resample(self._key, self._ns_results.samples, self._ns_results.log_dp_mean,
                           S=self._S, replace=True)
        marginalised = tree_map(lambda marg: jnp.nanmean(marg, axis=0), vmap(_eval)(**samples))
        return marginalised

    def __call__(self, U_star: jnp.ndarray, cov: bool = False):
        def _eval(**sample):
            conditional_predictive = self._conditional_predictive_factory(**sample)
            return conditional_predictive(U_star=U_star, cov=cov)

        samples = resample(self._key, self._ns_results.samples, self._ns_results.log_dp_mean,
                           S=self._S, replace=True)
        marginalised = tree_map(lambda marg: jnp.nanmean(marg, axis=0), vmap(_eval)(**samples))
        return marginalised
