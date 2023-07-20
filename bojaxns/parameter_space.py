from typing import Union, Literal, List, Generator, Annotated

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxns import Prior, PriorModelGen
from jaxns.prior import PriorModelType
from jaxns.special_priors import Categorical
from jaxns.types import float_type
from pydantic import BaseModel, Field, validator, confloat

from bojaxns.utils import build_example

tfpd = tfp.distributions

__all__ = [
    'ContinuousPrior',
    'IntegerPrior',
    'CategoricalPrior',
    'Parameter',
    'ParameterSpace',
    'build_prior_model'
]


class ContinuousPrior(BaseModel):
    type: Literal['continuous_prior'] = 'continuous_prior'
    lower: float = Field(
        description="The greatest lower bound of interval. Inclusive.",
        example=0.1
    )
    upper: float = Field(
        description="The least upper bound of interval. Inclusive.",
        example=5.5
    )
    mode: float = Field(
        description="The mode of the prior.",
        example=2.5
    )
    uncert: confloat(gt=0.) = Field(
        description="The uncertainty of the prior. Set to np.inf for the uniform prior over (lower, upper).",
        example=2.
    )


class IntegerPrior(BaseModel):
    type: Literal['integer_prior'] = 'integer_prior'
    lower: int = Field(
        description="The greatest lower bound of interval. Inclusive.",
        example=0
    )
    upper: int = Field(
        description="The least upper bound of interval. Inclusive.",
        example=5
    )
    mode: float = Field(
        description="The mode of the prior. Can be a float.",
        example=2.5
    )
    uncert: confloat(gt=0.) = Field(
        description="The uncertainty of the prior. Set to np.inf for the uniform prior over (lower, upper). Can be a float.",
        example=2.
    )


class CategoricalPrior(BaseModel):
    type: Literal['categorical_prior'] = 'categorical_prior'
    probs: List[confloat(ge=0.)] = Field(
        description="The unnormalised probabilities of categories. Must be >= 0, need not be normalised.",
        example=[0.1, 0.3, 0.6]
    )


ParamPrior = Annotated[
    Union[ContinuousPrior, IntegerPrior, CategoricalPrior],
    Field(
        description='The parameter prior, which defines the domain.',
        discriminator='type'
    )
]


class Parameter(BaseModel):
    name: str = Field(
        description="The name of the parameter",
        example='price'
    )
    prior: ParamPrior


class ParameterSpace(BaseModel):
    parameters: List[Parameter] = Field(
        description='The parameters of the problem.',
        example=[
            Parameter(
                name='continuous',
                prior=build_example(ContinuousPrior)
            ),
            Parameter(
                name='integers',
                prior=build_example(IntegerPrior)
            ),
            Parameter(
                name='categorical',
                prior=build_example(CategoricalPrior)
            )
        ]
    )

    @validator('parameters', always=True)
    def unique_parameters(cls, value):
        names = list(map(lambda param: param.name, value))
        if len(names) != len(set(names)):
            raise ValueError(f"parameter names must be unique. Got {names}.")
        return value


def translate_parameter(param: Parameter) -> Generator[Prior, jnp.ndarray, jnp.ndarray]:
    prior = param.prior
    if isinstance(prior, ContinuousPrior):
        underlying_dist = tfpd.TruncatedNormal(
            loc=jnp.asarray(prior.mode, float_type),
            scale=jnp.asarray(prior.uncert, float_type),
            low=jnp.asarray(prior.lower, float_type),
            high=jnp.asarray(prior.upper, float_type)
        )
        param_value = yield Prior(dist_or_value=underlying_dist, name=param.name)
        return param_value
    elif isinstance(prior, IntegerPrior):
        underlying_dist = tfpd.Normal(loc=jnp.asarray(prior.mode, float_type),
                                      scale=jnp.asarray(prior.uncert, float_type))
        int_options = jnp.arange(prior.lower, prior.upper + 1, dtype=float_type)
        logits = underlying_dist.log_prob(int_options)
        param_value_idx = yield Categorical(parametrisation='cdf', logits=logits)
        param_value = yield Prior(dist_or_value=param_value_idx + prior.lower, name=param.name)
        return param_value
    elif isinstance(prior, CategoricalPrior):
        param_value = yield Categorical(parametrisation='gumbel_max', probs=prior.probs, name=param.name)
        return param_value
    else:
        raise ValueError(f"Invalid prior {prior}")


def build_prior_model(parameter_space: ParameterSpace) -> PriorModelType:
    """
    Constructs a prior model given the parameter space.

    Args:
        parameter_space:

    Returns:

    """

    def prior_model() -> PriorModelGen:
        param_values = []
        for parameter in parameter_space.parameters:
            x = yield from translate_parameter(param=parameter)
            param_values.append(x)
        return tuple(param_values)

    return prior_model
