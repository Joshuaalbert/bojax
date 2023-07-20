from typing import Union, Literal, List, Tuple, Annotated

import tensorflow_probability.substrates.jax as tfp
from pydantic import BaseModel, Field, confloat

tfpd = tfp.distributions

__all__ = [
    'ContinuousAffineBetaPrior',
    'DiscreteAffineBetaPrior',
    'CategoricalLatentPrior',
    'Parameter',
    'ParameterSpace',
    'build_prior_model'
]


class ContinuousAffineBetaPrior(BaseModel):
    type: Literal['affine_beta_prior'] = 'affine_beta_prior'
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
    uncert: float = Field(
        description="The uncertainty of the prior. Set to np.inf for the uniform prior over (lower, upper).",
        example=2.
    )


class DiscreteAffineBetaPrior(BaseModel):
    type: Literal['discrete_affine_beta_prior'] = 'discrete_affine_beta_prior'
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
    uncert: float = Field(
        description="The uncertainty of the prior. Set to np.inf for the uniform prior over (lower, upper). Can be a float.",
        example=2.
    )


class CategoricalLatentPrior(BaseModel):
    type: Literal['categorical_latent_prior'] = 'categorical_latent_prior'
    categories: List[Tuple[str, float]] = Field(
        description="The categories, and affine parameters of the parameter.",
        example=[('a', 0.), ('b', 2.), ('c', 10.)]
    )
    probs: List[confloat(ge=0.)] = Field(
        description="The unnormalised probabilities of categories. Must be >= 0, need not be normalised.",
        example=[0.1, 0.3, 0.6]
    )


ParamPrior = Annotated[
    Union[ContinuousAffineBetaPrior, DiscreteAffineBetaPrior, CategoricalLatentPrior],
    Field(
        description='The parameter prior, which defines the domain.',
        discriminator='type'
    )
]


# from bojax.parameter_space import DiscreteAffineBetaPrior, CategoricalLatentPrior, ParameterSpace, Parameter, build_prior_model, ContinuousAffineBetaPrior
from bojax.utils import build_example


def test_serialisation():
    print(DiscreteAffineBetaPrior(lower=0, upper=1,mode=0.5, uncert=0.5))
    s = build_example(DiscreteAffineBetaPrior)
