from datetime import datetime
from typing import Literal, List, Dict, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator, conint, confloat

from bojax.parameter_space import ParameterSpace
from bojax.utils import current_utc, build_example

__all__ = [
    'FloatValue',
    'IntValue',
    'Trial',
    'TrialUpdate',
    'OptimisationExperiment',
    'NewExperimentRequest',
]


class FloatValue(BaseModel):
    type: Literal['float'] = 'float'
    value: float


class IntValue(BaseModel):
    type: Literal['int'] = 'int'
    value: int


class TrialUpdate(BaseModel):
    ref_id: str = Field(
        description="An identifier of the measurement, e.g. user UUID.",
        example=str(uuid4())
    )
    measurement_dt: datetime = Field(
        default_factory=current_utc,
        description='The datetime the objective_measurement was determined.',
        example=current_utc()
    )
    objective_measurement: float = Field(
        description="The measurement of trial objective function.",
        example=1.
    )


class Trial(BaseModel):
    trial_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description='UUID for this trial.',
        example=str(uuid4())
    )
    create_dt: datetime = Field(
        default_factory=current_utc,
        description='The datetime the param_value was determined.',
        example=current_utc()
    )
    param_values: Dict[str, Union[FloatValue, IntValue]] = Field(
        description="The parameter mapping for trial.",
        example={'price': FloatValue(value=1.)}
    )
    U_value: List[confloat(ge=0., le=1.)] = Field(
        description="The U-space value of parameters.",
        example=[0.2]
    )
    trial_updates: Dict[str, TrialUpdate] = Field(
        default_factory=dict,
        description="The measurement of trial updates.",
        example={"124": build_example(TrialUpdate)}
    )


class OptimisationExperiment(BaseModel):
    experiment_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description='UUID for this experiment.',
        example=str(uuid4())
    )
    parameter_space: ParameterSpace = Field(
        description='The parameter space that defines this experiment.',
        example=build_example(ParameterSpace)
    )
    trials: Dict[str, Trial] = Field(
        default_factory=dict,
        description="The mapping of trials that define the sequence of this experiment.",
        example={'12345': build_example(Trial)}
    )

    @validator('trials', always=True)
    def ensure_parameters_match_space(cls, value, values):
        parameter_space: ParameterSpace = values['parameter_space']
        names = list(map(lambda param: param.name, parameter_space.parameters))
        for trial_id in value:
            trial: Trial = value[trial_id]
            _names = list(trial.param_values)
            if set(_names) != set(names):
                raise ValueError(f"trial param_values {_names} don't match param space {names}.")
        return value


class NewExperimentRequest(BaseModel):
    parameter_space: ParameterSpace = Field(
        description='The parameter space that defines this experiment.',
        example=build_example(ParameterSpace)
    )
    init_explore_size: conint(ge=1)
