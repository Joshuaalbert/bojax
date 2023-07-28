from typing import Literal, Dict, Union, List

from pydantic import BaseModel, confloat


class FloatValue(BaseModel):
    type: Literal['float'] = 'float'
    value: float


class IntValue(BaseModel):
    type: Literal['int'] = 'int'
    value: int


ParamValues = Dict[str, Union[FloatValue, IntValue]]
UValue = List[confloat(ge=0., le=1.)]
