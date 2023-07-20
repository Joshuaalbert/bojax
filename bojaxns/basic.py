import inspect
from typing import Callable, Dict, Any, Type, List, TypeVar

import numpy as np
import ujson
from pydantic import BaseModel


# How many decimal places to retain. Note we use a fast but approximate method to compute.
# Note: results will vary after serialisation and loading an array due to the truncation applied.


class SerialisableBaseModel(BaseModel):
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        json_loads = ujson.loads  # can use because ujson decodes NaN and Infinity
        json_dumps = ujson.dumps  # (currently not possible because ujson doesn't encode NaN and Infinity like json)
        # json_dumps = lambda *args, **kwargs: json.dumps(*args, **kwargs, separators=(',', ':'))
        json_encoders = {np.ndarray: lambda x: x.tolist()}

    @classmethod
    def _deserialise(cls, kwargs):
        """Required for this class's __reduce__ method to be picklable."""
        return cls(**kwargs)

    def __reduce__(self):
        serialised_data = self.dict()
        return self.__class__._deserialise, (serialised_data,)


class HashableSerialisableBaseModel(SerialisableBaseModel):
    def __hash__(self):
        return hash(self.json)


def example_from_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Generate example from schema and return as dict.

    Args:
        model: BaseModel

    Returns: dict of example
    """
    example = dict()
    properties = model.schema().get('properties', dict())
    for field in model.__fields__:
        # print(model, model.__fields__[field])
        if inspect.isclass(model.__fields__[field]):
            if issubclass(model.__fields__[field], BaseModel):
                example[field] = example_from_schema(model.__fields__[field])
                continue
            example[field] = None
        example[field] = properties[field].get('example', None)
        # print(field, example[field])
    return example


_T = TypeVar('_T')


def build_example(model: Type[_T]) -> _T:
    return model(**example_from_schema(model))


def apply_validators(value, validators: List[Callable]):
    for validator in validators:
        value = validator(value)
    return value
