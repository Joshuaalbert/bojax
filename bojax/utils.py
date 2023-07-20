import inspect
from datetime import datetime, tzinfo
from typing import TypeVar, Type, Dict, Any

import numpy as np
from pyDOE2 import lhs
from pydantic import BaseModel

__all__ = [
    'latin_hypercube',
    'build_example',
    'current_utc'
]


def latin_hypercube(seed: int, num_samples: int, num_dim: int):
    """
    Sample from the latin-hypercube defined as the continuous analog of the discrete latin-hypercube.
    That is, if you partition each dimension into `num_samples` equal volume intervals then there is (conditionally)
    exactly one point in each interval. We guarantee that uniformity by randomly assigning the permutation of each dimension.
    The degree of randomness is controlled by `cube_scale`. A value of 0 places the sample at the center of the grid point,
    and a value of 1 places the value randomly inside the grid-cell.

    Args:
        key: PRNG key
        num_samples: number of samples in total to draw
        num_dim: number of dimensions in each sample
        cube_scale: The scale of randomness, in (0,1).

    Returns:
        latin-hypercube samples of shape [num_samples, num_dim]
    """
    np.random.seed(seed)
    return lhs(num_dim, samples=num_samples)


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
        if 'example' in properties[field]:
            example[field] = properties[field].get('example')
        else:
            example[field] = model.__fields__[field].get_default()
        # print(field, example[field])
    return example


_T = TypeVar('_T')


def build_example(model: Type[_T]) -> _T:
    return model(**example_from_schema(model))


def set_datetime_timezone(dt: datetime, offset: str | tzinfo) -> datetime:
    """
    Replaces the datetime object's timezone with one from an offset.

    Args:
        dt: datetime, with out without a timezone set. If set, will be replaced.
        offset: tzinfo, or str offset like '-04:00' (which means EST)

    Returns:
        datetime with timezone set
    """
    if isinstance(offset, str):
        dt = dt.replace(tzinfo=None)
        return datetime.fromisoformat(f"{dt.isoformat()}{offset}")
    if isinstance(offset, tzinfo):
        return dt.replace(tzinfo=offset)
    raise ValueError(f"offset {offset} not understood.")


def set_utc_timezone(dt: datetime) -> datetime:
    return set_datetime_timezone(dt, '+00:00')


def current_utc() -> datetime:
    return set_utc_timezone(datetime.utcnow())
