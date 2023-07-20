from typing import List, Union
from uuid import uuid4

import pylab as plt
from jax import random, numpy as jnp
from pydantic import BaseModel, Field

from bojax.experiment import NewExperimentRequest, Trial, TrialUpdate
from bojax.parameter_space import ParameterSpace, Parameter, ContinuousPrior, IntegerPrior, CategoricalPrior
from bojax.service import BayesianOptimisation


def test_bayesian_optimisation():
    num_steps = 11

    def objective(x):
        return -0.5 * jnp.sum(x ** 4 - 16 * x ** 2 + 5 * x)

    def test_example(ndim):

        lower_bound = 39.16616 * ndim
        upper_bound = 39.16617 * ndim
        print(f"Optimal value in ({lower_bound}, {upper_bound}).")

        x_max = -2.903534

        print(f"Global optimum at {jnp.ones(ndim) * x_max}")

        parameter_space = ParameterSpace(
            parameters=[
                Parameter(
                    name=f'x{i}',
                    prior=ContinuousPrior(
                        lower=-5,
                        upper=5.,
                        mode=0.,
                        uncert=10.
                    )
                )
                for i in range(ndim)
            ]
        )
        new_experiment_request = NewExperimentRequest(
            parameter_space=parameter_space,
            init_explore_size=num_steps - 1
        )
        bo_experiment = BayesianOptimisation.create_new_experiment(new_experiment=new_experiment_request)

        for i in range(num_steps):
            trial_id = bo_experiment.create_new_trial(
                key=random.PRNGKey(i),
                random_explore=False
            )
            trial = bo_experiment.get_trial(trial_id=trial_id)
            params = []
            for param_name in sorted(trial.param_values.keys()):
                param = trial.param_values[param_name]
                params.append(param.value)
            params = jnp.asarray(params)

            obj_val = float(objective(params))
            bo_experiment.post_measurement(
                trial_id=trial_id,
                trial_update=TrialUpdate(ref_id='a', objective_measurement=obj_val)
            )
            # fig = bo_experiment.visualise()
            # plt.show()
            # plt.close('all')

    test_example(2)


def test_bayesian_optimiser():
    parameter_space = ParameterSpace(parameters=[
        Parameter(
            name='churn_rate',
            prior=ContinuousPrior(
                lower=1 / 20.,
                upper=1 / 10.,
                mode=1 / 15.,
                uncert=1 / 15.
            )
        ),
        Parameter(
            name='integers',
            prior=IntegerPrior(
                lower=0,
                upper=5,
                mode=4.,
                uncert=1.3
            )
        ),
        Parameter(
            name='categorical',
            prior=CategoricalPrior(
                probs=[1., 1., 1.]
            )
        )
    ])
    bo = BayesianOptimisation.create_new_experiment(
        new_experiment=NewExperimentRequest(parameter_space=parameter_space, init_explore_size=10)
    )

    for trial in bo.experiment.trials.values():
        assert (1 / 20 <= trial.param_values['churn_rate'].value <= 1 / 10)

    class User(BaseModel):
        user_id: str = Field(default_factory=lambda: str(uuid4()))
        trial_id: str
        observable: Union[float, None] = None
        join_dt: float
        churn_dt: float

    users: List[User] = []

    t = 0.
    T = 200.
    n_per_trial = 10
    trial: Union[Trial, None] = None
    new_user_rate = 2
    key = random.PRNGKey(42)
    while t < T:
        # Create a new trial
        if trial is None:
            key, sample_key = random.split(key)
            trial_id = bo.create_new_trial(key=sample_key, random_explore=True)
            trial = bo.get_trial(trial_id)
        # Update time to next user join
        key, sample_key = random.split(key)
        diff_dt = float(jnp.abs(random.laplace(sample_key)) / new_user_rate)
        t += diff_dt
        key, sample_key = random.split(key)
        churn_dt = t + float(jnp.abs(random.laplace(sample_key)) / trial.param_values['churn_rate'].value)
        users.append(User(trial_id=trial.trial_id, join_dt=t, churn_dt=churn_dt))
        # Handle trial
        trial_count = bo.trial_size(trial_id=trial.trial_id)
        if trial_count >= n_per_trial:
            trial = None
        # Report data
        for user in users:
            if t > user.churn_dt:
                user.observable = user.churn_dt - user.join_dt
            if user.observable is None:
                continue
            trial_update = TrialUpdate(ref_id=user.user_id,
                                       objective_measurement=user.observable)
            bo.post_measurement(trial_id=user.trial_id, trial_update=trial_update)
    fig = bo.visualise()
    # fig.show()
    plt.close('all')
