import logging

import numpy as np
import pylab as plt
from jax import random, numpy as jnp
from jax._src.random import PRNGKey
from jaxns.prior import parse_prior, PriorModelType, transform
from matplotlib import dates as mdates

from bojax.experiment import OptimisationExperiment, NewExperimentRequest, Trial, FloatValue, \
    IntValue, TrialUpdate
from bojax.gaussian_process_formulation.bayesian_optimiser import BayesianOptimiser
from bojax.parameter_space import build_prior_model, ContinuousPrior, IntegerPrior, CategoricalPrior
from bojax.utils import latin_hypercube

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'InvalidTrial',
    'BayesianOptimisation'
]


class InvalidTrial(Exception):
    pass


class BayesianOptimisation:
    def __init__(self, experiment: OptimisationExperiment):
        self._experiment = experiment

    @property
    def experiment(self):
        return self._experiment

    @classmethod
    def create_new_experiment(cls, new_experiment: NewExperimentRequest) -> 'BayesianOptimisation':
        experiment = OptimisationExperiment(parameter_space=new_experiment.parameter_space)
        prior_model = build_prior_model(experiment.parameter_space)
        U_placeholder, _ = parse_prior(prior_model=prior_model)
        U_dims = U_placeholder.size

        U_seeds = latin_hypercube(
            seed=42,
            num_samples=new_experiment.init_explore_size,
            num_dim=U_dims
        )

        # Add trials for seeds.
        for U_seed in U_seeds:
            trial = BayesianOptimisation._create_trial(
                experiment=experiment,
                U=U_seed,
                prior_model=prior_model
            )
            experiment.trials[trial.trial_id] = trial

        return cls(experiment=experiment)

    @staticmethod
    def _create_trial(experiment: OptimisationExperiment, U: jnp.ndarray, prior_model: PriorModelType) -> Trial:
        prior_sample = transform(U=U, prior_model=prior_model)
        param_values = {}
        for param in experiment.parameter_space.parameters:
            val = prior_sample[param.name]
            if isinstance(param.prior, ContinuousPrior):
                param_values[param.name] = FloatValue(value=float(val))
                continue
            if isinstance(param.prior, IntegerPrior):
                param_values[param.name] = IntValue(value=int(val))
                continue
            if isinstance(param.prior, CategoricalPrior):
                param_values[param.name] = IntValue(value=int(val))
                continue
        trial = Trial(param_values=param_values, U_value=U.tolist())
        return trial

    def create_new_trial(self, key: PRNGKey, random_explore: bool = False, beta: float = 0.5) -> str:

        # Go through trials, and find one with unfilled values. Give that one.
        for trial in sorted(self._experiment.trials.values(), key=lambda t: t.create_dt):
            if len(trial.trial_updates) == 0:
                return trial.trial_id

        prior_model = build_prior_model(self.experiment.parameter_space)

        if random_explore:
            U_placeholder, _ = parse_prior(prior_model=prior_model)
            U_dims = U_placeholder.size
            U = random.uniform(key, shape=(U_dims,))
            trial = BayesianOptimisation._create_trial(
                experiment=self.experiment,
                U=U,
                prior_model=prior_model
            )
            self._experiment.trials[trial.trial_id] = trial
            return trial.trial_id

        # get new trial parameter from bojax
        bo = BayesianOptimiser(experiment=self._experiment, beta=beta, S=128)
        # U = bo.choose_next_U_multistep(
        #     key=key,
        #     batch_size=1,
        #     max_depth=2,
        #     num_simulations=1000,
        #     branch_factor=100
        # )
        U = bo.choose_next_U_toptwo(
            key=key,
            batch_size=10,
            num_search=100000
        )
        trial = BayesianOptimisation._create_trial(
            experiment=self.experiment,
            U=U,
            prior_model=prior_model
        )
        self._experiment.trials[trial.trial_id] = trial
        return trial.trial_id

    def get_trial(self, trial_id: str) -> Trial:
        if trial_id not in self._experiment.trials:
            raise InvalidTrial(trial_id)
        return self._experiment.trials[trial_id]

    def delete_trial(self, trial_id: str):
        if trial_id not in self._experiment.trials:
            raise InvalidTrial(trial_id)
        del self._experiment.trials[trial_id]

    def post_measurement(self, trial_id: str, trial_update: TrialUpdate):
        trial = self.get_trial(trial_id=trial_id)
        if (trial_update.ref_id in trial.trial_updates) and (
                trial_update.objective_measurement == trial.trial_updates[trial_update.ref_id].objective_measurement):
            return
        trial.trial_updates[trial_update.ref_id] = trial_update

    def trial_size(self, trial_id: str):
        trial = self.get_trial(trial_id=trial_id)
        return len(trial.trial_updates)

    def visualise(self,
                  main_color="#7e97bf",
                  grid_color="#969396",
                  ) -> plt.Figure:
        """
        Constructs a visual breakdown of condition

        Args:
            main_color: color of main axes
            grid_color: color of grid

        Returns:
            a pylab Figure

        Raises:
            NotEnoughData if not enough to compute a breakdown
        """
        # Plots scatter of trial outcomes over time.
        # Highlight, best.
        # For each trial use a colored line, with error bars that scale 1/sqrt(S)
        series = []
        for trial in self._experiment.trials.values():
            if len(trial.trial_updates) == 0:
                continue
            x, y, n = [], [], []
            for trial_update in sorted(trial.trial_updates.values(), key=lambda tu: tu.measurement_dt):
                x.append(trial_update.measurement_dt)
                if len(y) == 0:
                    y.append(trial_update.objective_measurement)
                    n.append(1)
                else:
                    y.append(y[-1] + trial_update.objective_measurement)
                    n.append(n[-1] + 1)
            y = list(map(lambda _y, _n: _y / _n, y, n))
            if len(y) < 3:
                y_std = [0.] * len(y)
            else:
                _mu = jnp.mean(jnp.asarray(y))
                _y_std = jnp.abs(y[0] - _mu)
                y_std = _y_std / jnp.sqrt(jnp.asarray(n))
                y_std = y_std.tolist()
            series.append((x, y, y_std))

        if len(series) == 0:
            raise RuntimeError("Nothing to visualise. Provide data to a trial first.")

        min_dt = min(min(x) for (x, y, y_std) in series)
        max_dt = max(max(x) for (x, y, y_std) in series)

        fig_width = 6
        fig_height = fig_width
        fig, ax = plt.subplots(figsize=(fig_height, fig_width), facecolor='None')
        ax.set_facecolor('none')
        np.random.seed(42)  # deterministic colors
        for x, y, y_std in series:
            color = [np.random.uniform(), np.random.uniform(), np.random.uniform(), 1.]
            ax.plot(x, y, c=color)
            color[-1] = 0.2
            ax.errorbar(x,
                        y,
                        y_std,
                        fmt='-o',
                        color=color
                        )

        agg_series = [(x[-1], y[-1], y_std[-1]) for (x, y, y_std) in series]
        agg_series = np.asarray(agg_series)

        idx_max = np.argmax(agg_series[:, 1])
        ax.scatter(agg_series[idx_max, 0], agg_series[idx_max, 1], s=100, fc='none', ec='black', marker='o',
                   label=f"Best {agg_series[idx_max, 1]}")

        # min_date_num = mdates.date2num(min_dt)
        # max_date_num = mdates.date2num(max_dt)
        #
        # interval = max((max_date_num - min_date_num) // 6, 1)
        #
        # ax.set_xticks(np.arange(min_date_num, max_date_num, interval))

        # xticks = []
        # ax.set_xticks(mdates.date2num(datetime.combine(today, time())))
        # restrict x lim
        ax.set_xlim(
            mdates.date2num(min_dt),
            mdates.date2num(max_dt)
        )
        # ax.set_ylim(0, 100)

        date_formatter = mdates.DateFormatter('%a, %-d %b')  # Customize the format as per your preference
        ax.xaxis.set_major_formatter(date_formatter)

        ax.set_title(f"Trial progression", color=main_color, fontsize=8)

        # Rotate value labels
        ax.tick_params(axis='x', rotation=45, labelsize=6)

        # Add legend
        ax.legend(loc='best', prop={'size': 6}, framealpha=0.1)
        # ax.legend(loc='upper center', prop={'size': 6}, framealpha=0.25, bbox_to_anchor=(0.5, -0.05))

        # Set xlim and ylim tight to data points

        ax.margins(0.01)

        # Invisible top, right
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set axis-colors
        ax.spines['left'].set_color(main_color)
        ax.spines['bottom'].set_color(main_color)
        ax.tick_params(axis='x', colors=main_color)
        ax.tick_params(axis='y', colors=main_color)

        # grid
        ax.grid(axis='y', linestyle='dashed', color=grid_color)

        # Make tight
        fig.tight_layout()

        return fig
