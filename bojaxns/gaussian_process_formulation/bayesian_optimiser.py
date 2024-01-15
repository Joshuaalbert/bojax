import os.path

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from chex import PRNGKey
from jax import random, numpy as jnp, vmap
from jax._src.lax.control_flow import scan
from jaxns import Model, DefaultNestedSampler
from jaxns.internals.types import float_type, NestedSamplerResults

from bojaxns.base import AbstractAcquisition, MarginalisedAcquisitionFunction, MarginalisationData
from bojaxns.experiment import OptimisationExperiment
from bojaxns.gaussian_process_formulation.distribution_math import GaussianProcessData, \
    GaussianProcessConditionalPredictiveFactory, ExpectedImprovementAcquisitionFactory, TopTwoAcquisitionFactory
from bojaxns.gaussian_process_formulation.multi_step_lookahead import run_multi_lookahead, convert_tree_to_graph

tfpb = tfp.bijectors


class BayesianOptimiser:
    def __init__(self, experiment: OptimisationExperiment, num_parallel_solvers: int = 1, beta: float = 0.5,
                 S: int = 512):
        self._experiment = experiment
        self._num_parallel_solvers = num_parallel_solvers
        self._beta = beta
        self._S = int(S)
        self._data = self._prepare_data()

    def _prepare_data(self) -> GaussianProcessData:
        U = []
        Y = []
        Y_var = []
        sample_size = []

        # handle nans ==> illegal value
        min_val, max_val = np.inf, -np.inf
        for trial_id, trial in self._experiment.trials.items():
            for ref_id, trial_update in trial.trial_updates.items():
                if not np.isfinite(trial_update.objective_measurement):
                    continue
                min_val = min(trial_update.objective_measurement, min_val)
                max_val = max(trial_update.objective_measurement, max_val)

        illegal_value = min_val - 0.1*(max_val - min_val)
        if not np.isfinite(illegal_value):
            illegal_value = 0.

        for trial_id, trial in self._experiment.trials.items():
            if len(trial.trial_updates) == 0:
                continue
            samples = []
            for ref_id, trial_update in trial.trial_updates.items():
                if not np.isfinite(trial_update.objective_measurement):
                    samples.append(illegal_value)
                else:
                    samples.append(trial_update.objective_measurement)
            U.append(trial.U_value)
            Y.append(np.mean(samples))
            if len(samples) < 2:
                Y_var.append(np.nan)
            else:
                Y_var.append(np.var(samples))
            sample_size.append(len(samples))
        U = jnp.asarray(U, float_type)
        Y = jnp.asarray(Y, float_type)
        Y_var = jnp.asarray(Y_var, float_type)
        sample_sizes = jnp.asarray(sample_size, float_type)
        data = GaussianProcessData(U=U, Y=Y, Y_var=Y_var, sample_size=sample_sizes)
        return data

    def posterior_solve(self, key: PRNGKey) -> NestedSamplerResults:
        print("Performing posterior solve")
        conditional_predictive_factory = GaussianProcessConditionalPredictiveFactory(data=self._data)

        prior_model = conditional_predictive_factory.build_prior_model()

        def log_likelihood(amplitude, length_scale, variance, mean, kernel_select):
            """
            P(Y|sigma, half_width) = N[Y, f, K]
            """
            conditional_predictive = conditional_predictive_factory(
                amplitude=amplitude,
                length_scale=length_scale,
                variance=variance,
                mean=mean,
                kernel_select=kernel_select
            )
            return conditional_predictive.marginal_likelihood()

        model = Model(
            prior_model=prior_model,
            log_likelihood=log_likelihood
        )

        ns = DefaultNestedSampler(
            model=model,
            parameter_estimation=True,
            max_samples=1e5
        )
        termination_reason, state = jax.jit(ns)(key=key)
        results = ns.to_results(termination_reason, state)
        ns.summary(results)
        ns.plot_diagnostics(results)
        ns.plot_cornerplot(results)
        return results

    @staticmethod
    def _random_search(search_key: PRNGKey, acquisition_function: AbstractAcquisition, ndims: int,
                       batch_size: int, num_search: int):
        vmap_acquisition_function = vmap(acquisition_function)

        def body(carry, key):
            (acquisition_best, u_best) = carry
            U_star = random.uniform(key, shape=(batch_size, ndims),
                                    dtype=float_type)
            acquisition = vmap_acquisition_function(U_star)
            idx_max = jnp.argmax(acquisition)
            acquisiton_max = acquisition[idx_max]
            u_star_max = U_star[idx_max]
            better = acquisiton_max > acquisition_best
            acquisition_best = jnp.where(better, acquisiton_max, acquisition_best)
            u_best = jnp.where(better, u_star_max, u_best)
            # # Gradient-based improvement
            # bij = tfpb.Sigmoid()
            # x0 = bij.inverse(u_best)
            # opt_res = minimize(lambda x: -acquisition_function(u_star=bij.forward(x)), x0=x0, method='BFGS')
            # u_best = bij.forward(opt_res.x)
            # acquisition_best = -opt_res.fun
            return (acquisition_best, u_best), (acquisition, U_star)

        num_batches = num_search // batch_size
        (acquisition_best, u_best), (acquisition, U_star) = \
            scan(body,
                 (-jnp.inf, jnp.zeros(ndims, float_type)),
                 random.split(search_key, num_batches)
                 )
        # concatenate from stack
        acquisition = jnp.reshape(acquisition, (-1,))
        U_star = jnp.reshape(U_star, (acquisition.size, -1))

        return (acquisition_best, u_best), (acquisition, U_star)

    @staticmethod
    def _multistep_lookahead_search(key: PRNGKey, data: GaussianProcessData, ns_results: MarginalisationData,
                                    batch_size: int, max_depth: int, num_simulations: int, branch_factor: int,
                                    S: int):
        print("Starting search.")
        u_best, policy_output = run_multi_lookahead(
            rng_key=key,
            data=data,
            ns_results=ns_results,
            batch_size=batch_size,
            max_depth=max_depth,
            num_actions=branch_factor,
            num_simulations=num_simulations,
            S=S
        )
        output_file = 'search_tree.png'
        if not os.path.exists(output_file):
            print("Saving tree diagram to:", output_file)
            graph = convert_tree_to_graph(policy_output.search_tree)
            graph.draw(output_file, prog="dot")
        return u_best

    def search_U_top1(self, key: PRNGKey, ns_results: MarginalisationData, batch_size: int, num_search: int):
        conditional_predictive_factory = GaussianProcessConditionalPredictiveFactory(data=self._data)
        acquisition_factory = ExpectedImprovementAcquisitionFactory(
            conditional_predictive_factory=conditional_predictive_factory
        )
        search_key, marginalise_key = random.split(key)
        marginalised_acquisition = MarginalisedAcquisitionFunction(
            key=marginalise_key,
            ns_results=ns_results,
            acquisition_factory=acquisition_factory,
            S=self._S
        )

        return BayesianOptimiser._random_search(
            search_key=search_key,
            acquisition_function=marginalised_acquisition,
            ndims=conditional_predictive_factory.ndims(),
            batch_size=batch_size,
            num_search=num_search
        )

    def search_U_top2(self, key: PRNGKey, ns_results: MarginalisationData, u1: jnp.ndarray, batch_size: int,
                      num_search: int):
        conditional_predictive_factory = GaussianProcessConditionalPredictiveFactory(data=self._data)
        acquisition_factory = TopTwoAcquisitionFactory(
            conditional_predictive_factory=conditional_predictive_factory,
            u1=u1
        )
        search_key, marginalise_key = random.split(key)
        marginalised_acquisition = MarginalisedAcquisitionFunction(
            key=marginalise_key,
            ns_results=ns_results,
            acquisition_factory=acquisition_factory,
            S=self._S
        )

        return BayesianOptimiser._random_search(
            search_key=search_key,
            acquisition_function=marginalised_acquisition,
            ndims=conditional_predictive_factory.ndims(),
            batch_size=batch_size,
            num_search=num_search
        )

    def choose_next_U_toptwo(self, key: PRNGKey, batch_size: int, num_search: int):
        ns_key, search_top1_key, search_top2_key, do_top2_key = random.split(key, 4)

        do_top2 = random.uniform(do_top2_key) < self._beta

        ns_results = self.posterior_solve(key=ns_key)

        ns_results = MarginalisationData(
            samples=ns_results.samples,
            log_dp_mean=ns_results.log_dp_mean
        )

        # search over U-domain space for top1

        (_, next_u), (acquisition, U_star) = self.search_U_top1(
            key=search_top1_key,
            ns_results=ns_results,
            batch_size=batch_size,
            num_search=num_search
        )
        if len(U_star.shape) > 1:
            import pylab as plt
            sc = plt.scatter(U_star[:, 0], U_star[:, 1], c=acquisition, s=1, cmap='jet')
            plt.colorbar(sc)
            plt.title("Search top1 plot")
            plt.show()

        if do_top2:
            u1 = next_u
            (_, next_u), (acquisition, U_star) = self.search_U_top2(
                key=search_top2_key,
                ns_results=ns_results,
                u1=u1,
                batch_size=batch_size,
                num_search=num_search
            )
            if len(U_star.shape) > 1:
                import pylab as plt
                sc = plt.scatter(U_star[:, 0], U_star[:, 1], c=acquisition, s=1, cmap='jet')
                plt.colorbar(sc)
                plt.title("Search top2 plot")
                plt.show()

        return next_u

    def choose_next_U_multistep(self, key: PRNGKey, batch_size: int, max_depth: int, num_simulations: int,
                                branch_factor: int):
        ns_key, search_key = random.split(key, 2)

        ns_results = self.posterior_solve(key=ns_key)
        ns_results = MarginalisationData(
            samples=ns_results.samples,
            log_dp_mean=ns_results.log_dp_mean
        )

        next_u = BayesianOptimiser._multistep_lookahead_search(
            key=search_key,
            data=self._data,
            ns_results=ns_results,
            batch_size=batch_size,
            max_depth=max_depth,
            num_simulations=num_simulations,
            branch_factor=branch_factor,
            S=self._S
        )

        return next_u
