import jax
import pylab as plt
from chex import PRNGKey
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from bojaxns.gaussian_process_formulation.distribution_math import GaussianProcessConditionalPredictive, \
    GaussianProcessData, ExpectedImprovementAcquisition

tfpd = tfp.distributions


def _expected_improvement(post_mu_x_max: jnp.ndarray, post_mu_s: jnp.ndarray,
                          post_var_s: jnp.ndarray) -> jnp.ndarray:
    post_stddev_s = jnp.sqrt(jnp.maximum(1e-6, post_var_s))
    posterior_pdf = tfpd.Normal(loc=0., scale=1.)
    u = (post_mu_s - post_mu_x_max) / post_stddev_s
    return post_stddev_s * (posterior_pdf.prob(u) + u * posterior_pdf.cdf(u))


def marginalised_mulitstep(key: PRNGKey, u1: jnp.ndarray, u2: jnp.ndarray, data: GaussianProcessData, S: int):
    max_depth = 2

    def _extend(x, value, extra_len):
        extra = jnp.full((extra_len,) + x.shape[1:], value)
        return jnp.concatenate([x, extra], axis=0)

    init_data_size = data.Y.size
    # Need up to depth-1
    data = GaussianProcessData(
        U=_extend(data.U, 0., max_depth - 1),
        Y=_extend(data.Y, 0., max_depth - 1),
        Y_var=_extend(data.Y_var, jnp.inf, max_depth - 1),
        sample_size=_extend(data.sample_size, jnp.inf, max_depth - 1),
    )

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=1.)
    variance = jnp.asarray(0.1)
    mean = jnp.asarray(0.)

    marginalise_idx = init_data_size

    def _sample_acquisition(key):
        conditional_predictive = GaussianProcessConditionalPredictive(
            data=data,
            kernel=kernel,
            variance=variance,
            mean=mean
        )
        marg_mu, marg_var = conditional_predictive(u1[None, :])
        Y_sample = marg_mu + jax.random.normal(key) * jnp.sqrt(jnp.maximum(1e-6, marg_var))
        Y_sample = jnp.reshape(Y_sample, ())
        _data = data._replace(
            U=data.U.at[marginalise_idx, :].set(u1),
            Y=data.Y.at[marginalise_idx].set(Y_sample),
            Y_var=data.Y_var.at[marginalise_idx].set(variance),
            sample_size=data.sample_size.at[marginalise_idx].set(1.)
        )
        conditional_predictive = GaussianProcessConditionalPredictive(
            data=_data,
            kernel=kernel,
            variance=variance,
            mean=mean
        )
        acquisition_function = ExpectedImprovementAcquisition(conditional_predictive=conditional_predictive)
        # marginalise_over
        return acquisition_function(u_star=u2)

    return jnp.nanmean(jax.vmap(_sample_acquisition)(jax.random.split(key, S)))

def marginalised_mulitstep_2(key: PRNGKey, u1: jnp.ndarray, u2: jnp.ndarray, data: GaussianProcessData, S: int):
    max_depth = 2

    def _extend(x, value, extra_len):
        extra = jnp.full((extra_len,) + x.shape[1:], value)
        return jnp.concatenate([x, extra], axis=0)

    init_data_size = data.Y.size
    # Need up to depth-1
    data = GaussianProcessData(
        U=_extend(data.U, 0., max_depth - 1),
        Y=_extend(data.Y, 0., max_depth - 1),
        Y_var=_extend(data.Y_var, jnp.inf, max_depth - 1),
        sample_size=_extend(data.sample_size, jnp.inf, max_depth - 1),
    )

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=1.)
    variance = jnp.asarray(0.1)
    mean = jnp.asarray(0.)

    marginalise_idx = init_data_size

    conditional_predictive = GaussianProcessConditionalPredictive(
        data=data,
        kernel=kernel,
        variance=variance,
        mean=mean
    )
    marg_mu, marg_var = conditional_predictive(u1[None, :])
    marg_mu = jnp.reshape(marg_mu, ())
    marg_var = jnp.reshape(marg_var, ())

    _data = data._replace(
        U=data.U.at[marginalise_idx, :].set(u1),
        Y=data.Y.at[marginalise_idx].set(marg_mu),
        Y_var=data.Y_var.at[marginalise_idx].set(marg_var),
        # sample_size=data.sample_size.at[marginalise_idx].set(jnp.inf)
    )
    conditional_predictive = GaussianProcessConditionalPredictive(
        data=_data,
        kernel=kernel,
        variance=variance,
        mean=mean
    )
    acquisition_function = ExpectedImprovementAcquisition(conditional_predictive=conditional_predictive)
    # marginalise_over
    return acquisition_function(u_star=u2)


if __name__ == '__main__':
    # mean_array = jnp.linspace(-5., 5., 100)
    # uncert_array = jnp.linspace(0., 20., 100)
    # X, Y = jnp.meshgrid(mean_array, uncert_array, indexing='ij')
    #
    # f_max = jnp.asarray(0.)
    # ac = jax.vmap(lambda mu, sigma: _expected_improvement(post_mu_x_max=f_max,
    #                                                       post_mu_s=mu,
    #                                                       post_var_s=sigma ** 2))(X.flatten(), Y.flatten())
    # ac = ac.reshape(X.shape)
    #
    # plt.imshow(ac,
    #            origin='lower',
    #            extent=(Y.min(), Y.max(), X.min(), X.max()),
    #            aspect='auto')
    # plt.xlabel('uncert')
    # plt.ylabel('mu')
    # plt.colorbar()
    # plt.show()

    U = jnp.linspace(0., 10., 20)[:, None]
    Y = jnp.sin(U[:, 0])
    Y_var = 0.1 * jnp.ones_like(Y)
    sample_size = jnp.ones_like(Y)
    data: GaussianProcessData = GaussianProcessData(
        U=U,
        Y=Y,
        Y_var=Y_var,
        sample_size=sample_size
    )

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=1.)
    variance = jnp.asarray(0.1)
    mean = jnp.asarray(0.)


    conditional_predictive = GaussianProcessConditionalPredictive(
        data=data,
        kernel=kernel,
        variance=variance,
        mean=mean
    )



    @jax.jit
    def _marg(U_test: jnp.ndarray):
        def _inner(u1: jnp.ndarray):
            def _outer(u2: jnp.ndarray):
                return marginalised_mulitstep_2(
                    key=jax.random.PRNGKey(42),
                    u1=u1,
                    u2=u2,
                    data=data,
                    S=400
                )
            return jax.vmap(_outer)(U_test)
        return jax.vmap(_inner)(U_test)

    U_test = jnp.linspace(-1., 2., 1000)[:, None]

    acquisition_function = ExpectedImprovementAcquisition(conditional_predictive=conditional_predictive)

    acq1 = jax.vmap(acquisition_function)(U_test)
    plt.plot(U_test[:,0], acq1)
    plt.show()

    marg_acq = _marg(U_test)
    (i, j) = jnp.where(marg_acq == jnp.max(marg_acq))
    print(U_test[i], U_test[j])
    plt.imshow(marg_acq,
               origin='lower',
               extent=(U_test.min(), U_test.max(), U_test.min(), U_test.max()),
               aspect='auto')
    plt.xlabel('u2')
    plt.ylabel('u1')
    plt.colorbar()
    plt.show()

    jnp.argmax(marg_acq, axis=0)
