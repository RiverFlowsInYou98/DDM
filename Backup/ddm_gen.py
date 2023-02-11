import jax
import jax.numpy as jnp
from jax import lax, random, jit, tree_util, vmap


@tree_util.register_pytree_node_class
class DDModel(object):
    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma

    def strong_Euler(self, init_cond, T, Nt=1000, key: random.PRNGKey = random.PRNGKey(0)):
        """
        Use Gaussian distributed increments to simulate the process
        """
        t0, u0 = init_cond
        dt = (T - t0) / Nt
        t_grid = jnp.arange(t0, T + dt, dt)
        dW = jnp.sqrt(dt) * random.normal(key, shape=(Nt + 1,))
        du = self.mu * dt + self.sigma * dW
        u_grid = jnp.cumsum(du)
        return t_grid, u_grid

    def weak_Euler(self, init_cond, T, Nt=1000, key: random.PRNGKey = random.PRNGKey(0)):
        """
        Use Bernoulli distributed increments to simulate the process
        """
        t0, u0 = init_cond
        dt = (T - t0) / Nt
        t_grid = jnp.arange(t0, T + dt, dt)
        dW = jnp.sqrt(dt) * (2 * random.bernoulli(key, shape=(Nt + 1,)) - 1)
        du = self.mu * dt + self.sigma * dW
        u_grid = jnp.cumsum(du)
        return t_grid, u_grid

    @jit
    def gen_datum(self, a, z, dt=0.001, key: random.PRNGKey = random.PRNGKey(0)):
        """
        generate a data pair of the form (T, C)
        where T is the first passage time
            C is the crossed boundary (0 or a)
        """

        def cond_fun(carry):
            step, pos, key = carry
            return jnp.logical_and(pos > 0, pos < a)

        def gaussian_walk(carry):
            prev_step, prev_pos, key = carry
            dW = jnp.sqrt(dt) * random.normal(key)
            _, key = random.split(key)
            du = self.mu * dt + self.sigma * dW
            return prev_step + 1, prev_pos + du, key

        step, uT, _ = lax.while_loop(cond_fun, body_fun=gaussian_walk, init_val=(0, z, key))
        return jnp.hstack((step * dt, (uT > a / 2) * a))

    @jit
    def gen_data(self, a, z, num=1000, dt=0.001, key: random.PRNGKey = random.PRNGKey(0)):
        """
        generate 'num' data pairs of the form (Ck, Tk)
        """
        keys = random.split(key, num=num)
        vec_gen = vmap(self.gen_datum, in_axes=(None, None, None, 0), out_axes=0)
        return vec_gen(a, z, dt, keys)

    def tree_flatten(self):
        children = (self.mu, self.sigma, )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    
if __name__ == '__main__':
    ddm = DDModel(mu=0.2, sigma=1)
    data = ddm.gen_data(a=4.0, z=1.5)
    jnp.save('data', data)
    print('data saved!')