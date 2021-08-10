import jax
import jax.numpy as jnp

@jax.custom_vjp
def clip_gradient(lo, hi, x):
    r"""
    Differentiable gradient clipping function.
    Taken from https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
    """
    return x

def clip_gradient_fwd(lo, hi, x):
    return x, (lo, hi)

def clip_gradient_bwd(res, g):
    lo, hi = res
    return (None, None, jnp.clip(g, lo, hi))

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)

def confine(bound: float,
            x    : jnp.ndarray) -> jnp.ndarray:
    r"""
    Rescale a vector if its norm is too large.
    
    Parameters
    ----------
    bound : float
        The maximum allowable norm
    x : jnp.ndarray
        The vector to rescale
    """
    norm = jnp.max(jnp.sqrt(jnp.square(x).sum(axis=-1)))
    return jnp.where(norm > bound,
                     x * bound / norm,
                     x)
