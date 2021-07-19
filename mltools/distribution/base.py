import jax
import jax.numpy as jnp

from typing import Any

class Distribution:
    """
    Distribution base class
    """

    def sample(rng : jax.random.PRNGKey) -> Any:
        raise NotImplementedError
