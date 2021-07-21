import jax
import jax.numpy as jnp

from typing import Any

Probability = float

class Distribution:
    """
    Distribution base class
    """

    def sample(self, rng : jax.random.PRNGKey) -> Any:
        raise NotImplementedError

    def pdf(self, x: Any) -> Probability:
        raise NotImplementedError
