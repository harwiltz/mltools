import jax
import jax.numpy as jnp

from mltools.distribution import Distribution
from typing import Any, List, Optional, Union

class Normal(Distribution):
    """ Univariate normal distribution """
    def __init__(self, loc: float, scale: float) -> Distribution:
        self.loc = loc
        self.scale = scale

    def sample(self, rng: jax.random.PRNGKey) -> float:
        return self.loc + self.scale * jax.random.normal(rng)

    def instantiate(params: str) -> Distribution:
        tokens = params.split()
        assert len(tokens) >= 2, \
            f"{__class__.__name__} requires 2 parameters: loc and scale"
        loc = float(tokens[0])
        scale = float(tokens[1])
        return Normal(loc, scale)

class MixedNormal(Distribution):
    """ Univariate mixture of gaussians """
    def __init__(self,
                 dists: List[Normal],
                 weights: Optional[List[float]]=None) -> Distribution:
        self.dists = dists
        self.n = len(dists)
        if weights is None:
            self.weights = jnp.ones(self.n, dtype=jnp.float32) / self.n
        else:
            weights = jnp.array(weights)
            assert jnp.all(weights >= 0), "Mixture weights must be nonnegative"
            assert len(weights) == self.n, \
                "Number of weights does not match number of distributions:\n" + \
                f"Expect {self.n}, got {len(weights)}"
            self.weights = weights / jnp.sum(weights)

    def sample(self, rng: jax.random.PRNGKey) -> float:
        rng, sub = jax.random.split(rng)
        i = jax.random.choice(rng, self.n, p=self.weights)
        return self.dists[i].sample(sub)

    def instantiate(params: str) -> Distribution:
        args = params.split(",")
        dists = []
        weights = []
        for arg in args:
            tokens = arg.split()
            dists.append(Normal.instantiate(" ".join(tokens[:2])))
            if len(tokens) >= 3:
                weights.append(float(tokens[2]))
            else:
                weights.append(None)
        assert jnp.all(weights is None) or jnp.all(weights is not None), \
            f"Error instantiating {__class__.__name__} with weights {weights}:\n" + \
            "Cannot leave a subset of weights unspecified"
        if jnp.all(weights is None):
            weights = None
        return MixedNormal(dists, weights)
