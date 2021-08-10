import jax
import jax.numpy as jnp
import unittest

import mltools.ops as ops

from functools import partial

class TestOps(unittest.TestCase):
    def setUp(self):
        self.bound = 10.
        self.small_vector = jnp.array([3.,4.])
        self.big_vector = jnp.array([12, 16.])

    def test_confine_vector(self):
        v = ops.confine(self.bound, self.small_vector)
        self.assertTrue(jnp.allclose(v, self.small_vector))
        V = ops.confine(self.bound, self.big_vector)
        self.assertFalse(jnp.allclose(V, self.big_vector))
        self.assertEqual(jnp.sqrt(jnp.square(V).sum()), self.bound)
        self.assertEqual(V[1] / V[0], self.big_vector[1] / self.big_vector[0])

    def test_confine_matrix(self):
        matrix = jnp.array([self.small_vector, self.big_vector])
        m = ops.confine(self.bound, matrix)
        self.assertTrue(norm(m) <= self.bound)
        self.assertFalse(jnp.allclose(m[0], self.small_vector))
        self.assertTrue(same_ratio(m[0], self.small_vector))
        self.assertFalse(jnp.allclose(m[1], self.big_vector))
        self.assertTrue(same_ratio(m[1], self.big_vector))
        self.assertEqual(norm(m[1]), self.bound)

    def test_confine_batch_vectors(self):
        batch = jnp.array([self.small_vector, self.big_vector])
        b = jax.vmap(partial(ops.confine, self.bound))(batch)
        self.assertTrue(jnp.allclose(b[0], self.small_vector))
        self.assertFalse(jnp.allclose(b[1], self.big_vector))
        self.assertEqual(norm(b[1]), self.bound)
        self.assertTrue(same_ratio(b[1], self.big_vector))

def same_ratio(v1, v2, eps=1e-6):
    return jnp.abs((v1[1] / v1[0]) - (v2[1] / v2[0])) < eps

def norm(v):
    return jnp.max(jnp.sqrt(jnp.square(v).sum(axis=-1)))

if __name__ == "__main__":
    unittest.main()
