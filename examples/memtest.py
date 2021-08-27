import jax
import jax.numpy as jnp

from haiku import PRNGSequence

from mltools.logging.html import HTMLExperimentLogger

@profile
def main():
    monitor = HTMLExperimentLogger("memtest",
                                   template_path="/home/harwiltz/lab/mltools/examples/",
                                   template_name="html-logger-template.html")
    rngs = PRNGSequence(jax.random.PRNGKey(0))
    i = 0
    monitor.log_metric("dummy", jax.random.uniform(next(rngs), shape=(4,)).sum(), i)
    i += 1
    monitor.log_metric("dummy", jax.random.uniform(next(rngs), shape=(4,)).sum(), i)
    i += 1
    monitor.log_metric("dummy", jax.random.uniform(next(rngs), shape=(4,)).sum(), i)

if __name__ == "__main__":
    main()
