import diffrax as dfx
import jax
import jax.numpy as jnp
import optuna

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)
