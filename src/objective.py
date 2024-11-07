import jax.numpy as jnp
from .simulation import simulate_ode, generate_y0

def compute_loss(simulated_data, experimental_data):
    """
    Computes the loss between the predicted values and observed data, considering the noise level.

    Args:
        simulated_data (numpy.ndarray): The predicted values from the ODE solve
        experimental_data (numpy.ndarray): Experimental data from pickled data object

    Returns:
        float: The computed loss value, which quantifies the discrepancy between the predicted and observed data.
    """

    return jnp.sum(jnp.divide(simulated_data - experimental_data, experimental_data) ** 2)

def objective(trial, data_pkl_lst):
    """
    Defines the objective function to be optimized, which is the loss between the simulated model predictions and observed data.

    Args:
        trial (Optuna.trial.Trial): Trial object in Optuna optimization routine
        data_pkl_lst (list of tuples): A list of tuples containing initial conditions and noisy data for each condition.

    Returns:
        float: The computed objective value, equivalent to loss.
    """

    # Define parameter space
    alpha = trial.suggest_float("alpha", 0., 1.)
    beta = trial.suggest_float("beta", 0., 0.05)
    delta = trial.suggest_float("delta", 0., 0.05)
    gamma = trial.suggest_float("gamma", 0., 1.)
    params = jnp.array([alpha, beta, delta, gamma], dtype=jnp.float64)

    # Define condition-specific parameters
    num_conditions = len(data_pkl_lst)
    condition_params = jnp.array([
        [trial.suggest_float(f"condition_{i}_x0", 0., 50.),
        trial.suggest_float(f"condition_{i}_y0", 0., 50.)]
        for i in range(num_conditions)
    ], dtype=jnp.float64)

    # Run simulation and calculate the loss
    total_loss = 0.
    for i in range(num_conditions):
        experimental_t, experimental_data = data_pkl_lst[i]
        y0 = generate_y0(params, condition_params[i])
        t0, t1, dt = experimental_t[0], experimental_t[-1], 1e-3*jnp.diff(experimental_t)[0]
        _, simulated_data = simulate_ode(params, condition_params[i], y0, t0, t1, dt, timepoints=experimental_t)
        loss = compute_loss(simulated_data, experimental_data)
        total_loss += loss

    return jnp.divide(total_loss, num_conditions)
