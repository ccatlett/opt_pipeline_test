import jax.numpy as jnp


def lotka_volterra(t, y, args):
    """
    Defines the Lotka-Volterra predator-prey system of ODEs.

    Args:
        t (float): Current time.
        y (jax.numpy.ndarray): Current values of prey and predator populations [x, y].
        args (tuple): Tuple containing params and condition_params.
            - params (numpy.ndarray): Parameters of the system [alpha, beta, delta, gamma].
            - condition_params (numpy.ndarray): Condition-specific parameters.

    Returns:
        jax.numpy.ndarray: Derivatives of prey and predator populations [dx/dt, dy/dt].
    """

    # Unpack args
    params, _ = args
    alpha, beta, delta, gamma = params
    x, y = y

    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return jnp.array([dxdt, dydt])
