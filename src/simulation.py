import diffrax as dfx
import jax.numpy as jnp
import numpy as np

from .ode_def import lotka_volterra


def generate_y0(params, condition_params):
    """
    Generates initial conditions for prey and predator populations based on parameters.

    Args:
        params (numpy.ndarray): Parameters of the Lotka-Volterra system, though not used directly here.
        condition_params (numpy.ndarray): Initial conditions for the populations, typically
                                          [initial_prey_population, initial_predator_population].

    Returns:
        jax.numpy.ndarray: Initial values [prey, predator] for the populations.
    """

    prey_init = condition_params[0]
    predator_init = condition_params[1]
    return jnp.array([prey_init, predator_init])


def simulate_ode(params, condition_params, y0, t0, t1, dt):
    """
    Simulates the Lotka-Volterra ODE system for given parameters and initial conditions.

    Args:
        params (numpy.ndarray): Parameters of the Lotka-Volterra system [alpha, beta, delta, gamma].
        condition_params (numpy.ndarray): Condition-specific parameters, passed to the ODE system function.
        y0 (jax.numpy.ndarray): Initial conditions for the populations [prey, predator].
        t0 (float): Start time of the simulation.
        t1 (float): End time of the simulation.
        dt (float): Initial step size for the solver.

    Returns:
        numpy.ndarray: Array of simulated values for prey and predator populations over time.
    """

    solver = dfx.Kvaerno5()
    term = dfx.ODETerm(lotka_volterra)
    stepsize_controller = dfx.PIDController(rtol=1e-6, atol=1e-8)  # adaptive step-size
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        args=(params, condition_params),  # Pass params and condition_params as tuple
        stepsize_controller=stepsize_controller,
    )

    print(f"Number of time points: {len(sol.ts)}")  # Check the number of time steps
    print(f"Final time point: {sol.ts[-1]}")
    print(f"Shape of the solution: {sol.ys.shape}")  # Should be (number of time points, 2)

    return np.array(sol.ys)
