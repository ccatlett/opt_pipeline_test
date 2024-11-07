import pickle

import numpy as np

from src import generate_y0, simulate_ode


def create_noisy_data(params, condition_params_list, t0, t1, dt, noise_percentage, output_file, timepoints=None):
    """
    Simulates the ODE system for multiple conditions, adds noise to the results,
    and saves the data to a pickle file.

    Args:
        params (numpy.ndarray): Parameters of the Lotka-Volterra system [alpha, beta, delta, gamma].
        condition_params_list (numpy.ndarray): Array of condition-specific parameters.
            Each entry specifies the initial populations for [prey, predator] for a specific condition.
        t0 (float): Start time of the simulation.
        t1 (float): End time of the simulation.
        dt (float): Initial step size for the solver.
        noise_percentage (float): Percentage of noise to add, relative to the maximum simulated values.
        output_file (str): Path to the file where the generated data will be saved as a pickle object.

    Returns:
        None. Saves data as a list of tuples (params, condition_params, y0, noised_data) to the specified output file.
    """

    data = []
    for condition_params in condition_params_list:
        # Ensure np arr
        condition_params = np.array(condition_params, dtype=np.float64)

        # Generate initial condition y0 based on condition_params
        y0 = generate_y0(params, condition_params)

        # Simulate ODE
        clean_data_t, clean_data = simulate_ode(params, condition_params, y0, t0, t1, dt, timepoints)

        # Add noise
        noise = (noise_percentage / 100
            * np.max(np.abs(clean_data))
            * np.random.normal(size=clean_data.shape))
        noised_data = clean_data + noise

        # Collect tuple of (params, condition_params, y0, noised_data)
        data.append((params, condition_params, y0, clean_data_t, noised_data))

    # Save to pickle file
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    params = np.array([0.5, 0.02, 0.01, 0.4], dtype=np.float64)  # Lotka-Volterra parameters: alpha, beta, delta, gamma
    condition_params_list = np.array([[40.0, 9.0], [10.0, 35.0], [8.0, 10.0]], dtype=np.float64)  # Initial populations of prey and predator
    t0, t1, dt = 0.0, 50.0, 1e-3
    timepoints = np.linspace(t0, t1, 500)
    noise_percentage = 0.0
    output_file = "data/syn_data_lv.pkl"

    create_noisy_data(params, condition_params_list, t0, t1, dt, noise_percentage, output_file, timepoints)

    # Test data creation (sanity check)
    with open(output_file, "rb") as f:
        loaded_data = pickle.load(f)

    print("\nConfirmation test:\n")
    print("\tLoaded data structure:", type(loaded_data))
    print("\tNumber of conditions:", len(loaded_data))

    # Check the structure of each dataset
    for idx, (loaded_params, loaded_condition_params, loaded_y0, loaded_t, loaded_noised_data) in enumerate(loaded_data):
        print(f"\n\tCondition {idx + 1}:")
        print("\t\tParameters:", loaded_params)
        print("\t\tCondition-specific params:", loaded_condition_params)
        print("\t\tInitial conditions (y0):", loaded_y0)
        print("\t\tTimes solved (sample):", loaded_t[:5])

        loaded_data_noise_str = np.array_str(loaded_noised_data[:5], max_line_width=50)  # Print the first few rows to confirm structure
        loaded_data_noise_str = loaded_data_noise_str.replace("\n", "\n\t\t\t\t      ")
        print("\t\tNoised data (sample):", loaded_data_noise_str) 

        assert np.allclose(params, loaded_params), "Params do not match!"
        assert np.allclose(condition_params_list[idx], loaded_condition_params), "Condition-specific params do not match!"
        assert len(loaded_noised_data) > 0, "Noised data is empty!"
        assert loaded_noised_data.shape[1] == len(loaded_y0), "Noised data has incorrect columns!"

    print("\nTest completed: Data structure and values appear correct.\n")
