import pickle

import jax
import optuna

from src import objective

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def optimize_parameters(data_file="data/syn_data_lv.pkl"):
    # Load data
    with open(data_file, "rb") as f:
        loaded_data = pickle.load(f)

    # Create, run trial
    sampler = optuna.samplers.CmaEsSampler(
        restart_strategy="ipop",
        inc_popsize=2,
        lr_adapt = True,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda tr : objective(tr, loaded_data), n_trials=10000, show_progress_bar=True)
    return study.best_params

if __name__ == "__main__":
    best_pars = optimize_parameters()
    print("\n")
    print(best_pars)