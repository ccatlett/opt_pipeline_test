import argparse
import csv
import os
import pickle

import pandas as pd
import jax
import optuna

from src import objective

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def optimize_parameters(study_name, data_file, n_trials):
    # Load data
    with open(data_file, "rb") as f:
        loaded_data = pickle.load(f)

    # Create, run trial
    sampler = optuna.samplers.CmaEsSampler(
        restart_strategy="ipop",
        inc_popsize=2,
        lr_adapt = True,
    )
    study = optuna.create_study(direction="minimize",
                                sampler=sampler,
                                storage=f"sqlite:////{os.getcwd()}/results/{study_name}/{study_name}_db.sqlite3",
                                study_name=study_name,)
    study.optimize(lambda tr : objective(tr, loaded_data),
                    n_trials=n_trials,
                    show_progress_bar=True,)
    return study # return whole study object

if __name__ == "__main__":
    # Collect and parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("study_name", type=str, help="name of study")
    parser.add_argument("data_file", type=str, help="location of pickled data")
    parser.add_argument("N", type=int, help="number of trials")
    args = parser.parse_args()

    # Create results dir
    os.makedirs(f"results/{args.study_name}", exist_ok=True)

    # Run optimization, save as pkl
    study_results = optimize_parameters(args.study_name, args.data_file, args.N)
    with open(f"results/{args.study_name}/{args.study_name}_results.pkl", "wb") as f:
        pickle.dump(study_results, f)

    # Export as df (csv)
    df = study_results.trials_dataframe()
    df.to_csv(f'results/{args.study_name}/{args.study_name}_df.csv', index=False)  

    # Get importances, save as csv
    evaluator = optuna.importance.PedAnovaImportanceEvaluator()
    importance = optuna.importance.get_param_importances(study_results, evaluator=evaluator)
    with open(f'results/{args.study_name}/{args.study_name}_importance.csv','w') as f:
        w = csv.writer(f)
        w.writerow(importance.keys())
        w.writerow(importance.values())
     