import itertools
import numpy as np
import random
from code_for_ACO_RCPSP_PP_V4 import AntColonyRCPSP, jobs
import pandas as pd

def run_single_config(alpha, beta, rho, c, gamma, elitist_forget_generations, seed_trials=5):
    best_makespans = []
    for seed in range(seed_trials):
        random.seed(seed)
        np.random.seed(seed)
        aco = AntColonyRCPSP(
            jobs,
            n_ants=60,
            n_iterations=15,
            alpha=alpha,
            beta=beta,
            rho=rho,
            c=c,
            gamma=gamma,
            elitist_forget_generations=elitist_forget_generations
        )
        _, best_ms = aco.run()
        best_makespans.append(best_ms)

    return {
        'alpha': alpha,
        'beta': beta,
        'rho': rho,
        'c': c,
        'gamma': gamma,
        'elitist_forget_generations': elitist_forget_generations,
        'mean_makespan': np.mean(best_makespans),
        'std_makespan': np.std(best_makespans),
        'min_makespan': np.min(best_makespans),
        'max_makespan': np.max(best_makespans)
    }

def main():
    # Define parameter grid
    alpha_range = np.round(np.arange(0.5, 2.1, 0.5), 2)
    beta_range = np.round(np.arange(1.0, 2.1, 0.5), 2)
    rho_range = np.round(np.arange(0.05, 0.21, 0.05), 2)
    c_range = np.round(np.arange(0.2, 1.01, 0.2), 2)
    gamma_range = np.round(np.arange(0.5, 1.01, 0.1), 2)
    elitist_range = list(range(5, 21, 5))

    # alpha_range = [1]
    # beta_range = [2]
    # rho_range = [0.1]
    # c_range = [0.5]
    # gamma_range = [1]
    # elitist_range = [10]


    param_grid = list(itertools.product(alpha_range, beta_range, rho_range, c_range, gamma_range, elitist_range))

    print(f"Total configurations: {len(param_grid)}")

    results = []

    for idx, (alpha, beta, rho, c, gamma, elitist) in enumerate(param_grid):
        print(f"\n[{idx + 1}/{len(param_grid)}] Testing: alpha={alpha}, beta={beta}, rho={rho}, c={c}, gamma={gamma}, elitist={elitist}")
        result = run_single_config(alpha, beta, rho, c, gamma, elitist)
        results.append(result)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.sort_values(by=['mean_makespan', 'std_makespan'], inplace=True)
    df.to_csv("aco_parameter_search_results.csv", index=False)
    print("\nTop 10 Configurations:")
    print(df.head(10))

if __name__ == "__main__":
    main()
