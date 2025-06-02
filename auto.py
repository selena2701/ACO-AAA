import itertools
import numpy as np
import random
from ACO_RCPSP import AntColonyRCPSP, jobs
import pandas as pd

def run_single_config(alpha, beta, rho, c, gamma, elitist_forget_generations,n_ants, n_iterations, seed_trials):
    best_makespans = []
    for seed in range(seed_trials):
        random.seed(seed)
        np.random.seed(seed)

        aco = AntColonyRCPSP(
            jobs,
            n_ants=n_ants, 
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            c=c,
            gamma=gamma,
            elitist_forget_generations=elitist_forget_generations
        )

        _, best_ms = aco.run()
        best_makespans.append(best_ms)

    print(np.min(best_makespans))

    return {
        'n_ants': n_ants,
        'n_iterations': n_iterations,
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


if __name__ == "__main__":

    # Define parameter grid
    n_ants_range = range(30, 90, 10) #[30, 60, 90] # Number of ants
    n_iterations_range = [None] #[5, 10, 15] #  Number of iterations
    elitist_range = [None] # [5, 10, 15] # Number of elitist generations

    # Fixed parameters
    alpha_range = [1]
    beta_range = [3]
    rho_range = [0.1]
    c_range = [0.5]
    gamma_range = [1]
    seed_trials = 3

    #Creating combinations
    param_grid = list(itertools.product(alpha_range, beta_range, rho_range, c_range, gamma_range, elitist_range, n_ants_range, n_iterations_range))
    print(f"Total configurations: {len(param_grid)}")

    #Running combinations
    results = []

    for idx, (alpha, beta, rho, c, gamma, elitist, ant, iteration) in enumerate(param_grid):
        print(f"\n[{idx + 1}/{len(param_grid)}] Testing: alpha={alpha}, beta={beta}, rho={rho}, c={c}, gamma={gamma}, elitist={elitist}, ant={ant}, iteration={iteration}")
        result = run_single_config(alpha, beta, rho, c, gamma, elitist, ant, ant, seed_trials)
        results.append(result)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.sort_values(by=['mean_makespan', 'std_makespan'], inplace=True)
    df.to_csv("experiment/aco_parameter_search_results.csv", index=False)
    print("\nTop 10 Configurations:")
    print(df.head(10))
