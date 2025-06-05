import itertools
import numpy as np
import random
import pandas as pd
import time

from ACO_RCPSP import AntColonyRCPSP, jobs


def run_single_config(alpha, beta, rho, c, gamma, elitist_forget_generations, n_ants, n_iterations, seed_trials):
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
    
    #### VARIABLE ranges for parameter grid ###
    n_ants_range = range(30, 90, 30) #[30, 60, 90] # Number of ants
    n_iterations_range = range(5, 15, 5) #[5, 10, 15] #  Number of iterations
    elitist_forget_range = range(0, 15, 5) # [5, 10, 15] # Number of generations forget elitist ants
    # alpha_range = range(1, 3, 1)  # Range for alpha
    beta_range = range(1, 3, 1) 
    # Fixed parameters
    # n_ants_range = [30]  # Fixed for this run
    # n_iterations_range = [10]  # Fixed for this run
    # elitist_range = [10]  # Fixed for this run

    ### FIXED parameters for this run ###   
    alpha_range = [1]
    # beta_range = [3]
    rho_range = [0.1]
    c_range = [0.5]
    gamma_range = [1]
    seed_trials = 10

    # Generate and deduplicate parameter grid
    raw_param_grid = list(itertools.product(
        alpha_range, beta_range, rho_range, c_range, gamma_range,
        elitist_forget_range, n_ants_range, n_iterations_range
    ))

    param_grid = sorted(set(raw_param_grid))
    print(f"Original configs: {len(raw_param_grid)} → Unique configs: {len(param_grid)}")

    # Run all unique configurations
    results = []

    for idx, (alpha, beta, rho, c, gamma, elitist, ant, iteration) in enumerate(param_grid):
        print(f"\n[{idx + 1}/{len(param_grid)}] Testing: alpha={alpha}, beta={beta}, rho={rho}, c={c}, "
              f"gamma={gamma}, elitist={elitist}, n_ants={ant}, n_iterations={iteration}")
        start = time.time()
        result = run_single_config(alpha, beta, rho, c, gamma, elitist, ant, iteration, seed_trials)
        print(f"Completed in {time.time() - start:.2f} seconds. Mean Makespan: {result['mean_makespan']:.2f}")
        results.append(result)

    # Save results
    df = pd.DataFrame(results)
    df.drop_duplicates(inplace=True)  # Optional: remove any accidental duplicates
    df.sort_values(by=['min_makespan', 'mean_makespan', 'std_makespan'], inplace=True)
    df.to_csv("results/aco_parameter_search_results_j6048_3.csv", index=False)

    # Show top configurations
    print("\nTop 10 Configurations:")
    print(df.head(10))
