
#Importing read-sm-files.py
import ReadSMFIles
from Resource import Resource
from Task import Task
import random
from typing import List, Tuple, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

print = lambda *args, **kwargs: None

class AntColonyRCPSP:
    def __init__(self, tasks: List[Task], resources: List[Resource], n_ants: int = 10, n_iterations: int = 50,
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1, c: float = 0.5, gamma: float = 1.0, elitist_forget_generations = None):
        self.tasks = tasks
        self.resources = resources
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # pheromone importance
        self.initial_beta = beta  # initial heuristic importance
        self.rho = rho      # evaporation rate
        self.elitist_forget_generations = elitist_forget_generations

        self.n_tasks = len(tasks)
        self.pheromone = np.random.uniform(0.1, 1.0, (self.n_tasks, self.n_tasks))
        self.best_schedule = None
        self.best_makespan = float('inf')
        self.elite_solutions = []
        self.makespan_history = []

    def _heuristic(self, from_task: Task, to_task: Task) -> float:
        duration_factor = 1.0 / (to_task.duration + 1)
        resource_factor = 1.0 / (sum(to_task.renewable_resources.values()) + 1)
        return duration_factor * resource_factor

    def _local_pheromone(self, from_task: Task, to_task: Task) -> float:
        from_idx = self.tasks.index(from_task)
        to_idx = self.tasks.index(to_task)
        return self.pheromone[from_idx][to_idx]

    def _summation_pheromone(self, from_task: Task, to_task: Task) -> float:
        from_idx = self.tasks.index(from_task)
        to_idx = self.tasks.index(to_task)
        # Sum pheromone values from all paths to the target task
        return sum(self.pheromone[i][to_idx] for i in range(self.n_tasks))
    
    def _combined_pheromone(self, schedule: List[Task], candidate: Task, gamma: float) -> Tuple[float, float]:
        """
        Compute direct and summation pheromone values for a candidate task.
        schedule: list of already scheduled tasks (defines positions 0..k-1)
        candidate: task being considered at position k
        gamma: decay factor for summation pheromone
        """
        task_idx = self.tasks.index(candidate)
        direct_pheromone = 0.0
        sum_pheromone = 0.0

        if schedule:
            # Last scheduled task index
            prev_task = schedule[-1]
            prev_idx = self.tasks.index(prev_task)
            direct_pheromone = self.pheromone[prev_idx][task_idx]

            # Summation pheromone
            for pos, prev_task in enumerate(schedule):
                prev_idx = self.tasks.index(prev_task)
                weight = gamma ** (len(schedule) - pos - 1)
                sum_pheromone += weight * self.pheromone[prev_idx][task_idx]

        return direct_pheromone, sum_pheromone

    def _probabilities(self, current_task: Task, eligible: List[Task], iteration: int) -> List[float]:
        # Calculate beta decay
        current_beta = self.initial_beta * (1.0 - (iteration / self.n_iterations) * 0.5)
        
        # Add more randomness in early iterations
        exploration_factor = 1.0 - (iteration / self.n_iterations) * 0.8
        random_factors = np.random.uniform(0.5, 1.5, len(eligible))
        
        # Calculate pheromone values with more exploration
        pheromone_vals = []
        for t in eligible:
            # Combine local and global pheromone information
            local_pher = self._local_pheromone(current_task, t)
            global_pher = self._summation_pheromone(current_task, t)
            # Add some randomness to pheromone values
            pher_val = 0.7 * local_pher + 0.3 * global_pher
            pher_val *= random_factors[len(pheromone_vals)]
            pheromone_vals.append(pher_val)
        
        # Calculate heuristic values
        heuristic_vals = [self._heuristic(current_task, t) for t in eligible]
        
        # Add small random noise to heuristic values
        heuristic_noise = np.random.uniform(0.8, 1.2, len(eligible))
        heuristic_vals = [h * n for h, n in zip(heuristic_vals, heuristic_noise)]
        
        # Calculate probabilities with exploration
        product = [(pheromone_vals[i] ** self.alpha) * 
                  (heuristic_vals[i] ** current_beta)
                  for i in range(len(eligible))]
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        product = [p + epsilon for p in product]
        total = sum(product)
        
        # Normalize probabilities
        probabilities = [p / total for p in product]
        
        # Add some minimum probability to all options
        min_prob = 0.05
        probabilities = [(1 - min_prob * len(probabilities)) * p + min_prob for p in probabilities]
        
        return probabilities

    def _get_eligible_tasks(self, scheduled: List[Task], unscheduled: List[Task]) -> List[Task]:
        eligible = []
        for i, task in enumerate(unscheduled):
            if all(pred in scheduled for pred in task.predecessors):
                eligible.append(task)
        return eligible

    def _schedule_makespan(self, task_order: List[Task]) -> int:
        max_time = 1000
        resource_availability = {
            resource.name: np.ones(max_time) * resource.per_period_availability
            for resource in self.resources
        }
        task_times = {task.name: {'start': 0, 'end': 0} for task in task_order}

        for task in task_order:
            if task.name == '1':
                continue
            earliest_start = 0
            for pred in task.predecessors:
                earliest_start = max(earliest_start, task_times[pred.name]['end'])
            start_time = earliest_start
            while True:
                resources_available = True
                for resource, amount in task.renewable_resources.items():
                    if amount > 0:
                        if np.any(resource_availability[resource.name][start_time:start_time + task.duration] < amount):
                            resources_available = False
                            break
                if resources_available:
                    break
                start_time += 1

            task_times[task.name]['start'] = start_time
            task_times[task.name]['end'] = start_time + task.duration
            for resource, amount in task.renewable_resources.items():
                if amount > 0:
                    resource_availability[resource.name][start_time:start_time + task.duration] -= amount

        makespan = max(task['end'] for task in task_times.values())
        return makespan

    def _2opt_swap(self, route: List[Task], i: int, j: int) -> List[Task]:
        new_route = route[:i]
        new_route.extend(reversed(route[i:j+1]))
        new_route.extend(route[j+1:])
        return new_route

    def _is_valid_sequence(self, seq: List[Task]) -> bool:
        pos = {t.name: idx for idx, t in enumerate(seq)}
        for task in seq:
            for pred in task.predecessors:
                if pos[pred.name] > pos[task.name]:
                    return False
        return True

    def _local_search(self, task_seq: List[Task]) -> Tuple[List[Task], int]:
        best_seq = task_seq
        best_makespan = self._schedule_makespan(task_seq)
        improved = True

        while improved:
            improved = False
            for i in range(1, len(task_seq)-1):
                for j in range(i+1, len(task_seq)):
                    new_seq = self._2opt_swap(best_seq, i, j)
                    if not self._is_valid_sequence(new_seq):
                        continue
                    new_makespan = self._schedule_makespan(new_seq)
                    if new_makespan < best_makespan:
                        best_seq = new_seq
                        best_makespan = new_makespan
                        improved = True
                        break
                if improved:
                    break

        return best_seq, best_makespan

    def _construct_solution(self, iteration: int) -> Tuple[List[Task], int]:
        scheduled = []
        unscheduled = self.tasks[1:]  # Start with all tasks except dummy start
        current_task = self.tasks[0]  # Start from dummy start task
        scheduled.append(current_task)
        i = 0   
        while unscheduled:
            eligible = self._get_eligible_tasks(scheduled, unscheduled)
            if not eligible:
                # If no eligible tasks, try to find any task that can be scheduled
                for task in unscheduled:
                    if all(pred in scheduled for pred in task.predecessors):
                        eligible.append(task)
                        break
                if not eligible:
                    break
            i += 1
            # Add some randomness to task selection
            if random.random() < 0.03:  # 10% chance of random selection
                next_task = random.choice(eligible)
            else:
                probs = self._probabilities(current_task, eligible, iteration)
                next_task = random.choices(eligible, weights=probs, k=1)[0]
            
            scheduled.append(next_task)
            unscheduled.remove(next_task)
            current_task = next_task

        # Add any remaining unscheduled tasks at the end
        if unscheduled:
            scheduled.extend(unscheduled)

        # Calculate makespan with resource constraints
        makespan = self._schedule_makespan(scheduled)
        return scheduled, makespan

    def _update_pheromones(self, solutions: List[Tuple[List[Task], int]], iteration: int):
        # Evaporate pheromones
        self.pheromone *= (1 - self.rho)
        
        # Add some random pheromone to encourage exploration
        random_pheromone = np.random.uniform(0, 0.1, self.pheromone.shape)
        self.pheromone += random_pheromone
        
        # Update pheromones based on solutions
        for task_seq, makespan in solutions:
            if makespan > 0:  # Avoid division by zero
                delta = 1.0 / makespan
                for i in range(len(task_seq) - 1):
                    from_idx = self.tasks.index(task_seq[i])
                    to_idx = self.tasks.index(task_seq[i + 1])
                    self.pheromone[from_idx][to_idx] += delta
        
        # Add decreasing randomness based on iteration
        random_factor = 0.2 * (1.0 - iteration / self.n_iterations)
        self.pheromone += np.random.uniform(0, random_factor, self.pheromone.shape)
        
        # Normalize pheromone values
        self.pheromone = np.clip(self.pheromone, 0.1, 10.0)

        # Elitist forgetting with more randomness
        if self.elitist_forget_generations and iteration > self.elitist_forget_generations:
            self.elite_solutions = []
            # Add more randomness to the reset
            self.pheromone = np.random.uniform(0.1, 2.0, (self.n_tasks, self.n_tasks))

    def run(self):
        for iteration in range(self.n_iterations):
            print(f"\nIteration {iteration + 1}")
            print("=" * 50)
            
            solutions = []
            for ant in range(self.n_ants):
                print(f"\nAnt {ant + 1}")
                print("-" * 30)
                task_seq, makespan = self._construct_solution(iteration)

                # need to deep dive why local search is violating the constraints
                # Only apply local search if the solution improves the current best
                if makespan < self.best_makespan:
                    print("\n Applying local search to improve promising solution...")
                    task_seq, makespan = self._local_search(task_seq)

                # Store the solution (even if not the best) for pheromone update
                solutions.append((task_seq, makespan))

                # Check again if this (possibly improved) solution is the best so far
                if makespan < self.best_makespan:
                    self.best_makespan = makespan
                    self.best_schedule = task_seq[:]
                    print(f"\n New best solution found after local search! Makespan: {self.best_makespan}")

                # Print detailed task order information
                print("\nTask Order Details:")
                print("Step | Task | Duration | Resource Requirements (R1, R2, R3, R4)")
                print("-" * 80)
                for step, task in enumerate(task_seq, 1):
                    resource_reqs = [task.renewable_resources.get(resource, 0) for resource in self.resources]
                    print(f"{step:4} | {task.name:4} | {task.duration:8} | {resource_reqs}")

                print("\nFinal Task Sequence:", " -> ".join([task.name for task in task_seq]))
                print(f"Makespan: {makespan}")

            self._update_pheromones(solutions, iteration)
            print(f"\nIteration {iteration+1} completed. Best Makespan so far: {self.best_makespan}")
            self.makespan_history.append(self.best_makespan)


        # Print the best solution found
        print("\nBest Solution Found:")
        print("=" * 50)
        print("Task Order Details:")
        print("Step | Task | Duration | Resource Requirements (R1, R2, R3, R4)")
        print("-" * 80)
        for step, task in enumerate(self.best_schedule, 1):
            resource_reqs = [task.renewable_resources.get(resource, 0) for resource in self.resources]
            print(f"{step:4} | {task.name:4} | {task.duration:8} | {resource_reqs}")
        
        print("\nFinal Task Sequence:", " -> ".join([task.name for task in self.best_schedule]))
        print(f"Final Makespan: {self.best_makespan}")

        # Plot the best schedule
        #self.plot_schedule(self.best_schedule, self.best_makespan, resources)
        self.validate_schedule(self.best_schedule)
        return self.best_schedule, self.best_makespan
    
    def validate_schedule(self, schedule):
        """
        Checks if every task in the schedule is scheduled after all its predecessors.
        Prints errors if found, otherwise confirms validity.
        """
        position = {task.name: i for i, task in enumerate(schedule)}
        valid = True
        for task in schedule:
            for pred in getattr(task, 'predecessors', []):
                if position[pred.name] >= position[task.name]:
                    print(f"Error: Task {task.name} is scheduled before its predecessor {pred.name}")
                    valid = False
        if valid:
            print("Schedule is valid! All precedence constraints are satisfied.")
        return valid
    
    # def plot_pheromone_matrix(self):
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(self.pheromone, 
    #                 cmap="YlGnBu", 
    #                 annot=False, 
    #                 linewidths=0.5, 
    #                 square=True, 
    #                 cbar_kws={'label': 'Pheromone Strength'})
    #     plt.title("Pheromone Matrix After Optimization", fontsize=14)
    #     plt.xlabel("To Task Index")
    #     plt.ylabel("From Task Index")
    #     plt.xticks(ticks=np.arange(self.n_tasks), 
    #             labels=[task.name for task in self.tasks], 
    #             rotation=90)
    #     plt.yticks(ticks=np.arange(self.n_tasks), 
    #             labels=[task.name for task in self.tasks], 
    #             rotation=0)
    #     plt.tight_layout()
    #     plt.show()

def create_jobs(sm_file_name):
    sm_file = ReadSMFIles.SMFileParser.parse_sm_file(sm_file_name)

    #Creating resources
    r1 = int(sm_file[4].R1[0])
    r2 = int(sm_file[4].R2[0])
    r3 = int(sm_file[4].R3[0])
    r4 = int(sm_file[4].R4[0])

    R1 = Resource('R1', r1)
    R2 = Resource('R2', r2)
    R3 = Resource('R3', r3)
    R4 = Resource('R4', r4)

    resources = [R1, R2, R3, R4]

    print([resource.name for resource in resources], [resource.per_period_availability for resource in resources])

    #Creating jobs
    jobs_enumerate = sm_file[3].jobnr
    jobs_duration = sm_file[3].duration
    jobs_resources = sm_file[3].resources
    jobs_successors = sm_file[2].successors

    jobs = [None for _ in jobs_enumerate]

    for i in jobs_enumerate:
        jobs[i - 1] = Task(str(i), jobs_duration[i - 1])

    for i in range(len(resources)):
        for j in range(len(jobs)):
            jobs[j].add_renewable_resource(resources[i], jobs_resources[j][i])

    for i in range(len(jobs)):
        successors = jobs_successors[i]
        for j in successors:
            jobs[i].add_sucessor(jobs[j - 1])
        
    # jobs = jobs[1:-1]

    # Ensure all tasks with no predecessors have the dummy start as a predecessor
    # (dummy start is jobs[0])
    dummy_start = jobs[0]
    for job in jobs[1:]:
        if len(job.predecessors) == 0:
            job.add_predecessor(dummy_start)

    # Print task information
    print("\nTask Information:")
    print("Task ID | Duration | Resource Requirements (R1, R2, R3, R4) | Successors")
    print("-" * 80)
    for job in jobs:
        resource_reqs = [job.renewable_resources.get(resource, 0) for resource in resources]
        successors = [s.name for s in job.sucessors]
        print(f"{job.name:7} | {job.duration:8} | {resource_reqs} | {successors}")

    print("\nResource Availabilities:")
    for resource in resources:
        print(f"{resource.name}: {resource.per_period_availability}")

    return jobs, resources

def run_aco_with_seed(sm_file_name, seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)

    jobs, resources = create_jobs(sm_file_name)

    aco = AntColonyRCPSP(
        jobs,
        resources,
        n_ants=10,
        n_iterations=10,
        alpha=1.0,
        beta=2.0,  # Adjusted beta for more exploration
        rho=0.1,
        c=0.5,
        gamma=1.0,
        elitist_forget_generations=15
    )

    _, best_makespan = aco.run()
    return best_makespan


if __name__ == "__main__":
    run_aco_with_seed("j60.sm/j601_3.sm", 1)
    

# Create and run the ACO algorithm with specified parameters
# aco = AntColonyRCPSP(jobs, n_ants = 60,  ## More ants explore more paths per iteration but increase computation.
#                      n_iterations = 10,  ##	Determines total search effort. Too low → premature convergence, see the covergence chart to define a good value.
#                             alpha = 1.0, ## Higher = more exploitation of learned pheromone.
#                              beta = 1.5, ## Higher = more greedy / heuristic guidance (decays over time).
#                               rho = 0.1, ## Pheromone evaporation rate ##recommended value is 0.1
#                                 c = 0.5, ## Relative pheromone weight ##recommended value is 0.5 ##	Balances direct vs. summation pheromone evaluation.
#                             gamma = 1.0, ## Summation pheromone weight ##recommended value is 1.0
#        elitist_forget_generations = 10   ## Number of generations before dropping best-so-far.
#                     )
# best_schedule, best_makespan = aco.run()

# print("Best Makespan:", best_makespan)
# print("Task Order:", [task.name for task in best_schedule])

### ACO Convergence Curve ###
# plt.figure(figsize=(15, 5))
# plt.plot(aco.makespan_history, marker='o', linestyle='-')
# plt.xlabel("Iteration")
# plt.ylabel("Best Makespan So Far")
# plt.title("ACO Convergence Curve")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

### Plot the pheromone matrix after optimization ##
# aco.plot_pheromone_matrix()

# === Variability Diagnostic: Run ACO multiple times ===
# seeding = 3
# makespans = []

# for i in range(seeding):
#     print(f"\n=== Trial {i + 1} of {seeding} ===")
#     ms = run_aco_with_seed(i)
#     makespans.append(ms)

# # === Statistics ===
# mean_ms = np.mean(makespans)
# std_ms = np.std(makespans)
# best = np.min(makespans)
# worst = np.max(makespans)

# print("\n Makespan Statistics:")
# print(f"Best   : {best}")
# print(f"Worst  : {worst}")
# print(f"Mean   : {mean_ms:.2f}")
# print(f"Std Dev: {std_ms:.2f}")
# print("Number of Runs:", seeding)

# plt.figure(figsize=(10, 5))
# plt.hist(makespans, bins=10, edgecolor='black', color='skyblue')
# plt.axvline(mean_ms, color='red', linestyle='--', label=f'Mean = {mean_ms:.2f}')
# plt.axvline(best, color='green', linestyle='--', label=f'Best = {best}')
# plt.axvline(worst, color='orange', linestyle='--', label=f'Worst = {worst}')
# plt.title("Distribution of Best Makespans Across Multiple Runs")
# plt.xlabel("Makespan")
# plt.ylabel("Frequency")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


