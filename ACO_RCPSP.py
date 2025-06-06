# %%
#Importing read-sm-files.py
import ReadSMFiles
from Resource import Resource
from Task import Task
from Schedule import Schedule
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

# Get the current directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the input file
# input_file = os.path.join(current_dir, "j60.sm", "j601_1.sm")

#reading j30.sm/j301_1.sm file
#reading j30.sm/j301_1.sm file
#sm_file = ReadSMFiles.SMFileParser.parse_sm_file("j30.sm/j301_1.sm")
sm_file = ReadSMFiles.SMFileParser.parse_sm_file("j60.sm/j6025_1.sm")

print(sm_file)


# %%
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


# %%
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


# %%
# %%
class AntColonyRCPSP:
    def __init__(self, tasks: List[Task], n_ants: int = 10, n_iterations: int = 50,
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1, c: float = 0.5, gamma: float = 1.0, elitist_forget_generations = None):
        self.tasks = tasks
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # pheromone importance
        self.initial_beta = beta  # initial heuristic importance
        self.rho = rho      # evaporation rate
        self.c = c  # Weight for direct vs summation evaluation
        self.gamma = gamma  # Decay factor for summation pheromone
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

    def _probabilities(self, current_task: Task, eligible: List[Task], iteration: int, schedule: List[Task]) -> List[float]:

        # Calculate beta decay
        current_beta = self.initial_beta * (1.0 - (iteration / self.n_iterations) * 0.5)
        
        # Add more randomness in early iterations
        exploration_factor = 1.0 - (iteration / self.n_iterations) * 0.8
        random_factors = np.random.uniform(0.5, 1.5, len(eligible))
        
        # Calculate pheromone values with more exploration
        pheromone_vals = []
        for t in eligible:
            # Combine local and global pheromone information
            local_pher, global_pher = self._combined_pheromone(schedule, t, self.gamma)
            pher_val = self.c * local_pher + (1 - self.c) * global_pher
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
                # print(f"Adding task {i} {task.name} to eligible tasks")
                eligible.append(task)
        return eligible

    def _schedule_makespan(self, task_order: List[Task]) -> int:
        max_time = 1000
        resource_availability = {
            resource.name: np.ones(max_time) * resource.per_period_availability
            for resource in resources
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

    # prob
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
                    # Wrong if else condition
                    new_seq = self._2opt_swap(best_seq, i, j)
                    if not self._is_valid_sequence(new_seq):
                        continue
                    print(f"New sequence: {([task.name for task in new_seq])}")
                    print(f"Old sequence: {([task2.name for task2 in best_seq])}")
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
                        print(f"Adding task {task.name} to eligible tasks")
                        eligible.append(task)
                        break
                if not eligible:
                    break
            i += 1
            # Add some randomness to task selection
            if random.random() < 0.03:  # 10% chance of random selection
                next_task = random.choice(eligible)
            else:
                probs = self._probabilities(current_task, eligible, iteration, scheduled)
                next_task = random.choices(eligible, weights=probs, k=1)[0]
            
            scheduled.append(next_task)
            unscheduled.remove(next_task)
            current_task = next_task

        # Add any remaining unscheduled tasks at the end
        # not the problem
        if unscheduled:
            raise ValueError("Unscheduled tasks remaining")

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
        if self.elitist_forget_generations is not None and iteration > self.elitist_forget_generations:
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
                    resource_reqs = [task.renewable_resources.get(resource, 0) for resource in resources]
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
            resource_reqs = [task.renewable_resources.get(resource, 0) for resource in resources]
            print(f"{step:4} | {task.name:4} | {task.duration:8} | {resource_reqs}")
        
        print("\nFinal Task Sequence:", " -> ".join([task.name for task in self.best_schedule]))
        print(f"Final Makespan: {self.best_makespan}")

        # Plot the best schedule
        self.plot_schedule(self.best_schedule, self.best_makespan, resources)
        self.validate_schedule(self.best_schedule)
        return self.best_schedule, self.best_makespan
    
    def plot_schedule(self, schedule: List[Task], makespan: int, resources: List[Resource]):
        """
        Plot a directed graph of the schedule showing task dependencies and resource usage.
        """
        # Create figure and axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
        fig.suptitle(f'Schedule Visualization (Makespan: {makespan})', fontsize=16)

        # Create directed graph
        G = nx.DiGraph()
        
        # First, add all nodes with their attributes
        for task in schedule:
            if task.name == '1' or task.name == '32':  # Skip dummy start/end tasks
                continue
            # Calculate resource requirements
            resource_reqs = [task.renewable_resources.get(resource, 0) for resource in resources]
            # Add node with all attributes
            G.add_node(task.name, 
                      duration=task.duration,
                      resource_reqs=resource_reqs)
        
        # Then add edges
        for task in schedule:
            if task.name == '1' or task.name == '32':  # Skip dummy start/end tasks
                continue
            for pred in task.predecessors:
                if pred.name != '1':  # Skip dummy start
                    G.add_edge(pred.name, task.name)

        # Calculate node positions using hierarchical layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw the graph
        node_colors = []
        for node in G.nodes():
            # Get resource requirements from node attributes
            resource_reqs = G.nodes[node].get('resource_reqs', [0, 0, 0, 0])
            if sum(resource_reqs) > 0:
                # Find the resource with highest requirement
                max_resource_idx = np.argmax(resource_reqs)
                colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
                node_colors.append(colors[max_resource_idx])
            else:
                node_colors.append('gray')

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=1200, alpha=0.7, ax=ax1)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                             arrows=True, arrowsize=20, ax=ax1)
        
        # Add labels with duration and resource info
        labels = {}
        for node in G.nodes():
            duration = G.nodes[node].get('duration', 0)
            resource_reqs = G.nodes[node].get('resource_reqs', [0, 0, 0, 0])
            # Format resource requirements
            resource_str = ' '.join([f'R{i+1}:{req}' for i, req in enumerate(resource_reqs) if req > 0])
            if not resource_str:
                resource_str = 'No resources'
            labels[node] = f'Task {node}\nDur: {duration}\n{resource_str}'
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax1)
        
        # Plot resource usage
        time_points = np.arange(0, makespan + 1)
        resource_usage = {r.name: np.zeros(makespan + 1) for r in resources}
        
        # Calculate resource usage over time
        for task in schedule:
            if task.name == '1' or task.name == '32':  # Skip dummy start/end tasks
                continue
                
            # Find start time based on predecessors
            start_time = 0
            for pred in task.predecessors:
                if pred.name != '1':  # Skip dummy start
                    pred_end = 0
                    for t in schedule:
                        if t.name == pred.name:
                            pred_end = max(pred_end, start_time + t.duration)
                    start_time = max(start_time, pred_end)
            
            # Update resource usage
            for resource, amount in task.renewable_resources.items():
                if amount > 0:
                    resource_usage[resource.name][start_time:start_time + task.duration] += amount

        # Plot resource usage
        resource_colors = {
            'R1': 'blue',
            'R2': 'green',
            'R3': 'red',
            'R4': 'yellow'
        }
        
        for i, (resource, usage) in enumerate(resource_usage.items()):
            ax2.plot(time_points, usage, label=f'{resource}', 
                    color=resource_colors[resource], linewidth=2)
            ax2.axhline(y=resources[i].per_period_availability, 
                       color=resource_colors[resource], linestyle='--', alpha=0.5)

        # Customize plots
        ax1.set_title('Task Dependencies Graph')
        ax1.axis('off')  # Hide axes for the graph
        
        ax2.set_ylabel('Resource Usage')
        ax2.set_xlabel('Time')
        ax2.set_title('Resource Usage Over Time')
        ax2.grid(True)
        ax2.legend()
        
        # Set x-axis limits for resource usage
        ax2.set_xlim(0, makespan)
        
        plt.tight_layout()
        plt.show()
    
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


def run_aco_with_seed(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)

    aco = AntColonyRCPSP(
        jobs,
        n_ants=60,
        n_iterations=15,
        alpha=1.0,
        beta=3,  # Adjusted beta for more exploration
        rho=0.1,
        c=0.5,
        gamma=1.0,
        elitist_forget_generations=None
    )

    best_schedule, best_makespan = aco.run()
    return best_makespan

# print = lambda *args, **kwargs: None

# === Variability Diagnostic: Run ACO multiple times ===
if __name__ == "__main__":
    num_runs = 1
    makespans = []

    for i in range(num_runs):
        print(f"\n=== Trial {i + 1} of {num_runs} ===")
        ms = run_aco_with_seed(i)
        makespans.append(ms)

    # === Statistics ===
    mean_ms = np.mean(makespans)
    std_ms = np.std(makespans)
    best = np.min(makespans)
    worst = np.max(makespans)

    print("\n Makespan Statistics:")
    print(f"Best   : {best}")
    print(f"Worst  : {worst}")
    print(f"Mean   : {mean_ms:.2f}")
    print(f"Std Dev: {std_ms:.2f}")
    print("Number of Runs:", num_runs)

    plt.figure(figsize=(10, 5))
    plt.hist(makespans, bins=10, edgecolor='black', color='skyblue')
    plt.axvline(mean_ms, color='red', linestyle='--', label=f'Mean = {mean_ms:.2f}')
    plt.axvline(best, color='green', linestyle='--', label=f'Best = {best}')
    plt.axvline(worst, color='orange', linestyle='--', label=f'Worst = {worst}')
    plt.title("Distribution of Best Makespans Across Multiple Runs")
    plt.xlabel("Makespan")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


