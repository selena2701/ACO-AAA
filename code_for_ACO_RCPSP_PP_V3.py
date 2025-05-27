# %%
#Importing read-sm-files.py
import ReadSMFIles
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
#sm_file = ReadSMFIles.SMFileParser.parse_sm_file("j30.sm/j301_1.sm")
sm_file = ReadSMFIles.SMFileParser.parse_sm_file("j60.sm/j601_1.sm")

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

# %% [markdown]
# 
# # ACO for RCPSP

# %%
# %%
class AntColonyRCPSP:
    def __init__(self, tasks: List[Task], n_ants: int = 10, n_iterations: int = 50,
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1, c: float = 0.5, gamma: float = 1.0, elitist_forget_generations = 30):
        self.tasks = tasks
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # pheromone importance
        self.initial_beta = beta  # initial heuristic importance
        self.rho = rho      # evaporation rate
        self.elitist_forget_generations = 30

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

    def _probabilities(self, schedule: List[Task], eligible: List[Task], iteration: int, c: float = 0.5, gamma: float = 1.0) -> List[float]:
        current_beta = self.initial_beta * (1.0 - (iteration / self.n_iterations) * 0.5)
        exploration_factor = 1.0 - (iteration / self.n_iterations) * 0.8
        random_factors = np.random.uniform(0.5, 1.5, len(eligible))

        pheromone_vals = []
        for i, task in enumerate(eligible):
            direct_pher, sum_pher = self._combined_pheromone(schedule, task, gamma)
            pher = (1 - c) * direct_pher + c * sum_pher
            pher *= random_factors[i]  # add noise to promote exploration
            pheromone_vals.append(pher)

        heuristic_vals = [self._heuristic(schedule[-1], t) for t in eligible]
        heuristic_noise = np.random.uniform(0.8, 1.2, len(eligible))
        heuristic_vals = [h * n for h, n in zip(heuristic_vals, heuristic_noise)]

        # Compute selection probabilities
        product = [(pheromone_vals[i] ** self.alpha) * (heuristic_vals[i] ** current_beta) for i in range(len(eligible))]
        epsilon = 1e-10
        product = [p + epsilon for p in product]
        total = sum(product)

        min_prob = 0.05
        probabilities = [(1 - min_prob * len(product)) * p / total + min_prob for p in product]

        return probabilities

    def _get_eligible_tasks(self, scheduled: List[Task], unscheduled: List[Task]) -> List[Task]:
        eligible = []
        for task in unscheduled:
            if all(pred in scheduled for pred in task.predecessors):
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

    def _2opt_swap(self, route: List[Task], i: int, j: int) -> List[Task]:
        new_route = route[:i]
        new_route.extend(reversed(route[i:j+1]))
        new_route.extend(route[j+1:])
        return new_route

    def _local_search(self, task_seq: List[Task]) -> Tuple[List[Task], int]:
        best_seq = task_seq
        best_makespan = self._schedule_makespan(task_seq)
        improved = True

        while improved:
            improved = False
            for i in range(1, len(task_seq)-1):
                for j in range(i+1, len(task_seq)):
                    # Check if swap maintains precedence constraints
                    if all(pred in best_seq[:i] for pred in task_seq[j].predecessors):
                        new_seq = self._2opt_swap(best_seq, i, j)
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

            # Add some randomness to task selection
            if random.random() < 0.1:  # 10% chance of random selection
                next_task = random.choice(eligible)
            else:
                probs = self._probabilities(scheduled, eligible, iteration)
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
        if iteration > self.elitist_forget_generations:
            self.elite_solutions = []
            # Add more randomness to the reset
            self.pheromone = np.random.uniform(0.1, 2.0, (self.n_tasks, self.n_tasks))

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
                             node_size=1000, alpha=0.7, ax=ax1)
        
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

    def run(self):
        for iteration in range(self.n_iterations):
            print(f"\nIteration {iteration + 1}")
            print("=" * 50)
            
            solutions = []
            for ant in range(self.n_ants):
                print(f"\nAnt {ant + 1}")
                print("-" * 30)
                task_seq, makespan = self._construct_solution(iteration)

                # Only apply local search if the solution improves the current best
                if makespan < self.best_makespan:
                    print("\n🔍 Applying local search to improve promising solution...")
                    task_seq, makespan = self._local_search(task_seq)

                # Store the solution (even if not the best) for pheromone update
                solutions.append((task_seq, makespan))

                # Check again if this (possibly improved) solution is the best so far
                if makespan < self.best_makespan:
                    self.best_makespan = makespan
                    self.best_schedule = task_seq[:]
                    print(f"\n🏆 New best solution found after local search! Makespan: {self.best_makespan}")

                # Print detailed task order information
                print("\nTask Order Details:")
                print("Step | Task | Duration | Resource Requirements (R1, R2, R3, R4)")
                print("-" * 80)
                for step, task in enumerate(task_seq, 1):
                    resource_reqs = [task.renewable_resources.get(resource, 0) for resource in resources]
                    print(f"{step:4} | {task.name:4} | {task.duration:8} | {resource_reqs}")

                print("\nFinal Task Sequence:", " -> ".join([task.name for task in task_seq]))
                print(f"Makespan: {makespan}")

                if makespan < self.best_makespan:
                    self.best_makespan = makespan
                    self.best_schedule = task_seq[:]
                    print(f"\nNew best solution found! Makespan: {self.best_makespan}")

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

        return self.best_schedule, self.best_makespan
    
    def plot_pheromone_matrix(self):
        

        plt.figure(figsize=(10, 8))
        sns.heatmap(self.pheromone, 
                    cmap="YlGnBu", 
                    annot=False, 
                    linewidths=0.5, 
                    square=True, 
                    cbar_kws={'label': 'Pheromone Strength'})
        plt.title("Pheromone Matrix After Optimization", fontsize=14)
        plt.xlabel("To Task Index")
        plt.ylabel("From Task Index")
        plt.xticks(ticks=np.arange(self.n_tasks), 
                labels=[task.name for task in self.tasks], 
                rotation=90)
        plt.yticks(ticks=np.arange(self.n_tasks), 
                labels=[task.name for task in self.tasks], 
                rotation=0)
        plt.tight_layout()
        plt.show()


# Create and run the ACO algorithm with specified parameters
aco = AntColonyRCPSP(jobs, n_ants = 60,  ## More ants explore more paths per iteration but increase computation.
                     n_iterations = 20,  ##	Determines total search effort. Too low → premature convergence, see the covergence chart to define a good value.
                            alpha = 1.0, ## Higher = more exploitation of learned pheromone.
                             beta = 3.0, ## Higher = more greedy / heuristic guidance (decays over time).
                              rho = 0.1, ## Pheromone evaporation rate ##recommended value is 0.1
                                c = 0.5, ## Relative pheromone weight ##recommended value is 0.5 ##	Balances direct vs. summation pheromone evaluation.
                            gamma = 1.0, ## Summation pheromone weight ##recommended value is 1.0
       elitist_forget_generations = 20   ## Number of generations before dropping best-so-far.
                    )
best_schedule, best_makespan = aco.run()

print("Best Makespan:", best_makespan)
print("Task Order:", [task.name for task in best_schedule])

### ACO Convergence Curve ###
plt.figure(figsize=(15, 5))
plt.plot(aco.makespan_history, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Best Makespan So Far")
plt.title("ACO Convergence Curve")
plt.grid(True)
plt.tight_layout()
plt.show()

### Plot the pheromone matrix after optimization ##
aco.plot_pheromone_matrix()
