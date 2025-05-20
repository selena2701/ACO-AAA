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
import time 
# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the input file
input_file = os.path.join(current_dir, "j30.sm", "j301_1.sm")

#reading j30.sm/j301_1.sm file
sm_file = ReadSMFIles.SMFileParser.parse_sm_file(input_file)

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
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1):
        self.tasks = tasks
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # pheromone importance
        self.initial_beta = beta  # initial heuristic importance
        self.rho = rho      # evaporation rate
        self.elitist_forget_generations = 30

        self.n_tasks = len(tasks)
        # Initialize pheromone matrix with small random values
        self.pheromone = np.random.uniform(0.1, 1.0, (self.n_tasks, self.n_tasks))
        self.best_schedule = None
        self.best_makespan = float('inf')
        self.elite_solutions = []  # Store elite solutions

    def _heuristic(self, from_task: Task, to_task: Task) -> float:
        # Improved heuristic considering multiple factors
        duration_factor = 1.0 / (to_task.duration + 1)
        resource_factor = 1.0 / (sum(to_task.renewable_resources.values()) + 1)
        return duration_factor * resource_factor

    def _get_eligible_tasks(self, scheduled: List[Task], unscheduled: List[Task]) -> List[Task]:
        eligible = []
        for task in unscheduled:
            if all(pred in scheduled for pred in task.predecessors):
                eligible.append(task)
        return eligible

    def _schedule_makespan(self, task_order: List[Task]) -> int:
        schedule = Schedule()
        schedule.add_tasks(task_order)
        return schedule.makespan_without_penalization()

    def _local_pheromone(self, from_task: Task, to_task: Task) -> float:
        from_idx = self.tasks.index(from_task)
        to_idx = self.tasks.index(to_task)
        return self.pheromone[from_idx][to_idx]

    def _summation_pheromone(self, from_task: Task, to_task: Task) -> float:
        from_idx = self.tasks.index(from_task)
        to_idx = self.tasks.index(to_task)
        # Sum pheromone values from all paths to the target task
        return sum(self.pheromone[i][to_idx] for i in range(self.n_tasks))

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
                probs = self._probabilities(current_task, eligible, iteration)
                next_task = random.choices(eligible, weights=probs, k=1)[0]
            
            scheduled.append(next_task)
            unscheduled.remove(next_task)
            current_task = next_task

        # Add any remaining unscheduled tasks at the end
        if unscheduled:
            scheduled.extend(unscheduled)

        # Apply local search
        scheduled, makespan = self._local_search(scheduled)
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
            'R1': 'lightblue',
            'R2': 'lightgreen',
            'R3': 'lightpink',
            'R4': 'lightyellow'
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
                solutions.append((task_seq, makespan))

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
start_time = time.time()
# Create and run the ACO algorithm with specified parameters
aco = AntColonyRCPSP(jobs, n_ants=1, n_iterations=5, alpha=1.0, beta=3.0, rho=0.1)
best_schedule, best_makespan = aco.run()
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print("Best Makespan:", best_makespan)
print("Task Order:", [task.name for task in best_schedule])


