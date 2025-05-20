import json
import random
from collections import defaultdict
import heapq
import time
import matplotlib.pyplot as plt
import numpy as np

# --- Problem data structures ----------------

class Activity:
    def __init__(self, idx, duration, reqs, preds):
        self.idx = idx            # activity ID
        self.dur = duration       # processing time
        self.reqs = reqs          # dict {res_id: demand}
        self.preds = set(preds)   # set of predecessor IDs
        self.succs = set()        # filled in after reading all activities

class RCPSP:
    def __init__(self, activities, resource_caps):
        # print(activities)
        self.activities = {a.idx: a for a in activities}
        # print(self.activities)
        self.res_caps = resource_caps
        # build successor sets
        for a in activities:
            for p in a.preds:
                self.activities[p].succs.add(a.idx)
        self.start = 0
        self.end = max(self.activities)
        # print(self.end)

# --- Serial Schedule Generation Scheme (SSGS) --

def serial_schedule(instance, act_list):
    """Given a permutation act_list, build a feasible schedule."""
    start_times = {}
    # track resource usage over time (simple discrete)
    usage = defaultdict(lambda: defaultdict(int))  # usage[t][res] = used
    for act_id in act_list:
        act = instance.activities[act_id]
        est = 0
        # precedence
        for p in act.preds:
            est = max(est, start_times[p] + instance.activities[p].dur)
        # find earliest t ≥ est where res caps suffice
        t = est
        while True:
            conflict = False
            for dt in range(act.dur):
                for r, d in act.reqs.items():
                    if usage[t+dt][r] + d > instance.res_caps[r]:
                        conflict = True; break
                if conflict: break
            if not conflict:
                # reserve resources
                for dt in range(act.dur):
                    for r, d in act.reqs.items():
                        usage[t+dt][r] += d
                start_times[act_id] = t
                break
            t += 1
    makespan = max(start_times[a] + instance.activities[a].dur 
                   for a in start_times)
    return start_times, makespan

# --- Heuristic: normalized LST -----------------

def compute_lft_lsts(instance):
    """Compute latest finish/start times by backward pass."""
    n = len(instance.activities)
    LFT = {i: float('inf') for i in instance.activities}
    LFT[instance.end] = 0
    # reverse topological
    topo = list(instance.activities)[::-1]
    for i in topo:
        act = instance.activities[i]
        if act.succs:
            LFT[i] = min(LFT[s] for s in act.succs) - act.dur
        else:
            LFT[i] = max(LFT.values())
    LST = {i: LFT[i] - instance.activities[i].dur for i in LFT}
    return LFT, LST

# --- ACO core ----------------------------------

class AntColony:
    def __init__(self, instance, n_ants=10, n_gen=5000):
        self.inst = instance
        self.n_ants = n_ants
        self.n_gen = n_gen
        self.tau = defaultdict(lambda: defaultdict(lambda: 1.0))  # τ[pos][act]
        self.alpha = 2.0   # pheromone exponent (will decay)
        self.beta = 2.0    # heuristic exponent (will decay)
        self.rho = 0.1     # evaporation (may increase later)
        self.elite = None
        # precompute heuristic values (nLST)
        _, LST = compute_lft_lsts(instance)
        maxlst = max(LST.values())
        self.eta = {i: (maxlst - LST[i]) + 1e-6
                    for i in instance.activities}

    def construct_solution(self):
        """Build one ant's activity list via SSGS eligibility."""
        scheduled = set([self.inst.start])
        act_list = [self.inst.start]  # Keep using the ID since that's what the rest of the code expects
        available = set(self.inst.activities.keys()) - scheduled
        while self.inst.end in available:
            available = {i for i in available 
                         if self.inst.activities[i].preds <= scheduled}
            pos = len(act_list)
            # compute combined tau and eta
            scores = {}
            for a in available:
                tval = self.tau[pos][a]
                eval = sum(self.tau[p][a] for p in range(pos))  # summation
                # combine direct and sum: w*direct + (1-w)*sum, w=0.5
                phi = 0.5*tval + 0.5*(eval/ max(1, pos))
                scores[a] = (phi**self.alpha) * (self.eta[a]**self.beta)
            # roulette-wheel selection
            tot = sum(scores.values())
            pick = random.random() * tot
            cum = 0
            act = None  # Initialize act
            for a,v in scores.items():
                cum += v
                if cum >= pick:
                    act = a
                    break
            if act is None:  # If no activity was selected (shouldn't happen but just in case)
                act = list(scores.keys())[0]  # Take the first available activity
            act_list.append(act)
            scheduled.add(act)
            available.remove(act)
        return act_list

    def update_pheromone(self, sol_best, make_best):
        """Evaporate and deposit."""
        # evaporation
        for pos in self.tau:
            for a in self.tau[pos]:
                self.tau[pos][a] *= (1 - self.rho)
        # deposit for elite solution
        if sol_best is not None and make_best > 0:  # Add safety check
            for pos, a in enumerate(sol_best):
                if a in self.inst.activities:  # Verify activity exists
                    self.tau[pos][a] += 1.0 / make_best

    def run(self):
        global_best = None
        best_makespan = float('inf')
        for gen in range(self.n_gen):
            # dynamic parameter control
            if gen < self.n_gen//2:
                self.beta = 2.0*(1 - gen/(self.n_gen//2))
            else:
                self.rho = min(0.5, self.rho + 0.0001)
            # construct
            iter_best = None
            best_iter_mk = float('inf')
            for _ in range(self.n_ants):
                al = self.construct_solution()
                _, mk = serial_schedule(self.inst, al)
                if mk < best_iter_mk:
                    iter_best, best_iter_mk = al, mk
            # update global
            if best_iter_mk < best_makespan:
                global_best, best_makespan = iter_best, best_iter_mk
            # optionally "forget" elitist after G generations
            if gen % 200 == 0 and self.elite and gen>0:
                self.elite = iter_best
            # pheromone
            elite_sol = self.elite if self.elite else global_best
            self.update_pheromone(elite_sol, best_makespan)
            self.elite = elite_sol
        return global_best, best_makespan

# --- Example of usage --------------------------

# --- Main: JSON Loading and Execution -------------

def visualize_schedule(instance, start_times, makespan):
    """Visualize the schedule using a Gantt chart."""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Get activities and their durations
    activities = instance.activities
    y_pos = 0
    
    # Plot each activity
    for act_id in range(len(activities)):
        if act_id in start_times:  # Skip if activity not in schedule
            start = start_times[act_id]
            duration = activities[act_id].dur
            if duration > 0:  # Only plot non-dummy activities
                # Create rectangle for activity
                rect = plt.Rectangle((start, y_pos), duration, 0.8, 
                                   facecolor=f'C{act_id % 10}', 
                                   edgecolor='black',
                                   alpha=0.7)
                ax.add_patch(rect)
                
                # Add activity label
                ax.text(start + duration/2, y_pos + 0.4, f'Act {act_id}',
                       ha='center', va='center')
                y_pos += 1
    
    # Customize the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Activities')
    ax.set_title('Project Schedule Gantt Chart')
    ax.set_xlim(0, makespan)
    ax.set_ylim(0, y_pos)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Load instance from JSON file
    with open('j12010_1.json', 'r') as f:
        data = json.load(f)

    # 2. Parse raw arrays
    num_jobs = data['njobs']
    num_res = data['nres']
    durations = data['durations']   # indexed by activity ID (0 … njobs)
    jobs = data['jobs']            # list of job IDs
    res = data['res']              # list of resource IDs
    succs = data['succs']          # succs[i] = list of successors of i
    demands = data['demands']      # demands[i] = [demand on each resource]
    capacities = data['capacities'] # capacities[r] = total units of resource r

    # 3. Build predecessor lists
    preds = {i: [] for i in range(num_jobs + 1)}  # +1 to include the last job
    for i, succ_list in enumerate(succs):
        for s in succ_list:
            preds[s].append(i)

    # 4. Create Activity objects
    activities = []
    for i in range(num_jobs + 1):  # +1 to include the last job
        dur = durations[i-1] if i > 0 else 0  # Adjust index for 1-based job IDs
        # only include nonzero demands
        reqs = {r: demands[i-1][r] for r in range(num_res)  # Adjust index for demands too
                if demands[i-1][r] > 0}
        print(reqs)
        print(preds[i])
        print(i)
        print(dur)
        activities.append(Activity(idx=i, duration=dur,
                                   reqs=reqs, preds=preds[i]))

    # 5. Build RCPSP instance
    start_time = time.time()
    resource_caps = {r: capacities[r] for r in range(num_res)}
    rcpsp = RCPSP(activities, resource_caps)

    # 6. Run ACO
    colony = AntColony(rcpsp, n_ants=1, n_gen=1)
    best_order, best_makespan = colony.run()
    
    # Get the schedule for the best solution
    start_times, makespan = serial_schedule(rcpsp, best_order)
    
    # Visualize the schedule
    visualize_schedule(rcpsp, start_times, makespan)
    
    end_time = time.time()
    print("Time taken:", end_time - start_time)
    print("Best makespan found:", best_makespan)
    print("Best activity sequence:", best_order)
