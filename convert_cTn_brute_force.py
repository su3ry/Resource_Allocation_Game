from scipy import optimize
import random
import numpy as np
from all_feasible_actions import *
import os, psutil
from multiprocessing import Pool, freeze_support

def brute_force_SWING(vul_num, feasible_actions, prob_vector):
    column_size = len(feasible_actions)
    A = np.zeros((vul_num, column_size), dtype = int)
    for col in range(column_size):
        for ro in feasible_actions[col]:
            A[ro][col] = 1
    c = np.ones(column_size)
    B = np.array(prob_vector)
    #print(feasible_actions)
    sol = optimize.linprog(c, A_ub = np.multiply(A, -1), b_ub = np.multiply(B, -1))
    #print(sol.get("x"))
    return sol

def filter_and_brute_force(vul_num, time_matrices, task_nums, prob_vector, capacity, agent_order):
    t1 = time.time()
    normal_form_actions = enumerate_normal_form_actions(vul_num)
        #print(normal_form_actions)
    feasible_actions = filter_feasible_actions(time_matrices, task_nums, prob_vector, capacity, agent_order, normal_form_actions)
    #print(len(feasible_actions))
    sol  = brute_force_SWING(vul_num, feasible_actions, prob_vector)
    print("Vul_num, ", vul_num)
    t2 = time.time()
    f = open("Running_time_bruteforce1", "a")
    f.write(str(vul_num) + " " + str(t2-t1) + "\n")
    f.close()
    return sol

if __name__ == "__main__":
    agent_num = 10
    task_num_per_vul = 8
    all_args = []
    for vul_num in range(5, 6):
        task_nums = []
        time_matrices = []
        prob_vector = []
        task_nums = [task_num_per_vul] * vul_num
        time_matrices = []
        for i in range(vul_num):
            temp_m = np.zeros((agent_num, task_nums[i]), dtype = int)
            for j in range(agent_num):
                for l in range(task_nums[i]):
                    temp_m[j][l] = random.randint(1, 8)
            time_matrices.append(temp_m)
        for i in range(vul_num):
            prob_vector.append(random.random())
        agent_order = range(agent_num)
        total_time_needed = (np.array(time_matrices).sum()/agent_num)/(3*agent_num)
        capacity = [int(total_time_needed)]*agent_num
        '''
        normal_form_actions = enumerate_normal_form_actions(vul_num)
        #print(normal_form_actions)
        feasible_actions = filter_feasible_actions(time_matrices, task_nums, prob_vector, capacity, agent_order, normal_form_actions)
        print(len(feasible_actions))
        brute_force_SWING(vul_num, feasible_actions, prob_vector)
        '''
        all_args.append((vul_num, time_matrices, task_nums, prob_vector, capacity, agent_order))
    freeze_support()
    with Pool() as pool:
        L = pool.starmap(filter_and_brute_force, all_args)



