import random
from itertools import *
import time
import numpy as np
from scipy import optimize
import sys
import os
from max_nfa import *
def enumerate_normal_form_actions(vul_num):
    l = list(range(vul_num))
    actions = []
    for i in range(len(l)+1):
        for subset in combinations(l, i):
            if len(subset)>0:
                actions.append(list(subset))
    return actions

def filter_feasible_actions(time_matrices, task_nums, prob_vector, capacity, agent_order, normal_form_actions):
    feasible_actions = []
    #time_matrix = weight_matrix2d(time_matrices)
    #values_long = values(prob_vector, task_nums)
    for ac in normal_form_actions:
        time_matrices_ac = []
        prob_vector_ac = []
        task_nums_ac = []
        for v in ac:
            time_matrices_ac.append(time_matrices[v])
            task_nums_ac.append(task_nums[v])
            prob_vector_ac.append(prob_vector[v])
        time_matrix_ac = weight_matrix2d(time_matrices_ac)
        values_long_ac = values(prob_vector_ac, task_nums_ac)
        assignment = initial_assign(time_matrix_ac, capacity, agent_order, values_long_ac)

        #print(assignment)
        if np.array(assignment).sum() >= sum(task_nums_ac):
            feasible_actions.append(ac)
    return feasible_actions

if __name__ == "__main__":
    vul_num = 4
    agent_num = 2
    task_nums = []
    prob_vector = []
    for i in range(vul_num):
        task_nums.append(random.randint(6, 9))
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
    capacity = [100]*agent_num
    normal_form_actions = enumerate_normal_form_actions(vul_num)
    print(normal_form_actions)
    feasible_actions = filter_feasible_actions(time_matrices, task_nums, prob_vector, capacity, agent_order, normal_form_actions)
    print(feasible_actions)

