import random
from itertools import *
import time
import numpy as np
from scipy import optimize
import sys
import os, psutil
from multiprocessing import Pool, freeze_support
#from task_assignment_algorithm import *
from max_nfa import *


def normal_action_value(compact_form_stra, normal_form_action):
    v = 0
    for poi in normal_form_action:
        v += compact_form_stra[poi]
    return v

def update(compact_form, row_actions, x):
    updated_c = compact_form.copy()
    for acti in range(len(row_actions)):
        for i in row_actions[acti]:
            updated_c[i] -= x[acti]
    return updated_c

def homotopy(task_nums, compact_form_stra, time_matrix, original_capacity):
    t1 = time.time()
    row_actions = []
    lam = []
    agent_num = len(time_matrix)
    vul_num = len(task_nums)
    prob_vector = compact_form_stra.copy()
    values_long = values(prob_vector, task_nums) 
    ########  Intialization ########
    current_row_act = max_normal_form_action(time_matrix, values_long, original_capacity, list(range(agent_num)), task_nums)
    current_value = normal_action_value(prob_vector, current_row_act)
    #print(prob_vector, current_row_act)
    #####################################

    row_actions.append(current_row_act)
    current_lam = current_value - 0.01
    lam.append(current_lam)
    A = overlap_POI(row_actions, row_actions)
    A = np.array(A)
    temp_v = []
    for act in row_actions:
        temp_v.append(normal_action_value(compact_form_stra, act))
    B = np.array(temp_v)-np.array([current_lam]*len(row_actions))
    sol = optimize.linprog(np.array([1]*len(row_actions)), A_ub = np.multiply(A, -1), b_ub = np.multiply(B, -1))
    x = sol.get('x')

    prob_vector = update(compact_form_stra, row_actions, x)
    #print("prob_vector: ", prob_vector)
    #print("compact form stra: ", compact_form_stra)
    #print(row_actions)
    while (len(lam) == 0 or lam[-1] >= 0.01):
        values_long = values(prob_vector, task_nums)
        #############################################################
        current_row_act = max_normal_form_action(time_matrix, values_long, original_capacity, list(range(agent_num)), task_nums)
        current_value = normal_action_value(prob_vector, current_row_act)
        #############################################################
        in_row_actions = False

        ###########################################################
        if in_row_actions == False:
            current_lam = current_value-0.01
            lam.append(current_lam)
            if sorted(current_row_act) not in row_actions:
                row_actions.append(sorted(current_row_act))

            A = overlap_POI(row_actions, row_actions)
            A = np.array(A)
            temp_v = []
            for act in row_actions:
                temp_v.append(normal_action_value(compact_form_stra, act))
            B = np.array(temp_v)-np.array([current_lam]*len(row_actions))
            sol = optimize.linprog(np.array([1]*len(row_actions)), A_ub = np.multiply(A, -1), b_ub = np.multiply(B, -1))
            x = sol.get('x')
            prob_vector = update(compact_form_stra, row_actions, x)
        #print(lam[-1])
    t2 = time.time()
    fi = open("SWING_varied_vul", "a")
    fi.write(str(vul_num) + "\t\t" + str(t2-t1) + "\n")
    fi.close()
    print(vul_num)
    return row_actions, x, lam
       
def overlap_POI(row_actions, column_actions):
    A = []
    for i in range(len(row_actions)):
        temp = []
        for j in range(len(column_actions)):
            intersection = [p for p in row_actions[i] if p in column_actions[j]]
            temp.append(len(intersection))
        A.append(temp)
    return A



if __name__ == "__main__":
    cate_num = 3
    agent_num = 10
    vul_num = 30
    task_num_per_vul = 8
    homotopy_time_list = []
    repeat_time = 20
    all_args = []
    fi = open("SWING_varied_vul", "a")
    fi.write("Vul number\t running time\n")
    fi.close()
        
    for fre in range(repeat_time):
        for vul_num in range(3, 36):
            ######################## Iniciate the parameters #########################
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
            time_matrix =weight_matrix2d(time_matrices) 
            agent_order = range(agent_num)
            total_time_needed = (np.array(time_matrices).sum()/agent_num)/(3*agent_num)
            capacity = [int(total_time_needed)]*agent_num
            compact_form_stra = prob_vector.copy()

            all_args.append((task_nums, compact_form_stra, time_matrix, capacity))
            ############################################################################
            '''
            t1 = time.time()
            valid_actions, x, lam = homotopy(task_nums, compact_form_stra, time_matrix, capacity)
            t2 = time.time()
            print("valid_actions:", valid_actions)
            print("solution:", x)
            print("lam:", lam)
            print("Homotopy time: ", t2-t1)
            temp_time += (t2-t1)
            '''
    freeze_support()
    with Pool() as pool:
        L = pool.starmap(homotopy, all_args)
    #homotopy_time_list.append(temp_time/repeat_time)
    #print(homotopy_time_list)
