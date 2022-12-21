import random
from itertools import *
import time
import numpy as np
from scipy import optimize
import sys
import os
from task_assignment_algorithm import *

def generate_POI(num_cate, num_POI):
    POI = []
    num_per_cate = [0]*num_cate
    for i in range(num_POI):
        temp = []
        for j in range(num_cate):
            temp.append(random.randint(0, 6))
            num_per_cate[j] += temp[j]
        POI.append(temp)
    return POI, num_per_cate
def generate_POI_0(num_cate, num_POI, task_per_POI): # In this function, we want every POI has 5 tasks
    POI = []
    num_per_cate = np.zeros(num_cate, dtype=int)
    for i in range(num_POI):
        temp = np.zeros(num_cate, dtype=int)
        for j in range(task_per_POI):
            temp[random.randint(0, num_cate-1)] += 1
        POI.append(list(temp))
        num_per_cate = np.add(num_per_cate, temp)
    return POI, list(num_per_cate)

def next_lambda(compact_form_stra, current_action, POI_cate_matrix, row_actions, agent_cate_matrix, original_capacity):
    #Use binary search to find a lambda that will give us a new action

    left = 0
    right = len(current_action) * find_mincomp_act(compact_form_stra, current_action)
    compact_form = compact_form_stra.copy()
    A = overlap_POI(row_actions, row_actions)
    A = np.array(A)
    temp_v = []
    new_action = current_action
    for act in row_actions:
        temp_v.append(normal_action_value(compact_form_stra, act))

    while (right - left > 0.01):
        temp = (left+right)/2
        B = np.array(temp_v)-np.array([temp]*len(row_actions))

        #B = np.array([temp]*len(row_actions))
        sol = optimize.linprog(np.array([1]*len(row_actions)), A_ub = np.multiply(A, -1), b_ub = np.multiply(B, -1))
        x = sol.get('x')
        
        compact_form = compact_form_stra.copy()
        for action in range(len(row_actions)):
            for poi in row_actions[action]:
                compact_form[poi] = compact_form[poi] - x[action]
        current_row_act = max_normal_form_action(POI_cate_matrix, compact_form, agent_cate_matrix, original_capacity, list(range(agent_num)))

        #current_v = normal_action_value(compact_form_stra, current_action)
        #next_v = normal_action_value(compact_form_stra, next_action)
        if overlap_POI([current_row_act], [current_action]) == [[len(current_action)]]:
            left = temp
        else:
            new_action = current_row_act
            right = temp


    B = np.array(temp_v)-np.array([right]*len(row_actions))

    sol = optimize.linprog(np.array([1]*len(row_actions)), A_ub = np.multiply(A, -1), b_ub = np.multiply(B, -1))
    x = sol.get('x')
    compact_form = compact_form_stra.copy()
    for action in range(len(row_actions)):
        for poi in row_actions[action]:
            compact_form[poi] = compact_form[poi] - x[action]
    row_actions.append(new_action)
    return right, compact_form, x

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

def homotopy(POI_cate_matrix, compact_form_stra, agent_cate_matrix, original_capacity):
    
    row_actions = []
    lam = []
    agent_num = len(agent_cate_matrix)
    cate_num = len(agent_cate_matrix[0])
    POI_num = len(POI_cate_matrix)
    prob_vector = compact_form_stra.copy()
    ########  Intialization ########
    current_row_act = max_normal_form_action(POI_cate_matrix, prob_vector, agent_cate_matrix, original_capacity, list(range(agent_num)))
    current_value = normal_action_value(prob_vector, current_row_act)
    print(prob_vector, current_row_act)
    #####################################

    row_actions.append(current_row_act)
    current_lam = current_value - 0.01
    lam.append(current_lam)
    A = overlap_POI(row_actions, row_actions)
    A = np.array(A)
    A_dual = np.transpose(A)
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
    while (len(lam) == 0 or lam[-1] >= 0.1):
        
        #############################################################
        current_row_act = max_normal_form_action(POI_cate_matrix, prob_vector, agent_cate_matrix, original_capacity, list(range(agent_num)))
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
            A_dual = np.transpose(A)
            temp_v = []
            for act in row_actions:
                temp_v.append(normal_action_value(compact_form_stra, act))
            B = np.array(temp_v)-np.array([current_lam]*len(row_actions))
            sol = optimize.linprog(np.array([1]*len(row_actions)), A_ub = np.multiply(A, -1), b_ub = np.multiply(B, -1))
            x = sol.get('x')
            prob_vector = update(compact_form_stra, row_actions, x)
        print(lam[-1])

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

def convert_matrix(agents_time_matrix, action): #action here is in the form of task category
    task_agent_time_matrix = np.zeros((sum(action), len(agents_time_matrix)), dtype = int)
    i = 0
    for cate in range(len(action)):
        for j in range(i, i+action[cate]):
            for agent in range(len(agents_time_matrix)):
                task_agent_time_matrix[j][agent] = agents_time_matrix[agent][cate]
        i += action[cate]
    return task_agent_time_matrix.tolist()


if __name__ == "__main__":
    cate_num = 3
    agent_num = 10
    POI_num = 20
    task_per_POI = 8
    homotopy_time_list = []
    repeat_time = 10
    file1 = open("data0", "w")
    file1.write("Running time of SWING with varied agent number\n")
    file1.close()
    for agent_num in range(2, 21, 2):
        time_bound = int(5*POI_num*task_per_POI/20)
        temp_time = 0
        file1 = open("data0", "a")
        file1.write("\n" + str(agent_num) + ": ")
        file1.close()
        for fre in range(repeat_time):
            POI_cate_matrix, num_per_cate = generate_POI_0(cate_num, POI_num, task_per_POI)
            POI_cate_matrix = np.array(POI_cate_matrix)
            print(POI_cate_matrix)
            agent_cate_matrix = np.zeros((agent_num, cate_num), dtype=int)
            compact_form_stra = []
            for i in range(agent_num):
                for j in range(cate_num):
                    agent_cate_matrix[i][j] = random.randint(1, 10)
            original_capacity = [time_bound] * agent_num
            for i in range(POI_num):
                compact_form_stra.append(random.random())
            t1 = time.time()
            valid_actions, x, lam = homotopy(POI_cate_matrix, compact_form_stra, agent_cate_matrix, original_capacity)
            t2 = time.time()
            print("valid_actions:", valid_actions)
            print("solution:", x)
            print("number of tasks per category: ", num_per_cate)
            print("lam:", lam)
            print("Homotopy time: ", t2-t1)
            temp_time += (t2-t1)
            t3 = t2-t1
            file1 = open("data0", "a")
            file1.write(str(t3) + ", ")
            file1.close()
        homotopy_time_list.append(temp_time/repeat_time)
        file1 = open("data0", "a")
        temp_data = temp_time/repeat_time
        file1.write("\navg time: "+str(temp_data))
        file1.close()
    print(homotopy_time_list)
