import numpy as np
import random
import sys

MAXIMUM = 10000

## Input: 1. probability vector
##        2. vul_num-dimensional vector task_nums where each entry represent the number of tasks each vul has
##        3. vul_num agent_num * task_nums[i] matrix to represent time each agent needs to complete the tasks in vulnerability i.
##        4. agent_num capacity vector

## Intermidiate input:
#  1. agent_num * task_total_num matrix to represent the time for each agent to complete each task
#  2. agent_num * time_total_num binary task assignment matrix

def dynamic_programming(values, weights, capacity): ## This is a subprotocol for GAP (generalized assignment problem). This is a generic dp algorthm to solve the 0-1 basic knapsack problem.
    ## Note that wieghts and capacity must be integer, but values can be floats
    #print(capacity)
    if len(values) == 1:
        if weights[0] <= capacity:
            return values[0], np.array([1])
        else:
            return 0, np.array([0])
    if capacity == 0:
        return 0, np.array([0]*len(values))
    n = len(values)
    assignments = np.zeros((n, capacity, n), dtype = int)
    matrix = np.zeros((n, capacity))
    temp0 = np.zeros(n, dtype = int)
    temp0[0] = 1
    for j in range(1, capacity):
        if weights[0] <= j:
            matrix[0][j] = values[0]
            assignments[0][j] = temp0
    for i in range(1, n):
        for j in range(1, capacity):
            if weights[i] > j:
                matrix[i][j] = matrix[i-1][j]
                assignments[i][j]  = assignments[i-1][j]

            else:
                if matrix[i-1][j]> (matrix[i-1][j-weights[i]]+values[i]):
                    matrix[i][j] = matrix[i-1][j]
                    assignments[i][j] = assignments[i-1][j]
                else:
                    temp = assignments[i-1][j-weights[i]]
                    temp[i] = 1
                    assignments[i][j] = temp
                    matrix[i][j] = matrix[i-1][j-weights[i]]+values[i]
    #print(n, capacity)
    print(matrix)
    return matrix[n-1][capacity-1], assignments[n-1][capacity-1]

def values(probability_vector, task_nums):
    # remaining_task_nums the number of unassigned tasks in each vulnerability

    ## !!! Do we really need this?
    v_long = []
    for i in range(len(probability_vector)):
        for j in range(task_nums[i]):
            if task_nums[i] > 0:
                v_long.append(probability_vector[i]/task_nums[i])
            else:
                v_long.append(-1) # If remaining_task_nums[i] = 0, all the tasks in vul i have been assigned. We make the value of the task negative to prevent reassignment in the future
    return  v_long # Output a total_task_num-dimensional vector where each entry is the current value of assigning individual task.

def weight_matrix2d(time_matrices): # Join the vul_num individual agent_num * task_num_in_vul time matrix and obtain a agent_num * total_task_num matrix
    agent_task_time_matrix = []
    vul_num = len(time_matrices)
    agent_num = len(time_matrices[0])
    for i in range(agent_num):
        temp = []
        for j in range(vul_num):
            temp.extend(time_matrices[j][i])
        agent_task_time_matrix.append(temp)
    return agent_task_time_matrix # This is the agent_num * total_task_num time matrix

def con_values(GAP_task_values, task_nums): # GAP_task_values is the output of function values(), extend it to total_task_num long
    vul_num = len(GAP_task_values)
    total_value_list = []
    for i in range(vul_num):
        for j in range(task_nums[i]):
            total_value_list.append(GAP_task_values[i])
    return total_value_list # This is the total_task_num-dimensional vector where each entry is the value to assign that task

def minimize_decreased_values(weights, values, lower_bound):
    n = len(values)
    #print(n, lower_bound)
    if n == 1:
        if lower_bound < weights[0]:
            return values[0], np.array([1])
        else:
            return 10000, np.array([0])
    assignments = np.zeros((n, lower_bound+1, n), dtype = int)
    matrix = np.full((n, lower_bound+1), 10000, dtype = float)
    for i in range(n):
        for j in range(lower_bound+1):
            if weights[i] >= j:
                if i == 0:
                    matrix[i][j] = values[i]
                    temp = np.zeros(n, dtype = int)
                    temp[0] = 1
                    assignments[i][j] = temp
                else:
                    matrix[i][j] = min(matrix[i-1][j], values[i])
                    if matrix[i][j] == matrix[i-1][j]:
                        assignments[i][j] = assignments[i-1][j]
                    else:
                        temp = np.zeros(n, dtype = int)
                        temp[i] = 1
                        assignments[i][j] = temp
            else:

                if matrix[i-1][j] > (matrix[i-1][j-weights[i]]+values[i]):
                    matrix[i][j] = matrix[i-1][j-weights[i]]+values[i]
                    temp = assignments[i-1][j-weights[i]]
                    temp[i] = 1
                    assignments[i][j] = temp
                elif matrix[i-1][j] <= (matrix[i-1][j-weights[i]]+values[i]):
                    matrix[i][j] = matrix[i-1][j]
                    assignments[i][j] = assignments[i-1][j]

    return matrix[n-1][lower_bound], assignments[n-1][lower_bound]

# assigned matrix is a agent_num * total_tasks_num matrix where 1 represents the task is assigned to the corresponding agent; 0 otherwise
def initial_assign(time_matrix, capacity, agent_order, values_long):
    assignment = np.zeros((len(time_matrix), len(time_matrix[0])), dtype = int)
    vals = values_long.copy()
    for i in agent_order:
        weights = time_matrix[i].copy()
        max_value, indi_assignment = dynamic_programming(vals, weights, capacity[i])
        for j in range(len(indi_assignment)):
            if indi_assignment[j] == 1:
                vals[j] = -1
        assignment[i] = indi_assignment
    return assignment

def reassign_value(current_assignment, time_matrix, capacity, task_nums, values_long, the_vul):
    agent_num = len(capacity)
    slack = compute_slack(current_assignment, time_matrix, capacity)
    covered_vuls, uncovered_vuls, divided_assignment = covered_vul(current_assignment, task_nums)
    if the_vul not in uncovered_vuls:
        return 1, 0, current_assignment
    uncovered_tasks_in_the_vul = []
    for i in range(task_nums[the_vul]):
        if divided_assignment[the_vul][i] == 0:
            uncovered_tasks_in_the_vul.append(i)
    

    all_agents_weights = []
    all_agents_values = []
    uncovered_vuls.remove(the_vul)

    available_agents = []
    for the_agent in range(agent_num):
        we, va = extract_unfixed_assignment(the_agent, current_assignment, uncovered_vuls, task_nums, values_long, time_matrix)
        if len(we) > 0:
            available_agents.append(the_agent)
        all_agents_weights.append(we)
        all_agents_values.append(va)
    
    total_decreased_value = 0
    print("Uncovered tasks in the vul, ", uncovered_tasks_in_the_vul)
    new_assignment = current_assignment.copy()
    for j in uncovered_tasks_in_the_vul:
        assigns = []
        decreased_values = []
        the_task = sum(task_nums[:the_vul])+j
        for k in range(agent_num):
            lower_bound = time_matrix[k][the_task] - slack[k]
                
            if lower_bound <= 0:
                print(decreased_values, "---------")
                decreased_values.append(0)
                assigns.append([0]*len(all_agents_weights[k]))
            elif k in available_agents:
                decreased_value, temp_assign = minimize_decreased_values(all_agents_weights[k], all_agents_values[k], lower_bound)
                assigns.append(temp_assign)
                decreased_values.append(decreased_value)
            else:
                decreased_values.append(MAXIMUM)
                assigns.append([])
        temp_min = min(decreased_values)
        if temp_min == MAXIMUM:
            print("--------------------")
            return 1, 0, current_assignment
        ind = decreased_values.index(temp_min)
        print("Decreased values: ", decreased_values)
        temp_unassign = assigns[ind]
        total_decreased_value -= temp_min
        total_decreased_value += values_long[the_task]
        
        new_assignment[ind][the_task] = 1
        pointer = 0
        for v in uncovered_vuls:
            for t in range(task_nums[v]):
                temp_t = sum(task_nums[:v])+t
                if new_assignment[ind][temp_t] == 1:
                    if temp_unassign[pointer] == 1:
                        new_assignment[ind][temp_t] = 0
                    pointer += 1

    return 0, total_decreased_value, new_assignment

                    
def extract_unfixed_assignment(the_agent, current_assignment, uncovered_vuls, task_nums, values_long, time_matrix):
    # Extract the weights and values from current_assignment of the uncovered_vul  for the_agent
    extracted_weights = []
    extracted_values = []
    for v in uncovered_vuls:
        for j in range(task_nums[v]):
            the_task = sum(task_nums[:v])+j
            if current_assignment[the_agent][the_task] == 1:
                extracted_weights.append(time_matrix[the_agent][the_task])
                extracted_values.append(values_long[the_task])
    return extracted_weights, extracted_values
        
def compute_slack(current_assignment, time_matrix, capacity):
    agent_num = len(capacity)
    slack = capacity.copy()
    total_task_num = len(current_assignment[0])
    for i in range(agent_num):
        for j in range(total_task_num):
            slack[i] -= (current_assignment[i][j] * time_matrix[i][j])
    return slack
    

def covered_vul(current_assignment, task_nums):
    # Given an assignment, we want to which vulnerbility is fully covered
    vul_num = len(task_nums)
    agent_num = len(current_assignment)
    total_task_num = len(current_assignment[0])
    
    cur_vul = 0
    cur_task = 0
    covered_vuls = []
    uncovered_vuls = []
    divided_assignment = []
    while cur_vul < vul_num:
        task_assignment_vul = []
        for j in range(cur_task, cur_task+task_nums[cur_vul]):
            assigned = 0
            for i in range(agent_num):
                assigned += current_assignment[i][j]
            task_assignment_vul.append(assigned)
        divided_assignment.append(task_assignment_vul)
        if sum(task_assignment_vul) == task_nums[cur_vul]:
            covered_vuls.append(cur_vul)
        else:
            uncovered_vuls.append(cur_vul)
        print(task_assignment_vul)
        cur_task += task_nums[cur_vul]
        cur_vul += 1
    return covered_vuls, uncovered_vuls, divided_assignment
        

if __name__ == "__main__":
    vul_num = 3
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
    capacity = [20]*agent_num
    time_matrix = weight_matrix2d(time_matrices)
    agent_order = range(agent_num)
    values_long = values(prob_vector, task_nums)
    #values_long = con_values(vals, task_nums)
    
    print(time_matrix)
    test_assign = initial_assign(time_matrix, capacity, agent_order, values_long)
    print(test_assign)
    covered_vuls, uncovered_vuls, di = covered_vul(test_assign, task_nums)
    print("Uncovered vuls: ", uncovered_vuls)
    print(reassign_value(test_assign, time_matrix, capacity, task_nums, values_long, uncovered_vuls[0]))
    
