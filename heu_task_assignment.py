import numpy as np

import random
import sys

## Input: 1. probability vector
###       2. POI_num * cate_num matrix, each entry the number of tasks in POI i of category j
###       3. Agent_num * cate_num time matrix, each entry the time agent i needed for category j task
###       Note i is the row index, and j is the column index

def dynamic_programming(values, weights, capacity): ## This is a subprotocol for GAP (generalized assignment problem), where we assign every agent at one time
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
    return matrix[n-1][capacity-1], assignments[n-1][capacity-1]

def generate_weights_values(n):
    temp1 = []
    temp2 = []
    for i in range(n):
        temp1.append(random.randint(1, 10))
        temp2.append(random.random())

    return temp1, temp2


def utility_matrix_fixed(POI_cate_matrix, prob_vector):
    # We fix each task utility with the value of p divided by the total number of tasks in that POI
    POI_num = len(POI_cate_matrix)
    cate_num = len(POI_cate_matrix[0])
    utility_matrix = np.zeros((POI_num, cate_num))
    for i in range(POI_num):
        temp = POI_cate_matrix[i].sum()
        for j in range(cate_num):
            if POI_cate_matrix[i][j] > 0:
                utility_matrix[i][j] = prob_vector[i]/temp
    return utility_matrix

def convert_cate_to_individual(unassigned_matrix, utility_matrix, weight_cate_vector):
    ## Weight cate vector is a vecor of an agent's time to complete different category of tasks
    POI_num = len(unassigned_matrix)
    cate_num = len(unassigned_matrix[0])
    weights = []
    values = []
    for i in range(POI_num):
        for j in range(cate_num):
            for k in range(unassigned_matrix[i][j]):
                weights.append(weight_cate_vector[j])
                values.append(utility_matrix[i][j])
    return weights, values

def reverse_indi_to_cate_assign(indi_assignment, unassigned_matrix):
    # we need to return a matrix in the following format and return a new unassigned_matrix:
        #  [Agent_0:        task_cate_0   ...   task_cate_n
        #           POI_i1 [[                               ],
        #           ...     [                               ],
        #           POI_im  [                               ]],
        #
        #   ...
        #
        #   Agent_p:        task_cate_0   ...   task_cate_n
        #           POI_i1 [[                               ],
        #           ...     [                               ],
        #           POI_im  [                               ]],
        

        if len(indi_assignment) != unassigned_matrix.sum():
            print(indi_assignment)
            print(unassigned_matrix)
            print("Unmatched individual assignment and unassigned matrix!")
            sys.exit(1)
        n = len(indi_assignment)
        POI_num = len(unassigned_matrix)
        cate_num = len(unassigned_matrix[0])
        assign_matrix = np.zeros((POI_num, cate_num), dtype=int)
        row = 0
        column = 0
        current = 0
        for i in range(POI_num):
            for j in range(cate_num):
                if unassigned_matrix[i][j] != 0:
                    l = unassigned_matrix[i][j]
                    for k in range(current, current+l):
                        if indi_assignment[k] == 1:
                            assign_matrix[i][j] += 1
                            unassigned_matrix[i][j] -= 1
                    current += l
        '''
        for i in range(n):
            if i < current:
                if indi_assignment[i] == 1:
                    assign_matrix[row][column] += 1
                    unassigned_matrix[row][column] -= 1
            else:
                column += 1
                if column % cate_num == 0:
                    row +=1
                    column = 0
                current += unassigned_matrix[row][column]
        '''
        return assign_matrix
                

            
def transfer(assignment, POI_cate_matrix, unassigned_matrix, capacity, agent_cate_matrix):
    # We need transfer the assignment to the final_assignment which will give a betther coverage of more POIs
    # Assignment is the 3D-matrix, and final_assignment should also a 3D-matrix
    agent_num = len(assignment)
    POI_num = len(assignment[0])
    cate_num = len(assignment[0][0])
    coverage = np.zeros((POI_num, cate_num), dtype = int)
    for j in range(POI_num):
        temp = np.zeros(cate_num, dtype = int)
        for i in range(agent_num):
            temp += assignment[i][j]
        coverage[j] = temp
        if (np.array(POI_cate_matrix[j])-temp).sum() > 1: # 5 is a value we could change
            for i in range(agent_num):
                #assignment[i][j] = np.zeros(cate_num, dtype = int)
                for k in range(cate_num):
                    unassigned_matrix[j][k] += assignment[i][j][k]
                assignment[i][j] = np.zeros(cate_num, dtype = int)
        else:
            for i in range(agent_num):
                for k in range(cate_num):
                    capacity[i] -= (assignment[i][j][k] * agent_cate_matrix[i][k])

    return coverage

def check_POI_coverage(assignment, POI_cate_matrix):
    count = 0
    POI_covered = []
    agent_num = len(assignment)
    POI_num = len(POI_cate_matrix)
    cate_num = len(POI_cate_matrix[0])
    for j in range(POI_num):
        temp = np.zeros(cate_num, dtype = int)
        for i in range(agent_num):
            temp += assignment[i][j]
        if (np.array(POI_cate_matrix[j] - temp).sum() == 0):
            count += 1
            POI_covered.append(j)
    return POI_covered

def compute_slack(assignment, capacity, agent_cate_matrix):
    # Return a vector of time slack for every agent after the assignment

    agent_num = len(agent_cate_matrix)
    temp = capacity.copy()
    slack= np.array(temp)
    cate_num = len(agent_cate_matrix[0])
    POI_num = len(assignment[0])
    for i in range(agent_num):
        for j in range(POI_num):
            for k in range(cate_num):
                slack[i] -= (assignment[i][j][k] * agent_cate_matrix[i][k])
    return slack
        
def check_correctness(final_assignment, agent_cate_matrix):
    ch = []
    agent_num = len(final_assignment)
    POI_num = len(final_assignment[0])
    cate_num = len(agent_cate_matrix[0])
    for ag in range(agent_num):
        a1 = np.reshape(final_assignment[ag], POI_num*cate_num)
        a2 = np.array(list(agent_cate_matrix[ag])*POI_num)
        ch.append(np.dot(a1,a2))
    return ch

def divide_assignment(assignment, covered_POI, POI):
    # Divide the 3-d asssignment into two 3d matrix, one is the fixed one which represent those with POI covered, the other is the unfiexed one

    agent_num = len(assignment)
    cate_num = len(assignment[0][0])
    POI_num = len(assignment[0])
    fixed_assignment = np.zeros((agent_num, POI_num, cate_num), dtype = int)
    for j in range(POI_num):
        if j in covered_POI or j == POI:
            for i in range(agent_num):
                fixed_assignment[i][j] = assignment[i][j]

    unfixed_assignment = assignment-fixed_assignment
    return fixed_assignment, unfixed_assignment

def reassign_value(fixed_assignment, unfixed_assignment, slack, POI, agent_cate_matrix, utility_matrix, POI_cate_matrix, prob_vector):
    # This returns the largest reassign value for the POI, and its corresponding assignment
    agent_num = len(agent_cate_matrix)
    cate_num = len(agent_cate_matrix[0])
    POI_num = len(POI_cate_matrix)
    assigned_tasks = np.zeros(cate_num, dtype = int)
    task_utility = prob_vector[POI]/(POI_cate_matrix[POI].sum())
    for i in range(agent_num):
        assigned_tasks += fixed_assignment[i][POI]
    unassigned_tasks = POI_cate_matrix[POI] - assigned_tasks
    total_decreased_value = 0
    
    for i in range(cate_num):
        if unassigned_tasks[i] > 0:
            for j in range(unassigned_tasks[i]):
                decreased_values = []
                assigns = []
                for k in range(agent_num):
                    lower_bound = agent_cate_matrix[k][i]
                    temp_weights = []
                    temp_values = []
                    for j1 in range(POI_num):
                        for l in range(cate_num):
                            if unfixed_assignment[k][j1][l] > 0:
                                for n in range(unfixed_assignment[k][j1][l]):
                                    temp_weights.append(agent_cate_matrix[k][l])
                                    temp_values.append(utility_matrix[j1][l])
                    if len(temp_weights) == 0:
                        if slack[k] >= agent_cate_matrix[k][i]:
                            decreased_values.append(0)
                            assigns.append([])
                        else:
                            decreased_values.append(10000)
                            assigns.append([])
                        continue
                    lower_bound -= slack[k]
                    if lower_bound <= 0:
                        decreased_value = 0
                        temp_assign = np.zeros(len(temp_weights), dtype = int)
                    else:
                        decreased_value, temp_assign = minimize_decreased_values(temp_weights, temp_values, lower_bound)
                    '''
                    if decreased_value == 10000:
                        print(temp_weights, temp_values, lower_bound)
                        print(temp_assign)
                        print("What happened!")
                        sys.exit(1)
                    '''
                    decreased_values.append(decreased_value)
                    assigns.append(temp_assign)
                    mu = np.multiply(temp_assign, temp_weights).sum()
                    '''
                    if mu > 0 and mu+slack[k] < agent_cate_matrix[k][i]:
                        print("wrong")
                        print(agent_cate_matrix[k][i], mu+slack[k], temp_weights, temp_assign, mu, slack[k], decreased_value)
                        sys.exit(1)
                    '''
                temp_min = min(decreased_values)
                if temp_min > 1000:
                    return 1, 0
                ind = decreased_values.index(temp_min)
                temp_unassign = assigns[ind]
                total_decreased_value -= temp_min
                total_decreased_value += task_utility
                #print(temp_min, task_utility)
                unassign_matrix = reverse_indi_to_cate_assign(temp_unassign, unfixed_assignment[ind])
                
                #cate_pos = np.where(unassign_matrix == 1)[-1][0]
                original_slack = slack[ind]
                for p0 in range(POI_num):
                    for c0 in range(cate_num):
                        if unassign_matrix[p0][c0] > 0:
                            slack[ind] += (agent_cate_matrix[ind][c0] * unassign_matrix[p0][c0])
                slack[ind] -= agent_cate_matrix[ind][i]
                if slack[ind] < 0:
                    print(agent_cate_matrix[ind])
                    print(original_slack, agent_cate_matrix[ind][i])
                    print(unassign_matrix)
                    
                    print("slack: ", slack[ind])
                    sys.exit(1)
                fixed_assignment[ind][POI][i] += 1
    return 0, total_decreased_value
                
                    

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

def assign(unassigned_matrix, prob_vector, agent_cate_matrix, capacity, agent_order, utility_matrix):
    
    agent_num = len(agent_cate_matrix)
    POI_num = len(unassigned_matrix)
    cate_num = len(unassigned_matrix[0])
    final_assignment = np.zeros((agent_num, POI_num, cate_num), dtype = int)
    for i in agent_order:
        if unassigned_matrix.sum() == 0:
            break
        #utility_matrix = utility_matrix_conversion(unassigned_matrix, prob_vector)
        
        weights, values = convert_cate_to_individual(unassigned_matrix, utility_matrix, agent_cate_matrix[i])
        if len(weights) != unassigned_matrix.sum():
            print("Unmatched length!")
            sys.exit(1)
        max_value,indi_assignment = dynamic_programming(values, weights, capacity[i])
        assign_matrix = reverse_indi_to_cate_assign(indi_assignment, unassigned_matrix)
        final_assignment[i] = assign_matrix

    return final_assignment



def max_normal_form_action(POI_cate_matrix, prob_vector, agent_cate_matrix, original_capacity, agent_order):
    agent_num = len(agent_cate_matrix)
    cate_num = len(agent_cate_matrix[0])
    POI_num = len(prob_vector)
    utility_matrix = utility_matrix_fixed(POI_cate_matrix, prob_vector)
    unassigned_matrix = POI_cate_matrix.copy()
    capacity = original_capacity.copy()
    final_assignment = assign(unassigned_matrix, prob_vector, agent_cate_matrix, capacity, list(range(agent_num)), utility_matrix)

    capacity = original_capacity.copy()
    covered_POI = check_POI_coverage(final_assignment, POI_cate_matrix)
    slack = compute_slack(final_assignment, capacity, agent_cate_matrix)
    while True:
        unused_POI = []
        candidate_POI = []
        all_decreased_values = []
        fixed_assignments = []
        unfixed_assignments = []
        for i in range(POI_num):
            if i not in covered_POI:
                #print(i)
                fixed_assignment, unfixed_assignment = divide_assignment(final_assignment, covered_POI, i)
                capacity = original_capacity.copy()
                slack = compute_slack(final_assignment, capacity, agent_cate_matrix)

                indicator, total_decreased_value = reassign_value(fixed_assignment, unfixed_assignment, slack, i, agent_cate_matrix, utility_matrix, POI_cate_matrix, prob_vector)
                if indicator == 1:
                    unused_POI.append(i)
                else:
                    candidate_POI.append(i)
                    all_decreased_values.append(total_decreased_value)
                    fixed_assignments.append(fixed_assignment)
                    unfixed_assignments.append(unfixed_assignment)
        if len(candidate_POI) == 0:
            break
        max_update = max(all_decreased_values)
        max_ind = all_decreased_values.index(max_update)
        unfixed_assignment = unfixed_assignments[max_ind]
        fixed_assignment = fixed_assignments[max_ind]
        final_assignment = fixed_assignment + unfixed_assignment
        covered_POI.append(candidate_POI[max_ind])
        #print(covered_POI, len(covered_POI))
    #capacity = original_capacity.copy()
    #slack = compute_slack(fixed_assignment, capacity, agent_cate_matrix)
    return covered_POI

if __name__ == "__main__":
    cate_num = 10
    #weights, values = generate_weights_values(20)
    agent_num = 10
    POI_num = 30
    prob_vector = []
    time_bound = 60


    final_assignment = np.zeros((agent_num, POI_num, cate_num), dtype = int)
    agent_cate_matrix = np.zeros((agent_num, cate_num), dtype=int)
    POI_cate_matrix = np.zeros((POI_num, cate_num), dtype = int)
    
    capacity = [time_bound] * agent_num
    for i in range(POI_num):
        prob_vector.append(random.random())
    for i in range(agent_num):
        for j in range(cate_num):
            agent_cate_matrix[i][j] = random.randint(1, 10)
    for i in range(POI_num):
        for j in range(cate_num):
            POI_cate_matrix[i][j] = random.randint(0, 3)
    normal_form_action = max_normal_form_action(POI_cate_matrix, prob_vector, agent_cate_matrix, capacity, list(range(agent_num)))
    #print(normal_form_action)
    '''
    unassigned_matrix = POI_cate_matrix.copy()
    utility_matrix = utility_matrix_fixed(POI_cate_matrix, prob_vector)    
    final_assignment = assign(unassigned_matrix, prob_vector, agent_cate_matrix, capacity, list(range(agent_num)), utility_matrix)
    #normal_action = normal_form_action(final_assignment, POI_cate_matrix, unassigned_matrix, capacity, agent_cate_matrix)
    capacity = [time_bound] * agent_num
    covered_POI = check_POI_coverage(final_assignment, POI_cate_matrix)
    print(covered_POI, len(covered_POI))
    slack = compute_slack(final_assignment, capacity, agent_cate_matrix)
    count0 = 0
    
    while True:
        unused_POI = []
        candidate_POI = []
        all_decreased_values = []
        fixed_assignments = []
        unfixed_assignments = []
        for i in range(POI_num):
            if i not in covered_POI:
                #print(i)
                fixed_assignment, unfixed_assignment = divide_assignment(final_assignment, covered_POI, i)
                slack = compute_slack(final_assignment, capacity, agent_cate_matrix)
            
                indicator, total_decreased_value = reassign_value(fixed_assignment, unfixed_assignment, slack, i, agent_cate_matrix, utility_matrix, POI_cate_matrix, prob_vector)
                if indicator == 1:
                    unused_POI.append(i)
                else:
                    candidate_POI.append(i)
                    all_decreased_values.append(total_decreased_value)
                    fixed_assignments.append(fixed_assignment)
                    unfixed_assignments.append(unfixed_assignment)
        if len(candidate_POI) == 0:
            break
        max_update = max(all_decreased_values)
        max_ind = all_decreased_values.index(max_update)
        unfixed_assignment = unfixed_assignments[max_ind]
        fixed_assignment = fixed_assignments[max_ind]
        final_assignment = fixed_assignment + unfixed_assignment
        covered_POI.append(candidate_POI[max_ind])
        print(covered_POI, len(covered_POI))
        print(unused_POI)
    slack = compute_slack(fixed_assignment, [time_bound]*agent_num, agent_cate_matrix)
    print(slack)
    
    for i in range(POI_num):
        capacity = [time_bound] * agent_num
        if i not in covered_POI:
            slack = compute_slack(fixed_assignment, capacity, agent_cate_matrix)
            unassigned_matrix = np.zeros((POI_num, cate_num), dtype = int)
            unassigned_matrix[i] = POI_cate_matrix[i]
            final_assignment = assign(unassigned_matrix, prob_vector, agent_cate_matrix, slack, list(range(agent_num)), utility_matrix)
            if final_assignment.sum() == POI_cate_matrix[i].sum():
                print("assigned ", i)
    '''
