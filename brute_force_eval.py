import random
from itertools import *
import time
import numpy as np
from scipy import optimize
import sys
import os
#from gap_new import *
from GAP import *

def pseudo_compact_form_strategy(num_POI):
    compact_form = []
    
    for i in range(num_POI):
        compact_form.append(random.random())
    return compact_form
    
    #return [0.1]*num_POI


def pseudo_normal_actions(num_cate, num_per_cate):
    big_list = []
    for i in range(num_cate):
        tem = []
        for j in range(num_per_cate[i]):
            tem.append(j)
        big_list.append(tem)

    normal_actions = []
    for element in product(*big_list):
        normal_actions.append(element)
    return normal_actions

def edge_actions(avg_proc_time, num_per_cate, all_actions):
    bound = multi_sum(avg_proc_time, num_per_cate)/2
    possible_actions = []
    for action in all_actions:
        if multi_sum(action, avg_proc_time)<=bound:
            possible_actions.append(action)
    #edges = []
    #for action in possible_actions:
    #    to_add = True
    #    for temp in possible_actions:
    #        if action != temp and compare_list(action, temp):
    #            to_add = False
    #            break
    #    if to_add == True:
    #        edges.append(action)
    return possible_actions


def multi_sum(list_a, list_b):
    a = 0
    for i in range(len(list_a)):
        a += (list_a[i]*list_b[i])
    return a

def compare_list(list_a, list_b):
    for i in range(len(list_a)):
        if list_a[i]>list_b[i]:
            return False
    return True

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

def action_in_lists(action, action_lists):
    for a in action_lists:
        if overlap_POI([action], [a]) == [[max(len(action), len(a))]]:
            return True
    return False

def multi_dynamic_programming(POI, compact_form, all_actions, current_actions):
    dic = {}
    dic[tuple([0, 0, 0])] = 0
    eq_action = {}
    eq_action[tuple([0, 0, 0])] = []
    for action in all_actions:
        #print("============================================================")
        #print(action)
        candidate = []
        candidate_act = []
        candidate_dual = []
        candidate_act_dual = []
        for p in range(len(compact_form)):
            m = tuple(list_a_minus_b(action, POI[p]))
            #print(m)
            if larger_than_zero(m):
                if m not in dic:
                    search = []
                    search_val = []
                    for ac in dic.keys():
                        if smaller_than_list(ac, m):
                            search.append(ac)
                            search_val.append(dic[ac])
                    ind = search_val.index(max(search_val))
                    m = search[ind]
                if (p not in eq_action[m]):
                    ####### If action already in valid action, we get rid of it from candidates ##########
                    temp_act = eq_action[m] + [p]
                    candidate_act.append(temp_act)
                    '''
                    if action_in_lists(temp_act, current_actions):
                        candidate.append(-1)
                    else:
                    '''
                    candidate.append(dic[m]+compact_form[p])
                #if (p not in eq_action[m][1]):
                #    candidate_dual.append(dic[m][1]+compact_form[p])
                #    temp_act_dual = eq_action[m][1]+[p]
                #    candidate_act_dual.append(temp_act_dual)
            #elif larger_than_zero(m): 
                elif (p in eq_action[m]):
                    '''
                    if eq_action[m] not in current_actions:
                        candidate.append(dic[m])
                        candidate_act.append(eq_action[m])
                    else:
                    '''
                    candidate.append(-1)
                    candidate_act.append(eq_action[m])
                #if (p in eq_action[m][1]):
                #    candidate_dual.append(dic[m][1])
                #    candidate_act_dual.append(eq_action[m][1])
        if len(candidate) != 0:
            temp = max(candidate)
            ind = candidate.index(temp)
            act = candidate_act[ind]
        else:
            temp = 0
            act = []
        #if len(candidate) != 0: 
        #    temp1 = max(candidate_dual)
        #    ind1 = candidate_dual.index(temp1)
        #    act1 = candidate_act[ind1]
            #dic[tuple(action)] = temp
            #eq_action[tuple(action)] =candidate_act[ind]

        #else:
        #    temp1 = 0
        #    act1 = []
        dic[tuple(action)] = temp
        eq_action[tuple(action)] = act
        #else:
        #    dic[tuple(action)] = 0
        #    eq_action[tuple(action)] = []
        #print(dic)
        #print(eq_action)
    return dic, eq_action
            
    
            
        
def list_a_minus_b(list_a, list_b):
    m = []
    for i in range(len(list_a)):
        m.append(list_a[i]-list_b[i])
    return m

def larger_than_zero(l):
    for i in range(len(l)):
        if l[i] < 0:
            return False
    return True

def used_actions(dic, eq_actions, edge_actions):
    objective_func = []
    normal_form_actions = []
    for action in edge_actions:
        if (tuple(action) in dic) and (eq_actions[tuple(action)] not in normal_form_actions):
            objective_func.append(dic[tuple(action)])
            normal_form_actions.append(eq_actions[tuple(action)])
    return objective_func, normal_form_actions
            
def second_largest_act_value(dic, eq_actions, row_actions):

    largest_val = max(dic.values())
    position1 = max(dic, key=dic.get)
    i = 1
    sorted_values = sorted(dic.values())
    sorted_values.reverse()
    while in_action_list(row_actions, eq_actions[position1]):
        v = sorted_values[i]
        if v < largest_val:
            position1 = list(dic.keys())[list(dic.values()).index(v)]
            largest_val = v
        i += 1
    max2 = 0
    for temp_v in dic.values():
        if temp_v > max2 and temp_v < largest_val:
            max2 = temp_v
    position2 = list(dic.keys())[list(dic.values()).index(max2)]
    print("largest_value, max2: ", largest_val, max2)
    '''
    values = list(dic.values())
    acts = list(dic.keys())
    v = values.copy()
    sorted_v = sorted(v)
    largest_val = sorted_v[-1]
    second_largest_val = sorted_v[-2]
    for val in reversed(sorted_v):
        if val < largest_val:
            second_largest_val = val
            break
    position1 = values.index(largest_val)
    position2 = values.index(second_largest_val)
    '''
    return eq_actions[position1], largest_val, eq_actions[position2], max2

def in_action_list(action_list, action):
    in_actions = False
    for Act in action_list:
        if len(Act) == len(action) and overlap_POI([Act], [action]) == [[len(Act)]]:

            in_actions = True
            break
    return in_actions

def find_mincomp_act(compact_form, action):
    x = compact_form[action[0]]
    for poi in action:
        if compact_form[poi] < x and compact_form[poi] > 0:
            x = compact_form[poi]
    return x

def remove_zero_act(compact_form, action):
    act = []
    for i in range(len(action)):
        if compact_form[action[i]] > 0:
            act.append(action[i])
    return act

def next_lambda(compact_form_stra, current_action, next_action, POI, possible_actions, row_actions):
    #Use binary search to find a lambda that will give us a new action

    left = 0
    right = len(current_action) * find_mincomp_act(compact_form_stra, current_action)
    compact_form = compact_form_stra.copy()
    A = overlap_POI(row_actions, row_actions)
    A = np.array(A)

    while (right - left > 0.0001):
        temp = (left+right)/2
        B = np.array([temp]*len(row_actions))
        sol = optimize.linprog(np.array([1]*len(row_actions)), A_ub = np.multiply(A, -1), b_ub = np.multiply(B, -1))
        delta_x = sol.get('x')
        compact_form = compact_form_stra.copy()
        for action in range(len(row_actions)):
            for poi in row_actions[action]:
                compact_form[poi] = compact_form[poi] - delta_x[action]
        current_v = normal_action_value(compact_form_stra, current_action)
        next_v = normal_action_value(compact_form_stra, next_action)

        if current_v >= next_v:
            left = temp
        else:
            rigth = temp
        '''
        dic, eq_action = multi_dynamic_programming(POI, compact_form, possible_actions, row_actions)
        current_row_act, current_value, next_row_act, next_value = second_largest_act_value(dic, eq_action)
        if overlap_POI([current_row_act], [current_action]) == [[len(current_action)]]:
            left = temp
        else:
            right = temp
        '''
        #print("current row action: ", current_row_act)
    B = np.array([right]*len(row_actions))
    sol = optimize.linprog(np.array([1]*len(row_actions)), A_ub = np.multiply(A, -1), b_ub = np.multiply(B, -1))
    delta_x = sol.get('x')
    compact_form = compact_form_stra.copy()
    for action in range(len(row_actions)):
        for poi in row_actions[action]:
            compact_form[poi] = compact_form[poi] - delta_x[action]
    return right, compact_form, delta_x

def normal_action_value(compact_form_stra, normal_form_action):
    v = 0
    for poi in normal_form_action:
        v += compact_form_stra[poi]
    return v

def smaller_than_list(list_a, list_b): # Return True if every element in list_a smaller than or equal to every element in list_b
    for i in range(len(list_a)):
        if list_a[i] > list_b[i]:
            return False
    return True

def homotopy(POI, compact_form_stra, possible_actions):
    print(POI)
    row_actions = []
    possible_actions()
    column_actions = []
    lam = []
    ########  Intialization ########
    dic, eq_action = multi_dynamic_programming(POI, compact_form_stra, possible_actions, row_actions)
    current_row_act, current_value, next_row_act, next_value = second_largest_act_value(dic, eq_action, row_actions)

    row_actions.append(current_row_act)
    column_actions.append(current_row_act)
    lam.append(next_value)
    x = [(current_value-next_value)/len(current_row_act)] # x is for column actions, y for row actions
    y = [1/len(current_row_act)]
    dual_comp = [0]* len(compact_form_stra)
    compact_form = compact_form_stra.copy()
    print(compact_form_stra)
    for poi in current_row_act:
        compact_form_stra[poi] -= x[0]
        dual_comp[poi] += y[0]
    current_action_len = len(current_row_act)
    while (len(lam) == 0 or lam[-1] >= 0.1):
        ##### Randomize 9 out of 10 iterations ##########
        if len(row_actions)%10 == 0:
            dic, eq_action = multi_dynamic_programming(POI, compact_form_stra, possible_actions, row_actions)
            current_row_act, current_value, next_row_act, next_value = second_largest_act_value(dic, eq_action, row_actions)
            current_action_len = len(current_row_act)
        else:
            random_choose_list = list(range(len(compact_form_stra)))
            chosen_action = random.choices(random_choose_list, k=current_action_len)
            temp_cate = np.zeros(len(POI[0]))
            for p in chosen_action:
                temp_cate = np.add(temp_cate, np.array(POI[p]))
            exist = False
            for act2 in row_actions:
                if len(act2) == current_action_len and overlap_POI([act2], [chosen_action]) == [[current_action_len]]:
                    exist = True
                    break
            feasibility = False
            if temp_cate in possible_actions:
                feasibility = True
            while exist or (not feasibility):
                print(chosen_action)
                random_choose_list = list(range(len(compact_form_stra)))
                chosen_action = random.choices(random_choose_list, k=current_action_len)
                temp_cate = np.zeros(len(POI[0]))
                for p in chosen_action:
                    temp_cate = np.add(temp_cate, np.array(POI[p]))
                exist = False
                for act2 in row_actions:
                    if len(act2) == current_action_len and overlap_POI([act2], [chosen_action]) == [[current_action_len]]:
                        exist = True
                        break
                feasibility = False
                if temp_cate in possible_actions:
                    feasibility = True
            current_row_act = chosen_action
                
        #print(dic)       
        print("======================================")
        print(current_row_act, next_row_act, row_actions)
        #print(current_column_act, column_actions)
        print("======================================")
        
        in_row_actions = False
        for rowAct in row_actions:
            if overlap_POI([rowAct], [current_row_act]) == [[len(rowAct)]]:
                
                in_row_actions = True
                break
        '''
        in_row_actions = False
        if overlap_POI([current_row_act], [row_actions[-1]]) == [[len(row_actions[-1])]]:
            in_row_actions = True
        '''
        #for columnAct in column_actions:
        #    if overlap_POI([columnAct], [current_column_act]) == [[len(columnAct)]]:
        #        in_row_actions = True
        #        break
        

        ###################
        '''
        notin_next = []
        notin_current = []
        for a1 in current_row_act:
            if a1 not in next_row_act:
                notin_next.append(compact_form_stra[a1])
        for a2 in next_row_act:
            if a2 not in current_row_act:
                notin_current.append(compact_form_stra[a2])
        x.append(sum(notin_next)-sum(notin_current))
        row_actions.append(current_row_act)
        print(compact_form_stra)
        current_lam = next_value
        '''
        ###################
        
        #if in_row_actions == True:
        #    print("-------------------------")
        '''
        current_lam = next_value
        A = overlap_POI(row_actions, row_actions)
        A = np.array(A)
        if len(row_actions)>1:
            temp_B = [lam[-1]]*(len(row_actions)-1)
            temp_B.append(current_value)
        else:
            temp_B = [current_value]
        B = np.array(list_a_minus_b(temp_B, [current_lam]*len(row_actions)))
        #X = np.linalg.solve(A, B)
        #delta_x = list(X)
        sol = optimize.linprog(np.array([1]*len(row_actions)), A_ub = np.multiply(A, -1), b_ub = np.multiply(B, -1))
        print(sol)
        delta_x = sol.get('x')
        '''
         #   delta_lambda, compact_form_stra, delta_x = next_lambda(compact_form_stra, current_row_act, next_row_act, POI, possible_actions, row_actions)
         #   current_lam = current_value - delta_lambda
        if in_row_actions == False:
            print("++++++++++++++++++++++++++")
            row_actions.append(current_row_act)
            #column_actions.append(current_column_act)
        temp_v = []
        for acti in row_actions:
            te = 0
            for proj in range(len(compact_form)):
                if proj in acti:
                    te += compact_form[proj]
            temp_v.append(te)
        current_lam = next_value
        x.append(0)
        A = overlap_POI(row_actions, row_actions)
        A = np.array(A)
        A_dual = np.transpose(A)
            
        temp_B = [lam[-1]]*(len(row_actions)-1)
        temp_B.append(current_value)
            #B = np.array(list_a_minus_b(temp_B, [current_lam]*len(row_actions))) # Positive
        B = np.array(list_a_minus_b(temp_v, [current_lam]*len(row_actions)))
        dual_C = np.add(B, np.matmul(A, x))
        sol = optimize.linprog(np.array([1]*len(row_actions)), A_ub = np.multiply(A, -1), b_ub = np.multiply(B, -1))
        sol_dual = optimize.linprog(np.multiply(dual_C, -1), A_ub = A, b_ub = np.array([1]*len(row_actions)))
        print(sol)
        print("---------------------------------")
        print(sol_dual)
        print("---------------------------------")
        #B_dual = np.array([1]*len(row_actions))
        #Y = np.linalg.solve(A_dual, B_dual)
        #X = np.linalg.solve(A, B)
        delta_x = sol.get('x')
        y = sol_dual.get('x')
        print("Max value for dual program:, ", dual_value_check(y, row_actions, possible_actions, POI))
        '''
        temp_row_actions = row_actions.copy()
        row_actions = []
        temp_x = x.copy()
        x = []
        for k in range(len(temp_x)):
            if temp_x[k] > 0:
                row_actions.append(temp_row_actions[k])
                x.append(temp_x[k])
        '''
        for action in range(len(row_actions)):
            for poi in row_actions[action]:
                compact_form_stra[poi] = compact_form[poi] - delta_x[action]

        #for r in range(len(x)):
        #    x[r] += delta_x[r]

        '''
        dual_comp = [0]*len(compact_form)
        for action in range(len(row_actions)):
            for poi in row_actions[action]:
                dual_comp[poi]  += y[action]
        '''
        lam.append(current_lam)
        print(compact_form_stra)
        print(x)
        print(lam)
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

def dual_value_check(y, row_actions, possible_actions, POI):
    compact_dual = [0] * len(POI)
    for a in range(len(row_actions)):
        for p in row_actions[a]:
            compact_dual[p] += y[a]
    dic, eq_action = multi_dynamic_programming(POI, compact_dual, possible_actions, row_actions)
    max_action, max_value, next_action, second_max_value = second_largest_act_value(dic, eq_action, row_actions)
    return max_action, max_value


def convert_matrix(agents_time_matrix, action): #action here is in the form of task category
    task_agent_time_matrix = np.zeros((sum(action), len(agents_time_matrix)), dtype = int)
    i = 0
    for cate in range(len(action)):
        for j in range(i, i+action[cate]):
            for agent in range(len(agents_time_matrix)):
                task_agent_time_matrix[j][agent] = agents_time_matrix[agent][cate]
        i += action[cate]
    return task_agent_time_matrix.tolist()

def all_normal_form_actions(POI_num): # Enumerate all normal form actions. Each normal form action is represented by a list containing all the program this action is gonna defend
    programs = range(POI_num)
    return list(powerset(programs))

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


if __name__ == "__main__":
    task_cate_num = 3
    agent_num = 10
    POI_num = 20
    GAP_exp_time = []
    homo_exp_time = []
    total_time = []
    bb_exp_time = []
    task_per_POI = 20
    possible_actions_num_bb = []
    possible_actions_num_heuristic = []
    for POI_num in range(20, 21, 5):
        POI, num_per_cate = generate_POI_0(task_cate_num, POI_num, task_per_POI)
        
        agents_limits = []
        agents_time_matrix = []
        for i in range(agent_num):
            agents_limits.append(int((5*task_per_POI*POI_num)/(2*agent_num)))
            t = []
            for j in range(task_cate_num):
                t.append(random.randint(1, 10))
            agents_time_matrix.append(t)
        
    
        print(num_per_cate)
        
        all_actions = pseudo_normal_actions(task_cate_num, num_per_cate)
        compact_form_stra = pseudo_compact_form_strategy(POI_num)
        possible_actions = []
        possible_actions_bb = []
        '''
        actions_num_cate = 1
        for num_ca in num_per_cate:
            actions_num_cate = actions_num_cate * num_ca

        possible_actions = []
        possible_actions_bb = []
        all_actions = []
        compact_form_stra = pseudo_compact_form_strategy(POI_num)
        if 2 ** POI_num > actions_num_cate:
            all_actions = pseudo_normal_actions(task_cate_num, num_per_cate)
        else:
            normal_actions_poi = all_normal_form_actions(POI_num)
            for ac in normal_actions_poi:
                act_cate = np.zeros(task_cate_num, dtype = int)
                for p in ac:
                    act_cate = np.add(act_cate, np.array(POI[p]))
                all_actions.append(list(act_cate))
        #print(convert_matrix(agents_time_matrix, all_actions[10]))
        '''
        
        t5 = time.time()
            
        
        for action in all_actions:
            print(action)
            task_agent_time_matrix = convert_matrix(agents_time_matrix, action)
            tem = normal_form_action(task_agent_time_matrix, sum(action), agent_num, agents_limits)
            if (tem.early_check_infeasibility()):
                continue
            is_feasible, assigned = tem.early_check_feasibility()
            if (is_feasible):
                possible_actions_bb.append(action)
            else:
                tem.fix_task_agent()
                if tem.solutions.sum() == sum(action):
                    possible_actions_bb.append(action)
        t6 = time.time()
        
        possible_actions_num_bb.append(len(possible_actions_bb))
        
        '''
        t3 = time.time()
        for action in all_actions:
            #print("action 2: ", action)
            if GAP(agents_limits, agents_time_matrix, list(action)) == True:
                possible_actions.append(action)
        t4 = time.time()
        possible_actions_num_heuristic.append(len(possible_actions))
        print("num_per_cate", num_per_cate)
        '''
        #possible_actions = edge_actions([3, 3, 5], num_per_cate, all_actions)
        #t1 = time.time()
        #valid_actions, x, lam = homotopy(POI, compact_form_stra, possible_actions)
        #t2 = time.time()
        #homo_exp_time.append(t2-t1)
        #GAP_exp_time.append(t4-t3)
        #bb_exp_time.append(t6-t5)
        #total_time.append(t2-t1+t4-t3)
        #print("GAP time", t4-t3, "homotopy time", t2-t1)
        #print("possible_actions: ", len(possible_actions_bb))
        '''
        print("valid_actions:", valid_actions)
        print("solution:", x)
        print("number of tasks per category: ", num_per_cate)
        print("lam:", lam)
        '''
    '''
    print("\n\nGAP time list: ", GAP_exp_time)
    #print("branch and bound time list: ", bb_exp_time)
    print("Homotopy time list: ", homo_exp_time)
    #print("Total time: ", total_time)
    '''
    #print("\n\nPossible actions number branch and bound: ", possible_actions_num_bb)
    #print("Possible actions number heuristic: ", possible_actions_num_heuristic)
    print("brute force time for 20 targets, 20 tasks/target, 10 agents:", t6-t5)
