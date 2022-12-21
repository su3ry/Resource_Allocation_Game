import random
import cvxpy as cp
import numpy as np

MAXIMUM = 10000


class normal_form_action:

    def __init__(self, task_agent_time_matrix, task_num, agent_num, agent_bound):
        self.task_agent_matrix = task_agent_time_matrix
        self.task_num = task_num
        self.agent_num = agent_num
        self.agent_bound = agent_bound
        self.diff = [0]*task_num
        self.solutions = np.zeros((task_num, agent_num))
        for i in range(task_num):
            self.diff[i] = second_smallest(
                task_agent_time_matrix[i]) - min(task_agent_time_matrix[i])

    def early_check_feasibility(self):
        b = self.agent_bound.copy()
        assigned_task = 0
        assigned_matrix = np.zeros((self.task_num, self.agent_num))
        for task in range(len(self.task_agent_matrix)):
            temp = []
            for agent in range(self.agent_num):
                temp.append((self.task_agent_matrix[task][agent], agent))
            st = sorted(temp)
            for p in st:
                if p[0] <= b[p[1]]:
                    assigned_task += 1
                    b[p[1]] -= p[0]
                    assigned_matrix[task][p[1]] = 1
                    break
        if assigned_task == len(self.task_agent_matrix):
            return True, assigned_matrix # There exists a solution
        return False, assigned_matrix # The solution cannot be found just by greedy assignment

    def early_check_infeasibility(self):
        b = sum(self.agent_bound)
        minimum_condition = []
        for i in range(len(self.task_agent_matrix)):
            minimum_condition.append(min(self.task_agent_matrix[i]))
        if sum(minimum_condition) > b:
            return True # Solution definitely doesn't exist
        return False # Pass the infeasiblity check


    def first_assignment(self):
        util = sum(self.agent_bound)
        first_assign_matrix = np.zeros((self.task_num, self.agent_num))
        for i in range(self.task_num):
            temp = min(self.task_agent_matrix[i])
            j = self.task_agent_matrix[i].index(temp)
            first_assign_matrix[i][j] = 1
            util -= self.task_agent_matrix[i][j]
        return util, first_assign_matrix

    def second_assignment(self):
        util, first_assign_matrix = self.first_assignment()
        dic = {}
        for i in range(self.agent_num):
            dic[i] = []
            for j in range(self.task_num):
                if first_assign_matrix[j][i] == 1:
                    dic[i].append(j)

        contains_zero = False
        fixed_agent = -1
        fixed_task = -1
        for i in range(self.agent_num):
            # print(self.agent_num)
            if dic[i] != []:
                w = []
                value = []
                b_i = -self.agent_bound[i]
                for j in dic[i]:
                    w.append(self.task_agent_matrix[j][i])
                    value.append(self.diff[j])
                    b_i += self.task_agent_matrix[j][i]
                result, utility = self.solver(w, value, b_i)
                util -= utility
                if 0 in result:
                    contains_zero = True
                if contains_zero == True:
                    r = list(result)
                    fixed_task = dic[i][r.index(0)]
                    fixed_agent = i
                    break
        # print("----------------------------------------")
        # Needed to decide termination
        while contains_zero == False and self.task_agent_matrix != [] and sum(self.task_agent_matrix[0]) < MAXIMUM * self.agent_num:
            for i in dic:
                for j in dic[i]:
                    self.task_agent_matrix[j][i] = MAXIMUM
            # print(contains_zero)
            contains_zero = self.second_assignment()[-1]
            # print("--------------")
        # print(result)
        return dic, fixed_task, fixed_agent, util, contains_zero

    def solver(self, weight, value, b_i):
        num = len(weight)
        result = [1] * num
        dyn_w = sum(weight)
        dyn_v = sum(value)
        dic = {}
        dic[dyn_w] = (dyn_v, result)
        if b_i <= 0:
            return [0]*num, 0
        else:
            for i in range(dyn_w-1, b_i-1, -1):
                w = []
                r = []
                for j in range(num):
                    if i+weight[j] > dyn_w:
                        w.append(dyn_v)
                        r.append(result)
                    else:
                        if dic[i+weight[j]][-1][j] == 1:
                            w.append(dic[i+weight[j]][0]-value[j])
                            temp = []
                            for k in range(len(result)):
                                if k != j:
                                    temp.append(dic[i+weight[j]][-1][k])
                                else:
                                    temp.append(0)
                            r.append(temp)
                        else:
                            w.append(dic[i+weight[j]][0])
                            r.append(dic[i+weight[j]][-1])
                temp0 = min(w)
                ind = w.index(temp0)
                dic[i] = (w[ind], r[ind])
        return dic[b_i][-1], dic[b_i][0]

        '''print("weight, value, b", weight, value, b_i)
		selection = cp.Variable(len(weight), boolean=True)
		weights = np.array(weight)
		values = np.array(value)
		constraints = weights @ selection >= b_i
		utility = values @ selection
		objective = cp.Minimize(utility)
		prob=cp.Problem(objective, [constraints])
		result = prob.solve(solver = "GLPK_MI")
		v = selection.value
		cost = utility.value
		return v, cost'''

    def fix_task_agent(self):

        for i in range(self.task_num):
            #######################################
            dic, fixed_task, fixed_agent, util, c = self.second_assignment()
            if fixed_agent == -1:
                break
            tm1, b1 = self.modify_matrix(1, fixed_agent, fixed_task)
            tm0, b0 = self.modify_matrix(0, fixed_agent, fixed_task)

            tr1 = normal_form_action(tm1, self.task_num-1, self.agent_num, b1)
            dic1, ft1, fa1, u1, c1 = tr1.second_assignment()

            tr0 = normal_form_action(tm0, self.task_num, self.agent_num, b0)
            dic0, ft0, fa0, u0, c0 = tr0.second_assignment()
            # if fa0 == -1 or fa1 == 0:
            #	break
            #print("self.solutions:", self.solutions)
            #print("u0, u1", u0, u1)
            if u1 >= u0:
                which_task = -1
                for i in range(len(self.solutions)):
                    if sum(self.solutions[i]) == 0:
                        which_task += 1
                    if which_task == fixed_task:

                        self.solutions[i][fixed_agent] = 1
                        self.task_agent_matrix = tm1
                        self.task_num -= 1
                        self.agent_bound = b1
                        fixed_task = ft1
                        fixed_agent = fa1
                        break
            else:
                self.task_agent_matrix = tm0
                self.agent_bound = b0
                fixed_task = ft0
                fixed_agent = fa0

            #print("fixed value 1:", dic1, ft1, fa1, u1)
            #print("fixed value 0:", dic0, ft0, fa0, u0)
            #print("ffd", tm, b)

    def modify_matrix(self, fixed_value, fixed_agent, fixed_task):
        tm = []
        b = []
        if fixed_value == 0:
            for i in range(self.task_num):
                temp = []
                for j in range(self.agent_num):
                    if j == fixed_agent and i == fixed_task:
                        temp.append(MAXIMUM)
                    else:
                        temp.append(self.task_agent_matrix[i][j])
                tm.append(temp)
            for i in range(self.agent_num):
                b.append(self.agent_bound[i])
        elif fixed_value == 1:
            for i in range(self.task_num):
                temp = []
                if i != fixed_task:
                    for j in range(self.agent_num):
                        temp.append(self.task_agent_matrix[i][j])
                    tm.append(temp)
            for i in range(self.agent_num):
                if i == fixed_agent:
                    b.append(
                        self.agent_bound[i]-self.task_agent_matrix[fixed_task][i])
                else:
                    b.append(self.agent_bound[i])
        return tm, b


def second_smallest(time_list):
    length = len(time_list)
    temp = []
    for i in range(len(time_list)):
        temp.append(time_list[i])
    temp.sort()
    return temp[1]


'''	def GAP():
		while len(self.assigned_tasks) != task_num:
'''

if __name__ == "__main__":
    agent = 120
    task = 200
    matrix = []
    for i in range(task):
        m = []
        for j in range(agent):
            m.append(random.randint(3, 6))
        matrix.append(m)
    bound = [8]*agent
    print(np.array(matrix))
    a = normal_form_action(matrix, task, agent, bound)
    # print(a.first_assignment())
    print(a.diff)
    #util, first_assign_matrix = a.first_assignment()
    # print(a.second_assignment())
    a.fix_task_agent()
    print(a.solutions)
    print(sum(sum(a.solutions)))
