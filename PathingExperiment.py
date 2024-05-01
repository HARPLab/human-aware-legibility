import sys
import os
import autograd.numpy as np
import decimal
import copy
from pathlib import Path
from datetime import timedelta, datetime

import utility_environ_descrip as resto

# from LegiblePathQRCost import LegiblePathQRCost
from DirectPathQRCost import DirectPathQRCost
from ObstaclePathQRCost import ObstaclePathQRCost
from LegibilityOGPathQRCost import LegibilityOGPathQRCost
from OALegiblePathQRCost import OALegiblePathQRCost
from OAObsPathQRCost import OAObsPathQRCost
from SocLegPathQRCost import SocLegPathQRCost
from UnderstandingPathQRCost import UnderstandingPathQRCost
from RelevantPathQRCost import RelevantPathQRCost

# from LegiblePathCost import LegiblePathCost
# from SocLegPathCost import SocLegPathCost

import utility_legibility as legib


PREFIX_EXPORT = 'experiment_outputs/'

# OPTIONS OF F FUNCTION
F_NONE              = 'f_none'
F_OG_LINEAR         = 'f_og_linear'
F_VIS_LIN           = 'f_vis_lin'
F_VIS_BIN           = 'f_vis_bin'

# OPTIONS FOR COST/SOLVER TYPE
COST_DIRECT         = 'cost_direct'
COST_LEGIB          = 'cost_legible'
COST_OA             = 'cost_oalegib'
COST_OBS            = 'cost_obstacles'
COST_OA_AND_OBS     = 'cost_oa_and_obs'
COST_UNDERSTANDING  = 'cost_understanding'

# # SOLVER TYPE
# SOLVER_LEGIB        = 'Solver=Legible'
# SOLVER_OBSTACLES    = 'Solver=Obstacles'
# SOLVER_OA           = 'Solver=OALegible'
# SOLVER_DIRECT       = 'Solver=Direct'

dupline = np.asarray([[ 0.,          0.        ],
    [-0.49906399, 0.26825858],
    [-0.98855055, 0.56214198],
    [-1.45796738, 0.8661427 ],
    [-1.89646097, 1.16553758],
    [-2.29334226, 1.4466766 ],
    [-2.63857395, 1.69727316],
    [-2.92320919, 1.90668009],
    [-3.13977168, 2.06613778],
    [-3.28256842, 2.16898242],
    [-3.34792763, 2.21080467],
    [-3.33435616, 2.18955116],
    [-3.2426128,  2.10556408],
    [-3.07569609, 1.96155635],
    [-2.83874743, 1.76252279],
    [-2.53887275, 1.51558997],
    [-2.18488811, 1.22981025],
    [-1.78699649, 0.91590764],
    [-1.35640491, 0.58598542],
    [-0.90489235, 0.25320699],
    [-0.44433972,  -0.06853642],
    [ 0.013766,    -0.36495693],
    [ 0.45884194,  -0.62168443],
    [ 0.88172235,  -0.82454814],
    [ 1.27516455,  -0.95980898],
    [ 1.63433941,  -1.01432608],
    [ 1.95730056,  -0.97564123]])

dupline1 = [[0.,         0.        ], [0.07591171, 0.49716833], [0.15182342, 0.76363696], [0.22773513, 0.94001898], [0.30364684, 1.06678282], [0.37955854, 1.16078189], [0.45547025, 1.23126017], [0.53138196, 1.28422314], [0.60729367, 1.32392959], [0.68320538, 1.35351389], [0.75911709, 1.37530548], [0.8350288,  1.39102491], [0.91094051, 1.40191794], [0.98685221, 1.40885061], [1.06276392, 1.41237535], [1.13867563, 1.41277341], [1.21458734, 1.41007656], [1.29049905, 1.40406958], [1.36641076, 1.39427363], [1.44232247, 1.37990938], [1.51823418, 1.35983749], [1.59414588, 1.33247177], [1.67005759, 1.29565669], [1.7459693,  1.24649101], [1.82188101, 1.1810503 ], [1.89779272, 1.09387312], [1.97370443, 0.97679227]]
dupline2 = [[0.,         0.        ], [0.07591171, -0.49716833], [0.15182342, -0.76363696], [0.22773513, -0.94001898], [0.30364684, -1.06678282], [0.37955854, -1.16078189], [0.45547025, -1.23126017], [0.53138196, -1.28422314], [0.60729367, -1.32392959], [0.68320538, -1.35351389], [0.75911709, -1.37530548], [0.8350288,  -1.39102491], [0.91094051, -1.40191794], [0.98685221, -1.40885061], [1.06276392, -1.41237535], [1.13867563, -1.41277341], [1.21458734, -1.41007656], [1.29049905, -1.40406958], [1.36641076, -1.39427363], [1.44232247, -1.37990938], [1.51823418, -1.35983749], [1.59414588, -1.33247177], [1.67005759, -1.29565669], [1.7459693,  -1.24649101], [1.82188101, -1.1810503 ], [1.89779272, -1.09387312], [1.97370443, -0.97679227]]


class PathingExperiment():
    label = "needslabel"
    ax = None

    # default values for solver
    solver_scale_term       = 100000.0 #.01
    solver_scale_stage      = 1.0
    solver_scale_obstacle   = 1.0

    lambda_cost_path_coeff  = 1.0

    TABLE_RADIUS            = .25
    OBSERVER_RADIUS         = .1
    GOAL_RADIUS             = .2
    OBSTACLE_BUFFER         = .1 #05


    # DEFAULT COST TYPE AND F TYPE
    cost_label  = COST_UNDERSTANDING
    f_label     = F_VIS_BIN

    state_size  = 4
    action_size = 2

    dt  = 1.0  #.5 #.025
    N   = int(21 * 2)
    Qf  = np.identity(state_size) # * 400.0
    Q   = 1.0 * np.eye(state_size)
    R   = np.eye(action_size) # * 100

    oa_on                   = True
    # heading_on              = True
    norm_on                 = False
    weighted_close_on       = False
    # mode_pure_heading       = False
    # mode_heading_err_sqr    = False

    mode_type_dist          = 'lin' # 'exp', 'sqr', 'lin'
    mode_type_heading       = 'lin'
    mode_type_blend         = 'mixed' #'min' #'mixed'

    mode_und_target         = None
    mode_und_secondary      = None

    local_distance          = -1

    observer_goal_pairs         = []
    has_observer_goal_pairs     = False

    J_hist = []
    best_xs = None
    best_us = None
    solve_status = None

    fn_note = ""
    run_filters = []
    mode_dist_legib_on = True

    ti = 0

    new_N = 1

    def __init__(self, label, restaurant, f_label=None, cost_label=None):
        self.exp_label  = label
        self.restaurant = restaurant

        self.start          = restaurant.get_start()
        self.target_goal    = restaurant.get_target_goal()
        self.goals          = restaurant.get_goals()

        self.observers      = restaurant.get_observers()
        self.tables         = restaurant.get_tables()

        if f_label is not None:
            self.f_label    = f_label
    
        if cost_label is not None:
            self.cost_label = cost_label

        self.setup_file_id()


    def __init__(self, label, start, target_goal, all_goals, restaurant=None, observers=[], table_pts=[], f_label=None, cost_label=None):
        self.exp_label = label

        self.start          = start
        self.target_goal    = target_goal
        self.goals          = all_goals

        if restaurant is None:
            restaurant = resto.Restaurant(resto.TYPE_CUSTOM, table_pts=table_pts, goals=all_goals, start=start, obs_pts=observers, dim=None)
        self.restaurant = restaurant

        if f_label is not None:
            self.f_label    = f_label
    
        if cost_label is not None:
            self.cost_label = cost_label

        self.observers      = restaurant.get_observers()
        self.table_pts = table_pts

        self.setup_file_id()

        max_dist = 0
        for goal in all_goals:
            dist = np.linalg.norm(np.asarray(start) - np.asarray(goal))
            if dist > max_dist:
                max_dist = dist

        self.max_dist = max_dist


    def on_iteration_exp(self, iteration_count, xs, us, J_opt, accepted, converged):
        self.J_hist.append(J_opt)
        info = "converged" if converged else ("accepted" if accepted else "failed")

        final_state = xs[-1]
        print("iteration", iteration_count, info, J_opt, final_state)
        
        if converged:
            self.best_xs = xs
            self.best_us = us

        most_recent_is_complete_packet = [converged, info, iteration_count, J_opt]

        print("TESTED PATH IS " + str(xs))
        self.solve_status = most_recent_is_complete_packet   

    def setup_cost(self, state_size, action_size, x_goal_raw, N):
        self.state_size = state_size
        self.action_size = action_size

        Xrefline = np.tile(x_goal_raw, (N+1, 1))
        # print(Xrefline.shape, state_size)
        Xrefline = np.reshape(Xrefline, (-1, state_size))

        Urefline = self.setup_Urefline(x_goal_raw, action_size)

        solver_label = self.cost_label

        if solver_label is COST_LEGIB:
            return LegiblePathQRCost(self, Xrefline, Urefline), Urefline
        elif solver_label is COST_OBS:
            return ObstaclePathQRCost(self, Xrefline, Urefline), Urefline
        elif solver_label is COST_OA:
            return OALegiblePathQRCost(self, Xrefline, Urefline), Urefline
        elif solver_label is COST_DIRECT:
            return DirectPathQRCost(self, Xrefline, Urefline), Urefline
        elif solver_label is COST_OA_AND_OBS:
            exit()
            return SocLegPathQRCost(self, Xrefline, Urefline), Urefline
        elif solver_label is COST_UNDERSTANDING:
            return RelevantPathQRCost(self, Xrefline, Urefline), Urefline
            # return UnderstandingPathQRCost(self, Xrefline, Urefline), Urefline

        print("ERROR, NO KNOWN SOLVER, PLEASE ADD A VALID SOLVER TO EXP")
        print("''''''" + str(solver_label) + "''''''")


    def setup_Urefline(self, goal, action_size):
        start = self.get_start()
        N = self.get_N()

        # u_blank  = np.asarray([0.0, 0.0])
        # Urefline = np.tile(u_blank, (N, 1))
        # Urefline = np.reshape(Urefline, (-1, action_size))

        # return Urefline

        # u_blank  = np.asarray([0.0, 0.0])
        # Urefline = np.tile(u_blank, (N, 1))
        # Urefline = np.reshape(Urefline, (-1, action_size))

        # return Urefline

        step_vector = [0.0, 0.0]
        u_blank  = np.asarray(step_vector)
        Urefline = np.tile(u_blank, (N, 1))
        Urefline = np.reshape(Urefline, (-1, action_size))

        # crow_flies_vector = [goal[0] - start[0], goal[1] - start[1]]
        # step_vector = [1.0 * crow_flies_vector[0] / N, 1.0 * crow_flies_vector[1] / N]


        ##### LONG EDGES
        goal_a, goal_b, goal_c, goal_d, goal_e, goal_f = self.get_goal_squad()

        if self.is_segment("A", "B"):
            # step_vector = [step_vector[0], step_vector[1] - .05] # Add a bias
            Urefline = [[0.37291543, 0.35940304], [0.2868237,  0.25326541], [0.21553565, 0.16057495], [0.15556699, 0.07936027], [0.10324814, 0.00549762], [ 0.06533833, -0.05333846], [ 0.07372553, -0.07929104], [ 0.07484242, -0.08588459], [ 0.0752228, -0.0861347], [ 0.07525216, -0.08619458], [ 0.07525428, -0.0862076 ], [ 0.07525458, -0.08620947], [ 0.07525462, -0.08620974], [ 0.07525463, -0.08620978], [ 0.07525463, -0.08620979], [ 0.07525463, -0.08620979]]
            Urefline = np.reshape(Urefline, (-1, action_size))
            return Urefline
        ##
        if self.is_segment("A", "C"):
            # return int(resolution * 2)
            step_vector = [step_vector[0] - 0.01, step_vector[1]] #[-0.01, 0] # Add a bias


        if self.is_segment("A", "D"):
            # step_vector = [step_vector[0], step_vector[1] - .05] # Add a bias
            Urefline = [[-0.35986786, -0.38822228], [-0.25321005, -0.30196323], [-0.16111047, -0.23213754], [-0.08358655, -0.1771138 ], [-0.01675805, -0.13235135], [ 0.03910463, -0.09656519], [ 0.07053413, -0.07702418], [ 0.07821819, -0.07234532], [ 0.07941543, -0.07163051], [ 0.07958545, -0.07153108], [ 0.07960898, -0.07151744], [ 0.07961233, -0.07151571], [ 0.0796127,  -0.07151549], [ 0.07961271, -0.07151548], [ 0.07961272, -0.07151547], [ 0.07961272, -0.07151547]]
            Urefline = np.reshape(Urefline, (-1, action_size))
            return Urefline

        if self.is_segment("A", "E"):
            # step_vector = [step_vector[0], step_vector[1] - .05] # Add a bias
            return Urefline

        if self.is_segment("A", "F"):
            # step_vector = [step_vector[0], step_vector[1] - .05] # Add a bias
            return Urefline

        if self.is_segment("B", "A"):
            # step_vector = [step_vector[0], step_vector[1] - .05] # Add a bias
            return Urefline

        if self.is_segment("B", "C"):
            # step_vector = [step_vector[0], step_vector[1] - .05] # Add a bias
            return Urefline

        if self.is_segment("B", "D"):
            # step_vector = [step_vector[0], step_vector[1] - .05] # Add a bias
            Urefline = [[0.11320196, -0.19995151], [0.10495989, -0.1877859], [0.0982124, -0.17345419], [0.09286647, -0.15712068], [0.08885552, -0.13897364], [0.08614075, -0.11925201], [0.08471647, -0.09846178], [0.08456125, -0.0787806], [0.08512151, -0.06712796], [0.08541253, -0.06425035], [0.08548036, -0.06380169], [0.08549297, -0.06373803], [0.08549513, -0.06372913], [0.08549548, -0.06372789], [0.08549554, -0.06372772], [0.08549555, -0.0637277], [0.08549555, -0.06372769], [0.08549555, -0.06372769], [0.08549555, -0.06372769], [0.08549555, -0.06372769], [0.08549555, -0.06372769], [0.08549555, -0.06372769]] #horizontal_flip_u(BF_u)
            Urefline = np.reshape(Urefline, (-1, action_size))
            return Urefline

        if self.is_segment("B", "E"):
            # step_vector = [step_vector[0], step_vector[1] - .05] # Add a bias
            return Urefline

        if self.is_segment("B", "F"):
            # step_vector = [step_vector[0], step_vector[1] - .05] # Add a bias
            Urefline = [[ 0.11320196, -0.19995151], [ 0.10495989, -0.1877859 ], [ 0.0982124,  -0.17345419], [ 0.09286647, -0.15712068], [ 0.08885552, -0.13897364], [ 0.08614075, -0.11925201], [ 0.08471647, -0.09846178], [ 0.08456125, -0.0787806 ], [ 0.08512151, -0.06712796], [ 0.08541253, -0.06425035], [ 0.08548036, -0.06380169], [ 0.08549297, -0.06373803], [ 0.08549513, -0.06372913], [ 0.08549548, -0.06372789], [ 0.08549554, -0.06372772], [ 0.08549555, -0.0637277 ], [ 0.08549555, -0.06372769], [ 0.08549555, -0.06372769], [ 0.08549555, -0.06372769], [ 0.08549555, -0.06372769], [ 0.08549555, -0.06372769], [ 0.08549555, -0.06372769]]
            Urefline = np.reshape(Urefline, (-1, action_size))
            return Urefline

        return Urefline


    def setup_file_id(self):
        # Create a new folder for this experiment, along with sending debug output there
        g_index = self.get_g_index()

        file_name = self.get_pretty_study_label(g_index, self.get_target_goal())

        self.file_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + "-" + self.exp_label + "-" + file_name

        # try:
        #     os.mkdir(PREFIX_EXPORT + self.file_id)
        # except:
        #     print("FILE ALREADY EXISTS " + self.file_id)
        
        # sys.stdout = open(PREFIX_EXPORT + self.file_id + '/output.txt','wt')

    def reinit_file_id(self):
        self.setup_file_id()

    def get_g_index(self):
        # g_index = self.goals.index(self.target_goal)
        g_index = -1
        for gi in range(len(self.goals)):
            g = self.goals[gi]
            if self.target_goal[0] == g[0] and self.target_goal[1] == g[1]:
                return gi

        return g_index

    def set_f_label(self, label):
        self.f_label = label

    def get_f_label(self):
        return self.f_label

    def get_exp_label(self):
        return self.exp_label

    def set_exp_label(self, label):
        # goal_index = str(self.goals.index(self.target_goal))

        self.exp_label = label # + "-g" + goal_index

    def get_goal_label(self):
        return "g" + str(self.goals.index(self.target_goal))

    def set_cost_label(self, label):
        self.cost_label = label

    def get_cost_label(self):
        return self.cost_label

    def set_state_size(self, label):
        self.state_size = label

    def get_state_size(self):
        return self.state_size

    def set_action_size(self, label):
        self.action_size = label

    def get_action_size(self):
        return self.action_size

    def set_observer_goal_pairs(self, observers, goals):
        self.observers = observers 
        self.goal = goals

        observer_goal_pairs = []

        for i in range(len(observers)):
            observer_goal_pairs.append([observers[i], goals[i]])

        self.observer_goal_pairs = observer_goal_pairs

        self.has_observer_goal_pairs = True


    def set_observers(self, x):
        if type(x[0]) is resto.Observer:
            restaurant = resto.Restaurant(resto.TYPE_CUSTOM, table_pts=self.table_pts, goals=self.goals, start=self.start, observers=x, dim=None)
        else:
            restaurant = resto.Restaurant(resto.TYPE_CUSTOM, table_pts=self.table_pts, goals=self.goals, start=self.start, obs_pts=x, dim=None)
        
        self.observers      = restaurant.get_observers()
        self.restaurant = restaurant

    def get_observers(self):
        return self.observers

    def set_tables(self, x):
        self.tables = x

    def get_tables(self):
        return self.restaurant.get_tables()

    def get_restaurant(self):
        return self.restaurant

    def get_start(self):
        return self.start

    def get_goals(self):
        return self.goals

    def get_target_goal(self):
        return self.target_goal

    def get_observer_for_goal(self, target_goal):
        print("LA TARGET GOAL: " + str(target_goal))
        min_dist = np.inf

        for o in self.get_observers():
            try:
                o_center = o.get_center()
            except:
                o_center = o

            dist = self.dist_between(target_goal, o_center)
            if dist < min_dist:
                min_dist = dist
                target_obs = o

        return target_obs


    def get_target_observer(self):
        target_goal = self.get_target_goal()
        return self.get_observer_for_goal(target_goal)


    def get_secondary_goals(self):
        print("SECONDARIES")
        print(self.goals)
        sec = [x for x in self.goals if x != self.target_goal]
        print(sec)
        return sec

    def get_secondary_observers(self):
        return [x for x in self.observers if x != self.target_observer]


    def set_QR_weights(self, Q, R, Qf):
        self.Q  = Q
        self.R  = R
        self.Qf = Qf

    def get_Q(self):
        return np.eye(self.state_size)

    def get_R(self):
        return np.eye(self.action_size)

    def get_Qf(self):
        return np.eye(self.state_size)

    def set_N(self, n):
        self.N = n


    def get_goal_squad(self):
        goal_a  = [1.0, -1.0]
        goal_b  = [3.0, -1.0] 
        goal_c  = [5.0, -1.0]

        goal_d  = [1.0, -3.0]
        goal_e  = [3.0, -3.0]
        goal_f  = [5.0, -3.0]

        return goal_a, goal_b, goal_c, goal_d, goal_e, goal_f


    def get_goal_for_label(self, start_label):
        if start_label == 'A' or start_label == 'a':
            return self.get_goal_squad()[0]
        elif start_label == 'B' or start_label == 'b':
            return self.get_goal_squad()[1]
        elif start_label == 'C' or start_label == 'c':
            return self.get_goal_squad()[2]
        elif start_label == 'D' or start_label == 'd':
            return self.get_goal_squad()[3]
        elif start_label == 'E' or start_label == 'e':
            return self.get_goal_squad()[4]
        elif start_label == 'F' or start_label == 'f':
            return self.get_goal_squad()[5]


    def is_segment(self, start_label, end_label):
        start       = self.get_goal_for_label(start_label)
        end         = self.get_goal_for_label(end_label)

        start_goal  = self.get_start()
        tar_goal    = self.get_target_goal()

        if self.dist_between(tar_goal, start_goal) == self.dist_between(start, end):

            if start == start_goal and end == tar_goal:
                return True

        return False




    def get_N(self):

        tar_goal    = self.get_target_goal()
        start_goal  = self.get_start()

        resolution = 16

        goal_a, goal_b, goal_c, goal_d, goal_e, goal_f = self.get_goal_squad()

        ##### SHORT EDGES
        if self.dist_between(tar_goal, start_goal) == self.dist_between(goal_a, goal_b):
            return int(resolution)

        ##### LONG EDGES
        if self.dist_between(tar_goal, start_goal) == self.dist_between(goal_a, goal_c):
            return int(resolution * 2)

        ##### SHORT DIAG
        if self.dist_between(tar_goal, start_goal) == self.dist_between(goal_a, goal_e):
            return int(resolution * np.sqrt(2))

        ##### LONG DIAG
        if self.dist_between(tar_goal, start_goal) == self.dist_between(goal_a, goal_f):
            return int(resolution * np.sqrt(3))


        # else
        return self.N #* 10



    def set_dt(self, dt):
        self.dt = dt

    def get_dt(self):
        return self.dt

    def get_file_id(self):
        return self.file_id

    def set_solver_scale_obstacle(self, scale):
        self.solver_scale_obstacle = scale

    def get_solver_scale_obstacle(self):
        return self.solver_scale_obstacle

    def set_solver_scale_stage(self, scale):
        self.solver_scale_stage = scale

    def get_solver_scale_stage(self):
        return self.solver_scale_stage

    def set_solver_scale_term(self, scale):
        self.solver_scale_term = scale

    def get_solver_scale_term(self):
        return self.solver_scale_term * (1.0 / self.get_dt())

    def set_solver_coeff_terminal(self, scale):
        self.solver_coeff_terminal = scale

    def get_solver_coeff_terminal(self):
        return self.solver_coeff_terminal

    def get_experiment_dict(self):
        return {}

    def get_table_radius(self):
        return self.TABLE_RADIUS

    def set_table_radius(self, trad):
        self.obstacle_table_radius = trad

    def get_observer_offset(self):
        return 1.0

    def get_observer_radius(self):
        return self.get_local_distance() / 2.0 # + self.get_observer_offset()
        # return self.OBSERVER_RADIUS

    def get_goal_radius(self):
        return 0 #self.get_local_distance()

    def get_obstacle_buffer(self):
        return self.get_observer_offset()

    def get_lambda_cost_path_coeff(self):
        return self.lambda_cost_path_coeff

    def set_lambda_cost_path_coeff(self, l):
        self.lambda_cost_path_coeff = l

    def set_heading_on(self, v):
        self.heading_on = v

    def get_is_heading_on(self):
        return self.heading_on

    def set_oa_on(self, v):
        self.oa_on = v

    def get_is_oa_on(self):
        return self.oa_on

    def set_ax(self, v):
        self.ax = v

    def get_ax(self):
        return self.ax

    def set_norm_on(self, v):
        self.norm_on = v

    def get_norm_on(self):
        return self.norm_on

    def set_mode_dist_legib_on(self, v):
        self.mode_dist_legib_on = v

    def get_mode_dist_legib_on(self):
        return self.mode_dist_legib_on

    def set_weighted_close_on(self, v):
        self.weighted_close_on = v

    def get_weighted_close_on(self):
        return self.weighted_close_on

    def set_mode_dist_type(self, v):
        self.mode_type_dist

    def get_mode_dist_type(self):
        return self.mode_type_dist

    def set_fn_note(self, label):
        self.fn_note = "-" + label

    def get_fn_note(self):
        return self.fn_note

    def set_run_filters(self, v):
        self.run_filters = v

    def get_run_filters(self):
        return self.run_filters

    def get_solver_status(self):
        return self.solve_status

    def get_solver_status_blurb(self, info_packet=None):
        if info_packet is None:
            info_packet = self.get_solver_status()

        # info_packet = [converged, info, iteration_count, J_opt]
        if info_packet is not None:
            converged_text = ""
            converged, info, iteration_count, J_opt = info_packet
            if converged is True:
                converged_text = "CONVERGED after " + str(iteration_count)
            elif info == 'accepted':
                converged_text = "ACCEPTED in " + str(iteration_count)
            else:
                converged_text = "INCOMPLETE after " + str(iteration_count)

            blurb = converged_text
        else:
            blurb = ""

        local_def       = self.get_local_distance()
        keepout_dist    = self.get_goal_keepout_distance()
        lam             = self.get_lambda()

        blurb += "\nlc_" + str(local_def) + "_s" + str(keepout_dist) + "_lm" + str(lam)

        blurb += "\nJ=" + str(J_opt)

        return blurb        

    def get_fn_notes_ada(self):
        local_def       = self.get_local_distance()
        keepout_dist    = self.get_goal_keepout_distance()
        lam             = self.get_lambda()

        return self.get_pretty_study_label(self.get_target_goal(), self.get_start()) + "_lc_" + str(local_def) + "_s" + str(keepout_dist) + "_lm" + str(lam)

    def set_target_goal_index(self, ti):
        self.target_goal        = self.goals[ti]
        self.target_observer    = self.observers[ti]
        self.ti = ti

    def set_test_options(self, test_setup):
        # 'heading-mode':'none', 'dist-mode':'lin', 'blend-type': 'none'
        KEY_HEADING_MODE = 'mode_heading'
        if KEY_HEADING_MODE in test_setup.keys():
            self.mode_type_heading = test_setup[KEY_HEADING_MODE]

        KEY_DIST_MODE = 'mode_dist'
        if KEY_DIST_MODE in test_setup.keys():
            self.mode_type_dist = test_setup[KEY_DIST_MODE]

        KEY_TYPE_BLEND = 'mode_blend'
        if KEY_TYPE_BLEND in test_setup.keys():
            self.mode_type_blend = test_setup[KEY_TYPE_BLEND]

        KEY_UND_TARGET = 'und_target'
        if KEY_UND_TARGET in test_setup.keys():
            self.mode_und_target = test_setup[KEY_UND_TARGET]

        KEY_UND_SECONDARY = 'und_secondary' 
        if KEY_UND_SECONDARY in test_setup.keys():
            self.mode_und_secondary = test_setup[KEY_UND_SECONDARY]

    def get_mode_type_heading(self):
        return self.mode_type_heading

    def get_mode_type_dist(self):
        return self.mode_type_dist

    def get_mode_type_blend(self):
        return self.mode_type_blend

    def get_mode_und_target(self):
        return self.mode_und_target

    def get_mode_und_secondary(self):
        return self.mode_und_secondary

    def angle_between_points(self, p1, p2):
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        angle = np.arctan2(y2 - y1, x2 - x1)

        # angle = np.arctan2(x2 - x1, y2 - y1)

        # ang1 = np.arctan2(*p1[::-1])
        # ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg(angle)


    def get_FOV(self):
        return 180

    # ADA MASTER VISIBILITY EQUATION
    # ILQR OBSERVER-AWARE EQUATION
    # RETURN HERE
    def get_visibility_of_pt_w_observer_ilqr(self, pt, observer, normalized=True, epsilon=.01, angle_fov=180, RETURN_ANGLE=False, also_evil=False):
        observers   = []
        score       = []

        obs_pt = copy.copy(observer.get_center())

        MAX_DISTANCE    =  np.inf
        obs_orient      = observer.get_orientation() 
        obs_FOV         = self.get_FOV()

        angle       = self.angle_between_points(obs_pt, pt)
        x1, x2      = pt, obs_pt
        x1          = (x1[0], x1[1])
        x2          = (x2[0], x2[1])

        distance    = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

        in_view     = False

        a                   = angle - obs_orient
        signed_angle_diff   = (a + 180) % 360 - 180
        angle_diff          = abs(signed_angle_diff)

        # print("Observer test")
        # print(observer.get_center(), observer.get_orientation())

        # print(str(pt) + " -> " + str(observer.get_center()) + " = angle " + str(angle))
        # print("observer looking at... " + str(obs_orient))
        # print("angle diff = " + str(angle_diff))

        half_fov = (obs_FOV / 2.0)
        if np.abs(angle_diff) < np.abs(half_fov):
            from_center = half_fov - angle_diff
            if normalized:
                from_center = from_center / (half_fov)

            in_view = True

        # print("Is in view? " + str(in_view))

        if RETURN_ANGLE:
            return in_view, angle_diff

        return in_view

    def get_vislocal_status_of_point(self, x_input):
        observers       = self.get_observers()
        target_obs      = self.get_target_observer()
        secondary_obs   = self.get_secondary_observers()
        goals           = self.get_goals()

        status_dict = {}

        x = x_input #np.asarray([x_input[1], x_input[0]])

        for g in goals:
            o                   = self.get_observer_for_goal(g)
            vis, vis_angle      = self.get_visibility_of_pt_w_observer_ilqr(x, o, normalized=True, RETURN_ANGLE=True)
            local_dist          = np.abs(self.dist_between(x, o.get_center()))
            is_local            = local_dist <= self.get_local_distance()

            status_dict[(g[0], g[1])] = (vis, is_local, vis_angle, local_dist)

            # print()
            # print("PE LOOKUP VISLOCAL")
            # # print(i, value1, j, value2)
            # print(x)
            # print("obs == " + str(o.get_center()) + " goal == " + str(g))
            # print(vis, is_local, local_dist, " < " + str(self.get_local_distance()))

            # print("CENTER IS ")
            # print(o.get_center())

        return status_dict

    # RETURN TRUE IF IN SIGHT, FALSE IF NO
    # TARGET, then list of SECONDARY
    def get_visibility_of_all(self, x_in, dict_on=False):
        is_vis_target = 0
        is_vis_secondary = []

        x = x_in[:2]

        observers       = self.get_observers()
        target_obs      = self.get_target_observer()
        secondary_obs   = self.get_secondary_observers()


        print("VIS TARGETS AND SEC")
        print(target_obs.get_center())
        print([so.get_center() for so in secondary_obs])

        # what to do if there are no observers
        if len(observers) == 0:
            return True, [True]

        # if self.exp.get_is_oa_on() is True:
        is_vis_observers = []
    
        vis_target, target_ang  = self.get_visibility_of_pt_w_observer_ilqr(x, target_obs, normalized=True, RETURN_ANGLE=True)

        for o in self.get_secondary_observers():
            vis_sec, secondary_ang  = self.get_visibility_of_pt_w_observer_ilqr(x, o, normalized=True, RETURN_ANGLE=True)
            is_vis_sec = vis_sec

            is_vis_observers.append(is_vis_sec)

        # if self.ti == 0:
        #     is_vis_target = not is_vis_target

        print("vis mathing")
        print(x, self.get_target_observer().get_center(), self.get_secondary_observers()[0].get_center())
        print(target_ang, secondary_ang)
        print(is_vis_target, is_vis_observers)

        print("END VIS")

        return vis_target, is_vis_observers

    def dist_between(self, x1, x2):
        # print(x1)
        # print(x2)
        # print(x1[0], x2[0], x1[1], x2[1])

        distance = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        return distance

    # RETURN TRUE IF LOCAL, FALSE IF NOT
    # TARGET, then list of SECONDARY
    def get_is_local_of_all(self, x_in):
        islocal_target      = False
        islocal_secondary   = []

        observers       = self.get_observers()
        target_obs      = self.get_target_observer()
        secondary_obs   = self.get_secondary_observers()

        x = x_in[:2]

        print("LOCAL CHECK")

        print("TARGET DIST BETWEEN " + str(x) + " and " +  str(target_obs.get_center()) + " is...")
        print(self.dist_between(x, target_obs.get_center()))
        print(self.dist_between(x, target_obs.get_center()) < self.get_local_distance())
        if self.dist_between(x, target_obs.get_center()) < self.get_local_distance():
            islocal_target = True
    
        for o in self.get_secondary_observers():
            print("SEC DIST BETWEEN " + str(x) + " and " +  str(o.get_center()) + " is...")
            print(self.dist_between(x, o.get_center()))
            print(self.dist_between(x, o.get_center()) < self.get_local_distance())
            if self.dist_between(x, o.get_center()) < self.get_local_distance():
                islocal_secondary.append(True)
            else:
                islocal_secondary.append(False)            


        print(islocal_target, islocal_secondary, self.get_local_distance())


        print("END LOCAL")
        return islocal_target, islocal_secondary


    def get_solve_quality_status(self, test_group, info_packet=None):
        if info_packet is None:
            info_packet = self.get_solver_status()

        # info_packet = [converged, info, iteration_count]
        if info_packet is not None:
            converged_text = ""
            converged, info, iteration_count, J_opt = info_packet
            if converged is True:
                converged_text = "CONV in " + str(iteration_count)
            elif info == 'accepted':
                converged_text = "OK in " + str(iteration_count)
            else:
                converged_text = "INC in " + str(iteration_count)
            
            blurb = converged_text
        else:
            blurb = ""

        scenario    = self.get_exp_label()
        purpose     = self.get_fn_note()[1:]

        return (scenario, self.get_pretty_study_label(self.goals.index(self.target_goal), self.get_target_goal()), test_group, purpose, converged_text, converged, iteration_count, info, J_opt)

    def get_solve_quality_columns(self):
        return ('scenario', 'goal', 'test', 'condition', 'status_summary', 'converged', 'num_iterations', 'info', 'J_opt')


    # def get_heading_code(self):
    #     return self.get_heading_mode()

    def get_dist_code(self):
        hm = 'none'

        if self.mode_dist_legib_on == False:
            return 'none'

        return self.get_mode_dist_type()


    def get_mode_understanding_target(self):
        return self.mode_und_target

    def get_mode_understanding_secondary(self):
        return self.mode_und_secondary

    # TODO make this more consistent
    def get_suptitle(self):
        title = self.exp_label + "  " + self.get_mode_type_blend()

        # if self.mode_type_heading is None and self.mode_type_dist is None:
        #     title += "No legibility"

        # if self.get_mode_type_heading() == 'sqr':
        #         title += "Heading: Square"
        # elif self.get_mode_type_heading() == 'lin':
        #         title += "Heading: Linear"
        # if self.get_mode_type_dist() == 'exp':
        #     title += " Dist: Exponential"
        # elif self.get_mode_type_dist() == 'sqr':
        #     title += " Dist: Squared"
        # elif self.get_mode_type_dist() == 'lin':
        #     title += " Dist: Linear"

        # if self.mode_type_blend == 'mixed':
        #     title += " blended"

        title += "Target understanding: " + str(self.get_mode_understanding_target()) + " ::: Secondary: " + str(self.get_mode_understanding_secondary())
        title += '\n Local dist: ' + str("{0:.3g}".format(self.get_local_distance()))

        return title

    # Covers the weird cases with decimal falloff better
    def safe_norm(self, x):
        xmax = np.max(x)
        return np.linalg.norm(x / xmax) * xmax

    def vector_mag(self, x):
        # x = decimal.Decimal(x)
        mag = np.sqrt(x.dot(x))
        return mag

    def get_dist_scalar_expected_step_size(self):
        k = np.linalg.norm(np.asarray(self.get_start()) - np.asarray(self.get_target_goal())) / (self.get_N())
        return k

    def get_max_dist(self):
        return self.max_dist


    def get_dist_scalar_k(self):
        k = 1.0 / self.max_dist
        return k

    def set_local_distance(self, val):
        self.local_distance = val

    def get_local_distance(self):
        return self.local_distance

    def get_pretty_study_label(self, g_index, goal_in):
        goal_a, goal_b, goal_c, goal_d, goal_e, goal_f = self.get_goal_squad()
        start   = self.get_start()
        goal    = goal_in[:2]

        label = ''

        if start == goal_a:
            label += 'a'
        if start == goal_b:
            label += 'b'
        if start == goal_c:
            label += 'c'
        if start == goal_d:
            label += 'd'
        if start == goal_e:
            label += 'e'
        if start == goal_f:
            label += 'f'

        if goal == goal_a:
            label += 'a'
        if goal == goal_b:
            label += 'b'
        if goal == goal_c:
            label += 'c'
        if goal == goal_d:
            label += 'd'
        if goal == goal_e:
            label += 'e'
        if goal == goal_f:
            label += 'f'

        if label == '':
            label = "g" + str(g_index)

        return label

    def get_relevance(self, x, goal):
        Q       = np.eye(2)

        goal_a, goal_b, goal_c, goal_d, goal_e, goal_f = self.get_goal_squad()

        start = self.get_start()

        start   = np.asarray(start)
        goal    = np.asarray(goal)

        goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        diff_curr   = start - x
        diff_goal   = x - goal
        diff_all    = np.asarray(goal_a) - np.asarray(goal_b)

        diff_curr_v = np.dot(np.dot(diff_curr.T, Q), diff_curr)
        diff_goal_v = np.dot(np.dot(diff_goal.T, Q), diff_goal)
        diff_all_v  = np.dot(np.dot(diff_all.T, Q), diff_all)

        total_steps = self.get_N()

        # diff_curr_v = self.get_estimated_cost(diff_curr_v, i_step)
        # diff_goal_v = self.get_estimated_cost(diff_goal_v, self.exp.get_N())
        # diff_all_v  = self.get_estimated_cost(diff_all_v, self.exp.get_N())

        n = - (diff_curr_v) - (diff_goal_v)
        d = diff_all_v

        J = np.exp(n) / np.exp(d)

        return J

    def get_closest_daj_goalp_to_x(self, x_in, num=1):
        x = x_in[:2]

        val_for_goals = []
        start = self.get_start()

        for goal in self.get_goals():
            dist = self.dist_between(x, goal)#: # + self.dist_between(x, start)

            if goal != self.get_target_goal():
                val_for_goals.append((dist, goal))

            # print(goal)
            # print(overall_target)

            # val = self.get_relevance(x_in, goal)

        print("format what")
        print(val_for_goals)

        val_for_goals.sort(key=lambda name: name[0])

        subset_goals = val_for_goals[:num]

        print(subset_goals)

        print("trm to goals")
        print( [t[1] for t in subset_goals])
        return  [t[1] for t in subset_goals]


    def is_backtracking(self, x_in, goal):

        start = self.get_start()

        # u = (goal - x_in)
        # v = (goal - self.get_start())

        # proj_of_u_on_v_goal = (np.dot(u, v)/v_norm**2)*v 

        u = (np.asarray(goal) - x_in[:2])
        v = (np.asarray(goal) - np.asarray(self.get_start()))

        v_norm = np.sqrt(sum(v**2))     
        proj_of_u_on_v = (np.dot(u, v)/v_norm**2)*v

        print("backtracking check")
        print(u, v, np.linalg.norm(proj_of_u_on_v))

        if np.linalg.norm(proj_of_u_on_v) > np.linalg.norm(v):
            return True


        return False
        pass


    def get_overshoot(self, x_in, goal):
        start = self.get_start()

        # u = (goal - x_in)
        # v = (goal - self.get_start())

        # proj_of_u_on_v_goal = (np.dot(u, v)/v_norm**2)*v 

        u = (np.asarray(goal) - x_in[:2])
        v = (np.asarray(goal) - np.asarray(self.get_start()))

        v_norm = np.sqrt(sum(v**2))     
        proj_of_u_on_v = (np.dot(u, v)/v_norm**2)*v

        print("backtracking check")
        print(u, v, np.linalg.norm(proj_of_u_on_v))

        return np.linalg.norm(proj_of_u_on_v), np.linalg.norm(v)


    def get_closest_nontarget_goalp_to_x_nobacktrack(self, x_in, num=1):
        x = x_in[:2]

        val_for_goals = []

        for goal in self.get_goals():
            dist = self.dist_between(x, goal)

            is_backtracking = self.is_backtracking(x_in, goal)

            if goal != self.get_target_goal(): # and not is_backtracking:
                val_for_goals.append((dist, goal))


        print("format what")
        print(val_for_goals)

        val_for_goals.sort(key=lambda name: name[0])

        subset_goals = val_for_goals[:num]

        print(subset_goals)

        print("trm to goals")
        print( [t[1] for t in subset_goals])
        return  [t[1] for t in subset_goals]



    def get_closest_nontarget_goalp_to_x(self, x_in, num=1):
        x = x_in[:2]

        val_for_goals = []

        for goal in self.get_goals():
            dist = self.dist_between(x, goal)

            if goal != self.get_target_goal():
                val_for_goals.append((dist, goal))

        print("format what")
        print(val_for_goals)

        val_for_goals.sort(key=lambda name: name[0])

        subset_goals = val_for_goals[:num]

        print(subset_goals)

        print("trm to goals")
        print( [t[1] for t in subset_goals])
        return  [t[1] for t in subset_goals]

    def get_closest_any_goal_to_x(self, x_in):
        x = x_in[:2]
        num = 1

        val_for_goals = []

        for goal in self.get_goals():
            dist = self.dist_between(x, goal)

            val_for_goals.append((dist, goal))

        val_for_goals.sort(key=lambda name: name[0])
        # subset_goals = val_for_goals[:num]
        closest_goal_info = val_for_goals[0]

        return  closest_goal_info[1]

    def get_second_closest_any_goal_to_x(self, x_in):
        x = x_in[:2]
        num = 1

        val_for_goals = []

        for goal in self.get_goals():
            dist = self.dist_between(x, goal)

            val_for_goals.append((dist, goal))

        val_for_goals.sort(key=lambda name: name[0])
        # subset_goals = val_for_goals[:num]
        closest_goal_info = val_for_goals[1]

        return  closest_goal_info[1]


    def set_goal_keepout_distance(self, dist):
        self.keepout_dist = dist

    def get_goal_keepout_distance(self):
        return self.keepout_dist

    def set_lambda(self, lam):
        self.lam = lam

    def get_lambda(self):
        tar_goal    = self.get_target_goal()
        start       = self.get_start()
        start_goal  = start

        goal_a, goal_b, goal_c, goal_d, goal_e, goal_f = self.get_goal_squad()

        # if self.dist_between(tar_goal, start_goal) != self.dist_between(goal_a, goal_b):
        #     return 100000.0

        ### MODE FOR FASTER SEARCH OF KEY SUBSECTIONS
        # If AB chunk
        if start[0] == goal_a[0] and start[1] == goal_a[1] and tar_goal[0] == goal_b[0] and tar_goal[0] == goal_b[0]:
            return self.lam

        if start[0] == goal_b[0] and start[1] == goal_b[1] and tar_goal[0] == goal_a[0] and tar_goal[0] == goal_a[0]:
            return self.lam

        if start[0] == goal_a[0] and start[1] == goal_a[1] and tar_goal[0] == goal_e[0] and tar_goal[0] == goal_e[0]:
            return self.lam

        if start[0] == goal_a[0] and start[1] == goal_a[1] and tar_goal[0] == goal_c[0] and tar_goal[0] == goal_c[0]:
            return self.lam

        # return 100000.0

        resolution = self.lam



        ##### SHORT EDGES
        if self.dist_between(tar_goal, start_goal) == self.dist_between(goal_a, goal_b):
            return self.lam

        ##### LONG EDGES
        if self.dist_between(tar_goal, start_goal) == self.dist_between(goal_a, goal_c):
            return self.lam

        ##### SHORT DIAG
        if self.dist_between(tar_goal, start_goal) == self.dist_between(goal_a, goal_e):
            return self.lam # * np.sqrt(2)

        ##### LONG DIAG
        if self.dist_between(tar_goal, start_goal) == self.dist_between(goal_a, goal_f):
            return self.lam # * np.sqrt(3))

        # else
        return self.N #* 10


    def set_num_iterations(self, n):
        self.new_N = n

    def get_num_iterations(self):
        return self.new_N


    def get_initial_path(self):
        state_size  = self.get_state_size()
        start   = self.get_start()
        goal    = self.get_target_goal()

        if state_size == 4:
            x0_raw          = np.asarray([start[0],    start[1],   start[0],    start[1]]).T
            x_goal_raw      = np.asarray([goal[0],     goal[1],    goal[0],     goal[1]]).T

        elif state_size == 3:
            x0_raw          = np.asarray([start[0],    start[1],   STATIC_ANGLE_DEFAULT]).T
            x_goal_raw      = np.asarray([goal[0],     goal[1],    STATIC_ANGLE_DEFAULT]).T

        elif state_size == 2:
            x_goal_raw = x_goal_raw[:2]
            x0_raw = x0_raw[:2]

        return x_goal_raw, x0_raw

        goal_a, goal_b, goal_c, goal_d, goal_e, goal_f = self.get_goal_squad()

        ##### SHORT EDGES
        if self.dist_between(tar_goal, start_goal) == self.dist_between(goal_a, goal_b):
            return int(resolution)

        ##### LONG EDGES
        if self.dist_between(tar_goal, start_goal) == self.dist_between(goal_a, goal_c):
            return int(resolution * 2)

        ##### SHORT DIAG
        if self.dist_between(tar_goal, start_goal) == self.dist_between(goal_a, goal_e):
            return int(resolution * np.sqrt(2))

        ##### LONG DIAG
        if self.dist_between(tar_goal, start_goal) == self.dist_between(goal_a, goal_f):
            return int(resolution * np.sqrt(3))


        return 
