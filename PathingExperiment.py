import sys
import os
import autograd.numpy as np
import decimal

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

from LegiblePathCost import LegiblePathCost
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
    solver_scale_term       = 1000.0 #.01
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

    def __init__(self, label, restaurant, f_label=None, cost_label=None):
        self.exp_label = label
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

        u_blank  = np.asarray([0.0, 0.0])
        Urefline = np.tile(u_blank, (N, 1))
        Urefline = np.reshape(Urefline, (-1, action_size))

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
            return UnderstandingPathQRCost(self, Xrefline, Urefline), Urefline

        print("ERROR, NO KNOWN SOLVER, PLEASE ADD A VALID SOLVER TO EXP")
        print("''''''" + str(solver_label) + "''''''")


    def setup_file_id(self):
        # Create a new folder for this experiment, along with sending debug output there
        g_index = self.get_g_index()
        self.file_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + "-" + self.exp_label + "-" + str(g_index)

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

    def get_target_observer(self):
        return self.target_observer

    def get_secondary_goals(self):
        return [x for x in self.goals if x != self.target_goal]

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

    def set_N(self, N):
        self.N = N

    def get_N(self):
        return self.N

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

    def get_observer_radius(self):
        return self.OBSERVER_RADIUS

    def get_goal_radius(self):
        return self.GOAL_RADIUS

    def get_obstacle_buffer(self):
        return self.OBSTACLE_BUFFER

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

        blurb += "\nJ=" + str(J_opt)

        return blurb

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

        # ang1 = np.arctan2(*p1[::-1])
        # ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg(angle)

    # ADA MASTER VISIBILITY EQUATION
    # ILQR OBSERVER-AWARE EQUATION
    # RETURN HERE
    def get_visibility_of_pt_w_observer_ilqr(self, pt, observer, normalized=True, epsilon=.01, angle_fov=120, RETURN_ANGLE=False):
        observers   = []
        score       = []

        MAX_DISTANCE    =  np.inf
        obs_orient      = observer.get_orientation() 
        obs_FOV         = angle_fov #observer.get_FOV()

        angle       = self.angle_between_points(observer.get_center(), pt)
        x1, x2      = pt, observer.get_center()
        distance    = distance = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

        in_view     = False

        a                   = angle - obs_orient
        signed_angle_diff   = (a + 180) % 360 - 180
        angle_diff          = abs(signed_angle_diff)

        print("Observer test")
        print(observer.get_center(), observer.get_orientation())

        print(str(pt) + " -> " + str(observer.get_center()) + " = angle " + str(angle))
        print("observer looking at... " + str(obs_orient))
        print("angle diff = " + str(angle_diff))

        half_fov = (obs_FOV / 2.0)
        if angle_diff < half_fov:
            from_center = half_fov - angle_diff
            if normalized:
                from_center = from_center / (half_fov)

            in_view = True

        if RETURN_ANGLE:
            return in_view, angle_diff

        return in_view

    # RETURN TRUE IF IN SIGHT, FALSE IF NO
    # TARGET, then list of SECONDARY
    def get_visibility_of_all(self, x_in):
        is_vis_target = 0
        is_vis_secondary = []

        x = x_in[:2]

        observers       = self.get_observers()
        target_obs      = self.get_target_observer()
        secondary_obs   = self.get_secondary_observers()

        # what to do if there are no observers
        if len(observers) == 0:
            return True, [True]

        # if self.exp.get_is_oa_on() is True:
        is_vis_observers = []
    
        vis_target, target_ang  = self.get_visibility_of_pt_w_observer_ilqr(x, self.get_target_observer(), normalized=True, RETURN_ANGLE=True)
        is_vis_target = vis_target

        for o in self.get_secondary_observers():
            vis_target, secondary_ang  = self.get_visibility_of_pt_w_observer_ilqr(x, o, normalized=True, RETURN_ANGLE=True)
            is_vis_target = vis_target

            is_vis_observers.append(is_vis_target)

        # if self.ti == 0:
        #     is_vis_target = not is_vis_target

        print("vis mathing")
        print(x, self.get_target_observer().get_center(), self.get_secondary_observers()[0].get_center())
        print(target_ang, secondary_ang)
        print(is_vis_target, is_vis_observers)

        print("END VIS")

        return is_vis_target, is_vis_observers

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

        if self.dist_between(x, target_obs.get_center()) < self.get_local_distance():
                print(self.dist_between(x, target_obs.get_center()) )
                islocal_target = True
    
        for o in self.get_secondary_observers():
            print(self.dist_between(x, o.get_center()))
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

        return (scenario, self.get_goal_label(), test_group, purpose, converged_text, converged, iteration_count, info, J_opt)

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
