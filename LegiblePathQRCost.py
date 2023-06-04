import os
import sys
module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt
import decimal
import copy
import time
from datetime import timedelta, datetime
import matplotlib.ticker as mtick

from ilqr import iLQR
from ilqr.cost import Cost
from ilqr.cost import QRCost
from ilqr.cost import PathQRCost, AutoDiffCost, FiniteDiffCost
from ilqr.dynamics import constrain
from ilqr.examples.pendulum import InvertedPendulumDynamics
from ilqr.dynamics import BatchAutoDiffDynamics, tensor_constrain

from scipy.optimize import approx_fprime

import utility_legibility as legib
import utility_environ_descrip as resto
import pipeline_generate_paths as pipeline
import pdb
from random import randint

import PathingExperiment as ex

PREFIX_EXPORT = 'experiment_outputs/'

FLAG_SHOW_IMAGE_POPUP = False

goal_colors = ['red', 'blue', 'purple', 'green']

# Base class for all of our legible pathing offshoots
class LegiblePathQRCost(FiniteDiffCost):
    PREFIX_EXPORT = PREFIX_EXPORT

    FLAG_DEBUG_J = False
    FLAG_DEBUG_STAGE_AND_TERM = False

    coeff_terminal = 10000.0
    scale_term = 0.01 # 1/100
    # scale_stage = 1.5
    scale_stage = 2
    scale_obstacle = 0

    state_size = 2
    action_size = 2

    # The coefficients weigh how much your state error is worth to you vs
    # the size of your controls. You can favor a solution that uses smaller
    # controls by increasing R's coefficient.
    Q = 1.0 * np.eye(state_size)
    R = 200.0 * np.eye(action_size)
    Qf = np.identity(state_size) * 400.0

    """Quadratic Regulator Instantaneous Cost for trajectory following."""
    def __init__(self, exp, x_path, u_path):
        self.legib_path_cost_make_self(
            exp,
            exp.get_Q(),
            exp.get_R(),
            exp.get_Qf(),
            x_path,
            u_path,
            exp.get_start(),
            exp.get_target_goal(),
            exp.get_goals(),
            exp.get_N(),
            exp.get_dt(),
            restaurant=exp.get_restaurant(),
            file_id=exp.get_file_id()
            )

    def legib_path_cost_make_self(self, exp, Q, R, Qf, x_path, u_path, start, target_goal, goals, N, dt, restaurant=None, file_id=None, Q_terminal=None):
        """Constructs a QRCost.
        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            x_path: Goal state path [N+1, state_size].
            u_path: Goal control path [N, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
        """

        self.exp = exp
        self.Q = np.array(Q)
        self.Qf = np.array(Qf)
        self.R = np.array(R)

        # self.R = np.eye(2)*10000

        self.x_path = np.array(x_path)

        self.start = np.array(start)
        self.goals = goals
        self.target_goal = target_goal
        self.N = N
        self.dt = dt

        if file_id is None:
            self.file_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        else:
            self.file_id = file_id

        # Create a restaurant object for using those utilities, functions, and print functions
        # dim gives the dimensions of the restaurant
        if restaurant is None:
            self.restaurant = resto.Restaurant(resto.TYPE_CUSTOM, tables=[], goals=goals, start=start, observers=[], dim=None)
        else:
            self.restaurant = restaurant

        self.f_func = self.get_f()

        state_size = self.Q.shape[0]
        action_size = self.R.shape[0]
        path_length = self.x_path.shape[0]

        x_eps = .05 #05
        u_eps = .01 #05


        # self._x_eps_hess = np.sqrt(self._x_eps)
        # self._u_eps_hess = np.sqrt(self._u_eps)

        self._state_size = state_size
        self._action_size = action_size

        if Q_terminal is None:
            self.Q_terminal = self.Q
        else:
            self.Q_terminal = np.array(Q_terminal)

        if u_path is None:
            self.u_path = np.zeros(path_length - 1, action_size)
        else:
            self.u_path = np.array(u_path)

        assert self.Q.shape == self.Q_terminal.shape, "Q & Q_terminal mismatch"
        assert self.Q.shape[0] == self.Q.shape[1], "Q must be square"
        assert self.R.shape[0] == self.R.shape[1], "R must be square"
        
        assert state_size == self.x_path.shape[1], "Q & x_path mismatch"
        assert action_size == self.u_path.shape[1], "R & u_path mismatch"
        assert path_length == self.u_path.shape[0] + 1, \
                "x_path must be 1 longer than u_path"

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Q + self.Q.T
        self._R_plus_R_T = self.R + self.R.T
        self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T

        FiniteDiffCost.__init__(
            self,
            self.l,
            self.term_cost,
            state_size,
            action_size,
            x_eps=x_eps,
            u_eps=u_eps,
        )


    def init_output_log(self, dash_folder):
        n = 5
        rand_id = ''.join(["{}".format(randint(0, 9)) for num in range(0, n)])
        sys.stdout = open(self.get_export_label(dash_folder) + str(rand_id) + '--output.txt','a')

    def get_f(self):
        f_label = self.exp.get_f_label()
        print(f_label)

        if f_label is ex.F_ANCA_LINEAR:
            def f(i):
                return self.N - i

        elif f_label is ex.F_VIS_LIN:
            def f(i):
                pt = self.x_path[i]
                restaurant  = self.exp.get_restaurant()
                observers   = restaurant.get_observers()

                visibility  = legib.get_visibility_of_pt_w_observers_ilqr(pt, observers, normalized=True)
                # Can I see this point from each observer who is targeted
                return visibility

        else:
            def f(i):
                return 1.0


        return f
        print("ERROR, NO KNOWN SOLVER, PLEASE ADD A VALID SOLVER TO EXP")


    # How far away is the final step in the path from the goal?
    def term_cost(self, x, i):
        start = self.start
        goal1 = self.target_goal
        
        # Qf = self.Q_terminal
        Qf = self.Qf
        R = self.R

        # x_diff = x - self.x_path[i]
        x_diff = x - self.x_path[self.N]
        squared_x_cost = .5 * x_diff.T.dot(Qf).dot(x_diff)
        # squared_x_cost = squared_x_cost * squared_x_cost

        terminal_cost = squared_x_cost

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("term cost squared x cost")
            print(squared_x_cost)

        # We want to value this highly enough that we don't not end at the goal
        # terminal_coeff = 100.0
        coeff_terminal = self.exp.get_solver_coeff_terminal()
        terminal_cost = terminal_cost * coeff_terminal

        # Once we're at the goal, the terminal cost is 0
        
        # Attempted fix for paths which do not hit the final mark
        # if squared_x_cost > .001:
        #     terminal_cost *= 1000.0

        return terminal_cost

    # original version for plain path following
    def l_og(self, x, u, i, terminal=False):
        """Instantaneous cost function.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        x_diff = x - self.x_path[i]
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

        if terminal:
            return squared_x_cost

        u_diff = u - self.u_path[i]
        return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

    # original version for plain path following
    def l(self, x, u, i, terminal=False, just_term=False, just_stage=False):
        """Instantaneous cost function.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Qf if terminal else self.Q
        R = self.R
        start = self.start
        goal = self.target_goal
        thresh = .0001

        scale_term = self.exp.get_solver_scale_term() #0.01 # 1/100
        scale_stage = self.exp.get_solver_scale_stage() #1.5

        if just_term:   
            scale_stage = 0

        if just_stage:
            scale_term  = 0


        term_cost = 0 #self.term_cost(x, i)
        x_diff = x - self.x_path[i]

        u_diff = 0
        if i < len(self.u_path):
            u_diff = u - self.u_path[i]


        if terminal or just_term: #abs(i - self.N) < thresh or
            # TODO verify not a magic number
            return scale_term * self.term_cost(x, i) # * 1000
        else:
            if self.FLAG_DEBUG_STAGE_AND_TERM:
                # difference between this step and the end
                print("x, N, x_end_of_path -> inputs and then term cost")
                print(x, self.N, self.x_path[self.N])
                # term_cost = self.term_cost(x, i)
                print(term_cost)

        term_cost = 0 #self.term_cost(x, i)

        f_func     = self.get_f()
        f_value    = f_func(i)
        J = self.michelle_stage_cost(start, goal, x, u, i, terminal) * f_value

        wt_legib     = -1.0
        wt_lam       = .001
        wt_control   = 3.0

        J =  (wt_legib       * J)
        J += (wt_control    * u_diff.T.dot(R).dot(u_diff))
        J += (wt_lam        * x_diff.T.dot(Q).dot(x_diff))

    
        stage_costs = J

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("STAGE,\t TERM")
            print(stage_costs, term_cost)

        total = (scale_term * term_cost) + (scale_stage * stage_costs)

        return float(total)

    def f(self, t):
        return 1.0 #self.N - t #1.0

    def get_total_stage_cost(self, start, goal, x, u, i, terminal):
        N = self.N
        R = self.R

        stage_costs = 0.0 #u_diff.T.dot(R).dot(u_diff)
        
        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("u = " + str(u))
            print("Getting stage cost")

        for j in range(i):
            u_diff = u - self.u_path[j]
            stage_costs += (0.5 * u_diff.T.dot(R).dot(u_diff))

        # f_func     = self.get_f()
        # f_value    = f_func(i)
        # stage_costs = self.michelle_stage_cost(start, goal, x, u, i, terminal) * f_value

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("total stage cost " + str(stage_costs))

        return stage_costs

    def michelle_stage_cost(self, start, goal, x, u, i, terminal=False):
        Q = self.Q_terminal if terminal else self.Q
        R = self.R

        all_goals = self.goals

        goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        u_diff = u - self.u_path[i]
        x_diff = x - self.x_path[i]

        if len(self.u_path) == 0:
            return 0

        a = (goal_diff.T).dot(Q).dot((goal_diff))
        b = (start_diff.T).dot(Q).dot((start_diff))
        c = (togoal_diff.T).dot(Q).dot((togoal_diff))

        # (start-goal1)'*Q*(start-goal1) - (start-x)'*Q*(start-x) +  - (x-goal1)'*Q*(x-goal1) 
        J_g1 = a - b - c
        # J_g1 *= .5

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("For point at x -> " + str(x))
            # print("Jg1 " + str(J_g1))

        log_sum = 0.0
        total_sum = 0.0


        ####### NOTE: J_g1 is an artefact, so is log_sum and total_sum, PARTS is now the only element that matters
        parts = []
        for alt_goal in all_goals:
            # n = - ((start-x)'*Q*(start-x) + 5) - ((x-goal)'*Q*(x-goal)+10)
            # d = (start-goal)'*Q*(start-goal)
            # log_sum += (exp(n )/exp(d))* scale

            
            diff_curr   = start - x
            diff_goal   = x - alt_goal
            diff_all    = start - alt_goal

            diff_curr   = diff_curr.T
            diff_goal   = diff_goal.T
            diff_all    = diff_all.T

            n = - (diff_curr.T).dot(Q).dot((diff_curr)) - ((diff_goal).T.dot(Q).dot(diff_goal))
            d = (diff_all).T.dot(Q).dot(diff_all)

            # this_goal = np.exp(n) / np.exp(d)
            this_goal = n + d

            # total_sum += this_goal

            if goal != alt_goal:
                log_sum += (-1 * this_goal)
                parts.append(-1 * this_goal)
                
                if self.FLAG_DEBUG_STAGE_AND_TERM:
                    # print("Value for alt target goal " + str(alt_goal))
                    print("This is the nontarget goal: " + str(alt_goal) + " -> " + str(this_goal))
            else:
                # print("Value for our target goal " + str(goal))
                # J_g1 = this_goal
                log_sum += this_goal
                parts.append(this_goal)
                
                if self.FLAG_DEBUG_STAGE_AND_TERM:
                    print("This is the target goal " + str(alt_goal) + " -> " + str(this_goal))
    
            # print(n + d) 

        # print("log sum")
        # print(log_sum)

        # ratio = J_g1 / (J_g1 + np.log(log_sum))
        # print("ratio " + str(ratio))

        # the log on the log sum actually just cancels out the exp
        # J = np.log(J_g1) - np.log(log_sum)
        # alt_goal_multiplier = 5.0

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("Jg1, total")
            print(J_g1, total_sum)

        print(J_g1, parts)

        # J = J_g1 - (np.log(total_sum))
        J = J_g1 + log_sum
        # J = -1.0 * J

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("overall J " + str(J))

        J = sum(parts)

        # # We want the path to be smooth, so we incentivize small and distributed u

        return J

    def path_following_stage_cost(self, start, goal, x, u, i, terminal=False):
        Q = self.Q_terminal if terminal else self.Q
        R = self.R

        # Q = np.eye(2) * 1
        # R = np.eye(2) * 1000

        all_goals = self.goals

        goal_diff = start - goal
        start_diff = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        if len(self.u_path) == 0:
            return 0

        a = np.abs(goal_diff.T).dot(Q).dot((goal_diff))
        b = np.abs(start_diff.T).dot(Q).dot((start_diff))
        c = (togoal_diff.T).dot(Q).dot((togoal_diff))

        # (start-goal1)'*Q*(start-goal1) - (start-x)'*Q*(start-x) +  - (x-goal1)'*Q*(x-goal1)
        J_g1 = c
        J_g1 *= .5

        return J_g1

    def goal_efficiency_through_point(self, start, x, goal, terminal=False):
        Q = self.Q_terminal if terminal else self.Q

        goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        a = (goal_diff.T).dot(Q).dot((goal_diff))
        b = (start_diff.T).dot(Q).dot((start_diff))
        c = (togoal_diff.T).dot(Q).dot((togoal_diff))
    
        return (a + b) / c

    # TODO switch this to be logs
    def goal_efficiency_through_point_relative(self, start, x, goal, terminal=False):
        all_goals = self.goals

        this_goal = self.goal_efficiency_through_point(start, x, goal)

        goals_total = 0.0
        for alt_goal in all_goals:
            sub_goal = self.goal_efficiency_through_point(start, x, alt_goal)
            goals_total += np.exp(sub_goal)
    
        ratio = this_goal / np.log(goals_total)
        print(ratio)
        return ratio

        # return np.log(this_goal) - np.log(goals_total)

        # return decimal.Decimal(this_goal / goals_total)
        # return np.log(this_goal) - np.log(goals_total)

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            dl/dx [state_size].
        """
        if terminal:
            return approx_fprime(x, lambda x: self._l_terminal(x, i),
                                 self._x_eps)

        val = approx_fprime(x, lambda x: self._l(x, u, i), self._x_eps)
        if self.FLAG_DEBUG_J:
            print("J_x at " + str(x) + "," + str(u) + ","  + str(i))
            print(val)

        return val

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        val = approx_fprime(u, lambda u: self._l(x, u, i), self._u_eps)
        if self.FLAG_DEBUG_J:
            print("J_u at " + str(x) + "," + str(u) + ","  + str(i))
            print(val)

        return val

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        eps = self._x_eps_hess
        Q = np.vstack([
            approx_fprime(x, lambda x: self.l_x(x, u, i, terminal)[m], eps)
            for m in range(self._state_size)
        ])

        if self.FLAG_DEBUG_J:
            print("J_xx at " + str(x) + "," + str(u) + ","  + str(i))
            print(Q)

        return Q

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        eps = self._x_eps_hess
        Q = np.vstack([
            approx_fprime(x, lambda x: self.l_u(x, u, i)[m], eps)
            for m in range(self._action_size)
        ])

        if self.FLAG_DEBUG_J:
            print("J_ux at " + str(x) + "," + str(u) + ","  + str(i))
            print(Q)

        return Q

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        eps = self._u_eps_hess
        Q = np.vstack([
            approx_fprime(u, lambda u: self.l_u(x, u, i)[m], eps)
            for m in range(self._action_size)
        ])

        if self.FLAG_DEBUG_J:
            print("J_uu at " + str(x) + "," + str(u) + ","  + str(i))
            print(Q)

        return Q

    def get_window_dimensions_for_envir(self, start, goals, pts):
        xmin, ymin = start
        xmax, ymax = start

        all_points = copy.copy(goals)
        all_points.append(start)
        all_points.extend(pts)
        
        for pt in all_points:
            x, y = pt[0], pt[1]

            if x < xmin:
                xmin = x
            if y < ymin:
                ymin = y
            if x > xmax:
                xmax = x
            if y > ymax:
                ymax = y

        xwidth      = xmax - xmin
        yheight     = ymax - ymin

        xbuffer     = .1 * xwidth
        ybuffer     = .1 * yheight

        if xbuffer < .3:
            xbuffer = .5

        if ybuffer < .3:
            ybuffer = .5

        return xmin - xbuffer, xmax + xbuffer, ymin - ybuffer, ymax + ybuffer

    def get_export_label(self, dash_folder=None):
        note = self.exp.get_fn_note()
        if dash_folder is not None:
            folder_name = dash_folder + "/" + self.file_id + note + "/"
        else:
            folder_name = PREFIX_EXPORT + self.file_id + note + "/" 
    
        overall_name = folder_name + self.file_id
        try:
            os.mkdir(folder_name)
        except:
            print("FILE ALREADY EXISTS " + self.file_id)

        return overall_name

    def get_prob_of_path_to_goal(self, verts, us, goal, label):
        prob_list = []

        start = self.start
        u = None
        terminal = False
        if len(verts) != self.N + 1:
            print("points in path does not match the solve N")

        resto_envir = self.restaurant
        goals = self.goals

        for i in range(len(verts)):
            # print(str(i) + " out of " + str(len(verts)))
            x = verts[i]
            if i < len(us):
                u = us[i - 1]
            else:
                u = [0, 0]

            aud = resto_envir.get_observers()
            bin_visibility = legib.get_visibility_of_pt_w_observers_ilqr(x, aud, normalized=True)
            if bin_visibility > 0:
                bin_visibility = 1.0

            if label is 'dist_exp':
                p = self.legibility_stage_cost(start, goal, x, u, i, terminal, bin_visibility, force_mode='exp', pure_prob=True)
            elif label is 'dist_sqr':
                p = self.legibility_stage_cost(start, goal, x, u, i, terminal, bin_visibility, force_mode='sqr', pure_prob=True)
            elif label is 'dist_lin':
                p = self.legibility_stage_cost(start, goal, x, u, i, terminal, bin_visibility, force_mode='lin', pure_prob=True)
            elif label is 'head_sqr':
                p = float(self.get_heading_stage_cost(x, u, i, goal, bin_visibility, force_mode='sqr', pure_prob=True))
            elif label is 'head_lin':
                p = float(self.get_heading_stage_cost(x, u, i, goal, bin_visibility, force_mode='lin', pure_prob=True))

            
            prob_list.append(p)

        return prob_list

    def get_legibility_of_path_to_goal(self, verts, us, goal):
        ls, scs, tcs, vs = [], [], [], []
        start = self.start
        u = None
        terminal = False
        if len(verts) != self.N + 1:
            print("points in path does not match the solve N")

        resto_envir = self.restaurant
        goals = self.goals

        for i in range(len(verts)):
            # print(str(i) + " out of " + str(len(verts)))
            x = verts[i]

            aud = resto_envir.get_observers()
            l = legib.f_legibility_ilqr(resto_envir, goal, goals, verts[:i], aud)
            
            if i < len(us):
                j = len(us) - 1
                u = us[j]
                sc = self.l(x, u, j, just_stage=True) #self.get_total_stage_cost(start, goal, x, u, j, terminal)
                tc = self.l(x, u, j, just_term=True)
            else:
                sc = 0.0

            scs.append(sc)
            tcs.append(tc)
            # tc = float(self.term_cost(x, i))
            ls.append(l)

        # TODO: alter this if we want to show vis from multiple observers
        for obs in self.exp.get_restaurant().get_observers():
            vis_log = []

            for i in range(len(verts)):
                x = verts[i]

                v = legib.get_visibility_of_pt_w_observers_ilqr(x, [obs], normalized=True)
                vis_log.append(v)

            vs.append(vis_log)

        return ls, scs, tcs, vs


    def get_debug_text(self, elapsed_time):
        debug_text_a = "stage scale: " + str(self.exp.get_solver_scale_stage()) + "        "
        debug_text_b = "term scale: " + str(self.exp.get_solver_scale_term()) + "        "
        debug_text_c = "coeff_terminal: " + str(self.exp.get_solver_coeff_terminal()) + "\n        "
        debug_text_d = "obstacle scale: " + str(self.exp.get_solver_scale_obstacle()) + "        " # + "\n"
        debug_text_e = "dt: " + str(self.exp.get_dt()) + "        "
        debug_text_f = "N: " + str(self.exp.get_N()) # + "\n"

        debug_text = debug_text_a + debug_text_b + debug_text_c + debug_text_d + debug_text_e + debug_text_f

        if elapsed_time is not None:
            # time.strftime("%M:%S:%f", time.gmtime())
            elapsed_time =  "%.2f seconds" % elapsed_time
            debug_text = elapsed_time + "        " + debug_text

        return debug_text

    def hex_to_RGB(self, hex_str):
        """ #FFFFFF -> [255,255,255]"""
        #Pass 16 to the integer function for change of base
        return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

    def get_color_gradient(self, c1, c2, n):
        """
        Given two hex colors, returns a color gradient
        with n colors.
        """
        assert n > 1
        c1_rgb = np.array(self.hex_to_RGB(c1))/255
        c2_rgb = np.array(self.hex_to_RGB(c2))/255
        mix_pcts = [x/(n-1) for x in range(n)]
        rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
        return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

    def get_overview_pic(self, verts, us, elapsed_time=None, ax=None, info_packet=None, fn_note="", dash_folder=None):
        axarr = ax

        axarr.set_aspect('equal')
        axarr.grid(axis='y')

        TABLE_RADIUS    = self.exp.get_table_radius()
        OBS_RADIUS      = self.exp.get_observer_radius()
        GOAL_RADIUS     = self.exp.get_goal_radius()

        TABLE_RADIUS_BUFFER     = self.exp.get_table_radius() + self.exp.get_obstacle_buffer()
        OBSERVER_RADIUS_BUFFER  = self.exp.get_observer_radius() + self.exp.get_obstacle_buffer()
        GOAL_RADIUS_BUFFER             = self.exp.get_goal_radius() + self.exp.get_obstacle_buffer()

        tables      = self.restaurant.get_tables()
        observers   = self.restaurant.get_observers()

        xs, ys = zip(*verts)
        gx, gy = zip(*self.goals)
        sx, sy = self.start

        for table in tables:
            table = plt.Circle(table.get_center(), TABLE_RADIUS_BUFFER, color='#AFE1AF', clip_on=False)
            axarr.add_patch(table)

            table = plt.Circle(table.get_center(), TABLE_RADIUS, color='#097969', clip_on=False)
            axarr.add_patch(table)


        for observer in observers:
            obs_color = 'orange'
            obs_color_outer = '#f8d568'
            obs_pt  = observer.get_center()
            obs     = plt.Circle(obs_pt, OBSERVER_RADIUS_BUFFER, color=obs_color_outer, clip_on=False)
            axarr.add_patch(obs)

            obs     = plt.Circle(obs_pt, OBS_RADIUS, color=obs_color, clip_on=False)
            axarr.add_patch(obs)

            ox, oy = obs_pt
            THETA_ARROW_RADIUS = 1
            r = THETA_ARROW_RADIUS

            # u, v = x * (np.cos(y), np.sin(y))

            angle = observer.get_orientation()
            angle_rads = angle * np.pi / 180
            # ax1.arrow(x, y, r*np.cos(theta), r*np.sin(theta), length_includes_head=False, color=obs_color, width=.06)
            # ax1.quiver(x, y, r*np.cos(theta), r*np.sin(theta), color=obs_color)
            axarr.arrow(ox, oy, r*np.cos(angle_rads), r*np.sin(angle_rads), color=obs_color)

            half_fov = 60
            obs_color = 'yellow'
            fov1 = (angle + half_fov) % 360
            fov2 = (angle - half_fov) % 360

            fov1_rads = fov1 * np.pi / 180
            fov2_rads = fov2 * np.pi / 180

            # print("fov1, angle, fov2")
            # print(fov1, angle, fov2)

            # print("arrow1")
            # print(ox, oy, r*np.cos(fov1_rads), r*np.sin(fov1_rads))
            # print("arrow2")
            # print(ox, oy, r*np.cos(fov2_rads), r*np.sin(fov2_rads))
                        
            axarr.arrow(ox, oy, r*np.cos(fov1_rads), r*np.sin(fov1_rads), color=obs_color)
            axarr.arrow(ox, oy, r*np.cos(fov2_rads), r*np.sin(fov2_rads), color=obs_color)

            # ax1.arrow(x, y, r*np.cos(theta - half_fov), r*np.sin(theta - half_fov), length_includes_head=False, color=obs_color, width=.03)
            # ax1.arrow(x, y, r*np.cos(theta + half_fov), r*np.sin(theta + half_fov), length_includes_head=False, color=obs_color, width=.03)

        # COLOR BASED ON VISIBILITY
        # Color code the goals for ease of reading graphs

        for j in range(len(self.goals)):
            goal = self.goals[j]
            color = goal_colors[j]
            if goal is self.target_goal:
                target_color = color

        color_grad_1 = self.get_color_gradient('#FFFFFF', '#f4722b', len(xs))
        color_grad_2 = self.get_color_gradient('#FFFFFF', '#13678A', len(xs))

        ### DRAW INDICATOR OF IF IN SIGHT OR NOT
        ls, scs, tcs, vs = self.get_legibility_of_path_to_goal(verts, us, self.exp.get_target_goal())
        
        in_vis = None
        if len(vs) > 0:
            in_vis = [i > 0 for i in vs[0]]
        else:
            in_vis = [True for i in ls]

        color_grad = []
        outline_grad = []
        for i in range(len(in_vis)):
            can_see = in_vis[i]
            print(can_see)

            if can_see == True:
                color_grad.append(color_grad_1[i])
                outline_grad.append('#13678A')
            else:
                color_grad.append(color_grad_2[i])
                outline_grad.append('#f4722b')

        axarr.plot(sx, sy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor="grey", lw=0, label="start")
        _ = axarr.set_xlabel("X", fontweight='bold')
        _ = axarr.set_ylabel("Y", fontweight='bold')
        blurb = self.exp.get_solver_status_blurb()
        _ = axarr.set_title("Path through space\n" + blurb, fontweight='bold')

        # Draw the legibility over time

        # Example of how to draw a polygon obstacle on the path
        if False:
            obs = [[.5, 1], [1, 1], [1, .5], [.5, .5], [.5, 1]]
            oxs, oys = zip(*obs)
            axarr.fill(oxs, oys, "c")

        target = self.target_goal
        # for each goal, graph legibility
        for j in range(len(self.exp.get_goals())):
            goal = self.exp.get_goals()[j]
            color = goal_colors[j]

            gx, gy = goal

            if gx == target[0] and gy == target[1]:
                axarr.plot(gx, gy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor=color, lw=0, label="target")
            else:
                g = plt.Circle(goal, GOAL_RADIUS_BUFFER, color='#aaaaaa', clip_on=False)
                axarr.add_patch(g)

                g = plt.Circle(goal, GOAL_RADIUS, color='#333333', clip_on=False)
                axarr.add_patch(g)
                axarr.plot(gx, gy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor=color, lw=0) #, label=goal)


        if self.exp.best_xs is not None and verts is not self.exp.best_xs:
            print("ALT TO PLOT")

            bxs, bys = zip(*self.best_xs)

            axarr.plot(xs, ys, linestyle='dashed', lw=1, color='purple', label="path", markersize=1)
            axarr.scatter(xs, ys, c=outline_grad, s=20)
            axarr.scatter(xs, ys, c=color_grad, s=10)


        # Draw the path itself
        axarr.plot(xs, ys, linestyle='dashed', lw=1, color='black', label="path", markersize=0)
        axarr.scatter(xs, ys, c=outline_grad, s=20)
        axarr.scatter(xs, ys, c=color_grad, s=10)


        axarr.legend(loc="upper left")
        axarr.grid(False)
        xmin, xmax, ymin, ymax = self.get_window_dimensions_for_envir(self.start, self.goals, verts)
        axarr.set_xlim([xmin, xmax])
        axarr.set_ylim([ymin, ymax])

        return axarr

    def graph_legibility_over_time(self, verts, us, elapsed_time=None, status_packet=None, dash_folder=None, suptitle=None):
        print("GRAPHING LEGIBILITY OVER TIME")
        sys.stdout = open('path_overview.txt','w')


        ts = np.arange(self.N) * self.dt

        print("verts")
        print(verts)
        print("Attempt to display this path")

        xs, ys = zip(*verts)
        gx, gy = zip(*self.goals)
        sx, sy = self.start

        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(3, 2, figsize=(8, 6.5), gridspec_kw={'height_ratios': [4, 4, 1]})
        
        # plt.figure(figsize=(12, 6))
        # ax1 = plt.subplot(2,3,1)
        # ax2 = plt.subplot(2,3,2)
        # ax3 = plt.subplot(2,3,3)

        # ax4 = plt.subplot(2,1,2)
        # axes = [ax1, ax2, ax3, ax4]

        # Ummm, incredible plot layout system for numpy
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot_mosaic.html
        # fig, axes = plt.subplot_mosaic("AAB;AAC;DEF;GHI;JKL;MMM", figsize=(8, 6), gridspec_kw={'height_ratios':[36, 36, 36, 36, 36, 1], 'width_ratios':[1, 1, 1]})
        fig, axes = plt.subplot_mosaic("AADEF;AAGHI;BCJKL;MMMMM", figsize=(12, 6), gridspec_kw={'height_ratios':[36, 36, 36, 1], 'width_ratios':[1, 1, 1, 1, 1]})
        # fig, axes = plt.subplot_mosaic("AAB;AAC;DEF")
        ax1 = axes['A'] # plot of movements in space
        ax2 = axes['B'] # old legibility
        ax3 = axes['D'] # stage cost
        ax4 = axes['E'] # terminal cost
        ax5 = axes['M'] # error text box
        ax6 = axes['F'] # U magnitude over time
        ax7 = axes['J'] # visibility
        ax_p_dist_exp    = axes['G'] # p_dist_exp
        ax_p_dist_sqr    = axes['H'] # p_dist_sqr
        ax_p_dist_lin    = axes['I'] # p_dist_lin
        ax_p_head_sqr    = axes['K'] # p_head_sqr
        ax_p_head_lin    = axes['L'] # p_head_lin

        ax_none = axes['C']
        ax_none.axis('off')

        ax1 = self.get_overview_pic(verts, us, elapsed_time=None, ax=ax1, info_packet=status_packet)

        ax5.axis('off')
        debug_text = self.get_debug_text(elapsed_time)
        ax5.annotate(debug_text, (0.5, 0), xycoords='axes fraction', ha="center", va="center", wrap=True, fontweight='bold') #, fontsize=6)

        fig.suptitle(str(suptitle))

        ax2.grid(axis='y')
        ax3.grid(axis='y')
        ax4.grid(axis='y')
        ax7.grid(axis='y')

        # ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
        # ax4.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))

        # each set of xs, ys happens at time t
        # we want to find the legibility at time t
        # and graph it
        # ideally even combine it into a graph with the drawing itself

        # # Color code the goals for ease of reading graphs
        # goal_colors = ['red', 'blue', 'purple', 'green']

        for j in range(len(self.goals)):
            goal = self.goals[j]
            color = goal_colors[j]
            if goal is self.target_goal:
                target_color = color

        # # '#9F2B68'
        # # '#60D497'

        color_grad_1 = self.get_color_gradient('#FFFFFF', '#f4722b', len(xs))
        color_grad_2 = self.get_color_gradient('#FFFFFF', '#13678A', len(xs))

        # ### DRAW INDICATOR OF IF IN SIGHT OR NOT
        ls, scs, tcs, vs = self.get_legibility_of_path_to_goal(verts, us, self.exp.get_target_goal())

        p_dist_exp    = self.get_prob_of_path_to_goal(verts, us, self.exp.get_target_goal(), 'dist_exp')
        p_dist_sqr    = self.get_prob_of_path_to_goal(verts, us, self.exp.get_target_goal(), 'dist_sqr')
        p_dist_lin    = self.get_prob_of_path_to_goal(verts, us, self.exp.get_target_goal(), 'dist_lin')
        p_head_sqr    = self.get_prob_of_path_to_goal(verts, us, self.exp.get_target_goal(), 'head_sqr')
        p_head_lin    = self.get_prob_of_path_to_goal(verts, us, self.exp.get_target_goal(), 'head_lin')
        
        in_vis = None
        if len(vs) > 0:
            in_vis = [i > 0 for i in vs[0]]
        else:
            in_vis = [True for i in ls]

        color_grad = []
        outline_grad = []
        for i in range(len(in_vis)):
            can_see = in_vis[i]
            print(can_see)

            if can_see == True:
                color_grad.append(color_grad_1[i])
                outline_grad.append('#13678A')
            else:
                color_grad.append(color_grad_2[i])
                outline_grad.append('#f4722b')

        target = self.target_goal
        # for each goal, graph legibility
        for j in range(len(self.exp.get_goals())):
            goal = self.exp.get_goals()[j]
            color = goal_colors[j]

            gx, gy = goal

            ls, scs, tcs, vs = self.get_legibility_of_path_to_goal(verts, us, goal)
            print(goal)
            print("LEGIB VALUES")
            print(ls)
            ts = np.arange(len(ls)) * self.dt

            ax2.plot(ts, ls, 'o--', lw=2, color=color, label=goal, markersize=3)


            ax3.plot(ts, scs, linestyle='dashed', lw=1, color='grey', label=goal, markersize=0)
            ax4.plot(ts, tcs, linestyle='dashed', lw=1, color='grey', label=goal, markersize=0)

            ax3.scatter(ts, scs, c=outline_grad, s=8)
            ax4.scatter(ts, tcs, c=outline_grad, s=8)

            # ax3.scatter(ts, scs, c=color_grad, s=4)
            # ax4.scatter(ts, tcs, c=color_grad, s=4)

            # ax2.plot(ts, ls, 'o--', lw=2, color=color, label=goal, markersize=3)
            # # print("plotted ax2")
            # ax3.plot(ts, scs, 'o--', lw=2, color=color, label=goal, markersize=3)
            # # print("plotted ax3")
            # ax4.plot(ts, tcs, 'o--', lw=2, color=color, label=goal, markersize=3)

            for v in vs:
                ax7.scatter(ts, v, c=outline_grad, s=8)
                # ax7.scatter(ts, v, c=color_grad, s=5)
                ax7.plot(ts, v, lw=1, color=color, label=goal, markersize=0)

                # ax7.plot(ts, v, 'o--', lw=2, color=color, label=goal, markersize=3)
            # print("plotted ax4")
            # print("Plotted all data")
        

        ax_p_dist_exp.scatter(ts, p_dist_exp, c=outline_grad, s=8)
        ax_p_dist_exp.plot(ts, p_dist_exp, lw=1, color=color, label=goal, markersize=0)
    
        ax_p_dist_sqr.scatter(ts, p_dist_sqr, c=outline_grad, s=8)
        ax_p_dist_sqr.plot(ts, p_dist_sqr, lw=1, color=color, label=goal, markersize=0)

        ax_p_dist_lin.scatter(ts, p_dist_lin, c=outline_grad, s=8)
        ax_p_dist_lin.plot(ts, p_dist_lin, lw=1, color=color, label=goal, markersize=0)

        ax_p_head_sqr.scatter(ts, p_head_sqr, c=outline_grad, s=8)
        ax_p_head_sqr.plot(ts, p_head_sqr, lw=1, color=color, label=goal, markersize=0)

        ax_p_head_lin.scatter(ts, p_head_lin, c=outline_grad, s=8)
        ax_p_head_lin.plot(ts, p_head_lin, lw=1, color=color, label=goal, markersize=0)

        _ = ax_p_dist_exp.set_xlabel("Time", fontweight='bold')
        _ = ax_p_dist_exp.set_ylabel("P(G | xi)", fontweight='bold')
        _ = ax_p_dist_exp.set_title("P dist exp", fontweight='bold')

        _ = ax_p_dist_sqr.set_xlabel("Time", fontweight='bold')
        _ = ax_p_dist_sqr.set_ylabel("P(G | xi)", fontweight='bold')
        _ = ax_p_dist_sqr.set_title("P dist sqr", fontweight='bold')

        _ = ax_p_dist_lin.set_xlabel("Time", fontweight='bold')
        _ = ax_p_dist_lin.set_ylabel("P(G | xi)", fontweight='bold')
        _ = ax_p_dist_lin.set_title("P dist lin", fontweight='bold')

        _ = ax_p_head_sqr.set_xlabel("Time", fontweight='bold')
        _ = ax_p_head_sqr.set_ylabel("P(G | xi)", fontweight='bold')
        _ = ax_p_head_sqr.set_title("P head sqr", fontweight='bold')

        _ = ax_p_head_lin.set_xlabel("Time", fontweight='bold')
        _ = ax_p_head_lin.set_ylabel("P(G | xi)", fontweight='bold')
        _ = ax_p_head_lin.set_title("P head lin", fontweight='bold')

        ax_p_dist_exp.set_ylim(0, 1)
        ax_p_dist_sqr.set_ylim(0, 1)
        ax_p_dist_lin.set_ylim(0, 1)
        ax_p_head_sqr.set_ylim(0, 1)
        ax_p_head_lin.set_ylim(0, 1)

        #####

        _ = ax2.set_xlabel("Time", fontweight='bold')
        _ = ax2.set_ylabel("Legibility", fontweight='bold')
        _ = ax2.set_title("Legibility according to old", fontweight='bold')
        ax2.legend() #loc="upper left")
        ax2.set_ylim([-0.05, 1.05])

        _ = ax3.set_xlabel("Time", fontweight='bold')
        _ = ax3.set_ylabel("Stage Cost", fontweight='bold')
        _ = ax3.set_title("Stage cost during path", fontweight='bold')
        # ax3.legend() #loc="upper left")

        _ = ax4.set_xlabel("Time", fontweight='bold')
        _ = ax4.set_ylabel("Term Cost", fontweight='bold')
        _ = ax4.set_title("Term cost during path", fontweight='bold')
        # ax4.legend() #loc="upper left")

        _ = ax7.set_xlabel("Visibility", fontweight='bold')
        _ = ax7.set_ylabel("Percent", fontweight='bold')
        _ = ax7.set_title("Vis over path", fontweight='bold')
        # ax7.legend() #loc="upper left")
        
        ax2.grid(False)
        ax3.grid(False)
        ax4.grid(False)
        ax7.grid(False)

        # F = ax6 = Graph of U magnitude
        ts = np.arange(len(us)) * self.dt
        u_mags = [np.linalg.norm(vector) for vector in us]

        _ = ax6.plot(ts, u_mags, lw=2, color='black',)
        _ = ax6.set_xlabel("time (s)", fontweight='bold')
        _ = ax6.set_ylabel("Magnitude of U", fontweight='bold')
        _ = ax6.set_title("Magnitude of U Over Path", fontweight='bold')
        # ax6.legend()

        if False:
            plt.plot(xs, ys, 'o--', lw=2, color='black', label="path", markersize=3)
            plt.plot(gx, gy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor="green", lw=0, label="goals")
            plt.plot(sx, sy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor="grey", lw=0, label="start")
            _ = plt.xlabel("X", fontweight='bold')
            _ = plt.ylabel("Y", fontweight='bold')
            _ = plt.title("Path through space" + blurb, fontweight='bold')
            plt.legend(loc="upper left")
            # plt.xlim([xmin, xmax])
            # plt.ylim([ymin, ymax])
            if FLAG_SHOW_IMAGE_POPUP:
                plt.show()
            plt.clf()

        if False:
            ts = np.arange(len(us)) * self.dt
            u_mags = [np.linalg.norm(vector) for vector in us]

            _ = plt.plot(ts, u_mags, lw=2, color='black',)
            _ = plt.xlabel("time (s)", fontweight='bold')
            _ = plt.ylabel("Magnitude of U", fontweight='bold')
            _ = plt.title("Magnitude of U Over Path", fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.get_export_label() + 'u_graph.png')
            if FLAG_SHOW_IMAGE_POPUP:
                plt.show()
            plt.clf()

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            for i in range(len(us)):
                # print("STAGE COSTS")
                print("xs,\t\t us,\t\t tcs,\t\t scs \t at " + str(i))
                print(str(xs[i]) + "\t" + str(us[i]) + "\t" + str(tcs[i]) + "\t" + str(scs[i]))

            # print("TERM COSTS")

        # _ = plt.plot(J_hist)
        # _ = plt.xlabel("Iteration")
        # _ = plt.ylabel("Total cost")
        # _ = plt.title("Total cost-to-go")
        # if FLAG_SHOW_IMAGE_POPUP:
        #      plt.show()
        # plt.clf()

        plt.tight_layout()
        plt.savefig(self.get_export_label(dash_folder) + '-overview.png')
        if FLAG_SHOW_IMAGE_POPUP:
            plt.show()
        plt.clf()


    # def goal_efficiency_through_path(self, start, goal, path, terminal=False):
    #     for i in path:
    #         J = np.log(goal_component) - np.log(log_sum)
    #     return J

    def stage_cost(self, x, u, i, terminal=False):
        print("DOING STAGE COST")
        start   = self.start
        goal    = self.target_goal

        x = np.array(x)
        J = self.goal_efficiency_through_point_relative(start, goal, x, terminal)
        return J


    # def l(self, x, u, i, terminal=False):
    #     """Instantaneous cost function.
    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size]. None if terminal.
    #         i: Current time step.
    #         terminal: Compute terminal cost. Default: False.
    #     Returns:
    #         Instantaneous cost (scalar).
    #     """
    #     Q = self.Q_terminal if terminal else self.Q
    #     R = self.R
    #     x_diff = x - self.x_path[i]
    #     squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

    #     if terminal:
    #         return squared_x_cost

    #     u_diff = u - self.u_path[i]
    #     return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

    # def l_x(self, x, u, i, terminal=False):
    #     """Partial derivative of cost function with respect to x.
    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size]. None if terminal.
    #         i: Current time step.
    #         terminal: Compute terminal cost. Default: False.
    #     Returns:
    #         dl/dx [state_size].
    #     """
    #     Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
    #     x_diff = x - self.x_path[i]

    #     val = x_diff.T.dot(Q_plus_Q_T)

    #     if self.FLAG_DEBUG_J:
    #         print("J_x")
    #         print(val)

    #     return val

    # def l_u(self, x, u, i, terminal=False):
    #     """Partial derivative of cost function with respect to u.
    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size]. None if terminal.
    #         i: Current time step.
    #         terminal: Compute terminal cost. Default: False.
    #     Returns:
    #         dl/du [action_size].
    #     """
    #     if terminal:
    #         return np.zeros_like(self.u_path)

    #     u_diff = u - self.u_path[i]
    #     val = u_diff.T.dot(self._R_plus_R_T)

    #     if self.FLAG_DEBUG_J:
    #         print("J_u")
    #         print(val)

    #     return val

    # def l_xx(self, x, u, i, terminal=False):
    #     """Second partial derivative of cost function with respect to x.
    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size]. None if terminal.
    #         i: Current time step.
    #         terminal: Compute terminal cost. Default: False.
    #     Returns:
    #         d^2l/dx^2 [state_size, state_size].
    #     """
    #     val = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        
    #     if self.FLAG_DEBUG_J:
    #         print("J_xx")
    #         print(val)

    #     return val

    # def l_ux(self, x, u, i, terminal=False):
    #     """Second partial derivative of cost function with respect to u and x.
    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size]. None if terminal.
    #         i: Current time step.
    #         terminal: Compute terminal cost. Default: False.
    #     Returns:
    #         d^2l/dudx [action_size, state_size].
    #     """
    #     val = np.zeros((self.R.shape[0], self.Q.shape[0]))
        
    #     if self.FLAG_DEBUG_J:
    #         print("J_ux")
    #         print(val)

    #     return val

    # def l_uu(self, x, u, i, terminal=False):
    #     """Second partial derivative of cost function with respect to u.
    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size]. None if terminal.
    #         i: Current time step.
    #         terminal: Compute terminal cost. Default: False.
    #     Returns:
    #         d^2l/du^2 [action_size, action_size].
    #     """
    #     if terminal:
    #         return np.zeros_like(self.R)

    #     val = self._R_plus_R_T

    #     if self.FLAG_DEBUG_J:
    #         print("J_uu")
    #         print(val)

    #     return val
