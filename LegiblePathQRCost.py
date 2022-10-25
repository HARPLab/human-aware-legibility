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

import PathingExperiment as ex

PREFIX_EXPORT = 'experiment_outputs/'

# Base class for all of our legible pathing offshoots
class LegiblePathQRCost(FiniteDiffCost):
    FLAG_DEBUG_J = False
    FLAG_DEBUG_STAGE_AND_TERM = False

    coeff_terminal = 1000000.0
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


    def get_f(self):
        f_label = self.exp.get_f_label()
        print(f_label)

        if f_label is ex.F_ANCA_LINEAR:
            def f(i):
                return self.N - i

        elif f_label is ex.F_VIS:
            def f(i):
                pt = x_path[i]
                # Can I see this point from each observer who is targeted
                return 1.0
        
        else:
            def f(i):
                return 1.0


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
        coeff_terminal = self.coeff_terminal
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
    def l(self, x, u, i, terminal=False):
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
        thresh = 1


        if terminal or abs(i - self.N) < thresh:
            return self.term_cost(x, i)*1000
        else:
            if self.FLAG_DEBUG_STAGE_AND_TERM:
                # difference between this step and the end
                print("x, N, x_end_of_path -> inputs and then term cost")
                print(x, self.N, self.x_path[self.N])
                # term_cost = self.term_cost(x, i)
                print(term_cost)

        term_cost = 0
        stage_costs = self.michelle_stage_cost(start, goal, x, u, i, terminal) * self.f(i)#
    
        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("STAGE,\t TERM")
            print(stage_costs, term_cost)

        # Log of remixes of term and stage cost weightings
        # term_cost      = decimal.Decimal.ln(decimal.Decimal(term_cost)) 
        # stage_costs    = decimal.Decimal.ln(stage_costs)
        # if i < 30:
        #     stage_scale = 200
        #     term_scale = 0.1
        # else:
        #     stage_scale = 10
        #     term_scale = 1
        # stage_scale = max([(self.N - i), 20])
        # term_scale = 100
        # stage_scale = 10
        # term_scale = 1
        # stage_scale = max([self.N-i, 10])
        # stage_scale = abs(self.N-i)
        # term_scale = i/self.N
        # term_scale = 1
        # stage_scale = 50

        scale_term = self.scale_term #0.01 # 1/100
        scale_stage = self.scale_stage #1.5

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

        if len(self.u_path) == 0:
            return 0

        a = (goal_diff.T).dot(Q).dot((goal_diff))
        b = (start_diff.T).dot(Q).dot((start_diff))
        c = (togoal_diff.T).dot(Q).dot((togoal_diff))

        # (start-goal1)'*Q*(start-goal1) - (start-x)'*Q*(start-x) +  - (x-goal1)'*Q*(x-goal1) 
        J_g1 = a - b - c
        J_g1 *= .5

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("For point at x -> " + str(x))
            # print("Jg1 " + str(J_g1))

        log_sum = 0.0
        total_sum = 0.0
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

            this_goal = np.exp(n) / np.exp(d)

            total_sum += this_goal

            if goal != alt_goal:
                log_sum += (1 * this_goal)
                if self.FLAG_DEBUG_STAGE_AND_TERM:
                    # print("Value for alt target goal " + str(alt_goal))
                    print("This is the nontarget goal: " + str(alt_goal) + " -> " + str(this_goal))
            else:
                # print("Value for our target goal " + str(goal))
                # J_g1 = this_goal
                log_sum += this_goal
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

        J = J_g1 - (np.log(total_sum))
        J = -1.0 * J

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("overall J " + str(J))

        J += (0.5 * u_diff.T.dot(R).dot(u_diff))

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
        xmin, xmax, ymin, ymax = 0.0, 0.0, 0.0, 0.0

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

        return xmin - xbuffer, xmax + xbuffer, ymin - ybuffer, ymax + ybuffer

    def get_export_label(self):
        return PREFIX_EXPORT + self.file_id + "/" + self.file_id

    def get_legibility_of_path_to_goal(self, verts, us, goal):
        ls, scs, tcs = [], [], []
        start = self.start
        u = None
        terminal = False

        if len(verts) != self.N + 1:
            print("points in path does not match the solve N")

        resto_envir = self.restaurant
        goals = self.goals

        exp_settings = pipeline.get_default_exp_settings(unique_key="ilqr_verif")

        for i in range(len(verts)):
            # print(str(i) + " out of " + str(len(verts)))
            x = verts[i]

            l = legib.f_legibility(resto_envir, goal, goals, verts[:i], [])
            
            if i < len(us):
                j = len(us) - 1
                u = us[j]
                sc = self.get_total_stage_cost(start, goal, x, u, j, terminal)
            else:
                sc = 0.0

            scs.append(sc)

            tc = float(self.term_cost(x, i))
            tcs.append(tc)
            
            ls.append(l)

        return ls, scs, tcs


    def get_debug_text(self, elapsed_time):
        debug_text_a = "stage scale: " + str(self.scale_stage) + "        "
        debug_text_b = "term scale: " + str(self.scale_term) + "\n"
        debug_text_c = "coeff_terminal: " + str(self.coeff_terminal) + "        "
        debug_text_d = "obstacle scale: " + str(self.scale_obstacle) # + "\n"

        debug_text = debug_text_a + debug_text_b + debug_text_c + debug_text_d

        if elapsed_time is not None:
            # time.strftime("%M:%S:%f", time.gmtime())
            elapsed_time =  "%.2f seconds" % elapsed_time
            debug_text = elapsed_time + "        " + debug_text

        
        return debug_text

    def graph_legibility_over_time(self, verts, us, elapsed_time=None):
        print("GRAPHING LEGIBILITY OVER TIME")
        sys.stdout = open('path_overview.txt','wt')


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
        fig, axes = plt.subplot_mosaic("AB;CD;EE", figsize=(8, 6), gridspec_kw={'height_ratios':[36, 36, 1], 'width_ratios':[1, 1]})
        # fig, axes = plt.subplot_mosaic("AAB;AAC;DEF")
        ax1 = axes['A']
        ax2 = axes['B']
        ax3 = axes['C']
        ax4 = axes['D']
        ax5 = axes['E']
        # ax6 = axes['F']
        # ax7 = axes['G']


        ax5.axis('off')
        debug_text = self.get_debug_text(elapsed_time)
        ax5.annotate(debug_text, (0.5, 0), xycoords='axes fraction', ha="center", va="center", wrap=True, fontweight='bold') #, fontsize=6)

        ax1.grid(axis='y')
        ax2.grid(axis='y')
        ax3.grid(axis='y')
        ax4.grid(axis='y')
        
        ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
        ax4.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))

        # each set of xs, ys happens at time t
        # we want to find the legibility at time t
        # and graph it
        # ideally even combine it into a graph with the drawing itself

        # Color code the goals for ease of reading graphs

        # Draw the path itself
        ax1.plot(xs, ys, 'o--', lw=2, color='black', label="path", markersize=10)
        ax1.plot(sx, sy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor="grey", lw=0, label="start")
        _ = ax1.set_xlabel("X", fontweight='bold')
        _ = ax1.set_ylabel("Y", fontweight='bold')
        _ = ax1.set_title("Path through space", fontweight='bold')


        TABLE_RADIUS = .5
        tables = self.restaurant.get_tables()
        for table in tables:
            table = plt.Circle(table.get_center(), TABLE_RADIUS, color='g', clip_on=False)
            ax1.add_patch(table)

        # Draw the legibility over time

        # Example of how to draw an obstacle on the path
        if False:
            obs = [[.5, 1], [1, 1], [1, .5], [.5, .5], [.5, 1]]
            oxs, oys = zip(*obs)
            ax1.fill(oxs, oys, "c")

        goal_colors = ['red', 'blue', 'purple', 'green']


        target = self.target_goal
        # for each goal, graph legibility
        for j in range(len(self.goals)):
            goal = self.goals[j]
            color = goal_colors[j]

            gx, gy = goal

            if gx == target[0] and gy == target[1]:
                ax1.plot(gx, gy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor=color, lw=0, label="target")
            else:
                ax1.plot(gx, gy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor=color, lw=0) #, label=goal)

            ls, scs, tcs = self.get_legibility_of_path_to_goal(verts, us, goal)
            ts = np.arange(len(ls)) * self.dt

            ax2.plot(ts, ls, 'o--', lw=2, color=color, label=goal, markersize=3)
            # print("plotted ax2")
            ax3.plot(ts, scs, 'o--', lw=2, color=color, label=goal, markersize=3)
            # print("plotted ax3")
            ax4.plot(ts, tcs, 'o--', lw=2, color=color, label=goal, markersize=3)
            # print("plotted ax4")
            # print("Plotted all data")


        ax1.legend() #loc="upper left")
        ax1.grid(False)
        xmin, xmax, ymin, ymax = self.get_window_dimensions_for_envir(self.start, self.goals, verts)
        ax1.set_xlim([xmin, xmax])
        ax1.set_ylim([ymin, ymax])
        # ax1.set_ylim(-5, 25)
        # ax1.set_xlim(-15, 15)

        _ = ax2.set_xlabel("Time", fontweight='bold')
        _ = ax2.set_ylabel("Legibility", fontweight='bold')
        _ = ax2.set_title("Legibility according to old", fontweight='bold')
        ax2.legend() #loc="upper left")
        ax2.set_ylim([0.0, 1.0])

        _ = ax3.set_xlabel("Time", fontweight='bold')
        _ = ax3.set_ylabel("Stage Cost", fontweight='bold')
        _ = ax3.set_title("Stage cost during path", fontweight='bold')
        ax3.legend() #loc="upper left")

        _ = ax4.set_xlabel("Time", fontweight='bold')
        _ = ax4.set_ylabel("Term Cost", fontweight='bold')
        _ = ax4.set_title("Term cost during path", fontweight='bold')
        ax4.legend() #loc="upper left")
        
        ax2.grid(False)
        ax3.grid(False)
        ax4.grid(False)


        plt.tight_layout()
        plt.savefig(self.get_export_label() + '-overview.png')
        plt.show()
        plt.clf()


        if False:
            plt.plot(xs, ys, 'o--', lw=2, color='black', label="path", markersize=3)
            plt.plot(gx, gy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor="green", lw=0, label="goals")
            plt.plot(sx, sy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor="grey", lw=0, label="start")
            _ = plt.xlabel("X", fontweight='bold')
            _ = plt.ylabel("Y", fontweight='bold')
            _ = plt.title("Path through space", fontweight='bold')
            plt.legend(loc="upper left")
            # plt.xlim([xmin, xmax])
            # plt.ylim([ymin, ymax])
            plt.show()
            plt.clf()

        if True:
            ts = np.arange(len(us)) * self.dt
            u_mags = [np.linalg.norm(vector) for vector in us]

            _ = plt.plot(ts, u_mags, lw=2, color='black',)
            _ = plt.xlabel("time (s)", fontweight='bold')
            _ = plt.ylabel("Magnitude of U", fontweight='bold')
            _ = plt.title("Magnitude of U Over Path", fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.get_export_label() + 'u_graph.png')
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
        # plt.show()
        # plt.clf()

        # sys.stdout = open('output.txt','wt')



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
