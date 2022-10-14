import os
import sys
module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt
import decimal
import copy

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

from LegiblePathQRCost import LegiblePathQRCost


class OALegiblePathQRCost(LegiblePathQRCost):
    FLAG_DEBUG_J = True

    """Quadratic Regulator Instantaneous Cost for trajectory following."""
    def __init__(self, Q, R, Qf, x_path, u_path, start, target_goal, goals, N, dt, restaurant=None, Q_terminal=None):
        """Constructs a QRCost.
        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            x_path: Goal state path [N+1, state_size].
            u_path: Goal control path [N, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
        """

        LegiblePathQRCost.__init__(
            self, Q, R, Qf, x_path, u_path, start, target_goal, goals, N, dt, restaurant=None, Q_terminal=None
        )


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

        print("term cost squared x cost")
        print(squared_x_cost)

        # We want to value this highly enough that we don't not end at the goal
        # terminal_coeff = 100.0
        terminal_coeff = 1000.0
        terminal_coeff = 1
        terminal_cost = terminal_cost * terminal_coeff

        # Once we're at the goal, the terminal cost is 0
        
        # Attempted fix for paths which do not hit the final mark
        # if squared_x_cost > .001:
        #     terminal_cost *= 1000.0

        return terminal_cost

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
            # difference between this step and the end
            print("x, N, x_end_of_path -> inputs and then term cost")
            print(x, self.N, self.x_path[self.N])
            # term_cost = self.term_cost(x, i)
            term_cost = 0
            print(term_cost)

        stage_costs = self.michelle_stage_cost(start, goal, x, u, i, terminal)
    
        print("STAGE,\t TERM")
        print(stage_costs, term_cost)

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

        term_scale = 0.01 # 1/100
        stage_scale = 1.5
        stage_scale = 2


        
        total = (term_scale * term_cost) + (stage_scale * stage_costs)
        # print("total stage cost l")
        # print(total)

        return float(total)

    def f(t):
        return 1.0

    def get_total_stage_cost(self, start, goal, x, u, i, terminal):
        N = self.N
        R = self.R

        stage_costs = 0.0 #u_diff.T.dot(R).dot(u_diff)
        
        print("u = " + str(u))


        print("Getting stage cost")
        for j in range(i):
            u_diff = u - self.u_path[j]
            stage_costs += (0.5 * u_diff.T.dot(R).dot(u_diff))

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

            # TODO verify if this is pointwise, or pathwise
            n = - (diff_curr.T).dot(Q).dot((diff_curr)) - ((diff_goal).T.dot(Q).dot(diff_goal))
            d = (diff_all).T.dot(Q).dot(diff_all)

            # this_goal = np.exp(n) / np.exp(d)
            this_goal = np.exp(n) / np.exp(d)
            # this_goal = np.abs(this_goal)

            total_sum += this_goal

            if goal != alt_goal:
                log_sum += (1 * this_goal)
                # print("Value for alt target goal " + str(alt_goal))
                print("This is the nontarget goal: " + str(alt_goal) + " -> " + str(this_goal))
            else:
                # print("Value for our target goal " + str(goal))
                # J_g1 = this_goal
                log_sum += this_goal
                print("This is the target goal " + str(alt_goal) + " -> " + str(this_goal))
            # print(n + d) 

        # print("Jg1, total")
        # print(J_g1, total_sum)
        J = J_g1 - (np.log(total_sum))
        # J -= this_goal
        # J += log_sum * alt_goal_multiplier
        # J *= -1

        J = -1.0 * J

        # print("overall J " + str(J))

        J += (0.5 * u_diff.T.dot(R).dot(u_diff))

        return J

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
