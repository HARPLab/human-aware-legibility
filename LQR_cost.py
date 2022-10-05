import os
import sys
module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt
import decimal

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

class LegiblePathQRCost(FiniteDiffCost):
    FLAG_DEBUG_J = False

    """Quadratic Regulator Instantaneous Cost for trajectory following."""
    def __init__(self, Q, R, x_path, u_path, start, target_goal, goals, N, dt, Q_terminal=None):
        """Constructs a QRCost.
        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            x_path: Goal state path [N+1, state_size].
            u_path: Goal control path [N, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
        """

        self.Q = np.array(Q)
        self.R = np.array(R)
        self.x_path = np.array(x_path)

        self.start = np.array(start)
        self.goals = goals
        self.target_goal = target_goal
        self.N = N
        self.dt = dt

        # Create a restaurant object for using those utilities, functions, and print functions
        # dim gives the dimensions of the restaurant
        self.restaurant = resto.Restaurant(resto.TYPE_CUSTOM, tables=[], goals=goals, start=start, observers=[], dim=None)

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


    # How far away is the final step in the path from the goal?
    def term_cost(self, x, i):
        start = self.start
        goal1 = self.target_goal
        
        Qf = self.Q_terminal
        R = self.R

        x_diff = x - self.x_path[i]
        squared_x_cost = x_diff.T.dot(Qf).dot(x_diff)

        terminal_cost = squared_x_cost

        print("term cost squared x cost")
        print(squared_x_cost)

        # We want to value this highly enough that we don't not end at the goal
        terminal_coeff = 1000.0
        terminal_cost = terminal_cost * terminal_coeff

        # Once we're at the goal, the terminal cost is 0
        
        # Attempted fix for paths which do not hit the final mark
        if squared_x_cost > .001:
            terminal_cost *= 1000.0

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
        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        start = self.start
        goal = self.target_goal

        if terminal:
            return self.term_cost(x, self.N)
        else:
            # difference between this step and the end
            print("x, N, x_end_of_path -> inputs and then term cost")
            print(x, self.N, self.x_path[self.N])
            term_cost = self.term_cost(x, i)
            print(term_cost)

        stage_costs = self.get_total_stage_cost(start, goal, x, u, i, terminal)
    
        print("STAGE,\t TERM")
        print(stage_costs, term_cost)

        # term_cost      = decimal.Decimal.ln(decimal.Decimal(term_cost)) 
        # stage_costs    = decimal.Decimal.ln(stage_costs)
        
        print(stage_costs, term_cost)

        total = term_cost + stage_costs

        # print("total stage cost l")
        # print(total)

        return float(total)

    def f(t):
        return 1.0

    def get_total_stage_cost(self, start, goal, x, u, i, terminal):
        N = self.N

        stage_costs = 0.0 #u_diff.T.dot(R).dot(u_diff)

        for j in range(N):
            stage_costs = stage_costs + self.michelle_stage_cost(start, goal, x, u, j, terminal)

            # stage_costs = stage_costs + self.goal_efficiency_through_point_relative(start, goal, x, terminal)

        return stage_costs



    def michelle_stage_cost(self, start, goal, x, u, i, terminal=False):
        Q = self.Q_terminal if terminal else self.Q
        R = self.R

        all_goals = self.goals

        goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        if len(self.u_path) == 0:
            return 0

        # print(self.u_path)
        # print(i)
        # print(len(self.u_path))

        a = (goal_diff.T).dot(Q).dot((goal_diff))
        b = (start_diff.T).dot(Q).dot((start_diff))
        c = (togoal_diff.T).dot(Q).dot((togoal_diff))

        # (start-goal1)'*Q*(start-goal1) - (start-x)'*Q*(start-x) +  - (x-goal1)'*Q*(x-goal1) 
        J_g1 = a - b - c
        J_g1 *= .5

        # print("For point at x -> " + str(x))

        log_sum = 0.0
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

            if goal != alt_goal:
                log_sum += np.exp(n) / np.exp(d)
                # print("Value for alt target goal " + str(alt_goal))
            else:
                # print("Value for our target goal " + str(goal))
                pass
            # print(n + d)
        
        # print("log sum")
        # print(np.log(log_sum))

        # the log on the log sum actually just cancels out the exp
        J = J_g1 - np.log(log_sum)

        if u is None:
            u_diff = 0.0
            u_diff_val = 0.0
        else:
            u_diff = np.array(u) - self.u_path[i]
            u_diff_val = (u_diff).dot(R).dot(u_diff).T
            # needs a smaller value of this u_diff_val in order to reach all the way to the goal
            u_diff_val = .5 * (u_diff_val)

        J *= -1
        J += u_diff_val

        # print("J_initial")
        # print(J)
        # print("u_diff_val")
        # print(u_diff_val)
        # print(J)

        return J

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
            goals_total += sub_goal
    
        return np.log(this_goal) - np.log(goals_total)

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
            print("J_x")
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
            print("J_x")
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

        print("J_xx")
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

        print("J_ux")
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

        print("J_uu")
        print(Q)

        return Q

    def get_legibility_of_path_to_goal(self, verts, goal):
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
            print(str(i) + " out of " + str(len(verts)))
            x = verts[i]

            l = legib.f_legibility(resto_envir, goal, goals, verts, [])
            sc = self.get_total_stage_cost(start, goal, x, u, i, terminal)
            scs.append(sc)

            tc = float(self.term_cost(x, i))
            tcs.append(tc)
            
            ls.append(l)

        return ls, scs, tcs



    def graph_legibility_over_time(self, verts):
        print("GRAPHING LEGIBILITY OVER TIME")
        ts = np.arange(self.N) * self.dt

        xs, ys = zip(*verts)
        gx, gy = zip(*self.goals)
        sx, sy = self.start

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

        ax1.grid(axis='y')
        ax2.grid(axis='y')
        ax3.grid(axis='y')
        ax4.grid(axis='y')
        
        # each set of xs, ys happens at time t
        # we want to find the legibility at time t
        # and graph it
        # ideally even combine it into a graph with the drawing itself

        # Color code the goals for ease of reading graphs

        # Draw the path itself
        ax1.plot(xs, ys, 'o--', lw=2, color='black', label="path", markersize=3)
        ax1.plot(sx, sy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor="grey", lw=0, label="start")
        _ = ax1.set_xlabel("X", fontweight='bold')
        _ = ax1.set_ylabel("Y", fontweight='bold')
        _ = ax1.set_title("Path through space", fontweight='bold')
        ax1.legend(loc="upper left")
        ax1.grid(False)
        # plt.xlim([xmin, xmax])
        # plt.ylim([ymin, ymax])

        # Draw the legibility over time

        goal_colors = ['red', 'blue', 'purple']

        # for each goal, graph legibility
        for j in range(len(self.goals)):
            print(j)
            goal = self.goals[j]
            color = goal_colors[j]

            gx, gy = goal
            ax1.plot(gx, gy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor=color, lw=0) #, label=goal)

            print("Getting legibility for this goal")
            ls, scs, tcs = self.get_legibility_of_path_to_goal(verts, goal)
            ts = np.arange(len(ls)) * self.dt
            print("Got all data")

            print(ts.shape)
            print(len(tcs))
            print(len(scs))
            print(len(ls))

            ax2.plot(ts, ls, 'o--', lw=2, color=color, label=goal, markersize=3)
            # print("plotted ax2")
            ax3.plot(ts, scs, 'o--', lw=2, color=color, label=goal, markersize=3)
            # print("plotted ax3")
            ax4.plot(ts, tcs, 'o--', lw=2, color=color, label=goal, markersize=3)
            # print("plotted ax4")
            # print("Plotted all data")


        _ = ax2.set_xlabel("Time", fontweight='bold')
        _ = ax2.set_ylabel("Legibility", fontweight='bold')
        _ = ax2.set_title("Legibility according to old code during path", fontweight='bold')
        ax2.legend(loc="upper left")

        _ = ax3.set_xlabel("Time", fontweight='bold')
        _ = ax3.set_ylabel("Stage Cost", fontweight='bold')
        _ = ax3.set_title("Stage cost during path", fontweight='bold')
        ax3.legend(loc="upper left")

        _ = ax4.set_xlabel("Time", fontweight='bold')
        _ = ax4.set_ylabel("Term Cost", fontweight='bold')
        _ = ax4.set_title("Term cost during path", fontweight='bold')
        ax4.legend(loc="upper left")

        
        ax2.grid(False)
        ax3.grid(False)
        ax4.grid(False)
        # plt.xlim([xmin, xmax])
        # plt.ylim([ymin, ymax])

        plt.tight_layout()
        plt.show()
        plt.clf()

        print("Showed graphs")

    # def goal_efficiency_through_path(self, start, goal, path, terminal=False):
    #     for i in path:
    #         J = np.log(goal_component) - np.log(log_sum)
    #     return J

    # def stage_cost(self, x, u, i, terminal=False):
    #     print("DOING STAGE COST")
    #     start   = self.start
    #     goal    = self.target_goal

    #     x = np.array(x)
    #     J = self.goal_efficiency_through_point_relative(start, goal, x, terminal)
    #     return J


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
