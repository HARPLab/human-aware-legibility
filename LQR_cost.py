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

class LegiblePathQRCost(FiniteDiffCost):
    FLAG_DEBUG_J = False

    """Quadratic Regulator Instantaneous Cost for trajectory following."""
    def __init__(self, Q, R, x_path, u_path, start, target_goal, goals, Q_terminal=None):
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

        state_size = self.Q.shape[0]
        action_size = self.R.shape[0]
        path_length = self.x_path.shape[0]

        x_eps = .05
        u_eps = .05

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


    # TODO ada why does this always come out to be 0???
    def term_cost(self, x, i):
        start = self.start
        goal1 = self.target_goal
        
        Qf = self.Q_terminal
        R = self.R

        # x_diff = (x - xref)
        x_diff = x - self.x_path[i]
        squared_x_cost = x_diff.T.dot(Qf).dot(x_diff)

        all_goals = self.goals

        terminal_cost = squared_x_cost

        # somehow this governs how far is explored
        terminal_coeff = 1000.0
        terminal_cost = terminal_cost * terminal_coeff

        # Once we're at the goal, the terminal cost is 0
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
            return self.term_cost(x, i)
        else:
            term_cost = self.term_cost(x, i)

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

    def get_total_stage_cost(self, start, goal, x, u, i, terminal):
        N = self.u_path.shape[0]
        u_diff = u - self.u_path[i]
        R = self.R

        stage_costs = u_diff.T.dot(R).dot(u_diff)

        for i in range(N):
            stage_costs = stage_costs + self.michelle_stage_cost(start, goal, x, u, i, terminal)

            # stage_costs = stage_costs + self.goal_efficiency_through_point_relative(start, goal, x, terminal)

        return stage_costs



    def michelle_stage_cost(self, start, goal, x, u, i, terminal=False):
        Q = self.Q_terminal if terminal else self.Q
        R = self.R

        all_goals = self.goals

        goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        u_diff = np.array(u) - self.u_path[i]

        a = (goal_diff.T).dot(Q).dot((goal_diff))
        b = (start_diff.T).dot(Q).dot((start_diff))
        c = (togoal_diff.T).dot(Q).dot((togoal_diff))

        J_g1 = a - b - c
        # print("J_g1")
        # print(J_g1)

        # this_goal = self.goal_efficiency_through_point(start, x, goal)

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

        u_diff_val = (u_diff).dot(R).dot(u_diff).T
        # needs a smaller value of this u_diff_val in order to reach all the way to the goal
        u_diff_val = .1 * (u_diff_val)

        J *= -1
        # print("J_initial")
        # print(J)
        # print("u_diff_val")
        # print(u_diff_val)

        J += u_diff_val

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
