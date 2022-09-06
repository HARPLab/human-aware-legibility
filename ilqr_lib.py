#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import os
import sys
module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt

import theano.tensor as T

from ilqr import iLQR
from ilqr.cost import QRCost
from ilqr.dynamics import constrain
from ilqr.examples.pendulum import InvertedPendulumDynamics
from ilqr.dynamics import FiniteDiffDynamics, BatchAutoDiffDynamics, tensor_constrain

from LQR_cost import LegiblePathQRCost

class NavigationDynamics(FiniteDiffDynamics):

    _state_size  = 2
    _action_size = 2

    def f(self, x, u, i):
        # # Constrain action space.
        # if constrain:
        #     u = tensor_constrain(u, min_bounds, max_bounds)

        # Moving a square
        A = np.eye(2) #np.asarray([[1, 0], [0, 1]])
        B = np.eye(2)

        v0 = A*x

        # print("A and x shape")
        # print(A.shape)
        # print(x.shape)
        # print(v0.shape)
        # print("Ax")
        # print(A)
        # print(x)

        v1 = B.dot(u)
        v0 = A.dot(x)

        # print(v0)
        
        #(2,2) * (2, 1) = (2,1)
        xnext = v0 + v1 #A * x + B*u
        # print(xnext)
        # print(xnext.shape)
        return xnext

    """Inverted pendulum auto-differentiated dynamics model."""
    def __init__(self,
                 dt,
                 constrain=True,
                 min_bounds=-1.0,
                 max_bounds=1.0,
                 m=1.0,
                 l=1.0,
                 g=9.80665,
                 **kwargs):
        """Constructs an InvertedPendulumDynamics model.
        Args:
            dt: Time step [s].
            constrain: Whether to constrain the action space or not.
            min_bounds: Minimum bounds for action [N m].
            max_bounds: Maximum bounds for action [N m].
            m: Pendulum mass [kg].
            l: Pendulum length [m].
            g: Gravity acceleration [m/s^2].
            **kwargs: Additional key-word arguments to pass to the
                BatchAutoDiffDynamics constructor.
        Note:
            state: [x, y]
            action: [torque]
            theta: 0 is pointing up and increasing counter-clockwise.
        """
        self.constrained = constrain
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

            # sin_theta = x[..., 0]
            # cos_theta = x[..., 1]
            # theta_dot = x[..., 2]
            # torque = u[..., 0]

            # # Deal with angle wrap-around.
            # theta = T.arctan2(sin_theta, cos_theta)

            # # Define acceleration.
            # theta_dot_dot = -3.0 * g / (2 * l) * T.sin(theta + np.pi)
            # theta_dot_dot += 3.0 / (m * l**2) * torque

            # next_theta = theta + theta_dot * dt

            # return T.stack([
            #     T.sin(next_theta),
            #     T.cos(next_theta),
            #     theta_dot + theta_dot_dot * dt,
            # ]).T

        super(NavigationDynamics, self).__init__(self.f, 2, 2)

    @classmethod
    def augment_state(cls, state):
        """Augments angular state into a non-angular state by replacing theta
        with sin(theta) and cos(theta).
        In this case, it converts:
            [theta, theta'] -> [sin(theta), cos(theta), theta']
        Args:
            state: State vector [reducted_state_size].
        Returns:
            Augmented state size [state_size].
        """
        return state

        # if state.ndim == 1:
        #     theta, theta_dot = state
        # else:
        #     theta = state[..., 0].reshape(-1, 1)
        #     theta_dot = state[..., 1].reshape(-1, 1)

        # return np.hstack([np.sin(theta), np.cos(theta), theta_dot])

    @classmethod
    def reduce_state(cls, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.
        In this case, it converts:
            [sin(theta), cos(theta), theta'] -> [theta, theta']
        Args:
            state: Augmented state vector [state_size].
        Returns:
            Reduced state size [reducted_state_size].
        """
        return state

        # if state.ndim == 1:
        #     sin_theta, cos_theta, theta_dot = state
        # else:
        #     sin_theta = state[..., 0].reshape(-1, 1)
        #     cos_theta = state[..., 1].reshape(-1, 1)
        #     theta_dot = state[..., 2].reshape(-1, 1)

        # theta = np.arctan2(sin_theta, cos_theta)
        # return np.hstack([theta, theta_dot])


# # Inverted Pendulum Problem
# 
# The state and control vectors $\textbf{x}$ and $\textbf{u}$ are defined as follows:
# 
# $$
# \begin{equation*}
# \textbf{x} = \begin{bmatrix}
#     \theta & \dot{\theta}
#     \end{bmatrix}
# \end{equation*}
# $$
# 
# $$
# \begin{equation*}
# \textbf{u} = \begin{bmatrix}
#     \tau
#     \end{bmatrix}
# \end{equation*}
# $$
# 
# The goal is to swing the pendulum upright:
# 
# $$
# \begin{equation*}
# \textbf{x}_{goal} = \begin{bmatrix}
#     0 & 0
#     \end{bmatrix}
# \end{equation*}
# $$
# 
# In order to deal with potential angle wrap-around issues (i.e. $2\pi = 0$), we
# augment the state as follows and use that instead:
# 
# $$
# \begin{equation*}
# \textbf{x}_{augmented} = \begin{bmatrix}
#     \sin\theta & \cos\theta & \dot{\theta}
#     \end{bmatrix}
# \end{equation*}
# $$
# 
# **Note**: The torque is constrained between $-1$ and $1$. This is achieved by
# instead fitting for unconstrained actions and then applying it to a squashing
# function $\tanh(\textbf{u})$. This is directly embedded into the dynamics model
# in order to be auto-differentiated. This also means that we need to apply this
# transformation manually to the output of our iLQR at the end.

# In[22]:


# In[4]:


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = dynamics.reduce_state(xs[-1])
    print("iteration", iteration_count, info, J_opt, final_state)


# In[1]:

dt = .025
N = 61

x0_raw          = [0.0, 0.0]    # initial state
x_goal_raw      = [6.0, 5.0]

true_goal       = [8.0, 2.0]
goal2           = [2.0, 1.0]
goal3           = [4.0, 1.0]
start           = [0.0, 0.0]

true_goal       = [4.0, 2.0]
goal3           = [1.0, 3.0]
    
all_goals   = [true_goal, goal3]
bounds0     = [0.0,     0.0]
bounds1     = [10.0,    10.0]

x = T.dscalar("x")
u = T.dscalar("u")
t = T.dscalar("t")

# dynamics = AutoDiffDynamics(f, [x], [u], t)
dynamics = NavigationDynamics(dt)

# Note that the augmented state is not all 0.
x0 = dynamics.augment_state(np.array(x0_raw)).T
x_goal = dynamics.augment_state(np.array(x_goal_raw)).T

# Q = np.eye(dynamics.state_size)
# Q[0, 1] = Q[1, 0] = pendulum_length
# Q[0, 0] = Q[1, 1] = pendulum_length**2
# Q[2, 2] = 0.0
# Q_terminal = 100 * np.eye(dynamics.state_size)
# R = np.array([[0.1]])

x_T = N
Xrefline = np.tile(x_goal_raw, (N+1, 1))
Xrefline = Xrefline

#[np.array(x_goal_raw)] * N
# upath3=np.full((nos,1),10)
Urefline = np.tile(x0_raw, (N, 1))
Urefline = Urefline

# Q = np.eye(dynamics.state_size)
# R = 0.1 * np.eye(dynamics.action_size)
Q = np.identity(2)
R = np.identity(2)
Qf = np.identity(2) * 10

print(Xrefline.shape)
# Q, R, x_path, u_path = None, Q_terminal=None
# Q, R, x_path, u_path=None, Q_terminal=None
cost = LegiblePathQRCost(Q, R, Xrefline, Urefline, all_goals)

# cost = PathQRCost(Q, R, traj, us_init)


# x_dot = (dt * t - u) * x**2
# f = T.stack([x + x_dot * dt])

us_init = np.random.uniform(0, 1, (N, dynamics.action_size))
ilqr = iLQR(dynamics, cost, N)


print(Xrefline.shape)
print(Urefline.shape)
# print(Xrefline)
# print(Urefline)

J_hist = []
xs, us = ilqr.fit(x0_raw, us_init, n_iterations=N, on_iteration=on_iteration)
# xs, us = ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)


# # Reduce the state to something more reasonable.
# xs = dynamics.reduce_state(xs)

# # Constrain the actions to see what's actually applied to the system.
# us = constrain(us, dynamics.min_bounds, dynamics.max_bounds)

# # Constrain the inputs.
# min_bounds = np.array([0.0, 0.0, 0.0])
# max_bounds = np.array([61.0, 10.0, 10.0])
# u_constrained_inputs = tensor_constrain(us, min_bounds, max_bounds)

# # Constrain the solution.
# xs, us_unconstrained = ilqr.fit(x0, us_init)
# us = constrain(us_unconstrained, min_bounds, max_bounds)

t = np.arange(N) * dt
theta = np.unwrap(xs[:, 0])  # Makes for smoother plots.
theta_dot = xs[:, 1]


_ = plt.plot(theta, theta_dot)
_ = plt.xlabel("theta (rad)")
_ = plt.ylabel("theta_dot (rad/s)")
_ = plt.title("Phase Plot")
plt.show()
plt.clf()

_ = plt.plot(t, us)
_ = plt.xlabel("time (s)")
_ = plt.ylabel("Force (N)")
_ = plt.title("Action path")
plt.show()
plt.clf()

_ = plt.plot(J_hist)
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Total cost")
_ = plt.title("Total cost-to-go")

# n_spline = fn_pathpickle_from_exp_settings(exp_settings) + 'sample-cubic_spline_imposed_tangent_direction.png'
# plt.savefig(fn_spline)
plt.show()
plt.clf()

