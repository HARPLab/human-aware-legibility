import numpy as np
import theano.tensor as T
from ilqr.dynamics import FiniteDiffDynamics, tensor_constrain, constrain, BatchAutoDiffDynamics, AutoDiffDynamics


class NavDynamicsAuto(AutoDiffDynamics):

    """Inverted pendulum auto-differentiated dynamics model."""

    def __init__(self,
                 exp,
                 constrain=True,
                 min_bounds=-0.001, #-1, 1
                 max_bounds=0.001,
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
            state: [sin(theta), cos(theta), theta']
            action: [torque]
        """
        k = exp.get_dist_scalar_k()

        self.constrained    = constrain
        self.min_bounds     = min_bounds  #* exp.get_dt() * k * .1 # .3
        self.max_bounds     = max_bounds    # * exp.get_dt() * k * .1 #.3

        print("DYNAM: min max bounds")
        print(self.min_bounds, self.max_bounds)

        self.exp            = exp
        self._has_hessians  = True

        dt = self.exp.get_dt()

        x_x = T.dscalar("x_x")
        x_y = T.dscalar("x_y")
        x_x_prev = T.dscalar("x_x_prev")
        x_y_prev = T.dscalar("x_y_prev")

        u_x = T.dscalar("u_x")
        u_y = T.dscalar("u_y")

        min_bounds, max_bounds = self.min_bounds, self.max_bounds

        # # Constrain action space.
        if False: #self.constrained:
            print("speed limit")
            # print([u[0], u[1]])
            u = tensor_constrain([u_x, u_y], min_bounds, max_bounds)
            # print(u)
            u_x = u[0]
            u_y = u[1]

        f = T.stack([
                x_x + u_x,
                x_y + u_y,
                x_x,
                x_y
            ])

        x_inputs = [x_x, x_y, x_x_prev, x_y_prev]
        u_inputs = [u_x, u_y]
        super(NavDynamicsAuto, self).__init__(f, x_inputs, u_inputs, hessians=True,
                                                **kwargs)

        # super(NavDynamics, self).__init__(f,    state_size=4,
        #                                         action_size=2,
        #                                         **kwargs)


    # def f_x(self, x, u, i):
    #     """Partial derivative of dynamics model with respect to x.

    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size].
    #         i: Current time step.

    #     Returns:
    #         df/dx [state_size, state_size].
    #     """
    #     z = np.hstack([x, u, i])
    #     return self._f_x(*z)

    # def f_u(self, x, u, i):
    #     """Partial derivative of dynamics model with respect to u.

    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size].
    #         i: Current time step.

    #     Returns:
    #         df/du [state_size, action_size].
    #     """
    #     z = np.hstack([x, u, i])
    #     return self._f_u(*z)

    # def f_xx(self, x, u, i):
    #     """Second partial derivative of dynamics model with respect to x.

    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size].
    #         i: Current time step.

    #     Returns:
    #         d^2f/dx^2 [state_size, state_size, state_size].
    #     """
    #     if not self._has_hessians:
    #         raise NotImplementedError

    #     z = np.hstack([x, u, i])
    #     return self._f_xx(*z)

    # def f_ux(self, x, u, i):
    #     """Second partial derivative of dynamics model with respect to u and x.

    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size].
    #         i: Current time step.

    #     Returns:
    #         d^2f/dudx [state_size, action_size, state_size].
    #     """
    #     if not self._has_hessians:
    #         raise NotImplementedError

    #     z = np.hstack([x, u, i])
    #     return self._f_ux(*z)

    # def f_uu(self, x, u, i):
    #     """Second partial derivative of dynamics model with respect to u.

    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size].
    #         i: Current time step.

    #     Returns:
    #         d^2f/du^2 [state_size, action_size, action_size].
    #     """
    #     if not self._has_hessians:
    #         raise NotImplementedError

    #     z = np.hstack([x, u, i])
    #     return self._f_uu(*z)
