import numpy as np
import theano.tensor as T
from ilqr.dynamics import FiniteDiffDynamics, tensor_constrain, constrain, BatchAutoDiffDynamics, AutoDiffDynamics


class NavDynamics(BatchAutoDiffDynamics):

    """Inverted pendulum auto-differentiated dynamics model."""

    def __init__(self,
                 exp,
                 constrain=True,
                 min_bounds=-0.5,
                 max_bounds=0.5,
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
        self.constrained    = constrain
        self.min_bounds     = min_bounds * exp.get_dt()
        self.max_bounds     = max_bounds * exp.get_dt()

        self.exp            = exp

        def f(x, u, i):
            min_bounds, max_bounds = self.min_bounds, self.max_bounds

            # Constrain action space.
            if constrain:
                u = tensor_constrain(u, min_bounds, max_bounds)

            x_current_x   = x[..., 0]
            x_current_y   = x[..., 1]
            x_prev_x      = x[..., 2]
            x_prev_y      = x[..., 3]

            u_x = u[..., 0]
            u_y = u[..., 1]

            dt = self.exp.get_dt()

            x_new_x = x_current_x + u_x * dt
            x_new_y = x_current_y + u_y * dt

            return T.stack([
                x_new_x,
                x_new_y,
                x_current_x,
                x_current_y
            ]).T

        super(NavDynamics, self).__init__(f,    state_size=4,
                                                action_size=2,
                                                **kwargs)
