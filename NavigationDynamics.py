import numpy as np
from ilqr.dynamics import FiniteDiffDynamics, tensor_constrain, constrain

### CLASS DESCRIBING THE DYNAMICS OF A SIMPLE ROBOT
### MOVING THOUGH SPACE
class NavigationDynamics(FiniteDiffDynamics):

    _state_size  = 2
    _action_size = 2

    x_eps = .1
    u_eps = .1

    def f(self, x, u, i):
        return self.dynamics(x, u)

    # hardcoded test function for being within an obstacle
    def in_object(self, x):

        obstacle = .5, 1.0
        xx, xy = x
        if xx > .5 and xx < 1.0 and xy > .5 and xy < 1.0:
            return True


        return False

    # Combine the existing state with 
    def dynamics(self, x, u, max_u=5.0):
        # # Constrain action space.
        if constrain:
            min_bounds, max_bounds = -1.0 * max_u, max_u
            
            # If we want to constrain movements to manhattan 
            # (straight lines and diagonals)
            # if False:
            #     diff = (max_bounds - min_bounds) / 2.0
            #     mean = (max_bounds + min_bounds) / 2.0
            #     ux = diff * np.tanh(u[0]) + mean
            #     uy = diff * np.tanh(u[1]) + mean
            #     u = ux, uy

            # norm1 = u / np.linalg.norm(u)
            # norm2 = normalize(u[:,np.newaxis], axis=0).ravel()
            # u = norm2 * max_u

            # downscaling the u
            # if np.linalg.norm(x) > max_u:
            #     scalar = max_u / np.linalg.norm(x)
            #     print("SCALING U")
            #     print(scalar)
            #     u = u * scalar
            #     print(u)

            # u = tensor_constrain(u, min_bounds, max_bounds)
            
        dt = self.dt

        # Apply a constraint that limits how much the robot can move per-timestep
        # TODO: apply to overall vector magnitude rather than x and y components alone

        # Moving a square
        A = np.eye(self._state_size)
        B = np.eye(self._action_size)
        v0 = A.dot(x)
        v1 = B.dot(u)
        xnext = v0 + v1     # A*x + B*u

        # Sample code that shows what would happen if we attempted raw obstacle avoidance within dynamics
        if False and self.in_object(xnext):
            # Bounce
            xnext = x

        print("xnext")
        print(str(x) + " -> " + str(xnext))

        return xnext

    """ Original based on inverted pendulum auto-differentiated dynamics model."""
    def __init__(self,
                 dt,
                 constrain=True,
                 min_bounds=-1.0,
                 max_bounds=1.0,
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


        self.dt = dt
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
