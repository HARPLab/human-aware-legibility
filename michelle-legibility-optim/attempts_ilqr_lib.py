    # stage cost
    def l_attempt(self, x, u, i, terminal=False):
        # return self.l_og(x, u, i, terminal)

        # def trajectory_cost(X, U, Xref, Uref, start, goal1, all_goals, nongoal_scale):
        # calculate the cost of a given trajectory 
        # N_len = len(Xref)
        Xref = self.x_path
        Uref = self.u_path

        end_of_path = self.x_path[-1]
        # end_goal    = Xref[N_len]

        # start with the term cost
        term_cost = self.term_cost(end_of_path, -1)

        # J = term_cost(X[N_len],Xref[N_len])
        N = Uref.shape[0]

        print("Uref shape")
        print(N)
        stage_costs = 0

        # currently has a value of about 10 * N steps
        for i in range(N):
            stage_costs = stage_costs + self.stage_cost(x, u, i, terminal=terminal)

        J = term_cost + stage_costs

        print("Total J for stage: term, stage, J")
        print(term_cost, stage_costs, J)

        return J



    def stage_cost_og(self, x, u, i, terminal=False):
        """Instantaneous cost function.
        #     Args:
        #         x: Current state [state_size].
        #         u: Current control [action_size]. None if terminal.
        #         i: Current time step.
        #         terminal: Compute terminal cost. Default: False.
        #     Returns:
        #         Instantaneous cost (scalar).
        #     """

         # NOTE: The terminal cost needs to at most be a function of x and i, whereas
         #  the non-terminal cost can be a function of x, u and i.

        nongoal_scale = 1 #50

        start = self.x_path[0]
        goal1 = self.target_goal

        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        all_goals = self.goals

        if terminal: #ie, u is not None:
        # TERMINAL COST FUNCTION
            return self.term_cost(x, -1)

        xref = self.x_path[i]
        uref = self.u_path[i]
        
        x_diff = x - xref
        u_diff = u - uref
        
        # Find the cost of going to the targeted goal
        # STAGE COST FUNCTION
        goal_diff   = start - goal1
        start_diff  = (start - x)
        togoal_diff = (x - goal1)

        # goal_diff   = np.reshape(goal_diff.T, (-1, 2))
        # start_diff  = np.reshape(start_diff.T, (-1, 2))
        # togoal_diff = np.reshape(togoal_diff.T, (-1, 2))

        a = (goal_diff.T).dot(Q).dot((goal_diff))
        b = (start_diff.T).dot(Q).dot((start_diff))
        c = (togoal_diff.T).dot(Q).dot((togoal_diff))
    
        J_g1 = a - b + - c
        print("Value for goal before others: " + str(J_g1))

        # J_g1 = (np.exp(a - b)/np.exp(c))

        # J_g1 *= 0.5

        # print("J_g1 = legibility for goal 1")
        # print(J_g1)

        # and then also find this ratio for all of the other goals and combine
        print("Finding ratio")
        all_components = []
        goal_component = 0

        log_sum = 0
        for i in range(len(all_goals)):
            goal = all_goals[i]
            # print("goal = " + str(goal))

            scale = 1
            if goal != goal1:
                scale = nongoal_scale

            alt_goal_diff               = (x - goal)
            alt_goal_from_start_diff    = (start - goal)

            n0 = (start_diff.T).dot(Q).dot((start_diff))
            n1 = (alt_goal_diff.T).dot(Q).dot((alt_goal_diff))


            # weight_before = 5
            # weight_after = 10
            weight_before = 0.0
            weight_after = 0.0

            n = - (n0 + weight_before) - (n1 + weight_after)
            d = (alt_goal_from_start_diff).dot(Q).dot((alt_goal_from_start_diff).T)
            d = d
            component = (np.exp(n)/np.exp(d))
            
            all_components.append(component)
            if goal == goal1:
                goal_component = component

            # add weighted value for this component
            log_sum += component * scale
        

        print("RATIO")
        print(all_components)
        # print(J_g1)
        ratio = goal_component / sum(all_components)
        print(ratio)
        if ratio > .5:
            print("Doing the thing!")

        J = np.log(goal_component) - np.log(log_sum)
        # J *= -1
        # J_addition = 0.5 * (u_diff.T.dot(R).dot(u_diff))
        # J += J_addition

        print("J components")
        print((goal_component), (log_sum))
        print(np.log(goal_component), -np.log(log_sum))
        print("stage cost~~~")
        print(J)

        return J


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


    # these are functions from michelle's implementation
    # def rk4(self, x, u, dt):
    #     # rk4 for integration
    #     k1 = dt * self.dynamics(x, u)
    #     k2 = dt * self.dynamics(x + k1/2, u)
    #     k3 = dt * self.dynamics(x + k2/2, u)
    #     k4 = dt * self.dynamics(x + k3, u)
    #     # print("rk4")
    #     # print(x)
    #     # print(u)
    #     # print(dt)
    #     return x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)


    # def dynamics_jacobians(self, x, u, dt):
    #     # returns the discrete time dynamics jacobians
    #     A = self.rk4(0, u, dt) # FD.jacobian(_x -> rk4(_x,u,dt),x)
    #     B = self.rk4(x, 0, dt) #FD.jacobian(_u -> rk4(x,_u,dt),u)
        
    #     return A,B