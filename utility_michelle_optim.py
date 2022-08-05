import numpy as np
from sympy import *
import copy
from autograd import grad, jacobian

nx = 2 
nu = 2



Q = np.identity(2)
R = np.identity(2)
Qf = np.identity(2) * 10

dt = 0.025 
N = 61


def d(x1, y1, x2, y2):
    c = np.sqrt((x2 - x2)*(x2 - x1) + (y2 - y1)*(y2 - y1))
    return c

def dynamics(x,u):
    A = Matrix([1,0],[0,1])
    B = Matrix([1,0],[0,1])

    xnext = A * x + B*u
    return xnext

# https://stackoverflow.com/questions/49553006/compute-the-jacobian-matrix-in-python
def rk4(x, u, dt):
    # rk4 for integration
    k1,k2,k3,k4=symbols("k1,k2,k3,k4")

    k1 = dt*dynamics(x, u)
    k2 = dt*dynamics(x + k1/2,u)
    k3 = dt*dynamics(x + k2/2,u)
    k4 = dt*dynamics(x + k3,u)
    print(k1)
    print(k2)
    print(k3)
    print(k4)

    return x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)


# # https://docs.sympy.org/latest/modules/holonomic/operations.html?highlight=kutta
# def rk4():
# 	#Runge-Kutta 4th order on e^x from 0.1 to 1. Exact solution at 1 is 2.71828182845905
# 	HolonomicFunction(Dx - 1, x, 0, [1]).evalf(r)


# Update call signs
def dynamics_jacobians(x, u, dt):
    # Example usage
    # X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
    # Y = Matrix([rho, phi])
    # X.jacobian(Y)
    x = Symbol('x')
    u = Symbol('u')
    _x = Symbol('_x')
    _u = Symbol('_u')

    rk4_A = rk4(_x,u,dt)
    rk4_B = rk4(x,_u,dt)

    # returns the discrete time dynamics jacobians
    # A = jacobian(_x -> rk4_A, x)
    # B = jacobian(_u -> rk4_B, u)
    rk4_A.jacobian(x)
    rk4_B.jacobian(u)

    return A,B


def stage_cost(x, u, xref, uref, start, goal1, all_goals, nongoal_scale):
    # Legibility LQR cost at each knot point (depends on both x and u)    

    goal_idx = 1
    
    # ' = .T
    J_g1 = (start-goal1).T * Q * (start-goal1) - (start-x).T * Q * (start-x) +  - (x-goal1).T * Q * (x-goal1) 
    J_g1 *= 0.5

    log_sum = 0
    for i in range(len(all_goals)):
        goal = all_goals[i]
        scale = 1
        if goal != goal1:
            scale = nongoal_scale

        n = - ((start-x).T * Q * (start-x) + 5) - ((x-goal).T * Q * (x-goal) +10)
        d = (start-goal).T*Q*(start-goal)
        log_sum += (exp(n )/exp(d)) * scale
    
    
    J = J_g1 - log(log_sum)

    J *= -1
    J += 0.5 *  (u-uref).T * R * (u-uref)

    return J


def term_cost(x, xref):
    diff = (x-xref)
    # LQR terminal cost (depends on just x)
    J = 0.5*diff.T * Qf * diff
    return (J*1000)


def trajectory_cost(X, U, Xref, Uref, start, goal1, all_goals, nongoal_scale):
    # calculate the cost of a given trajectory 
    N_len = length(Xref)

    J = term_cost(X[N_len], Xref[N_len])
    for i in range(len(Xref)-1):
        xref = Xref[i]
        uref = Uref[i]
        x = X[i]
        u = U[i]
        J = J + stage_cost(x,u,xref,uref, start, goal1, all_goals, nongoal_scale)
    return J
        
def stage_cost_expansion(x,u,xref,uref, start, goal1, all_goals, nongoal_scale):
    # if the stage cost function is J, return the following derivatives:
    # ∇²ₓJ,  ∇ₓJ, ∇²ᵤJ, ∇ᵤJ

    # Jxx = FD.hessian(dx -> stage_cost(dx,u,xref,uref, start, goal1, all_goals, nongoal_scale), x)
    # Jx = FD.gradient(dx -> stage_cost(dx,u,xref,uref, start, goal1, all_goals, nongoal_scale), x)
    dx = stage_cost(dx,u,xref,uref, start, goal1, all_goals, nongoal_scale)
    Jxx	= dx.hessian(x)
    Jx 	= dx.gradient(x)

        
    du = stage_cost(x,du,xref,uref, start, goal1, all_goals, nongoal_scale)

    Juu	= du.hessian(u)
    Ju 	= du.gradient(u)
        
    return Jxx, Jx, Juu, Ju

def term_cost_expansion(x, xref):
    # if the terminal cost function is J, return the following derivatives:
    # ∇²ₓJ,  ∇ₓJ
	# Jxx = FD.hessian(dx -> term_cost(dx, xref), x)
	# Jx = FD.gradient(dx -> term_cost(dx, xref), x)

    print(x)
    print(xref)
    print("~~~~~~~~~~")

    # lil_x	= Symbol('x')
    dx 		= term_cost(x, xref)

    print('dx ->')
    print(dx)
    print('x ->')
    print(x)

    Jxx 	= hessian(dx, x)
    Jx 		= gradient(dx, x)
    
    return Jxx, Jx


def backward_pass(X, U, Xref, Uref, start, goal1, all_goals, nongoal_scale):
    # P = [zeros(nx,nx) for i = 1:N]     # cost to go quadratic term
    # p = [zeros(nx) for i = 1:N]        # cost to go linear term 
    # d = [zeros(nu)*NaN for i = 1:N-1]  # feedforward control
    # K = [zeros(nu,nx) for i = 1:N-1]   # feedback gain
    del_J = 0.0                           # expected cost decrease

    P = make_array_copying(zeros(nx,nx), N)     	# cost to go quadratic term
    p = make_array_copying(zeros(nx), N)			# cost to go linear term 
    d = make_array_copying((ones(nu)*nan), (N-1))  	# feedforward control
    K = make_array_copying(zeros(nu,nx), (N-1))   	# feedback gain
    
    end = -1

    # print(X)
    # print(Xref)

    print(X.shape)
    print(Xref.shape)
    X_last 		= X[:, -1]
    Xref_last 	= Xref[:, -1]

    Jxx_term, Jx_term = term_cost_expansion(X_last, Xref_last)

    p[end] = Jx_term
    P[end] = Jxx_term
    
    for i in range(N-1)[::-1]:
        Jxx, Jx, Juu, Ju = stage_cost_expansion(X[i],U[i],Xref[i],Uref[i], start, goal1, all_goals, nongoal_scale)  
        
        A, B = dynamics_jacobians(X[i], U[i], dt)
        
        gx = Jx + A.T * p[i+1]
        gu = Ju + B.T * p[i+1]
        
        Gxx = Jxx + A.T * P[i+1] *A
        Guu = Juu + B.T * P[i+1] *B
        
        Gxu = A.T * P[i+1] *B
        Gux = B.T * P[i+1] *A
        
        # backslash operator divides the argument on its right by the one on its left, commonly used to solve matrix equations
        K_i = Guu / Gux 	# Guu\ Gux
        d_i = Guu / gu  	# Guu\ gu

        # slight tweak from the julia
        P_i = Gxx + K_i.T * Guu*K_i - Gxu*K_i - K_i.T*Gux
        p_i = gx - K_i.T * gu + K_i.T * Guu*d_i - Gxu*d_i
        
        K[i] += K_i
        d[i] += d_i
        P[i] += P_i
        p[i] += p_i
        del_J +=  (gu.T * d_i) 
        
    return d, K, P, del_J

def forward_pass(X,U,Xref,Uref,K,d,del_J, start, goal1, all_goals, nongoal_scale, max_linesearch_iters = 10):
    Xn = copy.deepcopy(X)
    Un = copy.deepcopy(U)
    Jn = NaN
    α = 1.0
    
    n_iters = 0
    c = 0.5
    
    X_prev = copy.deepcopy(X)
    U_prev = copy.deepcopy(U)
    J = trajectory_cost(X,U,Xref,Uref, start, goal1, all_goals, nongoal_scale)
    while n_iters <= max_linesearch_iters:
        for k in range(N-1):
            u_k = U_prev[k] - α * d[k] - K[k]*(Xn[k]-X_prev[k])
            
            x_k = rk4(Xn[k],u_k, dt)
            
            Xn[k+1] = x_k

            Un[k] += u_k
        
        Jn = trajectory_cost(Xn,Un,Xref,Uref, start, goal1, all_goals, nongoal_scale)
        
        if Jn < J - 0.01* α * del_J:
            break
        α = c * α
        n_iters += 1

    return Xn, Un, Jn, α


def make_array_copying(x, N):
	# print(x.shape[0])
	new_arr = zeros(x.shape[0], N)
	# print(new_arr)
	for i in range(N):
		new_arr[:][i] = x

	# print(new_arr)
	return new_arr


def iLQR(x0,U,Xref,Uref, start, goal1, all_goals, nongoal_scale, atol=1e-5, max_iters = 100, verbose = true):
    # X = [copy(x0) for i = 1:N]
    # U = deepcopy(U)
    # K = [zeros(nu,nx) for i = 1:N-1]
    # P = [zeros(nx,nx) for i = 1:N]

    # X,U,K,P = symbols("X,U,K,P")

    # Most useful overview I found is here
    # https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/
    print("iLQR")

    # States, starting with the initial state
    X = make_array_copying(x0, N)
    # Control sequence to be tested
    U = Matrix(copy.deepcopy(U))


    K = make_array_copying(zeros(nu,nx), N-1)
    

    P = make_array_copying(zeros(nx,nx), N)
    
    # We keep going
    iterator = -1
    is_complete = False

    while iterator < max_iters and not is_complete:
        iterator += 1
        print("Doing backward pass")
        d, K, P, del_J = backward_pass(X, U, Xref, Uref, start, goal1, all_goals, nongoal_scale)
    
        print("Doing forward pass")
        X, U, J, α = forward_pass(X, U, Xref, Uref, K, d, del_J, start, goal1, all_goals, nongoal_scale)

        if max(norm(d)) < atol:
            is_complete = True
        else:
            print(norm(d))
            print(max(norm(d)))

    return X,U,K,P,iterator


def trial_1():
	true_goal = [8.0, 2.0]
	true_goal = Matrix(true_goal)
	
	goal2 = [2.0, 1.0]
	goal2 = Matrix(goal2)

	goal3 = [4.0, 1.0]
	goal3 = Matrix(goal3)
	
	start = [0.0, 0.0]
	start = Matrix(start)

	true_goal = [4.0, 2.0]
	true_goal = Matrix(true_goal)
	goal3 = [1.0, 3.0]
	goal3 = Matrix(goal3)
	    
	all_goals = [true_goal, goal3]


	x0 = Matrix([0.0,0.0])
	xgoal = true_goal
	Xrefline = make_array_copying(copy.copy(xgoal), N)
	Urefline = make_array_copying(Matrix([0.0,0.0]), N-1)

	Xline, Uline, Kline, Pline, iterline = iLQR(x0,Urefline,Xrefline,Urefline, start, true_goal, all_goals, 50)
	print("done")


	xvals_leg = [Xline[:][1]]
	yvals_leg = [Xline[:][2]]

	p = plot()

	print(xvals_leg)
	print(yvals_leg)

	# scatter!([start[1]], [start[2]], label="start")
	# for i=1:length(all_goals)
	#     goal = all_goals[i]
	#     scatter!([goal[1]], [goal[2]], label=string("Goal ", i) )
	# end

	# plot!(xvals_leg, yvals_leg, legend=:topleft, label="legible path to goal 1")

	# xlabel!("X")
	# ylabel!("Y")

	# title!("Legible Motion with ILQR")

def main():
	x0 = [0.0, 0.0]						# initial state
	xgoal = [6.0, 5.0]                  # goal state

	trial_1()



if __name__ == "__main__":
	main()






