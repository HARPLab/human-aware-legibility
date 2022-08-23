import numpy as np
from sympy import *
from sympy.vector import *
import copy
from autograd import grad, jacobian
import tensorflow as tf
import sympy as sympy

nx = 2 
nu = 2



Q = np.identity(2)
R = np.identity(2)
Qf = np.identity(2) * 10

dt = 0.025 
N = 61


def nested_tuple(size, ndims, t=0):
    for i in range(ndims):
        t = (t,)*size

    return t

def compute_hessian(fn, vars):
    gradients = lambda f, v: Matrix([f]).jacobian(v)

    mat = []
    for v1 in vars:
        temp = []
        for v2 in vars:
            # computing derivative twice, first w.r.t v2 and then w.r.t v1
            temp.append(gradients(gradients(fn, v2)[0], v1)[0])
        temp = [cons(0) if t == None else t for t in temp] # tensorflow returns None when there is no gradient, so we replace None with 0
        # print(temp)
        # temp = Matrix.hstack(temp)
        mat.append(temp)
    mat = Matrix(mat)
    return mat

def d(x1, y1, x2, y2):
    c = np.sqrt((x2 - x2)*(x2 - x1) + (y2 - y1)*(y2 - y1))
    return c

def dynamics(x, u):
    A = Matrix([[1,0],[0,1]])
    B = Matrix([[1,0],[0,1]])

    n0 = A * x
    n1 = B * u

    xnext = A*x + B*u
     
    return xnext

# https://stackoverflow.com/questions/49553006/compute-the-jacobian-matrix-in-python
def rk4(x, u, dt):
    # rk4 for integration
    k1,k2,k3,k4 = symbols("k1,k2,k3,k4")

    # print(x.shape)
    # print(u.shape)

    if x.shape[1] > 1:
        x = x.T

    if u.shape[1] > 1:
        u = u.T

    print("x shape, u shape")
    print(x.shape)
    print(u.shape)

    k1 = dt*dynamics(x, u)
    k2 = dt*dynamics(x + k1/2, u)
    k3 = dt*dynamics(x + k2/2, u)
    k4 = dt*dynamics(x + k3, u)
    # print(k1)
    # print(k2)
    # print(k3)
    # print(k4)

    total = x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    return total.T


# # https://docs.sympy.org/latest/modules/holonomic/operations.html?highlight=kutta
# def rk4():
#     #Runge-Kutta 4th order on e^x from 0.1 to 1. Exact solution at 1 is 2.71828182845905
#     HolonomicFunction(Dx - 1, x, 0, [1]).evalf(r)


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

    if x[0] == 1:
        x = x.T
    if u[0] == 1:
        u = u.T

    print(x.shape)
    print(u.shape)
    print(dt.shape)

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
    if x.shape[1] > 1:
        x = x.T

    if u.shape[1] > 1:
        u = u.T

    if uref.shape[1] > 1:
        uref = uref.T

    J_g1 = (start-goal1).T * Q * (start-goal1) - (start-x).T * Q * (start-x) +  - (x-goal1).T * Q * (x-goal1) 
    J_g1 *= 0.5

    log_sum = 0
    for i in range(len(all_goals)):
        goal = all_goals[i]
        scale = 1
        if goal != goal1:
            scale = nongoal_scale

        n0 = (start-x).T * Q * (start-x)
        n1 = (x-goal).T * Q * (x-goal)

        n = - (n0[0,0] + 5) - (n1[0,0] + 10)
        d = (start-goal).T*Q*(start-goal)
        d = d[0,0]

        log_sum += (exp(n )/exp(d)) * scale
    
    J = J_g1 - Matrix([[log(log_sum)]])

    J *= -1

    J += 0.5 *  (u-uref).T * R * (u-uref)
    # J += 0.5 *  (u-uref).T * R * (u-uref)

    return J


def term_cost(x, xref):
    # verify orientation of points before this math
    if x.shape[0] > 1:
        x = x.T

    if xref.shape[0] > 1:
        xref = xref.T

    # print(x, xref)
    diff = (x - xref).T

    # LQR terminal cost (depends on just x)
    J = 0.5*diff.T * Qf * diff
    return (J*1000).as_mutable()

def term_cost_symbolic(x, xref):
    x = Matrix(x)
    xref = Matrix(xref)
    diff = (x - xref)
    # LQR terminal cost (depends on just x)
    J = 0.5*diff.T * Qf * diff
    return (J*1000), [x, xref]


def trajectory_cost(X, U, Xref, Uref, start, goal1, all_goals, nongoal_scale):
    # calculate the cost of a given trajectory 
    # N_len = len(Xref)

    J = term_cost(X[-1, :], Xref[-1, :])

    for i in range(Uref.shape[0]):
        # print(i)
        # print(Xref.shape)
        # print(Uref.shape)
        xref = Xref[i, :]
        uref = Uref[i, :]
        x = X[i, :]
        u = U[i, :]
        J = J + stage_cost(x, u, xref, uref, start, goal1, all_goals, nongoal_scale)
    return J
        
def stage_cost_expansion(x, u, xref, uref, start, goal1, all_goals, nongoal_scale):
    # if the stage cost function is J, return the following derivatives:
    
    # dx = x
    # Jxx = FD.hessian(dx -> stage_cost(dx,u,xref,uref, start, goal1, all_goals, nongoal_scale), x)
    # Jx = FD.gradient(dx -> stage_cost(dx,u,xref,uref, start, goal1, all_goals, nongoal_scale), x)
    dx = stage_cost(x, u, xref,uref, start, goal1, all_goals, nongoal_scale)

    Jxx    = dx.hessian(x)
    Jx     = dx.gradient(x)

        
    du = stage_cost(x,du,xref,uref, start, goal1, all_goals, nongoal_scale)

    Juu    = du.hessian(u)
    Ju     = du.gradient(u)
        
    return Jxx, Jx, Juu, Ju

def term_cost_expansion(x_input_t, xref_t):
    # if the terminal cost function is J, return the following derivatives:
    # delta2xJ,  deltaxJ
    # Jxx = FD.hessian(dx -> term_cost(dx, xref), x)
    # Jx = FD.gradient(dx -> term_cost(dx, xref), x)

    # print(x)
    # print(xref)
    # print("~~~~~~~~~~")
    x0 = Symbol('x0')
    x1 = Symbol('x1')
    x = Symbol('x')
    x = Matrix([x, x])

    x_input     = x_input_t.T
    xref        = xref_t.T

    print("x_ref")
    print(xref)
    print('x ->')
    print(x_input)


    # x_m           = Matrix([x, x]) #Matrix([x0, x1]) #Matrix([x0, x1]) #Matrix(x)
    dx            = term_cost(x_input, xref)

    print('term_cost ->')
    print(dx)

    x = MatrixSymbol('x', 2, 1)
    xr = MatrixSymbol('xr', 2, 1)
    dx_vars = [x, xr]

    # dx = lambdify(dx_vars, dx)

    # take the hessian of term_cost with respect to variable xref, at point x (from my input)
    # dx is actually wrt first input variable x, xref is held constant
    # eval hessian setting first input val to x passed in

    print("dx")
    print(dx)

    dx_symbolic, dx_vars = term_cost_symbolic(x, xr)
    # dx_symbolic = lambdify([x, xr], term_cost, "numpy")

    # 2x2 matrix
    # Note: this can also take additional constraints
    # hessian(fx.as_explicit(), x.as_explicit())
    # print(type(dx_symbolic))

    gradient = lambda f, v: Matrix([f]).jacobian(v).T
    f = dx_symbolic

    Jxx     = compute_hessian(f, [x, xr])
    Jxx     = Jxx.subs([(x,x_input), (xr, xref)])
    #hessian(dx_symbolic, (x, xr)) #x_input) # taken at 0
    Jx      = gradient(f, [x]) #x_input)
    Jx      = Jx.subs([(x,x_input), (xr, xref)])
    
    print("Jx")
    print(Jx)
    print("Jxx")
    print(Jxx)
    print("~")
    print()

    # exit()

    return Jxx, Jx


def backward_pass(X, U, Xrefline, Urefline, start, goal1, all_goals, nongoal_scale):
    # P = [zeros(nx,nx) for i = 1:N]     # cost to go quadratic term
    # p = [zeros(nx) for i = 1:N]        # cost to go linear term 
    # d = [zeros(nu)*NaN for i = 1:N-1]  # feedforward control
    # K = [zeros(nu,nx) for i = 1:N-1]   # feedback gain
    del_J = 0.0                           # expected cost decrease

    empty_P = sympy.zeros(nx,nx)
    P = make_array_copying(empty_P, N)         # cost to go quadratic term
    p = make_array_copying(sympy.zeros(nx, 1), N)            # cost to go linear term 
    d = make_array_copying(sympy.zeros(nu, 1), (N-1))        # feedforward control removed the NaN
    K = make_array_copying(sympy.zeros(nu, nx), (N-1))       # feedback gain

    end = -1

    # print(X.shape)
    # print(Xref.shape)
    X_last              = X[-1, :]
    Xref                = Xrefline[-1, :]

    Jxx_term, Jx_term = term_cost_expansion(X_last, Xref)

    # print(p.shape)
    # print(Jx_term.shape)

    # print(P.shape)
    # print(P[end].shape)

    p[end, 0]      = Jx_term[0,0]
    p[end, 1]      = Jx_term[1,0]
    P[end, :]   = Jxx_term
    
    for i in range(U.shape[0] - 1, 0, -1):
    
        Jxx, Jx, Juu, Ju = stage_cost_expansion(X[i, :], U[i, :], Xrefline[i, :], Urefline[i, :], start, goal1, all_goals, nongoal_scale)  
        
        A, B = dynamics_jacobians(X[i], U[i], dt)
        
        gx = Jx + A.T * p[i+1]
        gu = Ju + B.T * p[i+1]
        
        Gxx = Jxx + A.T * P[i+1] *A
        Guu = Juu + B.T * P[i+1] *B
        
        Gxu = A.T * P[i+1] *B
        Gux = B.T * P[i+1] *A
        
        # backslash operator divides the argument on its right by the one on its left, commonly used to solve matrix equations
        K_i = Guu / Gux     # Guu\ Gux
        d_i = Guu / gu      # Guu\ gu
        
        # slight tweak from the julia
        P_i = Gxx + K_i.T * Guu*K_i - Gxu*K_i - K_i.T*Gux
        p_i = gx - K_i.T * gu + K_i.T * Guu*d_i - Gxu*d_i
        
        K[i] += K_i
        d[i] += d_i
        P[i] += P_i
        p[i] += p_i
        del_J +=  (gu.T * d_i) 
        
    return d, K, P, del_J

def forward_pass(X, U, Xref, Uref, K, d, del_J, start, goal1, all_goals, nongoal_scale, max_linesearch_iters = 10):
    Xn = copy.deepcopy(X)
    Un = copy.deepcopy(U)
    Jn = np.NaN
    alpha = 1.0
    
    n_iters = 0
    c = 0.5
    
    X_prev = copy.deepcopy(X)
    U_prev = copy.deepcopy(U)
    J = trajectory_cost(X, U, Xref, Uref, start, goal1, all_goals, nongoal_scale)
    while n_iters <= max_linesearch_iters:
        for k in range(N-1):
            u_k = (U_prev[k, :].T) - (alpha * d[k, :].T) - K[k, :]*(Xn[k, :].T - X_prev[k, :].T)
            u_k = u_k.T
            
            print("Forward rk4")
            x_k = rk4(Xn[k, :], u_k, dt)
            
            # print("in")
            # print(u_k.shape)
            # print("slot")
            # print(Un[k, :].shape)
            # # print(Un[k].shape)
            # print(u_k)
            # print(u_k.shape)

            Xn[k+1, :] = x_k

            Un[k, :] = Un[k, :] + u_k
        
        Jn = trajectory_cost(Xn, Un, Xref, Uref, start, goal1, all_goals, nongoal_scale)
        # print(Jn.shape)        
        # print(J.shape)

        if Jn[0,0] < J[0,0] - (0.01 * alpha * del_J):
            break
        alpha = c * alpha
        n_iters += 1

    return Xn, Un, Jn, alpha


def make_array_copying(x, N):
    if x.shape[1] == 1:
        # Note: Matrix or the sympy matrix function does not like edits to arrays
        # So I'm doing it in numpy vectors and then converting it back
        print(x.shape[0])
        new_arr = np.zeros((N, x.shape[0]))

        # print(new_arr)
        for i in range(N):
            for j in range(x.shape[0]):
                new_arr[i][j] = x[j]

        # 3d arrays are used for logging, but 2d ones will be in matrix form for convenience
        new_arr = Matrix(new_arr).as_mutable()

    else:
        new_arr = np.zeros((N, x.shape[0], x.shape[1]))

        # print(new_arr)
        for i in range(N):
                new_arr[i] = x

        print(new_arr.shape)

    return new_arr

# x0, Urefline, Xrefline, Urefline, start, xgoal, all_goals, 50
def iLQR(x0, U, Xrefline, Urefline, start, goal1, all_goals, nongoal_scale, atol=1e-5, max_iters = 100, verbose = true):
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


    K = sympy.MutableDenseNDimArray(np.zeros((N-1, nu, nx))) #make_array_copying(zeros(nu,nx), N-1)
    # sympy.MutableDenseNDimArray(nested_tuple(nu,nx))

    P = sympy.MutableDenseNDimArray(np.zeros((N, nx, nx))) #make_array_copying(zeros(nx,nx), N)
    
    # We keep going
    iterator = -1
    is_complete = False

    while iterator < max_iters and not is_complete:
        iterator += 1
        print("Doing backward pass")
        d, K, P, del_J = backward_pass(X, U, Xrefline, Urefline, start, goal1, all_goals, nongoal_scale)
    
        print("Doing forward pass")
        X, U, J, alpha = forward_pass(X, U, Xrefline, Urefline, K, d, del_J, start, goal1, all_goals, nongoal_scale)

        print(d)
        print(norm(d))

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

    Xrefline = make_array_copying(xgoal, N)
    Urefline = make_array_copying(Matrix([0.0,0.0]), N-1)

    Xline, Uline, Kline, Pline, iterline = iLQR(x0, Urefline, Xrefline, Urefline, start, xgoal, all_goals, 50)
    print("done")


    xvals_leg = [Xline[:][0]]
    yvals_leg = [Xline[:][1]]

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
    x0 = [0.0, 0.0]                        # initial state
    xgoal = [6.0, 5.0]                  # goal state

    trial_1()



if __name__ == "__main__":
    main()






