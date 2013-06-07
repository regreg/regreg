"""
This module contains functions to solve two problems:

.. math::

        \minimize_{z,y} y^Ta

    subject to $(z,y) \in \mathbf{epi}({\cal P})$ and the equality constraint 
    $z-b^Ty=s$. 

.. math::

        \minimize_{z,y,w} y^Ta

    subject to $(z,w) \in \mathbf{epi}({\cal P})$ and the equality constraints
    $z-b^Ty=s, w=Dy$.

"""

import warnings
import numpy as np
import regreg.api as rr

def linear_fractional_admm(a, b, epigraph, sign=1., rho=.1, tol=1.e-5, max_its=500):
    """
    Solve the problem

    .. math::

        \minimize_{z,y} -s*y^Ta

    subject to $(z,y) \in \mathbf{epi}({\cal P})$ and the equality constraint 
    $z-b^Ty=s$ with s in [1,-1].

    Inputs
    ======

    a, b: np.float

    sign: np.float (usually [1,-1])

    epigraph : 
          epigraph constraint whose proximal map projects onto the epigraph

    rho : float > 0
          ADMM parameter

    tol : float
          When to stop, based on objective value decreasing.

    max_its : int
          how many ADMM iterations should we do

    Outputs
    =======

    y : np.float

    z : float
         Solution with z = zy[0], y = zy[1:], and x can be found as x = y / z

    value : float
         The optimal value
    """
    
    if sign not in  [1,-1]:
        raise ValueError('expecting a sign, got %f' % float(sign))

    a = a.reshape(-1)
    b = b.reshape(-1)
    p = a.shape[0]
    a_extended = np.zeros(p+1)
    a_extended[1:] = -a * sign
    
    b_extended = np.zeros(p+1)
    b_extended[1:] = -b
    b_extended[0] = 1
    b_extended *= sign

    # dual variable
    u = 0
    
    loss = rr.squared_error(sign*b_extended.reshape((1,p+1)), np.array([1.]), coef=rho)
    
    linear_term = rr.identity_quadratic(0,0,a_extended,0) # a^Ty

    problem = rr.simple_problem(loss, epigraph)
    value = np.inf
    
    for i in range(max_its):
        lagrangian_term = rr.identity_quadratic(0,0,u*b_extended,-u) # Lagrangian part
        total_term = linear_term + lagrangian_term
        soln = problem.solve(total_term, max_its=20)
        u += rho * ((soln*b_extended).sum()-1)
        update = (soln*a_extended).sum()

        if np.fabs(value - update) < tol * np.fabs(value):
            break
        value = update
        
        debug = False
        if i % 50 == 0 and debug:
            print value, update
            
    z = soln[0]
    y = soln[1:]
    return y, z, -sign*value

def linear_fractional_admm_D(a, b, epigraph, D, sign=1., rho=.1, tol=1.e-5, max_its=500):
    """
    Solve the problem

    .. math::

        \minimize_{z,y,w} -s*y^Ta

    subject to $(z,w) \in \mathbf{epi}({\cal P})$ and the equality constraints
    $z-b^Ty=s, w=Dy$ with s in [1,-1].

    Inputs
    ======

    a, b: np.float

    sign: np.float (usually [1,-1])

    epigraph : 
          epigraph constraint whose proximal map projects onto the epigraph

    rho : float > 0
          ADMM parameter

    tol : float
          When to stop, based on objective value decreasing.

    max_its : int
          how many ADMM iterations should we do

    Outputs
    =======

    y : np.float

    z : float
         Solution with z = zy[0], y = zy[1:], and x can be found as x = y / z

    value : float
         The optimal value
    
    The augmented Lagrangian is
    
    .. math::
            
            a^Ty + u (z-sign-y^Tb) + v^T(w-Dy) + 
            \frac{\rho}{2} \left(\|w-Dy\|^2_2 + (z-sign-y^Tb)^2 \right)
            
    subject to the constraint $(z,w) \in \text{epigraph}$.
    
    * The update for (z,w) projects (y^Tb+sign-u,Dy-v) onto the epigraph.

    * The update for u solves a linear regression with response
    
    * The update for y solves a least squares system. Note that the
    objective will be unbounded unless a is in row(D).
    
    .. math::
            
            \begin{pmatrix}
            z-sign+u / \rho \\
            w+v / \rho - D^{\dagger}a / rho \\
            \end{pmatrix}
            
    and design matrix
    
    .. math::
            
            \begin{pmatrix}
            
            \end{pmatrix}
    """
    
    if sign not in  [1,-1]:
        raise ValueError('expecting a sign, got %f' % float(sign))

    a = a.reshape(-1)
    b = b.reshape(-1)
    p = a.shape[0]
    
    a = -sign * a
    # variable w=Dy
    m, p = D.shape
    w = np.zeros(m)
    y = np.zeros(p)
    
    # we will solve many least squares problems with D
    
    Dinv = np.linalg.pinv(D)
    
    # check whether a is in row(D)?
    
    aDinv = np.dot(Dinv.T, a) 
    bDinv = np.dot(Dinv.T, b) 
    aD = np.dot(D.T, aDinv)

    debug = False
    if debug:
        print 'Is a in row(D)?', np.linalg.norm(aD-a) / np.linalg.norm(a)
            
    # dual variable for sign*(z-np.dot(b,y)) = 1
    u = 0
    
    # dual variable for w=Dy
    v = np.zeros(m)
    
    # augmented lagrangian term for y
    """
    a^Ty + u (z-sign-y^Tb) + v^T(w-Dy) + 
            \frac{\rho}{2} \left(\|w-Dy\|^2_2 + (z-sign-y^Tb)^2 \right)
            
    subject to the constraint $(z,w) \in \text{epigraph}$.
    
    * The update for (z,w) projects (y^Tb+sign-u,Dy-v) onto the epigraph.

    * The update for u solves a linear regression with response
    
    * The update for y solves a least squares system. Note that the
    objective will be unbounded unless a is in row(D).
    """

    design_y = np.vstack([b.reshape((1,p)), D])
    response_y = np.zeros(m+1)
    response_y[0] = -sign
    dinv_y = np.linalg.pinv(design_y)
    response_y -= np.dot(dinv_y.T, a)
    
    # the quadratic part of the prox for w

    linear_term = np.zeros(p+1)
    value = np.inf
    
    for i in range(max_its):
               
        # solve for (z,w)
        
        linear_term[1:] = -rho * np.dot(D, y) + v
        linear_term[0] = u - rho * ((y*b).sum() - sign)
        prox_term = rr.identity_quadratic(rho, 0, linear_term, 0)
        zw = epigraph.proximal(prox_term)
        z = zw[0]
        w = zw[1:]
        
        # solve for y
        
        data_y = response_y + zw
        data_y[1:] += v / rho
        data_y[0] += u / rho
        y = np.dot(dinv_y, data_y)
        
        # update dual variables
        
        u += rho * (z - (y*b).sum() - sign)
        v += rho * (w - np.dot(D,y))
        update = (w*a).sum() / z

        if np.fabs(value - update) < tol * np.fabs(value):
            break
        value = update
        
        debug = True
        if i % 50 == 0 and debug:
            print value, update
            
    return w, y, -sign*value


def linear_fractional_tfocs(a, b, epigraph, sign=1., epsilon=None, tol=1.e-5, max_its=500):
    """
    Solve the problem

    .. math::

        \minimize_{z,y} -s*y^Ta

    subject to $(z,y) \in \mathbf{epi}({\cal P})$ and the equality constraint 
    $z-b^Ty=s$ with s in [1,-1].

    Inputs
    ======

    a, b: np.float

    sign: np.float (usually [1,-1])

    epigraph : 
          epigraph constraint whose proximal map projects onto the epigraph

    epsilon : np.float
          a sequence of epsilons for the smoothing

    tol : float
          When to stop, based on objective value decreasing.

    max_its : int
          how many ADMM iterations should we do

    Outputs
    =======

    y : np.float

    z : float
         Solution with z = zy[0], y = zy[1:], and x can be found as x = y / z

    value : float
         The optimal value
    """
    
    if sign not in  [1,-1]:
        raise ValueError('expecting a sign, got %f' % float(sign))

    shape = a.shape
    a = a.reshape(-1)
    b = b.reshape(-1)
    p = a.shape[0]

    a = -sign * a

    # primal variables
    
    w = 0
    y = np.zeros(p)

    # dual variable
    u = 0

    # prox_center
    w0 = 0
    y0 = np.zeros(p)

    # for updates over (w,y)
    prox_center = np.zeros(p+1)
    linear_term = np.zeros(p+1)

    if epsilon is None:
        epsilon = np.array([0.8**j for j in range(70)])

    max_its = 500
    overall_value = np.inf
    value = np.inf

    for eps in epsilon:

        prox_center[1:] = y0
        prox_center[0] = w0
        debug = False

        for i in range(max_its):

            # update w, y

            linear_term[0] = -u
            linear_term[1:] = u*b+a
            quadratic = rr.identity_quadratic(eps, prox_center, linear_term, 0)
            wy = epigraph.proximal(quadratic)
            w = wy[0]
            y = wy[1:]

            # update u

            du = w - (b*y).sum() - sign

            # take a step of size epsilon

            u = u - eps * du 

            if np.fabs(value - u) < tol * min(np.fabs(value), 1):
                break

            value = u
            
        # update the prox_center
        w0 = w
        y0[:] = y.copy()

    return y, w, -sign*value

def linear_fractional_tfocs_D(a, b, epigraph, D, sign=1., epsilon=None, tol=1.e-5, max_its=500):
    """
    Solve the problem

    .. math::

        \minimize_{z,y} y^Ta

    subject to $(z,y) \in \mathbf{epi}({\cal P})$ and the equality constraint 
    $z-b^Ty=s$ with s in [1,-1].

    Inputs
    ======

    a, b: np.float

    sign: np.float (usually [1,-1])

    epigraph : 
          epigraph constraint whose proximal map projects onto the epigraph

    epsilon : np.float
          a sequence of epsilons for the smoothing

    tol : float
          When to stop, based on objective value decreasing.

    max_its : int
          how many ADMM iterations should we do

    Outputs
    =======

    y : np.float

    z : float
         Solution with z = zy[0], y = zy[1:], and x can be found as x = y / z

    value : float
         The optimal value
    """
    
    if sign not in  [1,-1]:
        raise ValueError('expecting a sign, got %f' % float(sign))

    a = a.reshape(-1)
    b = b.reshape(-1)
    m, p = D.shape

    # primal variables
    
    w = 0
    y = np.zeros(p)
    v = np.zeros(m)

    # dual variable
    u1 = 0
    u2 = np.zeros(m)

    # prox_center
    w0 = 0
    y0 = np.zeros(p)
    v0 = np.zeros(m)

    # for updates over (w,v)
    prox_center = np.zeros(p+1)
    linear_term = np.zeros(p+1)

    if epsilon is None:
        epsilon = np.array([0.9**j for j in range(10)])

    max_its = 2
    overall_value = np.inf
    value = np.inf

    for eps in epsilon:

        prox_center[1:] = v0
        prox_center[0] = w0
        debug = False

        for i in range(max_its):

            # update w, v

            linear_term[0] = -u1 
            linear_term[1:] = u2
            quadratic = rr.identity_quadratic(eps, prox_center, linear_term, 0)
            wv = epigraph.proximal(quadratic)
            w = wv[0]
            v = wv[1:]

            # update y

            y = y0 + (np.dot(D.T,u2) - a - u1 * b) / eps

            # update u

            du1 = (b*y).sum() - w - sign
            du2 = v - np.dot(D,y)

            # take a step of size epsilon

            u1 = u1 - eps * du1 
            u2 = u2 - eps * du2

            if np.fabs(value - u1) < tol * min(np.fabs(value), 1) and i > 50:
                break

            value = u1
        
        # update the prox_center
        w0 = w
        y0[:] = y.copy()
        v0[:] = v.copy()

    return y, w, value

def linear_fractional2(a, b, epigraph, sign=1., rho=.1, tol=1.e-5, max_its=100, mu=1., min_its=10):
    a, b = np.asarray(a), np.asarray(b)
    p = a.shape[0]
    w_0, w_prev = np.zeros((2,) + epigraph.shape)
    
    value = np.inf
    for j in range(min_its, max_its):
        w_next, updated = solve_dual_block(b, a, w_0, mu, epigraph, tol=tol, max_its=max_its, sign=sign)[2:]
        if np.fabs(value - updated) < tol * np.fabs(updated):
            break
        w_0 = update_w_0(w_next, w_prev, j)
        w_prev, value = w_next, updated
    return value

def eta_step(gh, b, a, w_0, mu, sign=1):
    y_0, z_0 = w_0[:-1], w_0[-1]
    g, h = gh[:-1], gh[-1]
    
    num = mu*z_0 - h + ((g+a-mu*y_0)*b).sum() - sign * mu
    den = 1 + (b**2).sum()
    return num / den

def gh_step(eta, b, a, w_0, mu, epigraph):
    y_0, z_0 = w_0[:-1], w_0[-1]
    W = np.zeros_like(w_0)
    W[:-1] = mu*y_0 + eta*b - a
    W[-1] = mu*z_0 - eta
    U = epigraph.cone_prox(W)
    return W - U

def solve_dual_block(b, a, w_0, mu, epigraph, sign=1, tol=1.e-6, max_its=1000):
    eta = 0
    gh = np.zeros_like(w_0)
    
    for idx in range(max_its):
        new_eta = eta_step(gh, b, a, w_0, mu, sign=sign)
        new_gh = gh_step(new_eta, b, a, w_0, mu, epigraph)
        if ((np.linalg.norm(new_gh - gh) < tol * max(1, np.linalg.norm(gh)))
            and (np.fabs(new_eta - eta) < tol * max(1, np.fabs(eta)))):
            break
        eta = new_eta
        gh = new_gh
        
    if idx == max_its-1:
        warnings.warn('algorithm did not converge after %d steps' % max_its)
    
    g, h = gh[:-1], gh[-1]
    primal = np.zeros_like(w_0)
    primal += w_0
    primal[:-1] -= (g -eta*b + a) / mu
    primal[-1] -= (h + eta) / mu
    
    return gh, eta, primal, (a*primal[:-1]).sum()

def update_w_0(w_star, w_star_last, j):
    return w_star + j / (j+3.) * (w_star - w_star_last)


def linear_fractional3(a, b, epigraph, sign=1., rho=.1, tol=1.e-5, max_its=100, mu=1., min_its=10):
    a, b = np.asarray(a), np.asarray(b)
    p = a.shape[0]
    
    A = np.zeros((p+2,p+1))
    A[:-1] = np.identity(p+1)
    A[-1] = np.hstack([-b,1])

    e = np.zeros(p+2)
    e[-1] = -sign

    d = np.zeros(p+1)
    d[:-1] = a

    cone = rr.separable((p+2,), [epigraph, rr.zero_constraint(1)], [slice(0,p+1),slice(p+1,p+2)])
    affine = rr.affine_transform(A, e)
    comp_cone = rr.linear_cone(cone, affine)
    w_0, w_prev = np.zeros((2,) + cone.shape)

    smoothed_cone = comp_cone.smoothed(rr.identity_quadratic(2.0, w_0, 0, 0))
    print 'cone', smoothed_cone
    smoothed_cone.quadratic = rr.identity_quadratic(1.e-6,0,d,0)

    #print smoothed_cone._repr_latex_()
    #solver = rr.FISTA(smoothed_cone)
    #solver.fit()

    return smoothed_cone
