import numpy as np
import sys

def secant(f, niter, x_tol, x, dx):
    """
    Method to carry out the secant search.
    Args:
    f: function for root search
    x0: point 1
    x1: point 2
    n: number of cycles

    Return: the point between x0 and x1
    """
    x1 = x+dx
    for i in range(niter):
        if abs(dx) < x_tol:   
            return x1
        d = f(x1) - f(x)
        x2 = x1 - f(x1)*(x1-x)/d
        x, x1 = x1, x2
        dx = x1 - x
        #print('search:  ', i, x1)
    return x1

def numerov(m, h, u0, u1, q, s):
    """
    Method to perform the Numerov integration
    Args:
    m
    h
    u0
    u1

    Return: 
    """
    g = h*h/12
    u = np.empty(m)
    u[0] = u0
    u[1] = u1
    #print('numerov:   ', m, len(q), len(s))
    for i in range(1, m-1):
        c0 = 1+g*q[i-1]
        c1 = 2-10*g*q[i]
        c2 = 1+g*q[i+1]
        d = g*(s[i+1]+s[i-1]+10*s[i])
        u[i+1] = (c1*u[i]-c0*u[i-1]+d)/c2
    return u

def f(x):
    """
    Method to provide the function for the root search
    """
    u, ur, ul, nr, nl = wave(x)
    if max([nr, nl]) > nx+1:
        return 0
    else:
        f0 = ur[nr-1]+ul[nl-1]-ur[nr-3]-ul[nl-3]
    return f0/(2*h*ur[nr-2])

def wave(energy):
    """
    Method to calculate the wavefunction.
    """
    y =  np.empty(nx+1)
    ul = np.zeros(nx+1)
    ur = np.zeros(nx+1)
    ql = np.zeros(nx+1)
    qr = np.zeros(nx+1)
    s =  np.zeros(nx+1)
    u =  np.zeros(nx+1)

    ua, ub = 0, 0.01

    # Set up function q(x) in the equation
    for i in range(nx+1):
        x = x1 + i*h
        ql[i] = 2*(energy-V(x))
        qr[nx-i] = ql[i]

    # Find the matching point at the right turning point
    im = 0
    for i in range(nx):
        if ql[i]*ql[i+1]<0 and ql[i]>0:
            im = i
            break
    nl, nr = im+2, nx-im+2
    if im > 0:
        #print('cannot find the turning point')
        #sys.exit()
        # Carry out the Numerov integrations
        ul = numerov(nl, h, ua, ub, ql, s)
        ur = numerov(nr, h, ua, ub, qr, s)
        #print('Numerov', ul, ur)
        # Find the wavefunction on the left 
        ratio = ur[nr-2]/ul[im]
        for i in range(im):
            u[i] = ratio*ul[i]
            y[i] = u[i]*u[i]
        ul[nl-1] *= ratio
        ul[nl-3] *= ratio

        # Find the wavefunction on the right
        for i in range(nr-1):
            u[i+im] = ur[nr-i-2]
            #print('right: ', u[i+im], i+im)
            y[i+im] = u[i+im]*u[i+im]

        # Normalize the wavefunction
        sum0 = integrate(y, h)
        u = u/sum0

    return u, ur, ul, nr, nl

def integrate(y, h):
    sum0 = 0
    for i in range(len(y)):
        if i==0 or i==len(y):
            coef=1/3
        elif i%2 == 1:
            coef = 4/3
        else: 
            coef = 2/3
        sum0 += coef*y[i]
    return h*sum0

def V(x, alpha=1, lambda0=4):
    def cosh(x):
        return (np.exp(x) + np.exp(-x))/2
    return alpha*alpha*lambda0*(lambda0-1)*(0.5-1/pow(cosh(alpha*x), 2))/2

"""
    def __init__(self, pots, e=2.4, de=0.1, dx=1e-6):
        x1, x2, nx = 10, -10, 500
        m = 10  #
        ni = 10 #
        h = (x2-x1)/nx
        X = np.linspace(x1, x2, nx+1)
        pots = V(X)
"""

if __name__ == "__main__":

    alpha = 1
    lambda0 = 4
    tmp = lambda0*(lambda0-1)/2
    for n in range(10):
        print(n, alpha*alpha*(tmp-(lambda0-1-n)*(lambda0-1-n))/2)
    de, e_tol = 0.1, 1e-6
    x1, x2, nx = 10, -10, 500
    m = 10  #
    ni = 10 #
    h = (x2-x1)/nx
    for e in np.linspace(-5, 5, 50):
        eigenvalue = secant(f, ni, e_tol, e, de)
        print("{:12.4f}: {:12.4f}".format(e, eigenvalue))
