import numpy as np
from math import cosh, pow, sqrt
import sys
import matplotlib.pyplot as plt
plt.style.use("bmh")


def numerov(m, h, q, s, u0=0, u1=0.01):
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

    for i in range(1, m-1):
        c0 = 1+g*q[i-1]
        c1 = 2-10*g*q[i]
        c2 = 1+g*q[i+1]
        d = g*(s[i+1]+s[i-1]+10*s[i])
        u[i+1] = (c1*u[i]-c0*u[i-1]+d)/c2
    return u


def simpson(y, h):
    """
    Method to achieve the evenly spaced Simpson rule.
    """
    n = len(y)-1
    s0 = 0
    s1 = 0
    s2 = 0
    for i in range(1, n, +2):
      s0 += y[i]
      s1 += y[i-1]
      s2 += y[i+1]
    
    s = (s1+4*s0+s2)/3

    # Add the last slice separately for an even n+1
    if ((n+1)%2 == 0):
        return h*(s+(5*y[n]+8*y[n-1]-y[n-2])/12)
    else:
        return h*s

# Method to provide the given potential in the problem.
def v(x, a=1.0, la=4.0):
    return a*a*la*(la-1)*(0.5-1/pow(cosh(a*x),2))/2

   
class potential():
    """
    A class to define the potential
    So far we only do the cosh function
    """
    def __init__(self, x, alpha=1, lambda0=4):
        def cosh(x):
            return (np.exp(x) + np.exp(-x))/2
        return alpha*alpha*lambda0*(lambda0-1)*(0.5-1/pow(cosh(alpha*x), 2))/2

class Solver():
    """
    A numerical solver for 1D Schordinger equation
    Args:
    V: 2D array to describe the potential function [position, values]

    Atrributes:

    """
    def __init__(self, V, e, ni=30, de=0.1, e_tol=1e-6):
        #self.x_min, self.x_max, self.nx = x_min, x_max, nx
        #self.xs = np.linspace(x_min, x_max, nx+1)
        self.parse_V(V)

        self.y =  np.empty(self.nx)
        self.ul = np.zeros(self.nx)
        self.ur = np.zeros(self.nx)
        self.ql = np.zeros(self.nx)
        self.qr = np.zeros(self.nx)
        self.s =  np.zeros(self.nx)
        self.u =  np.zeros(self.nx)
        
        self.ni = ni
        self.de = de
        self.e_tol = e_tol
        self.eigenvalue = self.secant(self.ni, self.e_tol, e, self.de)
        self.n = self.get_level()
        print("{:4d} {:12.4f} {:12.4f}".format(self.n, e, self.eigenvalue))

    def parse_V(self, V):
        """
        function to parse the potential array
        """
        self.V_array = V[:, 1]
        self.x_array = V[:, 0]
        self.nx = len(V)
        self.h = V[1, 0] - V[0, 0] 

    def plot_wavefunction(self, figname=None):
        plt.plot(self.xarray, self.u)
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)

    def get_level(self):
        # Find the matching point at the right turning point
        count = 0
        for i in range(self.nx-1):
            if self.u[i]*self.u[i+1]<0:
                count += 1
        return count

    def secant(self, n, dt, x, dx):
        k = 0
        x1 = x+dx
        while ((abs(dx)>dt) and (k<n)):
            d = self.f(x1)-self.f(x)
            x2 = x1-self.f(x1)*(x1-x)/d
            x, x1 = x1, x2
            dx = x1-x
            k += 1
            if (k==n):
                print("Convergence not found after ", n, " iterations")
        return x1

    def f(self, x):
        """
        Method to provide the function for the root search
        """
        nl, nr = self.wave(x)
        if max([nr, nl]) > self.nx:
            return 0
        else:
            f0 = self.ur[nr-1] + self.ul[nl-1] - self.ur[nr-3] - self.ul[nl-3]
        return f0/(2*self.h*self.ur[nr-2])

    def wave(self, energy):
        """
        Method to calculate the wavefunction based on the guess of energy.
        """
    
        # Set up function q(x) in the equation
        self.ql = 2*(energy-self.V_array)
        for i in range(self.nx):
            self.qr[self.nx-i-1] = self.ql[i]
    
        # Find the matching point at the right turning point
        im = 0
        for i in range(self.nx-1):
            if self.ql[i]*self.ql[i+1]<0 and self.ql[i]>0:
                im = i
                break
        nl, nr = im+2, self.nx-im+1
        if im > 0:
            # Carry out the Numerov integrations
            self.ul = numerov(nl, self.h, self.ql, self.s)
            self.ur = numerov(nr, self.h, self.qr, self.s)
            # Find the wavefunction on the left 
            ratio = self.ur[nr-2]/self.ul[im]
            for i in range(im):
                self.u[i] = ratio*self.ul[i]
                self.y[i] = self.u[i]*self.u[i]
            self.ul[nl-1] *= ratio
            self.ul[nl-3] *= ratio
    
            # Find the wavefunction on the right
            for i in range(nr-1):
                self.u[i+im] = self.ur[nr-i-2]
                self.y[i+im] = self.u[i+im]*self.u[i+im]
    
            # Normalize the wavefunction
            sum0 = simpson(self.y, self.h)
            self.u = self.u/sqrt(sum0)
        return nl, nr

if __name__ == "__main__":

    # create potential data, it must be evenly spaced
    vs = []
    for x in np.linspace(-10, 10, 501):
        vs.append([x, v(x)])
    vs = np.array(vs)

    minv = np.min(vs[:, 1])
    maxv = np.max(vs[:, 1]) - 0.1
    eigs, ns, waves = [], [], []
    for e in np.linspace(minv, maxv, 10):
        solver = Solver(vs, e)
        if solver.n not in ns:
            ns.append(solver.n)
            eigs.append(solver.eigenvalue)
            waves.append(solver.u)
    
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    ax1 = plt.subplot(211)
    for i, n in enumerate(ns):
        eig_str = "{:6.2f}".format(eigs[i])
        ax1.plot(vs[:, 0], waves[i], '--', label=str(n) + ': ' + eig_str)
        #solver.plot_wavefunction()
    ax1.set_ylabel('$\Psi(x)$')
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)


    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(vs[:, 0], vs[:, 1], 'b-')
    for eig in eigs:
        ax2.hlines(y=eig, xmin=vs[0,0], xmax=vs[-1,0], linewidth=1)
    ax2.set_ylabel('$V(x)$')
    ax2.set_xlabel('$x$')
    plt.tight_layout()
    plt.savefig('wavefunction.png')
    plt.close()
    #plt.show()
