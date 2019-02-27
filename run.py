import numpy as np
from math import sqrt, pi
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from monty.serialization import loadfn, MontyDecoder,MontyEncoder
import json

plt.style.use("bmh")

def process_potential(v0):
    #v0 = potential(omega=omega, m=m, n=n).data
    x, y = v0[:,0], v0[:, 1]
    #y += degree*omega*np.random.uniform(-1, 1, n)

    # shift the minimum to (0, 0)
    # fit returns Ax**2 + Bx + C
    fit = np.polyfit(x, y, 2)
    dx = -fit[1]/fit[0]/2
    dy = fit[2] - fit[0]*dx*dx
    x, y = x-dx, y-dy

    # interpolate the results to obtain smooth curves
    func = interp1d(x, y, kind='cubic')
    x1 = np.linspace(-8, 8, 501)
    y1 = func(x1)
    v = np.vstack((x1, y1))
    v = np.transpose(v)

    return v


def plot_results(json_data, lists=[0], figname='result.png'):
# plot the results
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    ax1 = plt.subplot(211)
    for i in lists:
        data = json_data[i]
        eig_str = "{:d} {:6.2f}".format(data['eigenvalue'][0], data['eigenvalue'][1])
        ax1.plot(data['potential'][:, 0], data['wavefunction'], '--', label=eig_str)
    ax1.set_ylabel('$\Psi(x)$')
    ax1.legend(loc=2)
    plt.setp(ax1.get_xticklabels(), visible=False)


    ax2 = plt.subplot(212, sharex=ax1)
    for i in lists:
        data = json_data[i]
        ax2.hlines(y=data['eigenvalue'][1], xmin=data['potential'][0,0], xmax=data['potential'][-1,0], linewidth=1)
        ax2.plot(data['potential'][:, 0], data['potential'][:, 1], 'b-')
    ax2.set_ylabel('$V(x)$')
    ax2.set_xlabel('$x$')
    #ax2.set_ylim([0, 2*max(eigs)])
    plt.tight_layout()
    plt.savefig('result.png')
    plt.close()

def numerov(m, h, q, s=None, u0=0, u1=0.01):
    """
    Method to perform the Numerov integration
    Numerov method is to solve the differential 
    equations of d2f/dx2 = -q(x)f(x) + s(x)
    more detials can be found in
    https://en.wikipedia.org/wiki/Numerov%27s_method

    Args:
    m: length of the wavefunction array
    h: step size 
    q: 1D array denotes the coefficient of f(x)

    Return: 
    """
    if s is None:
        s = np.zeros(m)

    g = h*h/12
    u = np.empty(m)
    u[0] = u0
    u[1] = u1
    
    for i in range(1, m-1):
        c0 = 1+g*q[i-1]
        c1 = 2-10*g*q[i]
        c2 = 1+g*q[i+1]
        d = g*(s[i+1]+s[i-1]+10*s[i])
        #u[i+1] = min([1e+32, (c1*u[i]-c0*u[i-1]+d)/c2])
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

class potential():
    """
    A class to define the potential
    """
    def __init__(self, xmin=-10, xmax=10, n=501, 
            func='oscilator', perturb=False, **kwargs):

        self.funcs = ['oscilator', 'cosineh']
        self.xs = np.linspace(xmin, xmax, n)
        self.func = func
        self.default_params = {'omega': 1.0, 
                               'm': 1.0, 
                               'a': 1.0, 
                               'la': 4.0,
                               'mu': 0,
                               'delta': 1, 
                               }
        if self.func == 'oscilator':
            self.vs = self.oscilator(**kwargs)
        elif self.func == 'cosineh':
            self.vs = self.cosineh()
        else:
            print('Error, the function'+ self.func + 'is not supported')
            print('Only the following funcs are supported')
            print(self.funcs)
        if perturb:
            self.vs += self.gaussian(**kwargs)
        self.data = np.vstack((self.xs, self.vs))
        self.data = self.data.transpose()

    def oscilator(self, **kwargs):
        keys = ['omega', 'm']
        out = self.get_params(keys, **kwargs)
        omega, m = out[0], out[1]
        return 0.5*m*omega*omega*np.power(self.xs, 2)

    def gaussian(self, **kwargs):
        keys = ['mu', 'delta']
        out = self.get_params(keys, **kwargs)
        mu, delta = out[0], out[1]
        return 1/delta/sqrt(2*pi)*np.exp(-0.5*np.power((self.xs-mu)/delta, 2))


    def cosineh(self, **kwargs):
        keys = ['a', 'la']
        out = self.get_params(keys, **kwargs)
        a, la = out[0], out[1]
        a2 = a*a
        c = a2*la*(la-1)
        return c*(0.5-1/np.power(np.cosh(a*self.xs),2))/2

    def get_params(self, dict_name, **kwargs):
        out = []
        for key in dict_name:
            if key in kwargs.keys():
                out.append(kwargs[key])
            else:
                out.append(self.default_params[key])
        return out

class Solver():
    """
    A numerical solver for 1D Schordinger equation
    Args:
    V: 2D array to describe the potential function [position, values]
    e: the initial guess of eigenvalue
    ni: the maximum number of iterations used in the secant search
    de_max: the initial step size of moves in the secant search
    e_tol: the termination condition for secant search

    Atrributes:

    """
    def __init__(self, V, e, ni=100, de_max=0.1, e_tol=1e-6):

        self.parse_V(V)

        self.ul = np.zeros(self.nx) # wavefunction from left
        self.ur = np.zeros(self.nx) # wavefunction from right
        self.ql = np.zeros(self.nx) # q function from left
        self.qr = np.zeros(self.nx) # q function from right
        self.u =  np.zeros(self.nx) # total wavefunction 
        
        self.ni = ni
        self.de_max = de_max
        self.e = e
        self.e_tol = e_tol
        self.eigenvalue = self.secant(self.ni, self.e_tol, e, self.de_max)
        self.n = self.get_level()
        self.valid = True
        #self.check_validity()

    def parse_V(self, V):
        """
        function to parse the potential array
        """
        self.V_array = V[:, 1]
        self.x_array = V[:, 0]
        self.nx = len(V)
        self.h = V[1, 0] - V[0, 0] 

    def plot_wavefunction(self, figname=None):
        plt.plot(self.x_array, self.u)
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)

    def check_validity(self):
        """
        check if the returned wavefunction is valid
        based on the symmetry
        """
        negative_sum = np.sum(self.u[self.x_array > 0])
        positive_sum = np.sum(self.u[self.x_array < 0])
        self.valid = False
        if abs(negative_sum) > 1e-2:
            if abs(negative_sum + positive_sum) < 1e-2 or \
               abs(negative_sum - positive_sum) < 1e-2:
                self.valid = True


    def get_level(self):
        # Find the matching point at the right turning point
        count = 0
        for i in range(self.nx-1):
            if self.u[i]*self.u[i+1]<0:
                count += 1
        return count

    def secant(self, n, dt, x, dx0):
        """
        Search for the root according to the secant rule:
        Arg
        n: max
        dt: the tolerance between f(x) and f(x+dx)
        x: the initial value of x
        dx: the initial jump of x
        """
        k = 0
        dx = dx0/10
        x1 = x+dx
        while ((abs(dx)>dt) and (k<n)):
            d = self.f(x1)-self.f(x)
            x2 = x1-self.f(x1)*(x1-x)/d
            x, x1 = x1, x2
            dx = np.sign(x1-x)*min(abs(x1-x), dx0/5)
            x1 = x+dx
            k += 1
            if (k==n):
                print("Convergence not found after ", n, " iterations")
                self.valid = False
            #print(x1)
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
        im = int(self.nx/2)
        for i in range(self.nx-1):
            if self.ql[i]*self.ql[i+1]<0 and self.ql[i]>0:
                im = i
                break
        nl, nr = im+2, self.nx-im+1
        if im > 0:
            # Carry out the Numerov integrations
            self.ul = numerov(nl, self.h, self.ql)
            self.ur = numerov(nr, self.h, self.qr)
            # Find the wavefunction on the left 
            ratio = self.ur[nr-2]/self.ul[im]
            for i in range(im):
                self.u[i] = ratio*self.ul[i]
            self.ul[nl-1] *= ratio
            self.ul[nl-3] *= ratio
    
            # Find the wavefunction on the right
            for i in range(nr-1):
                self.u[i+im] = self.ur[nr-i-2]
    
            # Normalize the wavefunction
            sum0 = simpson(np.power(self.u, 2), self.h)
            self.u = self.u/sqrt(sum0)
        return nl, nr

if __name__ == "__main__":

    omegas = np.linspace(1, 10, 10)
    ms = [1]
    degrees = range(10)
    n = 10 # random points

    json_data = []
    for omega in omegas:
        for m in ms:
            for degree in degrees:
                if degree == 0:
                    N_iter = 1
                else:
                    N_iter = 10
                for i in range(N_iter):
                    v0 = potential(omega=omega, m=m, n=n).data
                    v0[:, 1] += degree*omega*np.random.uniform(-1, 1, n)
                    v = process_potential(v0)

                    """ 
                    we attempt to solve the results starting from e=0
                    """

                    e = 0
                    count = 0
                    while True:
                        count += 1
                        if count > 10:
                            print('too many failures for this run')
                            break
                        else:
                            solver = Solver(v, e, de_max=0.2*omega)
                            if solver.valid and solver.n == 0:
                                print("{:4d} {:12.4f} {:12.4f}++++++++".format(solver.n, e, solver.eigenvalue))
                                data = {'potential': v,
                                        'wavefunction': solver.u,
                                        'eigenvalue': [solver.n, solver.eigenvalue],
                                        }
                                json_data.append(data)
                                break
                            else:
                                # if the results is not ground states, decrease e value
                                print("{:4d} {:12.4f} {:12.4f}".format(solver.n, e, solver.eigenvalue))
                                e -= 0.5*omega

json_file = 'trainingdata.json'
with open(json_file, "w") as f:
    json.dump(json_data, f, cls=MontyEncoder, indent=1)

plot_results(json_data, lists=[0], figname='result.png')

