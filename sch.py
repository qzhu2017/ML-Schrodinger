# An example of solving the eigenvalue problem of the one-dimensional
# Schroedinger equation via the secant and Numerov methods.
# T. Pang (February 13, 2019)

from math import cosh, pow, sqrt
from numpy import empty

nx = 500
ns = 10
x1 = -10.0
x2 = 10.0
h = (x2-x1)/nx
ul = empty([nx+1], float)
ur = empty([nx+1], float)
u = empty([nx+1], float)

# Method to perform the Numerov integration.
def numerov(m, h, u0, u1, q, s):
  u = empty([m+1],float)
  u[0] = u0
  u[1] = u1
  g = h*h/12

  for i in range(1, m-1, +1):
    c0 = 1+g*q[i-1]
    c1 = 2-10*g*q[i]
    c2 = 1+g*q[i+1]
    d = g*(s[i+1]+s[i-1]+10*s[i])
    u[i+1] = (c1*u[i]-c0*u[i-1]+d)/c2

  return u

# Method to provide the given potential in the problem.
def v(x):
  al = 1.0
  la = 4.0
  return al*al*la*(la-1)*(0.5-1/pow(cosh(al*x),2))/2

# Method to achieve the evenly spaced Simpson rule.
def simpson(y, h):
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

# Method to calculate the wavefunction.
def wave(energy):
  global u
  global ul
  global ur
  global nl
  global nr
  y = empty([nx+1],float)
  ql = empty([nx+1],float)
  qr = empty([nx+1],float)
  s = empty([nx+1],float)
  u0 = 0
  u1 = 0.01

# Set up function q(x) in the equation
  for i in range (nx+1):
    x = x1+i*h
    ql[i] = 2*(energy-v(x))
    qr[nx-i] = ql[i]

# Find the matching point at the right turning point
  im = 0
  for i in range (nx):
    if (((ql[i]*ql[i+1])<0) and (ql[i]>0)):
      im = i

# Carry out the Numerov integrations
  nl = im+2
  nr = nx-im+2
  ul = numerov(nl, h, u0, u1, ql, s)
  ur = numerov(nr, h, u0, u1, qr, s)

# Find the wavefunction on the left
  ratio = ur[nr-2]/ul[im]
  for i in range (im):
    u[i] = ratio*ul[i]
    y[i] = u[i]*u[i]

  ul[nl-1] *= ratio
  ul[nl-3] *= ratio

# Find the wavefunction on the right
  for i in range(nr-1):
    u[i+im] = ur[nr-i-2]
    y[i+im] = u[i+im]*u[i+im]

# Normalize the wavefunction
  sum = simpson(y, h)
  sum = sqrt(sum)
  for i in range(nx+1):
    u[i] =u[i]/sum

# Method to provide the function for the root search.
def f(x):
  wave(x)
  f0 = ur[nr-1]+ul[nl-1]-ur[nr-3]-ul[nl-3]
  return f0/(2*h*ur[nr-2])

# Method to carry out the secant search.
def secant(n, dt, x, dx):
  k = 0
  x1 = x+dx
  while ((abs(dx)>dt) and (k<n)):
    d = f(x1)-f(x)
    x2 = x1-f(x1)*(x1-x)/d
    x = x1
    x1 = x2
    dx = x1-x
    k=k+1
    if (k==n):
      print("Convergence not found after ", n, " iterations")
    return x1

def main():
  ni = 10
  dt = 1e-6
  e = 2.4
  de = 0.1

# Find the eigenvalue via the secant search
  e = secant(ni, dt, e, de)
  wo = open("wave.data", "w+")
  x = x1
  nh = ns*h

  for i in range (0, nx+1, +ns):
    wo.write("%10.6f  %10.6f\n" % (x,u[i]))
    x += nh

  print ("The eigenvalue is ", e)
# print ("The eigenvalue is %10.6f" % e)

if __name__=="__main__":
  main()
