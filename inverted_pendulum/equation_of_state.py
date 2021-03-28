#!/usr/bin/python3.6

import numpy as np
from numpy import cos
from numpy import sin

def f(x, u, params):
    M = params['M']
    m = params['m']
    l = params['l']
    Dx = params['Dx']
    Dtheta = params['Dtheta']
    g = params['g']
    x1, x2, x3, x4 = x
    u = u[0]
    return np.array( [x3, x4, (-(4/3)*Dx*l**2*m*x3 + (4/3)*l**3*m**2*x4**2*sin(x2) + (4/3)*l**2*m*u + Dtheta*l*m*x4*cos(x2) - g*l**2*m**2*sin(x2)*cos(x2))/((4/3)*l**2*m*(M + m) - l**2*m**2*cos(x2)**2), (-Dtheta*x4*(M + m) + Dx*l*m*x3*cos(x2) + g*l*m*(M + m)*sin(x2) - l**2*m**2*x4**2*sin(x2)*cos(x2) - l*m*u*cos(x2))/((4/3)*l**2*m*(M + m) - l**2*m**2*cos(x2)**2)] )

def A(x, params):
    M = params['M']
    m = params['m']
    l = params['l']
    Dx = params['Dx']
    Dtheta = params['Dtheta']
    g = params['g']
    x1, x2, x3, x4 = x
    u = 0
    return np.array( [[0, 0, 1, 0], [0, 0, 0, 1], [0, -2*l**2*m**2*(-(4/3)*Dx*l**2*m*x3 + (4/3)*l**3*m**2*x4**2*sin(x2) + (4/3)*l**2*m*u + Dtheta*l*m*x4*cos(x2) - g*l**2*m**2*sin(x2)*cos(x2))*sin(x2)*cos(x2)/((4/3)*l**2*m*(M + m) - l**2*m**2*cos(x2)**2)**2 + ((4/3)*l**3*m**2*x4**2*cos(x2) - Dtheta*l*m*x4*sin(x2) + g*l**2*m**2*sin(x2)**2 - g*l**2*m**2*cos(x2)**2)/((4/3)*l**2*m*(M + m) - l**2*m**2*cos(x2)**2), -(4/3)*Dx*l**2*m/((4/3)*l**2*m*(M + m) - l**2*m**2*cos(x2)**2), (2*(4/3)*l**3*m**2*x4*sin(x2) + Dtheta*l*m*cos(x2))/((4/3)*l**2*m*(M + m) - l**2*m**2*cos(x2)**2)], [0, -2*l**2*m**2*(-Dtheta*x4*(M + m) + Dx*l*m*x3*cos(x2) + g*l*m*(M + m)*sin(x2) - l**2*m**2*x4**2*sin(x2)*cos(x2) - l*m*u*cos(x2))*sin(x2)*cos(x2)/((4/3)*l**2*m*(M + m) - l**2*m**2*cos(x2)**2)**2 + (-Dx*l*m*x3*sin(x2) + g*l*m*(M + m)*cos(x2) + l**2*m**2*x4**2*sin(x2)**2 - l**2*m**2*x4**2*cos(x2)**2 + l*m*u*sin(x2))/((4/3)*l**2*m*(M + m) - l**2*m**2*cos(x2)**2), Dx*l*m*cos(x2)/((4/3)*l**2*m*(M + m) - l**2*m**2*cos(x2)**2), (-Dtheta*(M + m) - 2*l**2*m**2*x4*sin(x2)*cos(x2))/((4/3)*l**2*m*(M + m) - l**2*m**2*cos(x2)**2)]] )

def B(x, params):
    M = params['M']
    m = params['m']
    l = params['l']
    Dx = params['Dx']
    Dtheta = params['Dtheta']
    g = params['g']
    x1, x2, x3, x4 = x
    u = 0
    return np.array( [[0], [0], [(4/3)*l**2*m/((4/3)*l**2*m*(M + m) - l**2*m**2*cos(x2)**2)], [-l*m*cos(x2)/((4/3)*l**2*m*(M + m) - l**2*m**2*cos(x2)**2)]] )

