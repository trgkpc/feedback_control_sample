#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

fig = plt.figure()

def plot(x, theta0):
    theta = 0.5*np.pi + theta0
    ax = plt.axes()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    black = '#'+'0'*6
    
    # 地面の描写
    fy = -2.425
    plt.plot([-6,6], [fy, fy], color=black)
    h = 0.6
    c = hex(190)[2:]
    r = patches.Rectangle(xy=(-6, fy - h), width=12, height=h, color='#'+c*3)
    ax.add_patch(r)


    def circle(x, y, r, color='#999999'):
        c = patches.Circle(xy=(x, y), radius=r, fc=color, ec=black)
        ax.add_patch(c)
    
    def rectangle(x, y, w, h, color='#999999', theta=0.0):
        r = patches.Rectangle(xy=(x, y), width=w, height=h, fc=color, ec=black, angle=(theta*180/np.pi))
        ax.add_patch(r)

    y0 = -2
    dy = 0.3
    h = 1.0
    Y = y0 + dy + h
    small_r = 0.3
    circle(x, Y, small_r)
    circle(x-1, y0, 0.4)
    circle(x+1, y0, 0.4)
    rectangle(x-2, y0 + dy, 4, h)

    r = np.array([x, Y + 0.3*small_r])
    L = 3.5
    rectangle(r[0], r[1], L, 0.1, theta=theta)
    r += L * np.array([np.cos(theta), np.sin(theta)])
    circle(r[0], r[1], 0.4, color=black)

    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    plot(0., 0.1)
