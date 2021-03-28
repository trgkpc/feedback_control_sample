#!/usr/bin/python3
import numpy as np
import numpy.linalg as LA
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import equation_of_state as eos             # eosはこのディレクトリにあるファイル
fig = plt.figure()

def runge_kutta(f, x, u, h):
    k1 = f(x           , u)
    k2 = f(x + (h/2)*k1, u)
    k3 = f(x + (h/2)*k2, u)
    k4 = f(x + h*k3    , u)
    return (h/6) * (k1 + 2.0*k2 + 2.0*k3 + k4)

class DefaultController:
    def __init__(self):
        print("new Instance of Default Controller")
        
    def __call__(self, params, state, dt):
        return np.array([0.0])
        
class Pendulum:
    def __init__(self, controller=None, draw_cycle=3, interval_ms=10, T=10.0, x0=[0., 0.02, 0., 0.], params = {}):
        ##### 設定 ####
        self.draw_cycle = int(draw_cycle)   # 描画周期と制御周期の比率(整数)
        self.interval_ms = interval_ms      # 制御周期[ms]
        self.T = T                          # シミュレーション時間[s]
        self.params = {
            "M":params.get("M", 2.0),
            "m":params.get("m", 4.0),
            "l":params.get("l", 2.0),
            "Dx":params.get("Dx", 0.2),
            "Dtheta":params.get("Dtheta", 0.1),
            "g":params.get("g", 9.8),
        }
        self.noise = np.array([0.1, 0.1, 0.2, 0.2])

        #### 内部変数の初期化 ####
        self.dt = interval_ms * 1e-3    # 制御周期[s]
        self.t = 0.0                    # 現在時刻[s]
        self.x = np.array(x0)           # 状態
        if controller == None:          # 制御器
            self.controller = DefaultController()
        else:
            self.controller = controller

    def draw(self):
        x_,theta,v,omega = self.x
        theta += 0.5 * np.pi
        x = -1e3 * x_

        ax = plt.axes()
        ax.set_aspect('equal')
        X = 5000.0
        Y = 5000.0
        plt.xlim(-X, Y)
        plt.ylim(-Y, Y)

        black = '#'+'0'*6
        def triangle(p1, p2, p3, color='r'):
            t = patches.Polygon(xy = [p1, p2, p3], fc=color)
            ax.add_patch(t)

        def circle(x, y, r, color='#999999'):
            c = patches.Circle(xy=(x, y), radius=r, fc=color, ec=black)
            ax.add_patch(c)

        def rectangle(x, y, w, h, color='#999999', theta=0.0):
            r = patches.Rectangle(xy=(x, y), width=w, height=h, fc=color, ec=black, angle=(theta*180/np.pi))
            ax.add_patch(r)

        # 地面の描写
        fy = -2425
        plt.plot([-X*1.2,X*1.2], [fy, fy], color=black)
        h = 600
        c = hex(190)[2:]
        rectangle(-1.2*X, fy-h, 2.4*X, h, color='#'+c*3)
        r0 = np.array([0., fy])
        dr = np.array([250., -500.])
        triangle(r0, r0+dr, r0+np.array([-1.,1.])*dr)

        y0 = -2000
        dy = 300
        h = 1000
        y1 = y0 + dy + h
        small_r = 300
        circle(x, y1, small_r)
        circle(x-1000, y0, 400)
        circle(x+1000, y0, 400)
        rectangle(x-2000, y0 + dy, 4000, h)

        r = np.array([x, y1 + 0.3*small_r])
        L = 3500
        rectangle(r[0], r[1], L, 100, theta=theta)
        r += L * np.array([np.cos(theta), np.sin(theta)])
        circle(r[0], r[1], 400, color=black)


    ##### 倒立振子の定義 ####
    # 制御周期ごとに呼ばれる
    def control_callback(self):
        u = self.controller(self.params, self.x, self.dt)
        
        self.x += runge_kutta(lambda x,u:eos.f(x, u, self.params)+self.noise*np.random.randn(4), self.x, u, self.dt)
        self.t += self.dt

    # 描画周期ごとに呼ばれる
    def draw_callback(self):
        # animationでdrawするときに書く関数
        plt.cla()
        for i in range(self.draw_cycle):
            self.control_callback()
        self.draw()
    
    ##### 倒立振子を出力するレイヤ ####
    # 写真を作る
    def take_picture(self, name):
        self.draw()
        plt.savefig(name)
        plt.cla()

    # シミュレーションをする
    def sim(self):
        t_,y_=[],[]
        while self.t <= self.T:
            self.control_callback()
            t_.append(self.t)
            y_.append(self.x[0])
        plt.plot(t_,y_)
        plt.plot([0,self.T],[self.y0,self.y0])
        plt.show()
        plt.cla()

    # 倒立振子の動画を作る
    def gen_gif(self, name):
        draw_interval_ms = self.draw_cycle * self.interval_ms
        ani = animation.FuncAnimation(fig, lambda dat:self.draw_callback(), interval=draw_interval_ms, frames=int(self.T * 1000/draw_interval_ms))
        ani.save(name, writer="imagemagick")
        plt.cla()


    
#### 動画を作る ####
class PID():
    def __init__(self, Kp=3.0, Ki=1.0, Kd=2.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.I = 0.0

    def __call__(self, error, vel, dt):
        self.I += error*dt
        u = -self.Kp*error - self.Kd*vel - self.Ki*self.I
        return u

class ExtendedLQR:
    def __init__(self, Q_, R_):
        Q = np.array(Q_)
        self.Q = 0.5 * (Q+Q.T)
        R = np.array(R_)
        self.R = 0.5 * (R+R.T)


    def __call__(self, params, state, dt):
        x_center = np.array(state)
        A = eos.A(x_center, params)
        B = eos.B(x_center, params)
        P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
        F = -LA.inv(self.R)@(B.T)@P
        u = F@state
        return u

# 基本はここを書き換えて映像などを作る
def picture():
    pendulum = Pendulum()
    pendulum.take_picture("pendulum.png")

def no_control():
    pendulum = Pendulum(T=10.0)
    pendulum.gen_gif("no_control.gif")

def lqr():
    Q = np.array([
        [1.0, 0.1, 0.1, 0.1],
        [0.1, 2.0, 0.1, 0.1],
        [0.1, 0.1, 2.0, 0.1],
        [0.1, 0.1, 0.1, 1.2]])
    R = np.array([
            [0.05]])
    controller = ExtendedLQR(Q, R)
    pendulum = Pendulum(controller=controller, T=10.0)
    pendulum.gen_gif("lqr.gif")

if __name__ == '__main__':
    picture()
    no_control()
    lqr()
