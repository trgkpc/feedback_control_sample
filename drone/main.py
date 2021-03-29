#!/usr/bin/python3
import numpy as np
import numpy.linalg as LA
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import japanize_matplotlib
import subprocess as sp
fig = plt.figure()
vel_scale = 4.0

#### 系の定義 ####
def eos(state, u):
    # 状態方程式
    # Equation Of State
    y,v = state
    a = u - 0.9*v - 1000.0
    return np.array([v, a])


def runge_kutta(f, x, u, h):
    # ルンゲ食った方
    # ルンゲを食ってない方もあるかもしれない
    k1 = f(x           , u)
    k2 = f(x + (h/2)*k1, u)
    k3 = f(x + (h/2)*k2, u)
    k4 = f(x + h*k3    , u)
    return (h/6) * (k1 + 2.0*k2 + 2.0*k3 + k4)

class Drone:
    def __init__(self, controller, draw_cycle=10, interval_ms=10, T=10.0):
        ##### 設定 ####
        self.draw_cycle = draw_cycle    # 描画周期と制御周期の比率(整数)
        self.interval_ms = interval_ms  # 制御周期[ms]
        self.T = T                      # シミュレーション時間[s]

        #### 内部変数の初期化 ####
        self.dt = interval_ms * 1e-3    # 制御周期[s]
        self.t = 0.0                    # 現在時刻[s]
        self.x0 = [200., 0.]            # 初期状態
        self.x = np.array(self.x0)           # 状態
        self.y0 = 1500.0                # 目標高さ
        self.controller = controller    # 制御器

    # 描画する関数
    def draw(self):
        black = '#'+'0'*6
        y = int(self.x[0])

        # キャンバスの定義
        ax = plt.axes()
        ax.set_aspect('equal')
        plt.xlim(0, 3000)
        plt.ylim(0, 3000)

        # 地面の描写
        field_y = 20.0+self.x0[0]
        plt.plot([0,3000], [field_y, field_y], color=black)
        c = hex(190)[2:]
        r = patches.Rectangle(xy=(0, 0), width=3000, height=field_y, color='#'+c*3)
        ax.add_patch(r)

        # ドローンの目標位置の描画
        h = 2.0
        r = patches.Rectangle(xy=(0, self.y0 - h), width=3000, height=2*h, color='r')
        ax.add_patch(r)

        # ドローンの描画
        img = plt.imread('drone.png')[::-1]
        C = np.zeros((3000, 3000, 4))
        X = 114
        Y = y+len(img) # ドローンの上端
        if Y >= len(C): # ドローンの上端がはみ出ていたら切る
            d = Y - len(C) + 1
            img = img[:-d]
            Y -= d
        C[y:Y, X:X+len(img[0])] = img
        plt.imshow(C)

    ##### ドローンの定義 ####
    # 制御周期ごとに呼ばれる
    def control_callback(self):
        error = self.x[0] - self.y0
        vel = self.x[1]
        u = self.controller(error, vel, self.dt)
        
        self.x += runge_kutta(eos, self.x, u, self.dt)
        self.t += self.dt

    # 描画周期ごとに呼ばれる
    def draw_callback(self):
        # animationでdrawするときに書く関数
        plt.cla()
        for i in range(self.draw_cycle):
            self.control_callback()
        self.draw()
    
    ##### ドローンを出力するレイヤ ####
    # 写真を作る
    def take_picture(self, name):
        self.draw()
        plt.savefig(name)
        plt.cla()

    # シミュレーションをする
    def sim(self, name):
        t_,y_=[],[]
        while self.t <= self.T:
            self.control_callback()
            t_.append(self.t/vel_scale)
            y_.append(self.x[0])
        plt.plot(t_,y_)
        plt.plot([0,self.T/vel_scale],[self.y0,self.y0])
        plt.xlabel("時間[s]")
        plt.ylabel("高さ[mm]")
        plt.savefig(name)
        plt.pause(1.0)
        plt.cla()

    # ドローンの動画を作る
    def gen_gif(self, name):
        draw_interval_ms = self.draw_cycle * self.interval_ms
        ani = animation.FuncAnimation(fig, lambda dat:self.draw_callback(), interval=draw_interval_ms, frames=int(self.T * 1000/draw_interval_ms))
        ani.save("tmp.gif", writer="imagemagick")
        ns = (draw_interval_ms / 10.0) / vel_scale
        sp.run(" ".join(["convert","-delay",str(ns), "tmp.gif",name]), shell=True)
        sp.run(" ".join(["rm", "tmp.gif"]), shell=True)
        plt.cla()

    # simかgifかを統一的に呼び出すインターフェース
    def __call__(self, is_sim, name):
        if is_sim:
            self.sim(f'{name}.png')
        else:
            self.gen_gif(f'{name}.gif')

#### 動画を作る ####
# controllerにPID controllerをセットする
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

class BangBang():
    def __init__(self, u=1500, threshold=10.):
        self.u0 = u
        self.u = u
        self.threshold = threshold

    def __call__(self, error, vel, dt):
        if error > self.threshold:
            self.u = -self.u0
        elif error < -self.threshold:
            self.u = self.u0
        return self.u

# 基本はここを書き換えて映像などを作る
def picture():
    drone = Drone(lambda x,v,t:0.)
    drone.take_picture("world.png")

is_sim = True

def pid():
    # pid制御
    controller = PID()
    drone = Drone(controller, T=12.0)
    drone(is_sim, "PID")
    
def pi():
    # pi制御
    controller = PID(Kd=0.)
    drone = Drone(controller, T=20.0)
    drone(is_sim, "PI")

def p():
    # p制御
    controller = PID(Ki=0., Kd=0.)
    drone = Drone(controller, T=20.0)
    drone(is_sim, "P")

def bangbang():
    # BangBang制御
    controller = BangBang()
    drone = Drone(controller, T=20.0)
    drone(is_sim, "BangBang")

if __name__ == '__main__':
    #picture()

    #pid()
    #pi()
    #p()
    bangbang()

