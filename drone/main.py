#!/usr/bin/python3
import numpy as np
import numpy.linalg as LA
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import subprocess as sp
fig = plt.figure()

##### 設定 ####
draw_cycle = 10         # 描画周期と制御周期の比率(整数)
interval_ms = 10        # 制御周期[ms]
T = 10.0                # シミュレーション時間[s]

#### 内部変数の初期化 ####
dt = interval_ms * 1e-3 # 制御周期[s]
t = 0.0                 # 現在時刻[s]
x0 = [200., 0.]         # 初期状態
x = np.array(x0)        # 状態
y0 = 1500.0             # 目標高さ
controller = lambda:0.  # 制御器

def initialize(): # 再初期化の際に呼ぶ関数
    global t,x
    t  = 0.0
    x = np.array(x0)

#### 系の定義 ####
def eos(x, u):
    # 状態方程式
    # Equation Of State
    y,v = x
    a = u - 0.9*v - 1000.0
    return np.array([v, a])

def runge_kutta(x, u, h):
    # ルンゲ食った方
    # ルンゲを食ってない方もあるかもしれない
    k1 = eos(x           , u)
    k2 = eos(x + (h/2)*k1, u)
    k3 = eos(x + (h/2)*k2, u)
    k4 = eos(x + h*k3    , u)
    return (h/6) * (k1 + 2.0*k2 + 2.0*k3 + k4)

def control_callback():
    # 制御周期のたびに呼ばれるやつ
    global x,t
    u = controller()
    x += runge_kutta(x, u, dt)
    t += dt

def draw(y_):
    # 描画する関数
    black = '#'+'0'*6
    y = int(y_)

    # キャンバスの定義
    ax = plt.axes()
    ax.set_aspect('equal')
    plt.xlim(0, 3000)
    plt.ylim(0, 3000)

    # 地面の描写
    field_y = 20.0+x0[0]
    plt.plot([0,3000], [field_y, field_y], color=black)
    c = hex(190)[2:]
    r = patches.Rectangle(xy=(0, 0), width=3000, height=field_y, color='#'+c*3)
    ax.add_patch(r)

    # ドローンの目標位置の描画
    h = 2.0
    r = patches.Rectangle(xy=(0, y0 - h), width=3000, height=2*h, color='r')
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

def draw_callback(data):
    # animationでdrawするときに書く関数
    plt.cla()
    for i in range(draw_cycle):
        control_callback()
    draw(x[0])

##### 系を呼んでグラフなどを出力するレイヤ ####
def static(name, y):
    # ドローンの写真を取る
    draw(y)
    plt.savefig(name)
    plt.cla()

def sim():
    # 制御器のシミュレーションのみ
    initialize()
    t_,y_=[],[]
    while t <= T:
        control_callback()
        t_.append(t)
        y_.append(x[0])
    plt.plot(t_,y_)
    plt.plot([0,T],[y0,y0])
    plt.show()
    plt.cla()

def gen_gif(name):
    # ドローンの動画を作る
    initialize()
    draw_interval_ms = draw_cycle * interval_ms
    ani = animation.FuncAnimation(fig, draw_callback, interval=draw_interval_ms, frames=int(T * 1000/draw_interval_ms))
    ani.save("tmp.gif", writer="imagemagick")
    ms = draw_interval_ns / 40.0
    sp.run(" ".join(["convert","-delay",str(ns), "tmp.gif",name]), shell=True)
    sp.run(" ".join(["rm", "tmp.gif"]), shell=True)
    plt.cla()

#### 動画を作る ####
# controllerにPID controllerをセットする
class PID():
    def __init__(self, Kp=3.0, Ki=1.0, Kd=2.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.I = 0.0

    def __call__(self):
        y,v = x
        error = y-y0
        self.I += error*dt
        u = -self.Kp*error - self.Kd*v - self.Ki*self.I
        return u

# 基本はここを書き換えて映像などを作る
def picture():
    # ドローンの写真を取る
    static("world.png", x0[0])
    
def pid():
    # pid制御
    global T,controller
    T = 12.0
    controller = PID()
    gen_gif("PID.gif")

def pi():
    # pi制御
    global T,controller
    T = 20.0
    controller = PID(Kd=0.)
    gen_gif("PI.gif")

def p():
    # p制御
    global T,controller
    T = 20.0
    controller = PID(Ki=0., Kd=0.)
    gen_gif("P.gif")

if __name__ == '__main__':
    picture()

    pid()
    pi()
    p()

