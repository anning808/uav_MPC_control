import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpaches
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg


class Visualization(object):
    def __init__(self):

        self.fig = plt.figure(figsize=(20, 20))
        self.gs = gridspec.GridSpec(nrows=4, ncols=10)

        self.ax_3d = self.fig.add_subplot(self.gs[1:, 3:], projection='3d')
        self.ax_3d.set_xlim([-1, 1])
        self.ax_3d.set_ylim([-0.2, 0.2])
        self.ax_3d.set_zlim([-2, 2])

        self.ax_3d.set_xlabel("x")
        self.ax_3d.set_ylabel("y")
        self.ax_3d.set_zlabel("z")

        self.ax_3d.view_init(elev=18, azim=54)

        self.l_quad_pos, = self.ax_3d.plot([], [], [], 'b-')
        self.l_quad_pred_traj, = self.ax_3d.plot([], [], [], 'r*')

        self.l_quad_x, = self.ax_3d.plot([], [], [], 'r', linewidth=3)
        self.l_quad_y, = self.ax_3d.plot([], [], [], 'g', linewidth=3)
        self.l_quad_z, = self.ax_3d.plot([], [], [], 'b', linewidth=3)

        width, height = 5, 2
        g = Rectangle(xy=(-3, -3), width=6, height=6,
                      alpha=0.8, facecolor='gray', edgecolor='black')
        self.ax_3d.add_patch(g)
        art3d.pathpatch_2d_to_3d(g, z=0, zdir="z")

        self.quad_pos = []

    def init_animate(self):
        self.l_quad_pos.set_data_3d([], [], [])
        return [self.l_quad_pos]

    def update(self, data_info):
        self.quad_pos.append(data_info['quad_pos'])
        quad_axes_x, quad_axes_y, quad_axes_z = data_info['quad_axes']

        quad_pos_arr = np.array(self.quad_pos)
        self.l_quad_pos.set_data_3d(quad_pos_arr[:, 0], quad_pos_arr[:, 1], quad_pos_arr[:, 2])

        self.l_quad_x.set_data_3d([quad_pos_arr[-1, 0], quad_axes_x[0]], [quad_pos_arr[-1, 1], quad_axes_x[1]],
                                  [quad_pos_arr[-1, 2], quad_axes_x[2]])
        self.l_quad_y.set_data_3d([quad_pos_arr[-1, 0], quad_axes_y[0]], [quad_pos_arr[-1, 1], quad_axes_y[1]],
                                  [quad_pos_arr[-1, 2], quad_axes_y[2]])
        self.l_quad_z.set_data_3d([quad_pos_arr[-1, 0], quad_axes_z[0]], [quad_pos_arr[-1, 1], quad_axes_z[1]],
                                  [quad_pos_arr[-1, 2], quad_axes_z[2]])

        return [self.l_quad_pos, self.l_quad_x, self.l_quad_y, self.l_quad_z]



class Visualizationdby(object):
    def __init__(self):

        self.fig = plt.figure(figsize=(20, 20))
        self.ax_3d = self.fig.add_subplot(111,projection = '3d')
        self.fig.tight_layout()
        self.ax_3d.set_xlim([-1,1])
        self.ax_3d.set_ylim([-1,1])
        self.ax_3d.set_zlim([-1,1])

        self.ax_3d.set_xlabel("x")
        self.ax_3d.set_ylabel("y")
        self.ax_3d.set_zlabel("z")
        self.ax_3d.scatter(0,0,0, color='k') # black origin dot
        self.ax_3d.view_init(40,-45)
        self.ax_3d.plot([0,1], [0,0], [0,0], label='$X_0$', linestyle="dashed")
        self.ax_3d.plot([0,0], [0,-1], [0,0], label='$Y_0$', linestyle="dashed")
        self.ax_3d.plot([0,0], [0,0], [0,1], label='$Z_0$', linestyle="dashed")
        self.quad_pos = [1,0,0]

    def init_animate(self):
        self.ax_3d.legend(loc='best')
        self.vek = self.ax_3d.quiver(0, 0, 0, self.quad_pos[0], self.quad_pos[1], self.quad_pos[2], label='$g \cdot R$', pivot="tail", color="black")
        return [self.vek]

    def update(self, data_info):
        self.quad_pos=data_info['quad_pos']
        print(self.quad_pos)
        self.vek = self.ax_3d.quiver(0, 0, 0, self.quad_pos[0], self.quad_pos[1], self.quad_pos[2], label='$g \cdot R$', pivot="tail", color="black")

        return [self.vek]
    
