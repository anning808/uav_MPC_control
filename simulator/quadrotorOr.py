import numpy as np
from scipy.spatial.transform import Rotation as R

from simulator.parameter import *


class Quadrotor(object):
    def __init__(self, dt):
        self.s_dim = 10
        self.a_dim = 4

        self._state = np.zeros(shape=self.s_dim)
        self._state[kQuatW] = 1.0

        self._actions = np.zeros(shape=self.a_dim)

        self._gz = -G_z
        self._dt = dt

        self._arm_l = 0.3
        self._b_x = np.array([self._arm_l, 0, 0])
        self._b_y = np.array([0, self._arm_l, 0])
        self._b_z = np.array([0, 0, self._arm_l])

        # Sampling range of the quadrotor's initial position
        self._xyz_dist = np.array(
            [[-0.1,  0.1],  # x
             [-0.1,  0.1],  # y
             [-0.2, -0.1]]  # z
        )
        # Sampling range of the quadrotor's initial velocity
        self._vxyz_dist = np.array(
            [[-0.2,  0.2],  # vx
             [-0.2,  0.2],  # vy
             [-0.1,  0.1]]  # vz
        )

        #
        # self.obs_low = np.array([-10, -10, -10, -np.pi, -np.pi, -np.pi, -10, -10, -10])
        # self.obs_high = np.array([10, 10, 10, np.pi, np.pi, np.pi, 10, 10, 10])
        self.dstate = np.zeros(shape=self.s_dim)
        self.quat = np.zeros(4)
        self.reset()

    def reset(self):
        self._state = np.zeros(shape=self.s_dim)
        self._state[kQuatW] = 1.0

        # initialize position randomly
        self._state[kPosX] = np.random.uniform(
            low=self._xyz_dist[0, 0], high=self._xyz_dist[0, 1])
        self._state[kPosY] = np.random.uniform(
            low=self._xyz_dist[1, 0], high=self._xyz_dist[1, 1])
        self._state[kPosZ] = np.random.uniform(
            low=self._xyz_dist[2, 0], high=self._xyz_dist[2, 1])

        # initialize rotation randomly
        qxyz = np.random.uniform(low=-0.1, high=0.1, size=3)
        # normalize the quaternion
        self._state[kQuatW: kQuatZ+1] = np.array([1-np.linalg.norm(qxyz), qxyz[0], qxyz[1], qxyz[2]])

        # initialize velocity randomly
        self._state[kVelX] = np.random.uniform(
            low=self._vxyz_dist[0, 0], high=self._vxyz_dist[0, 1])
        self._state[kVelY] = np.random.uniform(
            low=self._vxyz_dist[1, 0], high=self._vxyz_dist[1, 1])
        self._state[kVelZ] = np.random.uniform(
            low=self._vxyz_dist[2, 0], high=self._vxyz_dist[2, 1])

        return self._state

    def run(self, action):
        # rk4 int
        M = 4
        DT = self._dt/M
        #
        X = self._state

        k1 = DT * self._f(X,            action)    # 1
        k2 = DT * self._f(X + 0.5 * k1, action)
        k3 = DT * self._f(X + 0.5 * k2, action)
        k4 = DT * self._f(X + k3,       action)
        X = X + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
        k1 = DT * self._f(X,            action)    # 2
        k2 = DT * self._f(X + 0.5 * k1, action)
        k3 = DT * self._f(X + 0.5 * k2, action)
        k4 = DT * self._f(X + k3,       action)
        X = X + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
        k1 = DT * self._f(X,            action)    # 3
        k2 = DT * self._f(X + 0.5 * k1, action)
        k3 = DT * self._f(X + 0.5 * k2, action)
        k4 = DT * self._f(X + k3,       action)
        X = X + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
        k1 = DT * self._f(X,            action)   # 4
        k2 = DT * self._f(X + 0.5 * k1, action)
        k3 = DT * self._f(X + 0.5 * k2, action)
        k4 = DT * self._f(X + k3,       action)
        X = X + (k1 + 2.0 * (k2 + k3) + k4) / 6.0

        self._state = X
        return self._state

    def _f(self, state, action):
        thrust, wx, wy, wz = action
        accel_z_sp = -( accel_min*(1-thrust) + accel_max*thrust )

        self.dstate[kPosX:kPosZ+1] = state[kVelX:kVelZ+1]

        self.quat = state[kQuatW:kQuatZ + 1]
        self.quat = self.quat / np.linalg.norm(self.quat)
        state[kQuatW:kQuatZ + 1] = self.quat
        qw, qx, qy, qz = self.quat
        self.dstate[kQuatW] = 0.5 * (-wx * qx - wy * qy - wz * qz)
        self.dstate[kQuatX] = 0.5 * (wx * qw + wz * qy - wy * qz)
        self.dstate[kQuatY] = 0.5 * (wy * qw - wz * qx + wx * qz)
        self.dstate[kQuatZ] = 0.5 * (wz * qw + wy * qx - wx * qy)

        self.dstate[kVelX] = 2 * (qw * qy + qx * qz) * accel_z_sp
        self.dstate[kVelY] = 2 * (qy * qz - qw * qx) * accel_z_sp
        self.dstate[kVelZ] = (qw * qw - qx * qx - qy * qy + qz * qz) * accel_z_sp - self._gz

        return self.dstate

    def get_position(self):
        return self._state[kPosX: kPosZ+1]

    def get_quaternion(self):
        quat = self._state[kQuatW: kQuatZ+1]
        quat = quat / np.linalg.norm(quat)
        return quat

    def get_axes(self):
        q = self.get_quaternion()
        rot_matrix = R.from_quat(np.array([q[3], q[2], q[1], q[0]])).as_matrix()
        quad_center = self.get_position()

        w_x = rot_matrix@self._b_x + quad_center
        w_y = rot_matrix@self._b_y + quad_center
        w_z = rot_matrix@self._b_z + quad_center
        return [w_x, w_y, w_z]
