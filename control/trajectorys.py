import numpy as np
import time
import csv

class CircleTrj(object):
    def __init__(self, center, radius, T, dt):
        self._center = center
        self._radius = radius
        self._T = T
        self._dt = dt
        self._N = int(T/dt)
        self._angular = 0
        self._angular_velocity = 0
        self._angular_velocity_target = 0
        self._angular_acceleration = 0.4
        self._t0 = 0
        self._start_time = 0

    def set_target_angular_velocity(self, a_v):
        self._angular_velocity_target = a_v;

    def reset(self, t):
        self._angular = 0
        self._angular_velocity = 0
        self._t0 = t
        self._start_time = t
    
    def get_running_time(self):
        return self._t0 - self._start_time
    
    def step(self, t):
        trj = np.zeros(13*self._N)

        t1 = t
        dt_true = t1 - self._t0;
        self._t0 = t1

        self._angular += self._angular_velocity*(dt_true)
        if self._angular_velocity - self._angular_velocity_target < -0.1:
            self._angular_velocity += self._angular_acceleration*dt_true
        if self._angular_velocity - self._angular_velocity_target > 0.1:
            self._angular_velocity -= self._angular_acceleration*dt_true

        angular = self._angular;
        angular_velocity = self._angular_velocity
        for i in range(self._N):
            angular += self._angular_velocity*self._dt
            if angular_velocity - self._angular_velocity_target < -0.1:
                angular_velocity += self._angular_acceleration*self._dt
            if angular_velocity - self._angular_velocity_target > 0.1:
                angular_velocity -= self._angular_acceleration*self._dt

            pos = self._center + np.array([self._radius*np.cos(angular), self._radius*np.sin(angular), 0])
            trj[i*13:(i+1)*13] = np.array([pos[0], pos[1], pos[2],
                                           1, 0, 0, 0,
                                           0, 0, 0,
                                           1, 0, 0])
        return trj

class TrackTrj(object):
    def __init__(self, path, T, dt):
        f = open(path)
        # f_csv = csv.reader(f)
        self._trj_data = np.loadtxt(f, delimiter=',', skiprows=1)
        self._trj_N = self._trj_data.shape[0]
        self._trj_t = self._trj_data[:,0]
        self._trj_dt = self._trj_t[1]-self._trj_t[0]
        self._trj_pos = self._trj_data[:,1:4]
        self._trj_vel = self._trj_data[:,8:11]

        self._T = T
        self._dt = dt
        self._N = int(T/dt)
        self._start_time = 0

        # private
        self._trj_index = 0
        self._trj_reach_endpoint = False
        self._dt_dt_ratio = self._dt/self._trj_dt

    def reset(self, t):
        self._start_time = t
        self._trj_index = 0
        self._trj_reach_endpoint = False

    def step(self, t):
        t = t - self._start_time
        trj = np.zeros(13*self._N)

        while (not self._trj_reach_endpoint) and (t > self._trj_t[self._trj_index+1]):
            self._trj_index += 1
            if self._trj_index == self._trj_N-1:
                self._trj_reach_endpoint = True
                break

        if self._trj_reach_endpoint:
            for i in range(self._N):
                trj[i*13 : (i+1)*13] = np.array([self._trj_pos[-1,0],self._trj_pos[-1,1],self._trj_pos[-1,2],
                                                1, 0, 0, 0,
                                                0, 0, 0,
                                                1, 0, 0])
        else:
            for i in range(self._N):
                index = self._trj_index + int(self._dt_dt_ratio*(i+1))
                if index > self._trj_N-1:
                    trj[i*13 : (i+1)*13] = np.array([self._trj_pos[-1,0],self._trj_pos[-1,1],self._trj_pos[-1,2],
                                                1, 0, 0, 0,
                                                0, 0, 0,
                                                1, 0, 0])
                else:
                    trj[i*13 : (i+1)*13] = np.array([self._trj_pos[index,0],self._trj_pos[index,1],self._trj_pos[index,2],
                                                1, 0, 0, 0,
                                                0, 0, 0,
                                                1, 0, 0])
        
        return trj


class TrackTrjDBY(object):
    def __init__(self, path, T, dt):
        f = open(path)
        # f_csv = csv.reader(f)
        self._trj_data = np.loadtxt(f)
        self._trj_N = self._trj_data.shape[0]
        self._trj_t = self._trj_data[:,0]
        self._trj_dt = self._trj_t[1]-self._trj_t[0]
        self._trj_pos = self._trj_data[:,1:4]

        self._T = T
        self._dt = dt
        self._N = int(T/dt)
        self._start_time = 0

        # private
        self._trj_index = 0
        self._trj_reach_endpoint = False
        self._dt_dt_ratio = self._dt/self._trj_dt

    def reset(self, t):
        self._start_time = t
        self._trj_index = 0
        self._trj_reach_endpoint = False

    def step(self, t):
        t = t - self._start_time
        trj = np.zeros(13*self._N)

        while (not self._trj_reach_endpoint) and (t > self._trj_t[self._trj_index+1]):
            self._trj_index += 1
            if self._trj_index == self._trj_N-1:
                self._trj_reach_endpoint = True
                break

        if self._trj_reach_endpoint:
            for i in range(self._N):
                trj[i*13 : (i+1)*13] = np.array([self._trj_pos[-1,0],self._trj_pos[-1,1],self._trj_pos[-1,2],
                                                1, 0, 0, 0,
                                                0, 0, 0,
                                                1, 0, 0])
        else:
            for i in range(self._N):
                index = self._trj_index + int(self._dt_dt_ratio*(i+1))
                if index > self._trj_N-1:
                    trj[i*13 : (i+1)*13] = np.array([self._trj_pos[-1,0],self._trj_pos[-1,1],self._trj_pos[-1,2],
                                                1, 0, 0, 0,
                                                0, 0, 0,
                                                1, 0, 0])
                else:
                    trj[i*13 : (i+1)*13] = np.array([self._trj_pos[index,0],self._trj_pos[index,1],self._trj_pos[index,2],
                                                1, 0, 0, 0,
                                                0, 0, 0,
                                                1, 0, 0])
        
        return trj    

