import casadi as ca
import numpy as np
# import time
from os import system
import math
from simulator.parameter import *

class QuadrotorMPC_dby(object):

    def __init__(self, T, dt):
        # Time constant
        self._T = T
        self._dt = dt
        self._N = int(self._T / self._dt)

        # state dimension
        # [px, py, pz,       # quadrotor position
        #  qw, qx, qy, qz,   # quadrotor quaternion
        #  vx, vy, vz]       # quadrotor linear velocity
        self._x_dim = 10
        # control input dimension [thrust, wx, wy, wz]
        self._u_dim = 4

        # cost matrix for tracking the path point position
        self._Q_track_pos = np.diag([100, 100, 100])
        # cost matrix for tracking the path point attitude
        self._Q_track_att = np.diag([100, 50, 50, 100])
        # cost matrix for tracking the path point velocity
        self._Q_track_vel = np.diag([10, 10, 10])

        # cost matrix for the control input
        self._Q_u = np.diag([50, 50, 50, 50])

        self._hover_thrust = (G_z - accel_min)/(accel_max - accel_min)
        self._quad_x0 = [1.0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]
        self._quad_u0 = [self._hover_thrust, 0, 0, 0]

        self._ipopt_options = {
            'verbose': False,
            'ipopt.tol': 1e-4,
            # 'ipopt.acceptable_tol': 1e-4,
            'ipopt.max_iter': 1000,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 0,
            'print_time': False
        }
        self.nlp_x0 = []  # initial guess of nlp variables
        self.nlp_lbx = []  # lower bound of the variables, lbx <= nlp_x
        self.nlp_ubx = []  # upper bound of the variables, nlp_x <= ubx
        self.nlp_lbg = []  # lower bound of constraint functions, lbg <= g
        self.nlp_ubg = []  # upper bound of constraint functions, g <= lbg

        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self._x_dim)]
        x_max = [+x_bound for _ in range(self._x_dim)]
        u_min = [0, -w_max_xy, -w_max_xy, -w_max_yaw]
        u_max = [1, w_max_xy, w_max_xy, w_max_yaw]
        g_min = [0 for _ in range(self._x_dim)]
        g_max = [0 for _ in range(self._x_dim)]

        self.nlp_x0 += self._quad_x0
        self.nlp_lbx += x_min
        self.nlp_ubx += x_max
        self.nlp_lbg += g_min
        self.nlp_ubg += g_max
        for i in range(self._N):
            self.nlp_x0 += self._quad_u0
            self.nlp_x0 += self._quad_x0
            self.nlp_lbx += u_min
            self.nlp_ubx += u_max
            self.nlp_lbx += x_min
            self.nlp_ubx += x_max
            self.nlp_lbg += g_min
            self.nlp_ubg += g_max

        self.sol = []
        self.define_solver_python()
        
    def define_solver_python(self):
                # states
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), ca.SX.sym('qz')
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        # control input
        thrust, wx, wy, wz = ca.SX.sym('thrust'), ca.SX.sym('wx'), ca.SX.sym('wy'), ca.SX.sym('wz')

        self._x = ca.vertcat(px, py, pz,
                             qw, qx, qy, qz,
                             vx, vy, vz)
        self._u = ca.vertcat(thrust, wx, wy, wz)

        accel_z_sp = -( accel_min*(1-thrust) + accel_max*thrust )
        x_dot = ca.vertcat(
            vx,
            vy,
            vz,
            0.5 * (-wx * qx - wy * qy - wz * qz),
            0.5 * (wx * qw + wz * qy - wy * qz),
            0.5 * (wy * qw - wz * qx + wx * qz),
            0.5 * (wz * qw + wy * qx - wx * qy),
            2 * (qw * qy + qx * qz) * accel_z_sp,
            2 * (qy * qz - qw * qx) * accel_z_sp,
            (qw * qw - qx * qx - qy * qy + qz * qz) * accel_z_sp + G_z
        )
        self.f = ca.Function('f', [self._x, self._u], [x_dot], ['x', 'u'], ['ode'])
        F = self.sys_dynamics(self._dt)
        fMap = F.map(self._N, 'openmp')

        #
        Delta_pos = ca.SX.sym('Delta_pos', 3)
        Delta_att = ca.SX.sym('Delta_att', 4)
        Delta_vel = ca.SX.sym('Delta_vel', 3)
        Delta_u = ca.SX.sym('Delta_u', self._u_dim)

        cost_pos = Delta_pos.T @ self._Q_track_pos @ Delta_pos
        cost_att = Delta_att.T @ self._Q_track_att @ Delta_att+100*math.atan2(2 * ( qx* qy +  qw*qz),  qw* qw +  qx* qx -  qy* qy -  qz* qz)
        cost_vel = Delta_vel.T @ self._Q_track_vel @ Delta_vel
        cost_u = Delta_u.T @ self._Q_u @ Delta_u
        f_cost_pos = ca.Function('cost_pos', [Delta_pos], [cost_pos])
        f_cost_att = ca.Function('cost_att', [Delta_att], [cost_att])
        f_cost_vel = ca.Function('cost_vel', [Delta_vel], [cost_vel])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])

        self.nlp_x = []  # nlp variables

        self.mpc_obj = 0  # objective
        self.nlp_g = []  # constraint functions
        

        P = ca.SX.sym('P', (self._x_dim + 3) * (self._N+1))
        X = ca.SX.sym('X', self._x_dim, self._N + 1)
        U = ca.SX.sym('U', self._u_dim, self._N)
        X_next = fMap(X[:, :self._N], U)

        # starting point
        self.nlp_x += [X[:, 0]]
        
        self.nlp_g += [X[:, 0] - P[0: self._x_dim]]
        

        for k in range(self._N):
            # add control input variables
            self.nlp_x += [U[:, k]]            

            p_index = (self._x_dim + 3) * (k + 1)
            delta_pos_k = (X[:3, k + 1] - P[p_index: p_index + 3])
            delta_att_k = (X[3:7, k + 1] - P[p_index + 3: p_index + 7])
            delta_vel_k = (X[7:10, k + 1] - P[p_index + 7: p_index + 10])
            w_pos_k = P[p_index + self._x_dim]
            w_att_k = P[p_index + self._x_dim + 1]
            w_vel_k = P[p_index + self._x_dim + 2]

            cost_track_k = f_cost_pos(delta_pos_k) * w_pos_k + f_cost_att(delta_att_k) * w_att_k +\
                           f_cost_vel(delta_vel_k) * w_vel_k

            delta_u_k = U[:, k] - [self._hover_thrust, 0, 0, 0]
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_track_k + cost_u_k
            #
            self.nlp_x += [X[:, k + 1]]

            self.nlp_g += [X_next[:, k] - X[:, k + 1]]

        nlp_dict = {
            'f': self.mpc_obj,
            'x': ca.vertcat(*self.nlp_x),
            'p': P,
            'g': ca.vertcat(*self.nlp_g)
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_dict, self._ipopt_options)

    def generate_solver_c(self, cname):
        self.solver.generate_dependencies(cname)
        ## system('gcc -fPIC -shared -O3' + cname + ' -o ' + './casadi_gen/mpc.so')


    def load_solver(self, path):
        self.solver = ca.nlpsol('solver', 'ipopt', path, self._ipopt_options)


    def solve(self, path):

        self.sol = self.solver(
            x0=self.nlp_x0,
            lbx=self.nlp_lbx,
            ubx=self.nlp_ubx,
            p=path,
            lbg=self.nlp_lbg,
            ubg=self.nlp_ubg
        )

        sol_x0 = self.sol['x'].full()
        opt_u = sol_x0[self._x_dim: self._x_dim + self._u_dim]

        # self.nlp_x0 = sol_x0
        self.nlp_x0 = list(sol_x0[self._x_dim + self._u_dim: 2 * (self._x_dim + self._u_dim)]) + \
                      list(sol_x0[self._x_dim + self._u_dim:])

        x0_array = np.reshape(sol_x0[:-self._x_dim], newshape=(-1, self._x_dim + self._u_dim))

        return opt_u, x0_array

    def sys_dynamics(self, dt):
        M = 4
        DT = dt / M
        X0 = ca.SX.sym('X', self._x_dim)
        U = ca.SX.sym('U', self._u_dim)

        X = X0
        for _ in range(M):
            k1 = DT * self.f(X, U)
            k2 = DT * self.f(X + 0.5 * k1, U)
            k3 = DT * self.f(X + 0.5 * k2, U)
            k4 = DT * self.f(X + k3, U)
            X = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        F = ca.Function('F', [X0, U], [X])
        return F
class QuadrotorMPC(object):

    def __init__(self, T, dt):
        # Time constant
        self._T = T
        self._dt = dt
        self._N = int(self._T / self._dt)

        # state dimension
        # [px, py, pz,       # quadrotor position
        #  qw, qx, qy, qz,   # quadrotor quaternion
        #  vx, vy, vz]       # quadrotor linear velocity
        self._x_dim = 10
        # control input dimension [thrust, wx, wy, wz]
        self._u_dim = 4

        # cost matrix for tracking the path point position
        self._Q_track_pos = np.diag([100, 100, 100])
        # cost matrix for tracking the path point attitude
        self._Q_track_att = np.diag([1, 10, 10, 10])
        # cost matrix for tracking the path point velocity
        self._Q_track_vel = np.diag([10, 10, 10])

        # cost matrix for the control input
        self._Q_u = np.diag([1, 0.1, 0.1, 0.1])

        self._hover_thrust = (G_z - accel_min)/(accel_max - accel_min)
        self._quad_x0 = [1.0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]
        self._quad_u0 = [self._hover_thrust, 0, 0, 0]

        self._ipopt_options = {
            'verbose': False,
            'ipopt.tol': 1e-4,
            # 'ipopt.acceptable_tol': 1e-4,
            'ipopt.max_iter': 1000,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 0,
            'print_time': False
        }
        self.nlp_x0 = []  # initial guess of nlp variables
        self.nlp_lbx = []  # lower bound of the variables, lbx <= nlp_x
        self.nlp_ubx = []  # upper bound of the variables, nlp_x <= ubx
        self.nlp_lbg = []  # lower bound of constraint functions, lbg <= g
        self.nlp_ubg = []  # upper bound of constraint functions, g <= lbg

        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self._x_dim)]
        x_max = [+x_bound for _ in range(self._x_dim)]
        u_min = [0, -w_max_xy, -w_max_xy, -w_max_yaw]
        u_max = [1, w_max_xy, w_max_xy, w_max_yaw]
        g_min = [0 for _ in range(self._x_dim)]
        g_max = [0 for _ in range(self._x_dim)]

        self.nlp_x0 += self._quad_x0
        self.nlp_lbx += x_min
        self.nlp_ubx += x_max
        self.nlp_lbg += g_min
        self.nlp_ubg += g_max
        for i in range(self._N):
            self.nlp_x0 += self._quad_u0
            self.nlp_x0 += self._quad_x0
            self.nlp_lbx += u_min
            self.nlp_ubx += u_max
            self.nlp_lbx += x_min
            self.nlp_ubx += x_max
            self.nlp_lbg += g_min
            self.nlp_ubg += g_max

        self.sol = []

    def define_solver_python(self):
                # states
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), ca.SX.sym('qz')
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        # control input
        thrust, wx, wy, wz = ca.SX.sym('thrust'), ca.SX.sym('wx'), ca.SX.sym('wy'), ca.SX.sym('wz')

        self._x = ca.vertcat(px, py, pz,
                             qw, qx, qy, qz,
                             vx, vy, vz)
        self._u = ca.vertcat(thrust, wx, wy, wz)

        accel_z_sp = -( accel_min*(1-thrust) + accel_max*thrust )
        x_dot = ca.vertcat(
            vx,
            vy,
            vz,
            0.5 * (-wx * qx - wy * qy - wz * qz),
            0.5 * (wx * qw + wz * qy - wy * qz),
            0.5 * (wy * qw - wz * qx + wx * qz),
            0.5 * (wz * qw + wy * qx - wx * qy),
            2 * (qw * qy + qx * qz) * accel_z_sp,
            2 * (qy * qz - qw * qx) * accel_z_sp,
            (qw * qw - qx * qx - qy * qy + qz * qz) * accel_z_sp + G_z
        )
        self.f = ca.Function('f', [self._x, self._u], [x_dot], ['x', 'u'], ['ode'])
        F = self.sys_dynamics(self._dt)
        fMap = F.map(self._N, 'openmp')

        #
        Delta_pos = ca.SX.sym('Delta_pos', 3)
        Delta_att = ca.SX.sym('Delta_att', 4)
        Delta_vel = ca.SX.sym('Delta_vel', 3)
        Delta_u = ca.SX.sym('Delta_u', self._u_dim)

        cost_pos = Delta_pos.T @ self._Q_track_pos @ Delta_pos
        cost_att = Delta_att.T @ self._Q_track_att @ Delta_att
        cost_vel = Delta_vel.T @ self._Q_track_vel @ Delta_vel
        cost_u = Delta_u.T @ self._Q_u @ Delta_u
        f_cost_pos = ca.Function('cost_pos', [Delta_pos], [cost_pos])
        f_cost_att = ca.Function('cost_att', [Delta_att], [cost_att])
        f_cost_vel = ca.Function('cost_vel', [Delta_vel], [cost_vel])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])

        self.nlp_x = []  # nlp variables

        self.mpc_obj = 0  # objective
        self.nlp_g = []  # constraint functions
        

        P = ca.SX.sym('P', (self._x_dim + 3) * (self._N+1))
        X = ca.SX.sym('X', self._x_dim, self._N + 1)
        U = ca.SX.sym('U', self._u_dim, self._N)
        X_next = fMap(X[:, :self._N], U)

        # starting point
        self.nlp_x += [X[:, 0]]
        
        self.nlp_g += [X[:, 0] - P[0: self._x_dim]]
        

        for k in range(self._N):
            # add control input variables
            self.nlp_x += [U[:, k]]            

            p_index = (self._x_dim + 3) * (k + 1)
            delta_pos_k = (X[:3, k + 1] - P[p_index: p_index + 3])
            delta_att_k = (X[3:7, k + 1] - P[p_index + 3: p_index + 7])
            delta_vel_k = (X[7:10, k + 1] - P[p_index + 7: p_index + 10])
            w_pos_k = P[p_index + self._x_dim]
            w_att_k = P[p_index + self._x_dim + 1]
            w_vel_k = P[p_index + self._x_dim + 2]

            cost_track_k = f_cost_pos(delta_pos_k) * w_pos_k + f_cost_att(delta_att_k) * w_att_k +\
                           f_cost_vel(delta_vel_k) * w_vel_k

            delta_u_k = U[:, k] - [self._hover_thrust, 0, 0, 0]
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_track_k + cost_u_k
            #
            self.nlp_x += [X[:, k + 1]]

            self.nlp_g += [X_next[:, k] - X[:, k + 1]]

        nlp_dict = {
            'f': self.mpc_obj,
            'x': ca.vertcat(*self.nlp_x),
            'p': P,
            'g': ca.vertcat(*self.nlp_g)
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_dict, self._ipopt_options)

    def generate_solver_c(self, cname):
        self.solver.generate_dependencies(cname)
        ## system('gcc -fPIC -shared -O3' + cname + ' -o ' + './casadi_gen/mpc.so')


    def load_solver(self, path):
        self.solver = ca.nlpsol('solver', 'ipopt', path, self._ipopt_options)


    def solve(self, path):

        self.sol = self.solver(
            x0=self.nlp_x0,
            lbx=self.nlp_lbx,
            ubx=self.nlp_ubx,
            p=path,
            lbg=self.nlp_lbg,
            ubg=self.nlp_ubg
        )

        sol_x0 = self.sol['x'].full()
        opt_u = sol_x0[self._x_dim: self._x_dim + self._u_dim]

        # self.nlp_x0 = sol_x0
        self.nlp_x0 = list(sol_x0[self._x_dim + self._u_dim: 2 * (self._x_dim + self._u_dim)]) + \
                      list(sol_x0[self._x_dim + self._u_dim:])

        x0_array = np.reshape(sol_x0[:-self._x_dim], newshape=(-1, self._x_dim + self._u_dim))

        return opt_u, x0_array

    def sys_dynamics(self, dt):
        M = 4
        DT = dt / M
        X0 = ca.SX.sym('X', self._x_dim)
        U = ca.SX.sym('U', self._u_dim)

        X = X0
        for _ in range(M):
            k1 = DT * self.f(X, U)
            k2 = DT * self.f(X + 0.5 * k1, U)
            k3 = DT * self.f(X + 0.5 * k2, U)
            k4 = DT * self.f(X + k3, U)
            X = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        F = ca.Function('F', [X0, U], [X])
        return F
