#!/home/ziyuzhou/anaconda3/bin/python3.9
import matplotlib.pyplot as plt
import matplotlib.animation

from simulator.parameter import *

from simulator.visualization import *
from control.trajectorys import *
from control.mpcOr import *
from simulator.quadrotorOr import *

# from simulator.quadrotor import *
# from control.mpc import *

import time

s_dt = 0.001  # simulation step

quad = Quadrotor(s_dt)
sim_visual = Visualization()
mpc_ctrl = QuadrotorMPC(1, 0.1)
# mpc_ctrl.define_solver_python()
mpc_ctrl.load_solver("./control/casadi_gen/mpc_step10.so")

start_pos = np.array([1, 0, -1])

trj_planner = CircleTrj(np.array([0.5,0,-1]), 0.5, 1, 0.1)
# trj_planner = TrackTrj("result.csv", 1, 0.1)

control_state = 0

def simulation_run(T=100):
    s_t = 0
    cnt = 0
    path_ref = np.zeros(13*11)
    path_ref[13 * 10:] = np.array([start_pos[0], start_pos[1], start_pos[2],
                                   1, 0, 0, 0,
                                   0, 0, 0,
                                   1, 1, 1])
    control_u0 = [0.5, 0, 0, 0]

    while s_t < T:
        s_t += s_dt
        quad_state = quad.run(control_u0)

        cnt = cnt + 1
        # 0.01s for mpc control
        if cnt % 10 == 0:
            t1 = time.time()
            path_ref[:10] = quad_state

            global control_state
            # state one: fly to start_point
            if control_state == 0:
                path_ref[13 * 10:] = np.array([start_pos[0], start_pos[1], start_pos[2],
                                              1, 0, 0, 0,
                                              0, 0, 0,
                                              1, 1, 1])
                if np.linalg.norm(quad_state[:3] - start_pos) < 0.1:
                    trj_planner.reset(s_t)
                    trj_planner.set_target_angular_velocity(5)
                    control_state = 1
            if control_state == 1:
                trj = trj_planner.step(s_t)
                path_ref[13:] = trj
                if trj_planner.get_running_time() > 40:
                    trj_planner.set_target_angular_velocity(0)
                    control_state = 2
            if control_state == 2:
                trj = trj_planner.step(s_t)
                path_ref[13:] = trj

            control_u0, a = mpc_ctrl.solve(path_ref)
            t2 = time.time()
            print(path_ref[0:2],path_ref[13*1:13*1+2],path_ref[13*2:13*2+2])
            # print("sim time:", s_t)
            # print("quad attitude:", quad_state[3:7])

        if cnt % 50 == 0:
            # print(q_state[:3], q_state[7:10])
            # print(control_u0)
            info = {
                'quad_pos': quad_state[:3],
                'quad_axes': quad.get_axes()
            }
            yield info



ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=simulation_run,
                              init_func=sim_visual.init_animate, interval=0, blit=True, repeat=False)

plt.tight_layout()
plt.show()
