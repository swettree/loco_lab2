'''
Author: Guohua Zhang
Date: 2025.1.16
Description: This script is used to deploy the RL policy in mujoco simulator for unitree go1.
Version: 1.0
'''

import numpy as np
import mujoco
import mujoco_viewer
from scipy.spatial.transform import Rotation as R
from utils.logger import Logger
import torch
import pygame
from threading import Thread

def keyboard_control_mode(keyboard_use, keyboard_opened, speed_step):
    global exit_flag
    if keyboard_use:
        pygame.init()
        screen = pygame.display.set_mode((600, 500))
        try:
            keyboard_opened = True
            print(f"Mode: keyboard control command")
        except Exception as e:
            print(f"cannot open keyboard")

        exit_flag = False
        def print_command():
            print("Keyboard Control: cmd.vx = {:.2f}, cmd.vy = {:.2f}, cmd.dyaw = {:.2f}. Press esc to out".format(cmd.vx, cmd.vy, cmd.dyaw))

        def handle_keyboard_input():
            global exit_flag
            while not exit_flag:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit_flag = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_w:
                            cmd.vx = min(cmd.vx + speed_step, cmd.x_vel_max)
                            print_command()
                        elif event.key == pygame.K_s:
                            cmd.vx = max(cmd.vx - speed_step, -cmd.x_vel_max)
                            print_command()
                        elif event.key == pygame.K_a:
                            cmd.vy = max(cmd.vy + speed_step, -cmd.y_vel_max)
                            print_command()
                        elif event.key == pygame.K_d:
                            cmd.vy = min(cmd.vy - speed_step, cmd.y_vel_max)
                            print_command()
                        elif event.key == pygame.K_j:
                            cmd.dyaw = min(cmd.dyaw + speed_step, cmd.yaw_vel_max)
                            print_command()
                        elif event.key == pygame.K_l:
                            cmd.dyaw = max(cmd.dyaw - speed_step, -cmd.yaw_vel_max)
                            print_command()
                pygame.time.delay(100)
            keyboard_thread.join()

        if keyboard_opened and keyboard_use:
            keyboard_thread = Thread(target = handle_keyboard_input)
            keyboard_thread.start()

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0
    x_vel_max = 1.0
    y_vel_max = 1.0
    yaw_vel_max = 1.0

_gait_command = np.zeros(5)
_gait_command[0] = 2.0 # Gait frequency range [Hz] [1.5-2.5]
_gait_command[1] = 0.5 # Contact duration range [0-1]
# _gait_command[1] = 0.4
_gait_command[2] = 0.5 # Phase offset2 range [0-1]
_gait_command[3] = 0.5 # Phase offset3 range [0-1]
_gait_command[4] = 0.0 # Phase offset4 range [0-1]

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''
    Calculates torques from position commands
    '''
    torque_out = (target_q - q) * kp + (target_dq - dq)* kd
    return torque_out

def compute_gait_phase(loop_count, loop_frequency, gait_command):
    """
    Computes the gait phase based on the current loop count and the gait period.
    """
    # Calculate gait indices
    gait_indices = torch.remainder(
        torch.tensor((loop_count / loop_frequency) * gait_command[0], dtype=torch.float32, device='cpu'), 
        1.0
    )
    # Convert to sin/cos representation
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)
    
    return torch.stack([sin_phase, cos_phase], dim=0)

def get_obs(data, model):
    '''
    Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    base_pos = q[:3]
    foot_positions = []
    foot_forces = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if 'calf' in body_name: 
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces.append(data.cfrc_ext[i][2].copy().astype(np.double)) 
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    print("Load mujoco xml from:", cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos
    mujoco.mj_step(model, data)
    for i in range(model.nbody): # nbody = 14
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(body_name)
    
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -45
    viewer.cam.lookat[:] =np.array([0.0,-0.25,0.824])

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    obs_buf = np.zeros((cfg.env.num_observations), dtype=np.double)

    dofmapping_mujoco2lab = cfg.robot_config.dofmapping_mujoco2lab
    dofmapping_lab2mujoco = cfg.robot_config.dofmapping_lab2mujoco

    count_lowlevel = 0
    logger = Logger(cfg.sim_config.dt)
    stop_state_log = 1000 # number of steps before plotting states
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    for i in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):

        # Obtain an observation
        q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data, model)
        q = q[-cfg.env.num_actions:] # joint position dim: 12
        dq = dq[-cfg.env.num_actions:] # joint velocity dim: 12
        base_z = base_pos[2]
        foot_z = foot_positions
        foot_force_z = foot_forces
        # print("before mapping:", q)
        q = q[dofmapping_mujoco2lab]
        # print("after mapping:", q)
        dq = dq[dofmapping_mujoco2lab]
        
        '''
        obs_buf: dim = 270
        current_obs = torch.cat((   
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel  * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
            ),dim=-1)
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        '''
        # 200hz -> 50hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            gait_phase = compute_gait_phase(i, cfg.sim_config.loop_frequency, _gait_command)
            gait_command = torch.tensor(_gait_command, dtype=torch.float32)
            # process the observation
            current_obs = np.zeros([1, cfg.env.num_one_step_observations], dtype=np.double)
            current_obs[0, 0] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            current_obs[0, 1] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            current_obs[0, 2] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            current_obs[0, 3:5] = gait_phase
            current_obs[0, 5:10] = gait_command

            current_obs[0, 10] = omega[0] * cfg.normalization.obs_scales.ang_vel
            current_obs[0, 11] = omega[1] * cfg.normalization.obs_scales.ang_vel
            current_obs[0, 12] = omega[2] * cfg.normalization.obs_scales.ang_vel
            current_obs[0, 13:16] = gvec
            current_obs[0, 16:28] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
            # current_obs[0, 9:21] = q * cfg.normalization.obs_scales.dof_pos
            current_obs[0, 28:40] = dq * cfg.normalization.obs_scales.dof_vel
            current_obs[0, 40:52] = action

            current_obs = np.clip(current_obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            obs_buf = np.concatenate((current_obs[0, :cfg.env.num_one_step_observations],
                                      obs_buf[:-cfg.env.num_one_step_observations],),
                                      axis=-1)
            # print(torch.tensor(obs_buf).float().unsqueeze(0).shape, torch.tensor(obs_buf).float().unsqueeze(0).dtype)
            action[:] = policy(torch.tensor(obs_buf).float().unsqueeze(0).detach()).detach().numpy()
            # print(action)
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            actions_scaled = action * cfg.control.action_scale
            # actions_scaled[[0, 3, 6, 9]] *= cfg.control.hip_reduction
            target_q = cfg.robot_config.default_dof_pos + actions_scaled
            # print(actions_scaled)

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Calc torques
        tau = pd_control(target_q, q, cfg.robot_config.kps, 
                             target_dq, dq, cfg.robot_config.kds)
        # Clamp torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        tau = tau[dofmapping_lab2mujoco]
        # print(tau)
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

        # logger
        joint_index = 1 # which joint is used for logging
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': action[joint_index].item() * cfg.control.action_scale + cfg.robot_config.default_dof_pos[joint_index].item(),
                    'dof_pos': q[joint_index].item(),
                    'dof_vel': dq[joint_index].item(),
                    'dof_torque': tau[joint_index].item(),
                    'command_x': cmd.vx,
                    'command_y': cmd.vy,
                    'command_yaw': cmd.dyaw,
                    'base_vel_x': v[0].item(),
                    'base_vel_y': v[1].item(),
                    'base_vel_z': v[2].item(),
                    'base_vel_yaw': omega[2].item(),
                    'contact_forces_z': foot_force_z
                }
            )
        elif i == stop_state_log:
            logger.plot_states()

    viewer.close()

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Deployment script of unitree go1 in mujoco simulator.')
    parser.add_argument('--load_model', type=str, default='./policy/go1/policy.pt', help='Run to load from.')
    parser.add_argument('--robot_load_path', type=str, default="./robot-descriptions/go1/xml/", help='Robot load path.')
    parser.add_argument('--terrain', action='store_true', default=False, help='Terrain or plane')
    parser.add_argument('--terrain_path', type=str, default="world_terrain.xml", help='Terrain path')
    parser.add_argument('--control_mode', type=str, default="auto", help='Keyboard or auto')
    args = parser.parse_args()
    args.load_model = './policy/go1/policy.pt'
    args.terrain = True
    args.control_mode = 'keyboard'
    args.terrain_path = 'world_heightfield.xml'
    args.robot_load_path = './robot-descriptions/go1/xml/'

    if args.control_mode == 'keyboard':
        keyboard_use = True
        keyboard_opened = False
        speed_step = 0.1
        keyboard_control_mode(keyboard_use, keyboard_opened, speed_step)
    elif args.control_mode == 'auto':
        cmd.vx = 1.0
        cmd.vy = 0.0
        cmd.dyaw = 0.0

    class Sim2simCfg:
        class sim_config:
            if args.terrain:
                mujoco_model_path = args.robot_load_path + args.terrain_path
            else:
                mujoco_model_path = args.robot_load_path + 'world.xml'
            sim_duration = 120.0 # s
            dt = 0.005 # s
            loop_frequency = 1 / dt
            decimation = 4
        
        class env:
            num_actions = 12
            num_one_step_observations = 45 + 2 + 5
            history_len = 5
            num_observations = num_one_step_observations * history_len

        class normalization:
            class obs_scales:
                lin_vel = 1.0
                ang_vel = 1.0
                dof_pos = 1.0
                dof_vel = 1.0
            clip_observations = 100.
            clip_actions = 100.
    
        class control:
            # PD Drive parameters:
            control_type = 'P'
            action_scale = 0.25
            decimation = 4
            hip_reduction = 1.0

        class robot_config:
            dofmapping_mujoco2lab = [
                0, 3, 6, 9,
                1, 4, 7, 10,
                2, 5, 8, 11
            ]
            dofmapping_lab2mujoco = [
                0, 4, 8, 
                1, 5, 9,
                2, 6, 10,
                3, 7, 11
            ]
            kp = 20.0
            kd = 0.5
            tau = 23.7
            tau_calf = 33.5
            kps = np.array([kp, kp, kp, kp, kp, kp, kp, kp, kp, kp, kp, kp], dtype=np.double)
            kds = np.array([kd, kd, kd, kd, kd, kd, kd, kd, kd, kd, kd, kd], dtype=np.double)
            tau_limit = np.array([tau, tau, tau_calf, tau, tau, tau_calf, tau, tau, tau_calf, tau, tau, tau_calf], dtype=np.double)
            default_dof_pos = np.array([
                0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5
            ])
            default_dof_pos = default_dof_pos[dofmapping_mujoco2lab]
            tau_limit = tau_limit[dofmapping_mujoco2lab]
            print("default_dof_pos:", default_dof_pos)
            print("tau_limit", tau_limit)
            # mujoco 
            # default_dof_vel/default_dof_pos/action/tau/tau_limit
            # [
            #   0 'FL_hip_joint', 1 'FL_thigh_joint', 2 'FL_calf_joint', \
            #   3 'FR_hip_joint', 4 'FR_thigh_joint', 5 'FR_calf_joint', \
            #   6 'RL_hip_joint', 7 'RL_thigh_joint', 8 'RL_calf_joint', \
            #   9 'RR_hip_joint', 10 'RR_thigh_joint', 11 'RR_calf_joint'
            # ]

            # isaaclab
            # default_dof_vel/default_dof_pos/action/tau/tau_limit
            # [
            #   0 'FL_hip_joint', 1 'FR_hip_joint', 2 'RL_hip_joint', 3 'RR_hip_joint', \
            #   4 'FL_thigh_joint', 5 'FR_thigh_joint', 6 'RL_thigh_joint', 7 'RR_thigh_joint', \
            #   8 'FL_calf_joint', 9 'FR_calf_joint', 10 'RL_calf_joint', 11 'RR_calf_joint'
            # ]

    policy = torch.jit.load(args.load_model)
    print("Load model from:", args.load_model)
    print(policy)
    run_mujoco(policy, Sim2simCfg())