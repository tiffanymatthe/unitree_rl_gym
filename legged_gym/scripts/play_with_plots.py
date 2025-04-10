import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import pprint
import pickle
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

NUM_ENVS = 1
HAS_LIN_VEL = False
PLOT = True

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, NUM_ENVS)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.add_control_freq = False
    env_cfg.domain_rand.add_delay = False
    env_cfg.domain_rand.randomize_damping = False
    env_cfg.domain_rand.randomize_stiffness = False

    vel = [0, 0.8, 0]
    vel_str = "_".join(map(str, vel))

    env_cfg.commands.ranges.lin_vel_x = [vel[0],vel[0]]
    env_cfg.commands.ranges.lin_vel_y = [vel[1],vel[1]]
    env_cfg.commands.ranges.ang_vel_yaw = [vel[2],vel[2]]
    env_cfg.commands.ranges.heading = [0,0]

    env_cfg.env.test = True

    env_cfg.terrain.plane = True
    
    # # UNCOMMENT WHEN YOU ONLY WANT TO SEE LINEAR VELOCITY TRACKING REWARDS
    # print("ENV CONFIG: removing all rewards except linear velocity tracking")
    # env_cfg.rewards.scales.termination = 0
    # env_cfg.rewards.scales.tracking_ang_vel = 0
    # env_cfg.rewards.scales.lin_vel_z = 0
    # env_cfg.rewards.scales.ang_vel_xy = 0
    # env_cfg.rewards.scales.orientation = 0
    # env_cfg.rewards.scales.torques = 0
    # env_cfg.rewards.scales.dof_vel = 0
    # env_cfg.rewards.scales.dof_acc = 0
    # env_cfg.rewards.scales.base_height = 0 
    # env_cfg.rewards.scales.feet_air_time = 0
    # env_cfg.rewards.scales.collision = 0
    # env_cfg.rewards.scales.feet_stumble = 0 
    # env_cfg.rewards.scales.action_rate = 0
    # env_cfg.rewards.scales.stand_still = 0

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    all_obs = []
    all_lin_vel_obs = []

    obs = env.get_observations()
    all_obs.append(obs.cpu().numpy())
    all_lin_vel_obs.append(obs[:, 0:3].cpu().numpy() * 0) # placeholder
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if RECORD_FRAMES:
        record_path = os.path.dirname(task_registry.resume_path) + f"/recordings_{vel_str}/"
        ppo_runner.env.set_recorder(record_path)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    all_rews = torch.zeros((NUM_ENVS,), device=args.rl_device)
    avg_rewards = 0

    all_lin_vel_errs = torch.zeros((NUM_ENVS,), device=args.rl_device)
    avg_lin_vel_errs = 0

    all_ang_vel_errs = torch.zeros((NUM_ENVS,), device=args.rl_device)
    avg_ang_vel_errs = 0

    num_finishes = 0
    num_terminated_failed = 0

    for i in tqdm(range(2 * int(env.max_episode_length))):
        if MOVE_CAMERA:
            x_pos = ppo_runner.env.root_states[0][0]
            y_pos = ppo_runner.env.root_states[0][1]
            ppo_runner.env.set_camera([x_pos,3+y_pos,1], [x_pos,0+y_pos,0.5])

        if (i % int(env.max_episode_length) == 0):
            # Additional Randomization
            for _ in range(20):
                env.gym.simulate(env.sim)

        actions = policy(obs.detach())
        obs, lin_vel_obs, rews, dones, infos = env.step(actions.detach())
        all_obs.append(obs.cpu().numpy())
        if HAS_LIN_VEL:
            all_lin_vel_obs.append(obs[:,0:3].cpu().numpy())
        else:
            all_lin_vel_obs.append(lin_vel_obs.cpu().numpy())

        # if (i % int(env.max_episode_length) == 1):
        #     input("press to play")

        all_rews += rews
        all_lin_vel_errs += infos["metrics"]["lin_vel_xy_error"]
        all_ang_vel_errs += infos["metrics"]["ang_vel_error"]
        done_rewards = all_rews[dones]
        done_lin_vel_errs = all_lin_vel_errs[dones]
        done_ang_vel_errs = all_ang_vel_errs[dones]
        if done_rewards.numel() != 0:
            num_terminated_failed += torch.sum(infos["metrics"]["terminated_from_contact"])
            num_finishes += done_rewards.numel()
            avg_rewards += torch.sum(done_rewards)
            done_length = infos["metrics"]["curr_episode_length"][dones]
            avg_lin_vel_errs += torch.sum(done_lin_vel_errs / done_length)
            avg_ang_vel_errs += torch.sum(done_ang_vel_errs / done_length)
            # plot all obs
            if PLOT:
                offset = 0 if not HAS_LIN_VEL else 3
                angular_velocities = [o[0][0+offset:offset+3] for o in all_obs]
                grav_vectors= [o[0][3+offset:offset+6] for o in all_obs]
                lin_x_y_yaw_commands = [o[0][6+offset:offset+9] for o in all_obs]
                dof_positions = [o[0][9+offset:offset+9+12] for o in all_obs]
                dof_velocities = [o[0][9+12+offset:offset+9+24] for o in all_obs]
                policy_output_actions = [o[0][9+24+offset:offset+9+36] for o in all_obs]
                fig, axs = plt.subplots(3, 2 , figsize=(12,8))
                axs[0, 0].plot(angular_velocities)
                axs[0, 0].set_title('Angular Velocities')

                axs[0, 1].plot(grav_vectors)
                axs[0, 1].set_title('Gravitational Vectors')

                axs[1, 0].plot(lin_x_y_yaw_commands)
                axs[1, 0].set_title('Linear X Y Yaw Commands')

                axs[1, 1].plot(dof_positions)
                axs[1, 1].set_title('DOF Positions')

                axs[2, 0].plot(dof_velocities)
                axs[2, 0].set_title('DOF Velocities')

                axs[2, 1].plot(policy_output_actions)
                axs[2, 1].set_title('Policy Output Actions')

                fig1, axs1 = plt.subplots(4, 3, figsize=(12,8))
                axs1 = axs1.flatten()

                REAL_JOINT_LABELS = np.array(["FR_0","FR_1","FR_2","FL_0","FL_1","FL_2","RR_0","RR_1","RR_2","RL_0","RL_1","RL_2"])
                REAL_TO_SIM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

                print(REAL_JOINT_LABELS[REAL_TO_SIM])

                JOINT_LIMITS = {
                    "FR_0": [-0.837758,0.837758],
                    "FR_1": [-1.5708,3.4907],
                    "FR_2": [-2.7227, -0.83776],
                    "FL_0": [-0.837758,0.837758],
                    "FL_1": [-1.5708,3.4907],
                    "FL_2": [-2.7227, -0.83776],
                    "RR_0": [-0.837758,0.837758],
                    "RR_1": [-0.5236,4.5379],
                    "RR_2": [-2.7227, -0.83776],
                    "RL_0": [-0.837758,0.837758],
                    "RL_1": [-0.5236,4.5379],
                    "RL_2": [-2.7227, -0.83776],
                }
                
                for i in range(12):
                    scaled_position = np.array([x[i] / env.obs_scales.dof_pos + env.default_dof_pos[0][i].cpu() for x in dof_positions])

                    scaled_action = np.array([x[i] * env.cfg.control.action_scale + env.default_dof_pos[0][i].cpu() for x in policy_output_actions])

                    axs1[i].plot(scaled_position, label="position (rad)") # use action_scale
                    axs1[i].plot(scaled_action, label="action (rad)")

                    label = REAL_JOINT_LABELS[REAL_TO_SIM[i]]

                    axs1[i].axhline(JOINT_LIMITS[label][0], linestyle="--", color="black")
                    axs1[i].axhline(JOINT_LIMITS[label][1], linestyle="--", color="black")
                    
                    axs1[i].set_title(label)
                    if i == 11:
                        axs1[i].legend()

                fig2, axs2 = plt.subplots(3, 1, figsize=(12,8))
                axs2 = axs2.flatten()
                labels = ["vel_x", "vel_y", "vel_z"]

                for i in range(3):
                    true_lin_vel = [x[0][i] / 2 for x in all_lin_vel_obs]
                    axs2[i].plot(true_lin_vel, label="true")

                    if i < 2:
                        target_lin_vel = [x[i] / env.obs_scales.lin_vel for x in lin_x_y_yaw_commands]
                        axs2[i].plot(target_lin_vel, label="target")

                    axs2[i].set_title(labels[i])

                pickle.dump([axs, axs1, axs2], open(f"{args.load_run}_{vel_str}.pickle", "wb"))
                print("DUMPED")

                # if RECORD_FRAMES:
                #     import subprocess
                #     rc = subprocess.call(["legged_gym/scripts/images2video.sh", f"{record_path}"])
                #     rc = subprocess.call(["rm", "-rf", f"{record_path}"])

                plt.show()
                input("Continue by entering.")
            all_obs = []
            all_lin_vel_obs = []
        all_rews *= ~dones
        all_lin_vel_errs *= ~dones
        all_ang_vel_errs *= ~dones

    to_print = {
        "finished runs": num_finishes,
        "avg. total episodic rew.": avg_rewards.item() / num_finishes,
        "avg. xy tracking err. per episode": avg_lin_vel_errs.item() / num_finishes,
        "avg. angular tracking err. per episode": avg_ang_vel_errs.item() / num_finishes,
        "percentage of failed episodes": num_terminated_failed.item() / num_finishes
    }

    pprint.pprint(to_print)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = True
    MOVE_CAMERA = True
    args = get_args()
    play(args)
