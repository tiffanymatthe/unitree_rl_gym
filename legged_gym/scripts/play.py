import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import pprint
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from rsl_rl.modules import ActorCritic

NUM_ENVS = 1

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

    env_cfg.env.test = True
    
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
    all_velocities = []

    obs = env.get_observations()
    all_obs.append(obs.cpu().numpy())
    all_velocities.ppaned(obs[:,0:3].cpu().numpy())
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    estimator_path = None
    estimator = ActorCritic(
        num_actor_obs=48,
        num_critic_obs=48,
        num_actions=3, # linear velocity x y z
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation='elu',
        init_noise_std=1.0
    )

    estimator.load_state_dict(torch.load(estimator_path, map_location=torch.device("cuda:0"))['model_state_dict'])
    for param in estimator.parameters():
        param.requires_grad = False
    
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

    for i in tqdm(range(10 * int(env.max_episode_length))):

        if (i % int(env.max_episode_length) == 0):
            # Additional Randomization
            for _ in range(20):
                env.gym.simulate(env.sim)

        lin_velocities = estimator(obs.detach()[:,3:]).detach()
        full_obs = torch.concatenate(lin_velocities, obs.detach()[:,3:])
        actions = policy(full_obs)
        obs, _, rews, dones, infos = env.step(actions.detach())
        all_obs.append(obs.cpu().numpy())
        all_velocities.append(lin_velocities.cpu().numpy())

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
            true_linear_velocities = [o[0][0:3] for o in all_obs]
            estimated_linear_velocities = [o[0][0:3] for o in all_velocities]
            angular_velocities = [o[0][3:6] for o in all_obs]
            grav_vectors= [o[0][6:9] for o in all_obs]
            lin_x_y_yaw_commands = [o[0][9:12] for o in all_obs]
            dof_positions = [o[0][12:9+12+3] for o in all_obs]
            dof_velocities = [o[0][9+12+3:9+24+3] for o in all_obs]
            policy_output_actions = [o[0][9+24+3:9+36+3] for o in all_obs]
            fig, axs = plt.subplots(4, 2 , figsize=(12,8))
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

            axs[3, 0].plot(true_linear_velocities, label="true")
            axs[3, 1].plot(estimated_linear_velocities, label="estimated")
            axs[3, 0].set_title('Linear Velocities (scaled by factor of 2 compared to command)')
            axs[3, 1].set_title('Linear Velocities (scaled by factor of 2 compared to command)')

            fig2, axs2 = plt.subplots(3, 1, figsize=(12,8))
            axs2 = axs2.flatten()
            labels = ["vel x","vel y", "vel z"]

            for i in range(3):
                true_lin_vel = [x[i] / 2 for x in true_linear_velocities]
                est_lin_vel = [x[i] / 2 for x in estimated_linear_velocities]
                axs2[i].plot(true_lin_vel, label="true")
                axs2[i].plot(est_lin_vel, label="est")

                if i < 2:
                    target_lin_vel = [x[i] for x in lin_x_y_yaw_commands]
                    axs2[i].plot(target_lin_vel, label="target cmd")

                axs2[i].set_title(labels[i])

            axs2.legend() 

            fig1, axs1 = plt.subplots(4, 3, figsize=(12,8))

            REAL_JOINT_LABELS = np.array(["FR_0","FR_1","FR_2","FL_0","FL_1","FL_2","RR_0","RR_1","RR_2","RL_0","RL_1","RL_2"])
            REAL_TO_SIM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

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
                scaled_position = [x[i] / env.obs_scales.dof_pos + env.default_dof_pos[i] for x in dof_positions]

                scaled_action = [x[i] * env.cfg.control.action_scale + env.default_dof_pos[i] for x in policy_output_actions]

                axs1[i].plot(scaled_position, label="position (rad)") # use action_scale
                axs1[i].plot(scaled_action, label="action (rad)")

                label = REAL_JOINT_LABELS[REAL_TO_SIM[i]]

                axs1[i].axhline(JOINT_LIMITS[label][0], linestyle="--", color="black")
                axs1[i].axhline(JOINT_LIMITS[label][1], linestyle="--", color="black")
                
                axs1[i].set_title(label)
                if i == 11:
                    axs1[i].legend()

            plt.show()
            input("Continue by entering.")
            all_obs = []
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
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
