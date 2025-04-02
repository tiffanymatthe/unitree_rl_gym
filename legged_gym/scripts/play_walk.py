import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play_walk(args, record_frames=False, record_name=""):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.plane = True
    env_cfg.env.env_spacing = 3
    # env_cfg.noise.add_noise = True
    # env_cfg.domain_rand.randomize_friction = True
    # env_cfg.domain_rand.push_robots = True
    # env_cfg.domain_rand.add_control_freq = True
    # env_cfg.domain_rand.add_delay = True
    # env_cfg.domain_rand.randomize_damping = True
    # env_cfg.domain_rand.randomize_stiffness = True
    env_cfg.commands.ranges.lin_vel_x = [0.5,0.5]
    env_cfg.commands.ranges.lin_vel_y = [0,0]
    env_cfg.commands.ranges.ang_vel_yaw = [0,0]
    env_cfg.commands.ranges.heading = [0,0]

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    if record_frames:
        ppo_runner.env.set_recorder(os.path.dirname(task_registry.resume_path) + f"/recordings/play_walk_{record_name}")
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = True
    MOVE_CAMERA = False
    args = get_args()
    play_walk(args, RECORD_FRAMES)
