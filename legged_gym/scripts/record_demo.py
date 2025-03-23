import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, get_load_path, task_registry, Logger
import numpy as np
import torch
from pathlib import Path
from tqdm import *


def _create_task(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    # env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.add_control_freq = False
    # env_cfg.domain_rand.add_delay = False
    # env_cfg.domain_rand.randomize_damping = False
    # env_cfg.domain_rand.randomize_stiffness = False
    env_cfg.commands.ranges.lin_vel_x = [0.5,0.5]
    env_cfg.commands.ranges.lin_vel_y = [0,0]
    env_cfg.commands.ranges.ang_vel_yaw = [0,0]
    env_cfg.commands.ranges.heading = [0,0]

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    return env, train_cfg

def _get_ppo_runner(env, args, train_cfg):
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    return ppo_runner

def _play_walk(ppo_runner, env):
    policy = ppo_runner.get_inference_policy(device=env.device)

    env.reset()

    for i in range(int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

def demo(args):
    env, train_cfg = _create_task(args)
    ppo_runner = _get_ppo_runner(env, args, train_cfg)

    path = os.path.dirname(task_registry.resume_path)
    # print("path:", path)
    for pt in  tqdm([k for k in os.listdir(path) if 'curriculum_' in k]):
        # print("pt:", pt)
        resume_path = os.path.join(path, pt)
        print(f"Loading model from: {resume_path}")
        ppo_runner.load(resume_path)

        record_path = os.path.join(path, "recording", Path(pt).stem)
        # print("recording path", record_path)
        ppo_runner.env.set_recorder(record_path)

        policy = ppo_runner.get_inference_policy()
        env.reset()
        obs = env.get_observations()

        for i in trange(int(env.max_episode_length)):
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    args = get_args()
    demo(args)
