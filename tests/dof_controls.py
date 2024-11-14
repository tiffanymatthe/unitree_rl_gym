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
import math

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    # if EXPORT_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    #     print('Exported policy as jit script to: ', path)


    dt = env_cfg.sim.dt 
    expected = []
    real = []
    for _ in range(12):
        expected.append([])
        real.append([])

    print(expected[1])

    for j in range(12):
        # for i in range(10*int(env.max_episode_length)):
        for i in range(1000):
            # actions = policy(obs.detach())
            actions = torch.zeros([1,12])
            actions[0, j] = math.sin(i*2*math.pi/25)*math.pi*5
            # actions[0,1] = 1
            # actions[0,1]= math.sin(i*0.1)*math.pi/2
            # print("Actions:", actions, i)
            obs, _, rews, dones, infos = env.step(actions.detach())
            poses = obs[0, 12:24]*4

            # if len(expected[0]) < 500:
            #     for i in range(12):
            expected[j].append(actions[0, j])
            real[j].append(poses[j])
            # elif len(expected[0]) == 500:
        with open(f'data/joint_{j}.pkl','wb') as data:
            torch.save({"action": expected,
                        "result": real}, data)
        # with open(f'data/real_{j}.pkl','wb') as data:
        #     torch.save(real, data)
                # raise RuntimeError("stop")

        
            # print(obs)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
