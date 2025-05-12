import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict, CurriculumEnvManager
from rsl_rl.runners import OnPolicyRunner
import torch
from tqdm import trange

class CurriculumTrainer():
    def __init__(self):
        self.i = 0
        
    def curriculum_train(self, args):
        self.env, env_cfg = task_registry.make_env(name=args.task, args=args)
        self.ppo_runner, self.train_cfg = task_registry.make_alg_runner(env=self.env, name=args.task, args=args)

        manager = CurriculumEnvManager(self.env)

        self.env.enable_viewer_sync = False
        
        if not self.train_cfg.runner.resume:
            self._train()  # Initial training
        self.args = args

        while manager.step():
            self._train(param=manager.get_title())

    def _train(self, param="base"):
        self.i+=1
        self.ppo_runner.env = self.env
        max_its = self.train_cfg.runner.max_iterations
        if self.i == 1 and not self.train_cfg.runner.resume:
        #     max_its = 500
        # else:
            max_its = 750
        self.ppo_runner.learn(num_learning_iterations=max_its, init_at_random_ep_len=True)
        self.ppo_runner.save(os.path.join(self.ppo_runner.log_dir, f'curriculum_{self.i}.pt'))


    def _demo(self, record_frames=False, record_name=""):
        if self.env.headless: # no need to play if headless
            return
        
        env_cfg, train_cfg = task_registry.get_cfgs(name=self.args.task)
        env_cfg.terrain.curriculum = False
        env_cfg.commands.ranges.lin_vel_x = [0.3,0.9]
        env_cfg.commands.ranges.lin_vel_y = [0,0]
        env_cfg.commands.ranges.ang_vel_yaw = [0,0]
        env_cfg.commands.ranges.heading = [0,0]

        env_cfg.env.test = True
        self.env.enable_viewer_sync = True

        # prepare environment
        # env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)s
        tmp = self.env.cfg
        self.env._update_cfg(env_cfg)

        obs = self.env.get_observations()
        # load policy
        # train_cfg.runner.resume = True
        # policy = self.ppo_runner.get_inference_policy(device=self.env.device)
        
        with torch.no_grad(): # switch to evaluation mode (dropout for example)
            actor = self.ppo_runner.alg.actor_critic 

            if self.env.device is not None:
                actor.to(self.env.device)
            policy =  actor.act_inference


            if record_frames:
                self.ppo_runner.env.set_recorder(self.ppo_runner.log_dir + f"/recordings/play_walk_{record_name}")

            for i in trange(int(self.env.max_episode_length)):
                actions = policy(obs.detach())
                obs, _, rews, dones, infos = self.env.step(actions.detach())

            self.ppo_runner.env.stop_recorder()

        self.env._update_cfg(tmp)
        self.env.enable_viewer_sync = False


if __name__ == '__main__':
    args = get_args()
    ct = CurriculumTrainer()
    ct.curriculum_train(args)
