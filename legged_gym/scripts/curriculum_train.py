import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict
from rsl_rl.runners import OnPolicyRunner
import torch
from tqdm import trange

class CurriculumTrainer():
    def __init__(self):
        self.i = 0
        
    def curriculum_train(self, args):
        self.env, env_cfg = task_registry.make_env(name=args.task, args=args)
        self.ppo_runner, self.train_cfg = task_registry.make_alg_runner(env=self.env, name=args.task, args=args)

        self.env.enable_viewer_sync = False
        
        # if not self.train_cfg.runner.resume:
        #     self._train()  # Initial training
        self.args = args
        # self._demo(True, "base")

        # Define curriculum modifications
        curriculum_steps = [
            [("rewards.scales.torques", -0.0002),
             ("rewards.scales.dof_pos_limits", -10.0),
             ("rewards.scales.tracking_lin_vel", 5),
             ("rewards.scales.tracking_ang_vel", 3),
             ("noise.noise_scales.lin_vel", 0.2),
             ("rewards.scales.orientation", -2),
             ("domain_rand.randomize_mass", True),
             ("domain_rand.randomize_inertia", True),
             ("domain_rand.randomize_base_com", True),
            ("domain_rand.randomize_stiffness", True),
             ("domain_rand.randomize_damping", True),
            ("domain_rand.randomize_motor_strength", True),
             ("domain_rand.randomize_motor_offset", True),
            # ("domain_rand.randomize_gravity", True),
            ("domain_rand.add_control_freq", True),
            ("domain_rand.add_delay", True),]
            # [("domain_rand.randomize_friction", True)],
        ]

        for attributes in curriculum_steps:
            title = ""
            for attr_path, value in attributes:
                # Set attribute dynamically
                obj = self.env.cfg
                *parents, attr = attr_path.split(".")
                for parent in parents:
                    obj = getattr(obj, parent)
                title = f"{title}_{attr}"
                setattr(obj, attr, value)
                self.env._update_cfg(self.env.cfg) # actually updates reward functions
                print(f"SET attribute {attr_path} to {value}. TRAINING.")
            
                if "rewards" in attr_path:
                    self.env.reward_scales[attr] = value

            self._train(param=title)  # Train after each change
            # self._demo(True, title)

    def _train(self, param="base"):
        self.i+=1
        self.ppo_runner.env = self.env
        max_its = self.train_cfg.runner.max_iterations
        if self.i == 1 and not self.train_cfg.runner.resume:
            max_its = 1500
        self.ppo_runner.learn(num_learning_iterations=max_its, init_at_random_ep_len=True)
        self.ppo_runner.save(os.path.join(sTrueelf.ppo_runner.log_dir, f'curriculum_{self.i}_{param}.pt'))


    def _demo(self, record_frames=False, record_name=""):
        if self.env.headless: # no need to play if headless
            return
        
        env_cfg, train_cfg = task_registry.get_cfgs(name=self.args.task)
        # override some parameters for testing
        # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
        # env_cfg.terrain.num_rows = 5
        # env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
        # env_cfg.noise.add_noise = False
        # env_cfg.domain_rand.randomize_friction = False
        # env_cfg.domain_rand.push_robots = False
        # env_cfg.domain_rand.add_control_freq = False
        # env_cfg.domain_rand.add_delay = False
        # env_cfg.domain_rand.randomize_damping = False
        # env_cfg.domain_rand.randomize_stiffness = False
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
