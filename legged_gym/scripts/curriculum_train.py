import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict
from rsl_rl.runners import OnPolicyRunner
import torch

class CurriculumTrainer():
    def __init__(self):
        self.i = 0
        
    def curriculum_train(self, args):
        self.env, env_cfg = task_registry.make_env(name=args.task, args=args)
        self.ppo_runner, self.train_cfg = task_registry.make_alg_runner(env=self.env, name=args.task, args=args)
        self._train()  # Initial training

        # Define curriculum modifications
        curriculum_steps = [
            [("rewards.scales.orientation", -20)], # helpful to prevent robot from falling onto its head
            [("rewards.scales.stand_still", -50)], # helpful to learn standing behaviors
            [("domain_rand.randomize_mass", True),
            ("domain_rand.randomize_inertia", True)],
            [("domain_rand.randomize_stiffness", True),
            ("domain_rand.randomize_damping", True)],
            [("domain_rand.add_control_freq", True)],
            [("domain_rand.add_delay", True)],
            [("domain_rand.randomize_friction", True)],
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
                print(f"SET attribute {attr_path} to {value}. TRAINING.")
            
                if "rewards" in attr_path:
                    self.env.reward_scales[attr] = value

            self._train(param=title)  # Train after each change



    def _train(self, param="base"):
        self.i+=1
        self.ppo_runner.env = self.env
        max_its = self.train_cfg.runner.max_iterations
        if self.i == 1:
            max_its = 1000
        self.ppo_runner.learn(num_learning_iterations=max_its, init_at_random_ep_len=True)
        self.ppo_runner.save(os.path.join(self.ppo_runner.log_dir, f'curriculum_{self.i}_{param}.pt'))

if __name__ == '__main__':
    args = get_args()
    ct = CurriculumTrainer()
    ct.curriculum_train(args)
