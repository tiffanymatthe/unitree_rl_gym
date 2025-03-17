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
            ("rewards.scales.orientation", -20),
            ("rewards.scales.stand_still", -50),
            ("domain_rand.randomize_mass", True),
            ("domain_rand.randomize_inertia", True),
            ("domain_rand.randomize_stiffness", True),
            ("domain_rand.randomize_damping", True),
            ("domain_rand.add_control_freq", True),
            ("domain_rand.add_delay", True),
            ("domain_rand.randomize_friction", True),
        ]

        for attr_path, value in curriculum_steps:
            # Set attribute dynamically
            obj = self.env.cfg
            *parents, attr = attr_path.split(".")
            for parent in parents:
                obj = getattr(obj, parent)
            setattr(obj, attr, value)
            print(self.env.cfg.rewards.scales.stand_still)
            self._train()  # Train after each change



    def _train(self):
        self.i+=1
        self.ppo_runner.env = self.env
        self.ppo_runner.learn(num_learning_iterations=self.train_cfg.runner.max_iterations, init_at_random_ep_len=True)
        self.ppo_runner.save(os.path.join(self.ppo_runner.log_dir, f'curriculum_{self.i}.pt'))

if __name__ == '__main__':
    args = get_args()
    ct = CurriculumTrainer()
    ct.curriculum_train(args)
