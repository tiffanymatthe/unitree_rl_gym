"""
Update SAVE_PATH and POLICY_PATH.

Run this script after training a policy to walk (and that takes in its linear velocity as input).

Will output an estimator policy which takes in observations (not including linear velocity or the command) and outputs estimated base linear velocity (before simulation scaling).
"""

import isaacgym
from legged_gym.envs import * # required to prevent circular imports
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from rsl_rl.modules import ActorCritic
from tqdm import trange, tqdm
import os

import torch
import torch.nn.functional as F

from legged_gym.utils.actor import Actor

POLICY_PATH = "/home/lukas/rl_gym/unitree_rl_gym/logs/rough_go2/Apr11_16-41-42_/model_2700.pt" # TODO CHANGE THIS
SAVE_PATH = POLICY_PATH.replace("model_", "state_estimator_")
SAVE_PATH = SAVE_PATH.replace(".pt", "")
os.makedirs(SAVE_PATH, exist_ok=True)

class Trainer:
    def __init__(
            self,
            learning_rate=0.001,
            batch_size=1024,
            mini_batch_size=512,
            weight_decay=0.0005,
            data_gathering_steps=5000,
            num_epochs=500,
            device="cuda:0",
            args=None,
        ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.weight_decay = weight_decay
        self.data_gathering_steps = data_gathering_steps
        self.num_epochs = num_epochs
        self.device = device

        self.env, env_cfg = self._load_env(args)
        self.policy_num_obs = env_cfg.env.num_observations

        # remove linear velocity and command
        self.estimator_num_obs = self.policy_num_obs - 6

        self.estimator_num_actions = 3
        self.policy_num_actions = env_cfg.env.num_actions

        self.estimator = Actor(
            num_actor_obs=self.estimator_num_obs,
            num_actions=self.estimator_num_actions,
            actor_hidden_dims=[256, 128],
            activation="elu",
            init_noise_std=1.0,
            noise_std_type="scalar"
        )

        self.estimator.to(device)

        self.num_steps = self.batch_size // env_cfg.env.num_envs + 1

        self.buffer_policy_observations = torch.zeros(
            (self.num_steps + 1, env_cfg.env.num_envs, self.policy_num_obs,), device=self.device
        )
        self.buffer_estimator_actions = torch.zeros(self.num_steps, env_cfg.env.num_envs, self.estimator_num_actions, device=self.device)

    def _load_env(self, args):
        cfg = GO2RoughCfg()

        curriculum_steps = [ #TODO CHANGE THIS
            [("rewards.scales.torques", -0.0002),
             ("rewards.scales.dof_pos_limits", -10.0),
             ("rewards.scales.tracking_lin_vel", 5),
             ("rewards.scales.tracking_ang_vel", 3),
             ("noise.noise_scales.lin_vel", 0.2), ],
            [("domain_rand.randomize_mass", True),
             ("domain_rand.randomize_inertia", True),
             ("domain_rand.randomize_base_com", True)],
            [("domain_rand.randomize_stiffness", True),
             ("domain_rand.randomize_damping", True)],
            [("domain_rand.randomize_motor_strength", True),
             ("domain_rand.randomize_motor_offset", True)],
            [("domain_rand.add_control_freq", True)],
            [("domain_rand.add_delay", True)],
        ]

        for attributes in curriculum_steps:
            title = ""
            for attr_path, value in attributes:
                # Set attribute dynamically
                obj = cfg
                *parents, attr = attr_path.split(".")
                for parent in parents:
                    obj = getattr(obj, parent)
                title = f"{title}_{attr}"
                setattr(obj, attr, value)
                print(f"SET attribute {attr_path} to {value}. TRAINING.")

        env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=cfg)
        return env, env_cfg
    
    def load_model(self, model_path, device="cuda:0"):
        model = ActorCritic(
                num_actor_obs=self.policy_num_obs,
                num_critic_obs=self.policy_num_obs,
                num_actions=self.policy_num_actions,
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
                activation='elu',
                init_noise_std=1.0
            )
        
        model.to(device)

        if model_path is not None:
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])

        model.eval()
        return model

    def gather_data(self):
        policy = self.load_model(POLICY_PATH)

        obs = self.env.reset()[0]
        self.buffer_policy_observations[-1].copy_(obs.to(self.device))

        for step in trange(self.data_gathering_steps):
            with torch.no_grad():
                self.buffer_policy_observations[0].copy_(self.buffer_policy_observations[-1])
                for step in range(self.num_steps):
                    action = policy.act(
                        self.buffer_policy_observations[step].to(args.rl_device)
                    )

                    obs, privileged_obs, _, _, _ = self.env.step(action)

                    self.buffer_policy_observations[step + 1].copy_(obs.to(self.device))
                    # no noise?
                    self.buffer_estimator_actions[step].copy_(privileged_obs[:,0:3].to(self.device))

        num_mini_batch = self.batch_size // self.mini_batch_size
        shuffled_indices = torch.randperm(
            num_mini_batch * self.mini_batch_size, generator=None, device=self.device
        )
        shuffled_indices_batch = shuffled_indices.view(num_mini_batch, -1)

        observations_shaped = self.buffer_policy_observations.view(-1, self.policy_num_obs)
        actions_shaped = self.buffer_estimator_actions.view(-1, self.estimator_num_actions)

        return observations_shaped, actions_shaped, shuffled_indices_batch

    def train(self, args):
        observations_shaped, actions_shaped, shuffled_indices_batch = self.gather_data()
        
        idx = int(len(shuffled_indices_batch) * 0.9)
        training_batches = shuffled_indices_batch[:idx]
        validation_batches = shuffled_indices_batch[idx:]
        print(f"Training batches: {len(training_batches)}, Validation batches: {len(validation_batches)}")

        optimizer = torch.optim.Adam(self.estimator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        ep_action_loss = torch.tensor(0.0, device=args.rl_device).float()

        for epoch in trange(self.num_epochs):
            ep_action_loss = torch.tensor(0.0, device="cuda:0")

            for indices in training_batches:
                optimizer.zero_grad()

                observations_batch = observations_shaped[indices]
                actions_batch = actions_shaped[indices]

                # remove linear velocity and command
                pred_actions = self.estimator.act_inference(
                    torch.cat((observations_batch[:, 3:9], observations_batch[:, 12:]), dim=1).to("cuda:0")
                )

                # Compute loss and update
                loss = F.mse_loss(pred_actions, actions_batch.to("cuda:0"))
                loss.backward(retain_graph=True)
                optimizer.step()

                ep_action_loss += loss.detach()

            L = len(shuffled_indices_batch)
            avg_loss = ep_action_loss.item() / L

            if epoch % 5 == 0:
                with torch.no_grad():
                    for indices in validation_batches:
                        observations_batch = observations_shaped[indices]
                        actions_batch = actions_shaped[indices]
                        pred_actions = self.estimator.act_inference(
                                torch.cat((observations_batch[:, 3:9], observations_batch[:, 12:]), dim=1).to("cuda:0")
                            )
                        
                        loss = F.mse_loss(pred_actions, actions_batch.to("cuda:0"))
                        tqdm.write(f"Epoch {epoch}/{self.num_epochs}, Validation Loss: {loss.item():.6f}, Loss: {avg_loss:.6f}")
            
            if epoch % 25 == 0:
                torch.save({
                    'model_state_dict': self.estimator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter': epoch,
                    "infos": None,
                }, f"{SAVE_PATH}/state_estimator_model_{epoch}.pt")

                
        torch.save({
                    'model_state_dict': self.estimator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter': epoch,
                    "infos": None,
                    }, f"{SAVE_PATH}/state_estimator_model_{self.num_epochs}.pt")
        
        print(f"Saved model to {SAVE_PATH}/state_estimator_model_{self.num_epochs}.pt")

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(
        learning_rate=0.001, # could try 3e-4 instead
        batch_size=1024 * 10, # unsure about this parameter
        mini_batch_size=512,
        weight_decay=0.0005,
        data_gathering_steps=5000,
        num_epochs=250,
        device="cuda:0",
        args=args,
    )
    trainer.train(args)