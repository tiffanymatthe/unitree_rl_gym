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

from torch.utils.data import TensorDataset, DataLoader, random_split

import h5py

import torch
import torch.nn.functional as F

from legged_gym.utils.actor import Actor

POLICY_PATH = "/home/lukas/rl_gym/unitree_rl_gym/logs/rough_go2/Apr22_22-20-45_/model_2300.pt" # TODO CHANGE THIS
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

        self.env, self.env_cfg = self._load_env(args)
        self.policy_num_obs = self.env_cfg.env.num_observations
        self.num_envs = self.env_cfg.env.num_envs

        # remove linear velocity and command
        self.estimator_num_obs = self.policy_num_obs - 6

        self.estimator_num_actions = 3
        self.policy_num_actions = self.env_cfg.env.num_actions

        self.estimator = Actor(
            num_actor_obs=self.estimator_num_obs,
            num_actions=self.estimator_num_actions,
            actor_hidden_dims=[256, 128],
            activation="elu",
            init_noise_std=1.0,
            noise_std_type="scalar"
        )

        self.estimator.to(device)


    def _load_env(self, args):
        cfg = GO2RoughCfg()
        ppo_cfg = GO2RoughCfgPPO()
        cfg.seed = ppo_cfg.seed

        curriculum_steps = [
            [
             ("rewards.scales.orientation", -1),
             ("rewards.scales.torques", -0.0002),
             ("rewards.scales.dof_pos_limits", -10.0),
             ("rewards.scales.tracking_lin_vel", 5),
             ("rewards.scales.tracking_ang_vel", 3),
             ("noise.noise_scales.lin_vel", 0.2),],
            # ("rewards.scales.feet_air_time", 2),
             [("domain_rand.randomize_mass", True),
             ("domain_rand.randomize_inertia", True),
             ("domain_rand.randomize_base_com", True),],
             [("domain_rand.randomize_stiffness", True),
             ("domain_rand.randomize_damping", True),
             ("domain_rand.randomize_motor_strength", True),
             ("domain_rand.randomize_motor_offset", True),],
            [("domain_rand.randomize_gravity", True),
            ("domain_rand.add_control_freq", True),
            ("domain_rand.add_delay", True),],
            # [("domain_rand.randomize_friction", True)],
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

        h5_path = f"{SAVE_PATH}/dataset.h5"
        with h5py.File(h5_path, 'a') as f:
            obs = self.env.reset()[0]
            self.buffer_policy_observations[-1].copy_(obs.to(self.device))

            observations_shaped = self.buffer_policy_observations.view(-1, self.policy_num_obs)

            num_samples = self.env_cfg.env.num_envs*self.data_gathering_steps
            obs_dim = torch.cat((observations_shaped[:, 3:9], observations_shaped[:, 12:]), dim=1).shape[1]
            label_dim = self.buffer_estimator_actions.view(-1, self.estimator_num_actions).shape[1]

            obs_ds = f.create_dataset("observations", shape=(num_samples, obs_dim), dtype='f4')
            lbl_ds = f.create_dataset("labels", shape=(num_samples, label_dim), dtype='f4')

            with torch.no_grad():
                obs = self.env.get_observations()
                for s in trange(self.data_gathering_steps):
                    action = policy.act(
                        obs.to(args.rl_device)
                    )
                    obs, privileged_obs, _, _, _ = self.env.step(action)
                    obs_ds[s*self.num_envs:(s+1)*self.num_envs] = torch.cat((obs[:, 3:9], obs[:, 12:]), dim=1).cpu()
                    lbl_ds[s*self.num_envs:(s+1)*self.num_envs] = privileged_obs[:,0:3].cpu()

        with h5py.File(h5_path, "r") as f:
            observations = torch.tensor(f['observations'][:])
            labels = torch.tensor(f['labels'][:])

        # Wrap in TensorDataset
        dataset = TensorDataset(observations, labels)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        # Split it randomly
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        return train_loader, val_loader 
    
    def get_data(self):

        try:
            if os.path.exists(f"{SAVE_PATH}/dataset.h5"):
                with h5py.File(f"{SAVE_PATH}/dataset.h5", "r") as f:
                    observations = torch.tensor(f['observations'][:])
                    labels = torch.tensor(f['labels'][:])

                # Wrap in TensorDataset
                dataset = TensorDataset(observations, labels)

                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size

                # Split it randomly
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                
                return train_loader, val_loader 
            else:
                return self.gather_data()
        except:
            return self.gather_data()

    

    def train(self, args):  
        train_dl, val_dl = self.get_data()

        print(f"Training batches: {len(train_dl)}, Validation batches: {len(val_dl)}")

        optimizer = torch.optim.Adam(self.estimator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        ep_action_loss = torch.tensor(0.0, device=args.rl_device).float()

        for epoch in trange(self.num_epochs):
            ep_action_loss = torch.tensor(0.0, device=self.device)

            for x, y in tqdm(train_dl):
                optimizer.zero_grad()


                # remove linear velocity and command
                pred_actions = self.estimator.act_inference(x.to(self.device))

                # Compute loss and update
                loss = F.mse_loss(pred_actions, y.to(self.device))
                loss.backward(retain_graph=True)
                optimizer.step()

                ep_action_loss += loss.detach()

            avg_loss = ep_action_loss.item() / len(train_dl)


            if epoch % 1 == 0:
                with torch.no_grad():
                    val_loss = torch.tensor(0.0, device=self.device)
                    for x, y in val_dl:
                        pred_actions = self.estimator.act_inference(x.to(self.device))
                        
                        loss = F.mse_loss(pred_actions, y.to(self.device))

                        val_loss += loss.detach()

                tqdm.write(f"Epoch {epoch}/{self.num_epochs}, Validation Loss: {(val_loss.item() / len(val_dl)):.6f}, Loss: {avg_loss:.6f}")
            
            if epoch % 5 == 0:
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
        num_epochs=10,
        device="cuda:0",
        args=args,
    )
    trainer.train(args)