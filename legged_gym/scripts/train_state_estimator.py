import isaacgym
from legged_gym.envs import * # required to prevent circular imports
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO

import torch
import torch.nn.functional as F

from legged_gym.utils.actor import Actor
class Trainer:
    def __init__(
            self,
            num_obs,
            num_actions,
            learning_rate=0.001,
            batch_size=1024,
            mini_batch_size=512,
            weight_decay=0.0005,
            num_epochs=500,
            num_processes=4096,
            device="cuda:0",
        ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

        self.estimator = Actor(
            num_actor_obs=num_obs,
            num_actions=num_actions,
            actor_hidden_dims=[256, 128],
            activation="elu",
            init_noise_std=1.0,
            noise_std_type="scalar"
        )

        self.estimator.to(device)

        self.num_steps = self.batch_size // num_processes + 1

        self.self.bservations = torch.zeros(
            self.num_steps + 1, num_processes, (num_obs,), device="cpu"
        )
        self.buffer_actions = torch.zeros(self.num_steps, num_processes, num_actions, device="cpu")
        self.num_actions = num_actions
        self.num_obs = num_obs

    def _load_env(self, args):
        cfg = GO2RoughCfg()

        env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=cfg)
        return env

    def train(self, args):
        env = self._load_env(args)

        obs = env.reset()[0]
        self.buffer_observations[-1].copy_(obs[:,self.num_actions:].to("cpu"))

        optimizer = torch.optim.Adam(self.estimator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for epoch in range(self.num_epochs):
            with torch.no_grad():
                self.buffer_observations[0].copy_(self.buffer_observations[-1])
                for step in range(self.num_steps):
                    action = self.estimator.act(
                        self.buffer_observations[step].to(args.rl_device)
                    )

                    obs, _, _, _, _ = env.step(action)

                    self.buffer_observations[step + 1].copy_(obs[:,3:].to("cpu"))
                    self.buffer_actions[step].copy_(action.to("cpu"))

        num_mini_batch = self.batch_size // self.mini_batch_size
        shuffled_indices = torch.randperm(
            num_mini_batch * self.mini_batch_size, generator=None, device="cpu"
        )
        shuffled_indices_batch = shuffled_indices.view(num_mini_batch, -1)

        observations_shaped = self.buffer_observations.view(-1, self.num_obs)
        actions_shaped = self.buffer_actions.view(-1, self.num_actions)

        ep_action_loss = torch.tensor(0.0, device=args.rl_device).float()

        for indices in shuffled_indices_batch:
            observations_batch = observations_shaped[indices]
            actions_batch = actions_shaped[indices]

            pred_actions = self.estimator.act_inference(observations_batch.to("cuda:0"))

            # Forward pass
            optimizer.zero_grad()
            loss = F.mse_loss(pred_actions, actions_batch.to("cuda:0"))
            loss.backward()
            optimizer.step()

            ep_action_loss.add_(loss.detach())

        L = shuffled_indices_batch.shape[0]
        ep_action_loss.div_(L)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{self.num_epochs}, Loss: {ep_action_loss.item()}")

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(num_obs=..., num_actions=...)
    trainer.train(args)