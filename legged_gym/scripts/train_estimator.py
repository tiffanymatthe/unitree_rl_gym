import isaacgym
from legged_gym.envs import * # required to prevent circular imports
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic

import torch
import csv
import time
import torch.nn.functional as F

NUM_EPOCHS = 300
BATCH_SIZE = 100000
MINI_BATCH_SIZE = 512

SAVE_PATH = "logs/behavior_cloning/walking_estimator"
TEACHER_PATH = "logs/rough_go2/walking/walking_model.pt"

def load_model(model_path, num_obs, device="cuda:0"):
    model = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_obs,
            num_actions=12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation='elu',
            init_noise_std=1.0
        )
    
    model.to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])

    return model

def load_estimator_model(model_path, num_obs, device="cuda:0"):
    model = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_obs,
            num_actions=3, # linear velocity x y z
            actor_hidden_dims=[256, 128],
            critic_hidden_dims=[256, 128],
            activation='elu',
            init_noise_std=1.0
        )
    # ignore critic, just used to initialize
    
    model.to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])

    return model

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # env1, env_cfg1 = task_registry.make_env(name=args.task1, args=args)
    obs_shape = (48,)
    action_shape = (3,) # for estimator
    obs_dim = 48
    action_dim = 3
    estimator_obs_shape = (obs_dim - 3 + 3 * 2 * 12,)
    estimator_obs_dim = obs_dim - 3 + 3 * 2 * 12
    
    num_processes = 4096
    num_steps = BATCH_SIZE // num_processes + 1

    buffer_observations = torch.zeros(
        num_steps + 1, num_processes, *obs_shape, device="cpu"
    )
    buffer_estimator_observations = torch.zeros(
        num_steps + 1, num_processes, *estimator_obs_shape, device="cpu"
    )
    buffer_actions = torch.zeros(
        num_steps, num_processes, *action_shape, device="cpu"
    )

    actor_critic = load_model(model_path=TEACHER_PATH, num_obs=obs_dim, device=args.rl_device)
    for param in actor_critic.parameters():
        param.requires_grad = False

    # remove linear velocity (3) but add a history of size 3 of past joint positions and velocities (6)
    estimator = load_model(model_path=None, num_obs=estimator_obs_dim, device=args.rl_device)

    past_joint_positions = torch.zeros(num_processes, 12 * 3, device="cpu")
    past_joint_velocities = torch.zeros(num_processes, 12 * 3, device="cpu")

    optimizer = torch.optim.Adam(estimator.parameters(), lr=3e-4)

    obs = env.reset()[0]
    buffer_observations[-1].copy_(obs.to("cpu"))
    past_joint_positions = obs[:,12:24].repeat(3)
    past_joint_velocities = obs[:,24:36].repeat(3)
    est_obs = torch.concatenate(obs[:,3:], past_joint_positions, past_joint_velocities, axis=1)
    buffer_estimator_observations[-1].copy_(est_obs.to("cpu"))

    file = open(f"{SAVE_PATH}/estimator_results.csv", mode="w", newline='')
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Elapsed Time", "Action Loss"])

    start = time.time()
    for epoch in range(NUM_EPOCHS):
        with torch.no_grad():
            buffer_observations[0].copy_(buffer_observations[-1])
            buffer_observations[0].copy_(buffer_observations[-1])
            for step in range(num_steps):
                stochastic_action = actor_critic.act(
                    buffer_observations[step].to(args.rl_device)
                )

                obs, priv_obs, _, _, _ = env.step(stochastic_action)

                est_obs = torch.concatenate(obs[:,3:], past_joint_positions, past_joint_velocities, axis=1)
                buffer_estimator_observations[step + 1].copy_(est_obs.to("cpu"))
                buffer_observations[step + 1].copy_(obs.to("cpu"))
                buffer_actions[step].copy_(priv_obs[:,0:3].to("cpu"))

                past_joint_positions = torch.concatenate(past_joint_positions, obs[:,12:24], axis=1)
                past_joint_positions = past_joint_positions[:,12:] # remove the first 12, too past
                past_joint_velocities = torch.concatenate(past_joint_velocities, obs[:,24:36], axis=1)
                past_joint_velocities = past_joint_velocities[:,12:]

        num_mini_batch = BATCH_SIZE // MINI_BATCH_SIZE
        shuffled_indices = torch.randperm(
            num_mini_batch * MINI_BATCH_SIZE, generator=None, device="cpu"
        )
        shuffled_indices_batch = shuffled_indices.view(num_mini_batch, -1)

        observations_shaped = buffer_observations.view(-1, obs_dim)
        estimator_observations_shaped = buffer_estimator_observations.view(-1, estimator_obs_dim)
        actions_shaped = buffer_actions.view(-1, action_dim)

        ep_action_loss = torch.tensor(0.0, device=args.rl_device).float()

        for indices in shuffled_indices_batch:
            optimizer.zero_grad()
            observations_batch = observations_shaped[indices]
            actions_batch = actions_shaped[indices] # get "ground truth" linear velocity (but it contains noise!)

            pred_actions = estimator.act_inference(estimator_observations_shaped.to("cuda:0")) # remove linear velocities
            action_loss = F.mse_loss(pred_actions, actions_batch.to("cuda:0"))
            action_loss.backward()

            optimizer.step()

            ep_action_loss.add_(action_loss.detach())

        L = shuffled_indices_batch.shape[0]
        ep_action_loss.div_(L)

        elapsed_time = time.time() - start

        torch.save({
            'model_state_dict':estimator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': epoch,
            "infos": None,
            }, f"{SAVE_PATH}/model.pt")

        print(
            (
                f"Epoch {epoch+1:4d}/{NUM_EPOCHS:4d} | "
                f"Elapsed Time {elapsed_time:8.2f} |"
                f"Action Loss: {ep_action_loss.item():8.4f} | "
            )
        )

        writer.writerow(
            [
                epoch+1,
                f"{elapsed_time:8.2f}",
                f"{ep_action_loss.item():8.4f}",
            ]
        )


if __name__ == '__main__':
    args = get_args()
    train(args)