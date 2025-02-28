import isaacgym
from legged_gym.envs import * # required to prevent circular imports
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic

import os
import torch
import csv
import time
import torch.nn.functional as F

NUM_EPOCHS = 300
BATCH_SIZE = 100000
MINI_BATCH_SIZE = 512

SAVE_PATH = "logs/behavior_cloning/walking_estimator_hist_len_6_no_cmd"
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
    history_len = 6
    dof_len = 12
    estimator_obs_shape = (obs_dim - 6 + history_len * 2 * dof_len,)
    estimator_obs_dim = obs_dim - 6 + history_len * 2 * dof_len
    
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

    # remove linear velocity and commands (6) but add a history of size 3 of past joint positions and velocities (6 + 6)
    estimator = load_estimator_model(model_path=None, num_obs=estimator_obs_dim, device=args.rl_device)

    past_joint_positions = torch.zeros(num_processes, dof_len * history_len, device="cpu")
    past_joint_velocities = torch.zeros(num_processes, dof_len * history_len, device="cpu")

    optimizer = torch.optim.Adam(estimator.parameters(), lr=3e-4)

    obs = env.reset()[0]
    buffer_observations[-1].copy_(obs.to("cpu"))
    past_joint_positions = obs[:,12:24].repeat(1,history_len).detach().clone()
    past_joint_velocities = obs[:,24:36].repeat(1,history_len).detach().clone()
    est_obs = torch.concatenate((obs[:,3:9], obs[:,12:], past_joint_positions, past_joint_velocities), dim=1)
    buffer_estimator_observations[-1].copy_(est_obs.to("cpu"))
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

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

                obs, priv_obs, _, dones, _ = env.step(stochastic_action)

                # remove command (9 to 12)
                est_obs = torch.concatenate((obs[:,3:9], obs[:,12:], past_joint_positions, past_joint_velocities), dim=1)
                buffer_estimator_observations[step + 1].copy_(est_obs.to("cpu"))
                buffer_observations[step + 1].copy_(obs.to("cpu"))
                buffer_actions[step].copy_(priv_obs[:,0:3].to("cpu"))

                # if not done, update history
                # first shift last two to first two to leave a space for the last one
                past_joint_positions[~dones,:-12].copy_(past_joint_positions[~dones,12:])
                past_joint_positions[~dones,-12:].copy_(obs[~dones,12:24])
                past_joint_velocities[~dones,:-12].copy_(past_joint_velocities[~dones,12:])
                past_joint_velocities[~dones,-12:].copy_(obs[~dones,24:36])

                # if done, repeat history
                past_joint_positions[dones].copy_(obs[dones,12:24].repeat(1,history_len))
                past_joint_velocities[dones].copy_(obs[dones,24:36].repeat(1,history_len))

                # print(f"Past joint positions and velocities: {past_joint_positions[0]} and {past_joint_velocities[0]} for step {step} in epoch {epoch}")

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
            estimator_observations_batch = estimator_observations_shaped[indices]
            pred_actions = estimator.act_inference(estimator_observations_batch.to("cuda:0")) # remove linear velocities
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