import isaacgym
from legged_gym.envs import * # required to prevent circular imports
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO

import torch
import csv
import time
import torch.nn.functional as F
import os

NUM_EPOCHS = 200
BATCH_SIZE = 100000
MINI_BATCH_SIZE = 512

NUM_TEACHER_EPOCHS = 1

lin_vel_x = [-1.0, 1.0] # min max [m/s]
lin_vel_y = [-1.0, 1.0]   # min max [m/s]
ang_vel_yaw = [-1, 1]    # min max [rad/s]
heading = [-3.14, 3.14]

SAVE_PATH = f"logs/simple_bc/teacher_{NUM_TEACHER_EPOCHS}_epochs_x_{lin_vel_x[0]}_{lin_vel_x[1]}_y_{lin_vel_y[0]}_{lin_vel_y[1]}_yaw_{ang_vel_yaw[0]}_{ang_vel_yaw[1]}_heading_{heading[0]}_{heading[1]}"

TEACHER_PATH = "logs/rough_go2/walking/walking_model.pt"

# Initialize lists to store the values
cmd_vel_x_list = []
cmd_vel_y_list = []
cmd_vel_yaw_list = []
cmd_heading_list = []

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

def train(args):
    ppo_cfg = GO2RoughCfgPPO()
    cfg = GO2RoughCfg()
    cfg.commands.ranges.lin_vel_x = lin_vel_x
    cfg.commands.ranges.lin_vel_y = lin_vel_y
    cfg.commands.ranges.ang_vel_yaw = ang_vel_yaw
    cfg.commands.ranges.heading = heading
    cfg.seed = ppo_cfg.seed

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=cfg)
    # env1, env_cfg1 = task_registry.make_env(name=args.task1, args=args)
    obs_shape = (48,)
    obs_dim = 48
    act_dim = 12
    
    num_processes = 4096
    num_steps = BATCH_SIZE // num_processes + 1

    buffer_observations = torch.zeros(
        num_steps + 1, num_processes, *obs_shape, device="cpu"
    )
    buffer_actions = torch.zeros(num_steps, num_processes, act_dim, device="cpu")
    buffer_values = torch.zeros(num_steps, num_processes, 1, device="cpu")

    actor_critic = load_model(model_path=TEACHER_PATH, num_obs=48, device=args.rl_device)
    for param in actor_critic.parameters():
        param.requires_grad = False

    student_actor_critic = load_model(model_path=None, num_obs=48-3, device=args.rl_device)

    optimizer = torch.optim.Adam(student_actor_critic.parameters(), lr=3e-4)

    obs = env.reset()[0]
    buffer_observations[-1].copy_(obs.to("cpu"))

    os.makedirs(SAVE_PATH, exist_ok=True)

    file = open(f"{SAVE_PATH}/bc_results.csv", mode="w", newline='')
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Elapsed Time", "Action Loss", *[f"Dof {i} Action Loss" for i in range(12)], "Value Loss"])

    # Open the CSV file in write mode initially to write the header
    cmd_vel_file = open(f"{SAVE_PATH}/cmd_vel_data.csv", mode="w", newline='')
    cmd_vel_writer = csv.writer(cmd_vel_file)
    cmd_vel_writer.writerow(["cmd_vel_x", "cmd_vel_y", "cmd_vel_yaw", "cmd_heading"])

    start = time.time()
    for epoch in range(NUM_EPOCHS):
        with torch.no_grad():
            buffer_observations[0].copy_(buffer_observations[-1])
            for step in range(num_steps):
                action = actor_critic.act_inference(
                    buffer_observations[step].to(args.rl_device)
                )
                value = actor_critic.evaluate(buffer_observations[step].to(args.rl_device))

                if epoch >= NUM_TEACHER_EPOCHS:
                    stochastic_action = student_actor_critic.act(
                        buffer_observations[step,:,3:].to(args.rl_device)
                    )
                else:
                    stochastic_action = actor_critic.act(
                        buffer_observations[step].to(args.rl_device)
                    )

                cpu_actions = stochastic_action # if epoch > 0 else action

                obs, _, _, _, _ = env.step(cpu_actions)

                cmd_vel_x = obs[:,9]
                cmd_vel_y = obs[:,10]
                cmd_vel_yaw = obs[:,11]
                cmd_heading = obs[:,12]

                # Append the values to the lists
                cmd_vel_x_list.extend(cmd_vel_x.cpu().numpy())
                cmd_vel_y_list.extend(cmd_vel_y.cpu().numpy())
                cmd_vel_yaw_list.extend(cmd_vel_yaw.cpu().numpy())
                cmd_heading_list.extend(cmd_heading.cpu().numpy())

                buffer_observations[step + 1].copy_(obs.to("cpu"))
                buffer_actions[step].copy_(action.to("cpu"))
                buffer_values[step].copy_(value.to("cpu"))

        num_mini_batch = BATCH_SIZE // MINI_BATCH_SIZE
        shuffled_indices = torch.randperm(
            num_mini_batch * MINI_BATCH_SIZE, generator=None, device="cpu"
        )
        shuffled_indices_batch = shuffled_indices.view(num_mini_batch, -1)

        observations_shaped = buffer_observations.view(-1, obs_dim)
        actions_shaped = buffer_actions.view(-1, act_dim)
        values_shaped = buffer_values.view(-1, 1)

        ep_action_loss = torch.tensor(0.0, device=args.rl_device).float()
        ep_dof_action_losses = [
            torch.tensor(0.0, device=args.rl_device).float()
            for _ in range(12)
        ]
        ep_value_loss = torch.tensor(0.0, device=args.rl_device).float()

        for indices in shuffled_indices_batch:
            optimizer.zero_grad()
            observations_batch = observations_shaped[indices]
            actions_batch = actions_shaped[indices]
            values_batch = values_shaped[indices]

            pred_actions = student_actor_critic.act_inference(observations_batch[:,3:].to("cuda:0"))
            pred_values = student_actor_critic.evaluate(observations_batch[:,3:].to("cuda:0"))

            # action_loss = F.mse_loss(pred_actions, actions_batch.to("cuda:0"))
            dof_action_losses = F.mse_loss(pred_actions, actions_batch.to("cuda:0"), reduction="none")
            (d1,d2) = dof_action_losses.shape
            dof_action_losses = torch.sum(dof_action_losses, dim=0)
            dof_action_losses /= d1 * d2
            action_loss = dof_action_losses.sum()

            value_loss = F.mse_loss(pred_values, values_batch.to("cuda:0"))
            (action_loss + value_loss).backward()

            optimizer.step()

            for i in range(12):
                ep_dof_action_losses[i].add_(dof_action_losses[i].detach())
            ep_action_loss.add_(action_loss.detach())
            ep_value_loss.add_(value_loss.detach())

        L = shuffled_indices_batch.shape[0]
        ep_action_loss.div_(L)
        ep_value_loss.div_(L)
        for i in range(12):
            ep_dof_action_losses[i].div_(L)

        elapsed_time = time.time() - start

        torch.save({
            'model_state_dict':student_actor_critic.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': epoch,
            "infos": None,
            }, f"{SAVE_PATH}/model.pt")

        dof_action_losses_str = ", ".join([f"{ep_dof_action_losses[i].item():8.4f}" for i in range(12)])

        print(
            (
                f"Epoch {epoch+1:4d}/{NUM_EPOCHS:4d} | "
                f"Elapsed Time {elapsed_time:8.2f} |"
                f"Action Loss: {ep_action_loss.item():8.4f} | "
                f"DOF Losses: {dof_action_losses_str} | "
                f"Value Loss: {ep_value_loss.item():8.2f}"
            )
        )

        writer.writerow(
            [
                epoch+1,
                f"{elapsed_time:8.2f}",
                f"{ep_action_loss.item():8.4f}",
                *[f"{ep_dof_action_losses[i].item():8.4f}" for i in range(12)],
                f"{ep_value_loss.item():8.2f}"
            ]
        )

        rows = zip(cmd_vel_x_list, cmd_vel_y_list, cmd_vel_yaw_list, cmd_heading_list)
        cmd_vel_writer.writerows(rows)

        # Clear the lists after writing to the file
        cmd_vel_x_list.clear()
        cmd_vel_y_list.clear()
        cmd_vel_yaw_list.clear()
        cmd_heading_list.clear()


if __name__ == '__main__':
    args = get_args()
    train(args)