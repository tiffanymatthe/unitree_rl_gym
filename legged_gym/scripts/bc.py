import isaacgym
from legged_gym.envs import * # required to prevent circular imports
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic

import torch
import csv
import time
import torch.nn.functional as F

# python3 legged_gym/scripts/bc.py
# python legged_gym/scripts/play.py --task=go2_less --experiment_name=behavior_cloning  --load_run=distilled_policy
# 1034 finished runs, with total avg rewards of 22.570068359375

# python3 legged_gym/scripts/bc.py # with dagger new commit
# python legged_gym/scripts/play.py --task=go2_less --experiment_name=behavior_cloning --load_run=dagger
# 922 finished runs, with total avg rewards of 25.417621612548828

# for comparison
# python legged_gym/scripts/play.py --task=go2 --load_run=Dec04_15-02-59_normal_walk 
# 904 finished runs, with total avg rewards of 27.316011428833008

NUM_EPOCHS = 400
BATCH_SIZE = 100000
MINI_BATCH_SIZE = 512

SAVE_PATH = "logs/behavior_cloning/walking_dagger_multi_task_w10_teach_epochs_10"
TEACHER_PATH = "logs/rough_go2/walking/walking_model.pt"
NUM_TEACHER_EPOCHS = 10

def load_model(model_path, num_obs, num_actions, device="cuda:0"):
    model = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_obs,
            num_actions=num_actions,
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
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # env1, env_cfg1 = task_registry.make_env(name=args.task1, args=args)
    obs_shape = (48,)
    obs_dim = 48
    act_dim = 12 + 3
    
    num_processes = 4096
    num_steps = BATCH_SIZE // num_processes + 1

    buffer_observations = torch.zeros(
        num_steps + 1, num_processes, *obs_shape, device="cpu"
    )
    buffer_actions = torch.zeros(num_steps, num_processes, act_dim, device="cpu")
    buffer_values = torch.zeros(num_steps, num_processes, 1, device="cpu")

    actor_critic = load_model(model_path=TEACHER_PATH, num_obs=48, num_actions=act_dim-3, device=args.rl_device)
    for param in actor_critic.parameters():
        param.requires_grad = False

    student_actor_critic = load_model(model_path=None, num_obs=48-3, num_actions=act_dim, device=args.rl_device)

    optimizer = torch.optim.Adam(student_actor_critic.parameters(), lr=3e-4)

    obs = env.reset()[0]
    buffer_observations[0].copy_(obs.to("cpu"))

    file = open(f"{SAVE_PATH}/bc_results.csv", mode="w", newline='')
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Elapsed Time", "Action Loss", "Value Loss", "Action Loss Lin Vel"])

    start = time.time()
    for epoch in range(NUM_EPOCHS):
        with torch.no_grad():
            buffer_observations[0].copy_(buffer_observations[-1])
            for step in range(num_steps):
                action = actor_critic.act_inference(
                    buffer_observations[step].to(args.rl_device)
                )
                # copying the input linear velocity as the output linear velocity prediction
                observed_lin_velocity = buffer_observations[step,:,0:3].to(args.rl_device)
                action = torch.cat((action, observed_lin_velocity), dim=1)

                value = actor_critic.evaluate(buffer_observations[step].to(args.rl_device))

                # QUESTION: if epoch == 0, should I use action from act_inference (deterministic) or stochastic action?
                if epoch >= NUM_TEACHER_EPOCHS:
                    stochastic_action = student_actor_critic.act(
                        buffer_observations[step,:,3:].to(args.rl_device)
                    )
                else:
                    stochastic_action = actor_critic.act(
                        buffer_observations[step].to(args.rl_device)
                    )
                    observed_lin_velocity = buffer_observations[step,:,0:3].to(args.rl_device)
                    stochastic_action = torch.cat((stochastic_action, observed_lin_velocity), dim=1)

                cpu_actions = stochastic_action # if epoch > 0 else action
                
                obs, _, _, _, _ = env.step(cpu_actions[:,:-3])

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
        ep_action_loss_lin_vel = torch.tensor(0.0, device=args.rl_device).float()
        ep_value_loss = torch.tensor(0.0, device=args.rl_device).float()

        for indices in shuffled_indices_batch:
            optimizer.zero_grad()
            observations_batch = observations_shaped[indices]
            actions_batch = actions_shaped[indices]
            values_batch = values_shaped[indices]

            # pred_actions: 12 of actual motor commands, and 3 of linear velocity predictions
            pred_actions = student_actor_critic.act_inference(observations_batch[:,3:].to("cuda:0"))
            pred_values = student_actor_critic.evaluate(observations_batch[:,3:].to("cuda:0"))

            action_loss = F.mse_loss(pred_actions[:,:-3], actions_batch[:,:-3].to("cuda:0"))
            action_loss_lin_vel = F.mse_loss(pred_actions[:,-3:], actions_batch[:,-3:].to("cuda:0"))
            value_loss = F.mse_loss(pred_values, values_batch.to("cuda:0"))
            (action_loss + action_loss_lin_vel * 10 + value_loss).backward()

            optimizer.step()

            ep_action_loss.add_(action_loss.detach())
            ep_action_loss_lin_vel.add_(action_loss_lin_vel.detach())
            ep_value_loss.add_(value_loss.detach())

        L = shuffled_indices_batch.shape[0]
        ep_action_loss.div_(L)
        ep_action_loss_lin_vel.div_(L)
        ep_value_loss.div_(L)

        elapsed_time = time.time() - start

        torch.save({
            'model_state_dict':student_actor_critic.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': epoch,
            "infos": None,
            }, f"{SAVE_PATH}/model.pt")

        print(
            (
                f"Epoch {epoch+1:4d}/{NUM_EPOCHS:4d} | "
                f"Elapsed Time {elapsed_time:8.2f} |"
                f"Action Loss: {ep_action_loss.item():8.4f} | "
                f"Action Loss Lin Vel: {ep_action_loss_lin_vel.item():8.4f} | "
                f"Value Loss: {ep_value_loss.item():8.2f}"
            )
        )

        writer.writerow(
            [
                epoch+1,
                f"{elapsed_time:8.2f}",
                f"{ep_action_loss.item():8.4f}",
                f"{ep_value_loss.item():8.2f}",
                f"{ep_action_loss_lin_vel.item():8.4f}",
            ]
        )


if __name__ == '__main__':
    args = get_args()
    train(args)