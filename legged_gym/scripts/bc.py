from legged_gym.utils import get_args, task_registry
import torch
import time
import torch.nn.functional as F
from rsl_rl.modules import ActorCritic

NUM_EPOCHS = 500
BATCH_SIZE = 20000
MINI_BATCH_SIZE = 512

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

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])

    return model

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # env1, env_cfg1 = task_registry.make_env(name=args.task1, args=args)

    obs_shape = env.observation_space.shape
    obs_shape = (obs_shape[0]-3, *obs_shape[1:])
    obs_dim = obs_shape[0]
    act_dim = env.action_space.shape[0]

    num_steps = BATCH_SIZE // args.num_processes
    num_processes = args.num_processes

    buffer_observations = torch.zeros(
        num_steps + 1, num_processes, *obs_shape, device=args.device
    )
    buffer_actions = torch.zeros(num_steps, num_processes, act_dim, device=args.device)
    buffer_values = torch.zeros(num_steps, num_processes, 1, device=args.device)

    actor_critic = load_model(model_path="logs/rough_go2/Dec04_15-02-59_normal_walk/model_1050.pt", num_obs=48, device=args.device)
    student_actor_critic = load_model(model_path=None, num_obs=48-3, device=args.device)

    optimizer = torch.optim.Adam(student_actor_critic.parameters(), lr=3e-4)

    obs = env.reset()
    buffer_observations[0].copy_(torch.from_numpy(obs[3:]))

    start = time.time()
    for epoch in range(NUM_EPOCHS):
        with torch.no_grad():
            for step in range(num_steps):
                _, stochastic_action, _ = actor_critic.act(
                    buffer_observations[step], deterministic=False
                )
                value, action, _ = actor_critic.act(
                    buffer_observations[step], deterministic=True
                )
                cpu_actions = stochastic_action #.cpu().numpy()

                obs, _, _, _ = env.step(cpu_actions)

                buffer_observations[step + 1].copy_(torch.from_numpy(obs[3:]))
                buffer_actions[step].copy_(action)
                buffer_values[step].copy_(value)

        num_mini_batch = BATCH_SIZE // MINI_BATCH_SIZE
        shuffled_indices = torch.randperm(
            num_mini_batch * args.mini_batch_size, generator=None, device=args.device
        )
        shuffled_indices_batch = shuffled_indices.view(num_mini_batch, -1)

        observations_shaped = buffer_observations.view(-1, obs_dim)
        actions_shaped = buffer_actions.view(-1, act_dim)
        values_shaped = buffer_values.view(-1, 1)

        ep_action_loss = torch.tensor(0.0, device=args.device).float()
        ep_value_loss = torch.tensor(0.0, device=args.device).float()

        for indices in shuffled_indices_batch:
            optimizer.zero_grad()

            observations_batch = observations_shaped[indices]
            actions_batch = actions_shaped[indices]
            values_batch = values_shaped[indices]

            pred_actions = student_actor_critic.actor(observations_batch)
            pred_values = student_actor_critic.get_value(observations_batch)

            action_loss = F.mse_loss(pred_actions, actions_batch)
            value_loss = F.mse_loss(pred_values, values_batch)
            (action_loss + value_loss).backward()

            optimizer.step()

            ep_action_loss.add_(action_loss.detach())
            ep_value_loss.add_(value_loss.detach())

        L = shuffled_indices_batch.shape[0]
        ep_action_loss.div_(L)
        ep_value_loss.div_(L)

        elapsed_time = time.time() - start
        torch.save(student_actor_critic, "logs/behavior_cloning/distilled.pt")

        print(
            (
                f"Epoch {epoch+1:4d}/{args.num_epochs:4d} | "
                f"Elapsed Time {elapsed_time:8.2f} |"
                f"Action Loss: {ep_action_loss.item():8.4f} | "
                f"Value Loss: {ep_value_loss.item():8.2f}"
            )
        )


if __name__ == '__main__':
    args = get_args()
    train(args)