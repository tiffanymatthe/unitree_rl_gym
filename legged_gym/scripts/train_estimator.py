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
HISTORY_LEN = 6
NUM_TEACHER_EPOCHS = 1

SAVE_PATH = "logs/behavior_cloning/walking_chained_hist_len_6"
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

    dof_len = 12

    teacher_obs_shape = (48,)
    teacher_obs_dim = 48
    teacher_action_shape = (12,)
    teacher_action_dim = 12

    student_obs_shape = (45,) # for env
    student_obs_dim = 45
    student_action_shape = (12,) # for estimator
    student_action_dim = 12

    # remove linear velocity and commands (6) but add a history of size HISTORY_LEN of past joint positions and velocities (12 + 12)
    estimator_obs_shape = (teacher_obs_dim - 6 + HISTORY_LEN * 2 * dof_len,)
    estimator_obs_dim = teacher_obs_dim - 6 + HISTORY_LEN * 2 * dof_len
    estimator_action_shape = (3,)
    estimator_action_dim = 3
    
    num_processes = 4096
    num_steps = BATCH_SIZE // num_processes + 1

    buffer_observations = torch.zeros(
        num_steps + 1, num_processes, *teacher_obs_shape, device="cpu"
    )
    buffer_estimator_observations = torch.zeros(
        num_steps + 1, num_processes, *estimator_obs_shape, device="cpu"
    )
    buffer_estimator_actions = torch.zeros(
        num_steps, num_processes, *estimator_action_shape, device="cpu"
    )
    buffer_student_actions = torch.zeros(
        num_steps, num_processes, *student_action_shape, device="cpu"
    )

    actor_critic = load_model(model_path=TEACHER_PATH, num_obs=teacher_obs_dim, device=args.rl_device)
    for param in actor_critic.parameters():
        param.requires_grad = False

    estimator = load_estimator_model(model_path=None, num_obs=estimator_obs_dim, device=args.rl_device)

    student_actor_critic = load_model(model_path=None, num_obs=student_obs_dim, device=args.rl_device)

    past_joint_positions = torch.zeros(num_processes, dof_len * HISTORY_LEN, device="cpu")
    past_joint_velocities = torch.zeros(num_processes, dof_len * HISTORY_LEN, device="cpu")

    optimizer = torch.optim.Adam(estimator.parameters(), lr=3e-4)

    obs = env.reset()[0]
    buffer_observations[-1].copy_(obs.to("cpu"))
    past_joint_positions = obs[:,12:24].repeat(1,HISTORY_LEN).detach().clone()
    past_joint_velocities = obs[:,24:36].repeat(1,HISTORY_LEN).detach().clone()
    est_obs = torch.concatenate((obs[:,3:9], obs[:,9:], past_joint_positions, past_joint_velocities), dim=1)
    buffer_estimator_observations[-1].copy_(est_obs.to("cpu"))

    file = open(f"{SAVE_PATH}/combined_results.csv", mode="w", newline='')
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Elapsed Time", "Action Loss", "Estimator Action Loss"])

    start = time.time()
    for epoch in range(NUM_EPOCHS):
        with torch.no_grad():
            buffer_observations[0].copy_(buffer_observations[-1])
            buffer_observations[0].copy_(buffer_observations[-1])
            for step in range(num_steps):
                action = actor_critic.act_inference(
                    buffer_observations[step].to(args.rl_device)
                )

                if epoch >= NUM_TEACHER_EPOCHS:
                    modified_observation = buffer_observations[step].clone()
                    estimated_lin_vels = estimator.act(
                        buffer_estimator_observations[step].to(args.rl_device)
                    )
                    modified_observation[:,0:3] = estimated_lin_vels
                    stochastic_action = student_actor_critic.act(
                        modified_observation.to(args.rl_device)
                    )
                else:
                    stochastic_action = actor_critic.act(
                        buffer_observations[step].to(args.rl_device)
                    )

                obs, priv_obs, _, dones, _ = env.step(stochastic_action)

                # remove command (9 to 12)
                est_obs = torch.concatenate((obs[:,3:9], obs[:,12:], past_joint_positions, past_joint_velocities), dim=1)
                buffer_estimator_observations[step + 1].copy_(est_obs.to("cpu"))
                buffer_observations[step + 1].copy_(obs.to("cpu"))
                buffer_estimator_actions[step].copy_(priv_obs[:,0:3].to("cpu"))
                buffer_student_actions[step].copy_(action.to("cpu"))

                # if not done, update history
                # first shift last two to first two to leave a space for the last one
                past_joint_positions[~dones,:-12].copy_(past_joint_positions[~dones,12:])
                past_joint_positions[~dones,-12:].copy_(obs[~dones,12:24])
                past_joint_velocities[~dones,:-12].copy_(past_joint_velocities[~dones,12:])
                past_joint_velocities[~dones,-12:].copy_(obs[~dones,24:36])

                # if done, repeat history
                past_joint_positions[dones].copy_(obs[dones,12:24].repeat(1,HISTORY_LEN))
                past_joint_velocities[dones].copy_(obs[dones,24:36].repeat(1,HISTORY_LEN))

        num_mini_batch = BATCH_SIZE // MINI_BATCH_SIZE
        shuffled_indices = torch.randperm(
            num_mini_batch * MINI_BATCH_SIZE, generator=None, device="cpu"
        )
        shuffled_indices_batch = shuffled_indices.view(num_mini_batch, -1)

        observations_shaped = buffer_observations.view(-1, teacher_obs_dim)
        estimator_observations_shaped = buffer_estimator_observations.view(-1, estimator_obs_dim)
        estimator_actions_shaped = buffer_estimator_actions.view(-1, estimator_action_dim)
        student_actions_shaped = buffer_student_actions.view(-1, student_action_dim)

        ep_action_loss = torch.tensor(0.0, device=args.rl_device).float()
        ep_estimator_action_loss = torch.tensor(0.0, device=args.rl_device).float()

        for indices in shuffled_indices_batch:
            optimizer.zero_grad()
            observations_batch = observations_shaped[indices]
            estimator_actions_batch = estimator_actions_shaped[indices]
            student_actions_batch = student_actions_shaped[indices]
            estimator_observations_batch = estimator_observations_shaped[indices]
            pred_estimator_actions = estimator.act_inference(estimator_observations_batch.to("cuda:0"))
            modified_observations_batch = observations_batch.clone()
            modified_observations_batch[:,0:3] = pred_estimator_actions #.detach()
            pred_student_actions = student_actor_critic.act_inference(modified_observations_batch.to("cuda:0"))
            estimator_action_loss = F.mse_loss(pred_estimator_actions, estimator_actions_batch.to("cuda:0"))
            student_action_loss = F.mse_loss(pred_student_actions, student_actions_batch.to("cuda:0"))
            (estimator_action_loss + student_action_loss).backward()

            optimizer.step()

            ep_action_loss.add_(student_action_loss.detach())
            ep_estimator_action_loss.add_(estimator_action_loss.detach())

        L = shuffled_indices_batch.shape[0]
        ep_action_loss.div_(L)

        elapsed_time = time.time() - start

        torch.save({
            'model_state_dict':estimator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': epoch,
            "infos": None,
            }, f"{SAVE_PATH}/estimator_model.pt")
        
        torch.save({
            'model_state_dict':student_actor_critic.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': epoch,
            "infos": None,
            }, f"{SAVE_PATH}/policy_model.pt")

        print(
            (
                f"Epoch {epoch+1:4d}/{NUM_EPOCHS:4d} | "
                f"Elapsed Time {elapsed_time:8.2f} |"
                f"Action Loss: {ep_action_loss.item():8.4f} | "
                f"Estimator Action Loss: {ep_estimator_action_loss.item():8.4f} | "
            )
        )

        writer.writerow(
            [
                epoch+1,
                f"{elapsed_time:8.2f}",
                f"{ep_action_loss.item():8.4f}",
                f"{ep_estimator_action_loss.item():8.4f}",
            ]
        )


if __name__ == '__main__':
    args = get_args()
    train(args)