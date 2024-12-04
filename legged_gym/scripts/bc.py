# behavior cloning
# want to do the same, but without the linear velocity observation

from legged_gym.utils import get_args, task_registry

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # TODO: create two envs and create behavior cloning

if __name__ == '__main__':
    args = get_args()
    train(args)