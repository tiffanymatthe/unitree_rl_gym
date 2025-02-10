```bash
# in unitree_rl_gym folder
conda create -n unitree_rl_gym_env python=3.8
conda activate unitree_rl_gym_env
pip3 install torch==1.10.0

# first clone rsl_rl somewhere
cd rsl_rl && git checkout v1.0.2 && pip install -e .

# in unitree_rl_gym folder, first remove isaacgym from setup.py
pip install -e .

# issues install numpy==1.20, so do this
conda config --env --add channels conda-forge
conda install numpy==1.20 

# and finish what was cancelled
pip install tensorboard
```