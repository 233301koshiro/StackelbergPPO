# Morphology–Control Co-Design via Stackelberg PPO

Official implementation of the paper: 

**“Efficient Morphology-Control Co-Design via Stackelberg Proximal Policy Optimization”**, Yanning Dai*, Yuhui Wang*, Dylan R. Ashley, Jürgen Schmidhuber, International Conference on Learning Representations (ICLR), 2026.

[Paper](https://openreview.net/pdf?id=sJ0vOOkclw) | [Code](https://github.com/YanningDai/StackelbergPPO.git) | [Project Page](https://yanningdai.github.io/stackelberg-ppo-co-design/)

<img src="static/m.png" alt="description">

## 📦 Installation

**System Requirements**

- Tested OS: Linux Ubuntu 24.04.3 LTS
- Python >= 3.9
- PyTorch == 2.0.1

**Dependencies**

1. Clone this GitHub repository and enter the project directory:

```bash
cd StackelbergPPO
```
2. Create the Conda environment and install dependencies
```bash
conda create -n StackelPPO python=3.9 -y
conda activate StackelPPO
conda install mesalib glew glfw patchelf -c conda-forge -y
pip install -r requirements.txt
```

3. Install MuJoCo 2.1.0

```bash
mkdir -p ~/.mujoco
cd /tmp
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

4. Add Environment Variables

```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia' >> ~/.bashrc
source ~/.bashrc
```

## 🚀 Quick Start

### MuJoCo backend (original)

```bash
OMP_NUM_THREADS=1 python -m design_opt.train cfg=pusher
```

### Choreonoid backend (`akita_sp` Docker image)

Training runs inside the Choreonoid process because the simulator requires the Qt application context:

```bash
USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=1 hydra.run.dir=single_run/pusher_cnoid
```

**Available environments**: cheetah, crawler, glider-hard, glider-medium, glider-regular, pusher, stepper-hard, stepper, swimmer, terraincrosser, walker-hard, walker-medium, walker-regular

## ⚙️ Advanced Usage

**Resume training from checkpoint**

```bash
# MuJoCo
OMP_NUM_THREADS=1 python -m design_opt.train cfg=pusher +restore_dir="single_run/test"

# Choreonoid
USE_CHOREONOID=1 choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=1 +restore_dir=single_run/test
```

Load only the morphology prior without controller weights:
```bash
# Choreonoid scratch training with morphology transfer from MuJoCo checkpoint
USE_CHOREONOID=1 choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=1 \
  +restore_dir=single_run/pusher morph_prior=true reset_epoch=true
```

**MuJoCo → Choreonoid migration**

```bash
python3 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher
python3 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher --auto-scratch
```

**Evaluation (Choreonoid)**

```bash
USE_CHOREONOID=1 choreonoid --no-window --python \
  scripts/eval_cnoid_numerical.py -- \
  --restore_dir single_run/pusher_cnoid --num_episodes 5
```

**Configuration**

This project uses Hydra for configuration management. Key configuration files are located in `design_opt/conf/`. Modify via YAML files or command-line overrides:

```bash
# MuJoCo
python -m design_opt.train cfg=pusher lamda=5 gradient_ratio_limit=1.0

# Choreonoid
choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher lamda=5 gradient_ratio_limit=1.0 num_threads=1
```

## 📊 Visualization and Results

All experimental visualizations are hosted at this project page: <https://yanningdai.github.io/stackelberg-ppo-co-design>

## 🙏 Acknowledgements

This project builds upon and is inspired by [BodyGen](https://github.com/Josh00-Lu/BodyGen) and [Transform2Act](https://github.com/Khrylx/Transform2Act). We thank the authors for their excellent work! 

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{dai2026stackelbergppo,
  title     = {Efficient Morphology--Control Co-Design via Stackelberg Proximal Policy Optimization},
  author    = {Dai, Yanning and Wang, Yuhui and Ashley, Dylan R. and Schmidhuber, Jürgen},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
