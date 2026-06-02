# import argparse
import os
import sys
sys.path.append(os.getcwd())

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent
from design_opt.utils.tools import set_global_seed
try:
    import wandb
except Exception:
    wandb = None
import hydra
from omegaconf import DictConfig

project_path = os.getcwd()

def main_loop(FLAGS, job_dir):
    if FLAGS.render:
        FLAGS.num_threads = 1
        
    cfg = Config(FLAGS, project_path, job_dir)

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=FLAGS.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(FLAGS.gpu_index)
    set_global_seed(cfg.seed)

    # load_epoch: which checkpoint file to load (0 = none, int = epoch_NNNN.p, str = <str>.p)
    load_epoch = int(FLAGS.epoch) if isinstance(FLAGS.epoch, str) and FLAGS.epoch.isnumeric() else FLAGS.epoch
    # start_epoch: where the training loop begins
    # reset_epoch=true: load the checkpoint but restart the counter from 0 (MuJoCo→Choreonoid transfer)
    if getattr(FLAGS, 'reset_epoch', False):
        start_epoch = 0
    elif isinstance(load_epoch, int):
        start_epoch = load_epoch
    else:
        start_epoch = 0


    """create agent"""
    agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device, seed=cfg.seed, num_threads=FLAGS.num_threads, training=True, checkpoint=load_epoch)

    if FLAGS.render:
        agent.pre_epoch_update(start_epoch)
        agent.sample(1e8, mean_action=not FLAGS.show_noise, render=True)
    else:
        for epoch in range(start_epoch, cfg.max_epoch_num):
            agent.optimize(epoch)
            agent.save_checkpoint(epoch)

            """clean up gpu memory"""
            torch.cuda.empty_cache()

        agent.logger.info('training done!')
        if hasattr(agent, '_worker_pool') and agent._worker_pool is not None:
            agent._worker_pool.close()
        if hasattr(agent, 'env') and hasattr(agent.env, 'close'):
            agent.env.close()


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    FLAGS = cfg
    
    job_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if FLAGS.enable_wandb:
        wandb.login()
        wandb.init(
            project=str(FLAGS.project),
            group=str(FLAGS.group),
            name=str(FLAGS.job_name),
            resume=False,
            dir=job_dir,
        )
    
    main_loop(FLAGS, job_dir)
    
    if FLAGS.enable_wandb:
        wandb.finish()
    
if __name__ == '__main__':
    main()
