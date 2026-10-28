import yaml
import os
import glob
import numpy as np


class Config:

    def __init__(self, FLAG, project_path, job_dir):
        self._flags = FLAG          # OmegaConf DictConfig をそのまま保持（ワーカー直列化用）
        cfg_id = FLAG.cfg
        self.id = cfg_id
        self.project_path = project_path
        cfg_path = os.path.join(project_path, "design_opt", "cfg", f"{cfg_id}.yml")
        files = glob.glob(cfg_path, recursive=True)
        assert(len(files) == 1)
        cfg = yaml.safe_load(open(files[0], 'r'))
        # create dirs
        self.job_dir = job_dir 

        if not os.path.exists(self.job_dir):
            os.makedirs(self.job_dir)

        self.model_dir = '%s/models' % self.job_dir
        self.log_dir = '%s/log' % self.job_dir
        self.tb_dir = '%s/tb' % self.job_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        
        restore_dir = FLAG.get('restore_dir', None)
        if restore_dir is not None:
            self.restore_dir = os.path.join(self.project_path, restore_dir)
            if not os.path.exists(self.restore_dir):
                raise FileNotFoundError(
                    f"Restore directory not found: {self.restore_dir}\n"
                    f"Please check the path and try again."
                )
        
        # training config
        self.epoch = FLAG.get('epoch', 0)
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.95)
        self.agent_specs = cfg.get('agent_specs', dict())
        self.policy_specs = cfg.get('policy_specs', dict())
        self.policy_specs.update(FLAG.get('policy_specs', dict()))
        self.obs_specs = cfg.get('obs_specs', dict())
        self.obs_specs.update(FLAG.get('obs_specs', dict()))
        self.adv_clip = cfg.get('adv_clip', np.inf)
        self.eval_batch_size = FLAG.get('eval_batch_size', 10000)
        self.seed_method = cfg.get('seed_method', 'deep')
        
        # training config (from global flag)
        self.lr_decay = FLAG.get('lr_decay', False)
        self.policy_optimizer = FLAG.get('policy_optimizer', 'Adam')
        self.policy_lr = FLAG.get('policy_lr', 5e-5)
        self.policy_momentum = FLAG.get('policy_momentum', 0.0)
        self.policy_weightdecay = FLAG.get('policy_weightdecay', 0.0)
        self.value_specs = cfg.get('value_specs', dict())
        self.value_specs.update(FLAG.get('value_specs', dict()))
        self.value_optimizer = FLAG.get('value_optimizer', 'Adam')
        self.value_lr = FLAG.get('value_lr', 3e-4)
        self.value_momentum = FLAG.get('value_momentum', 0.0)
        self.value_weightdecay = FLAG.get('value_weightdecay', 0.0)
        self.clip_epsilon = FLAG.get('clip_epsilon', 0.2)
        self.num_optim_epoch = int(FLAG.get('num_optim_epoch', 10))
        self.min_batch_size = int(FLAG.get('min_batch_size', 50000))
        self.mini_batch_size = int(FLAG.get('mini_batch_size', self.min_batch_size))
        self.max_epoch_num = int(FLAG.get('max_epoch_num', 1000))
        self.seed = FLAG.get('seed', 0)
        self.save_model_interval = FLAG.get('save_model_interval', 100)
        self.norm_advantage = FLAG.get('norm_advantage', False)
        self.max_grad_norm = FLAG.get('max_grad_norm', 40)
        self.uni_obs_norm = FLAG.get('uni_obs_norm', False)
        self.norm_return = FLAG.get('norm_return', True)
        self.reward_shift = FLAG.get('reward_shift', 0.0)
        self.xml_name = FLAG.get('xml_name', 'default')
        self.planner_demean = FLAG.get('planner_demean', False)
        
        self.enable_wandb = FLAG.get('enable_wandb', True)
        self.group = FLAG.get('group', 'group')

        # anneal parameters
        self.scheduled_params = cfg.get('scheduled_params', dict())
        self.skel_entropy_coef = FLAG.get('skel_entropy_coef', 0.0)
        self.attr_entropy_coef = FLAG.get('attr_entropy_coef', 0.0)
        self.control_entropy_coef = FLAG.get('control_entropy_coef', 0.0)

        # env
        self.env_name = cfg.get('env_name', 'hopper')
        self.done_condition = cfg.get('done_condition', dict())
        self.env_specs = cfg.get('env_specs', dict())
        self.env_specs.update(FLAG.get('env_specs', dict()) or {})
        self.reward_specs = cfg.get('reward_specs', dict())
        self.reward_specs.update(FLAG.get('reward_specs', dict()) or {})
        self.add_body_condition = cfg.get('add_body_condition', dict())
        self.max_body_depth = cfg.get('max_body_depth', 4)
        self.min_body_depth = cfg.get('min_body_depth', 1)
        self.enable_remove = cfg.get('enable_remove', True)
        self.skel_transform_nsteps = FLAG.get('skel_transform_nsteps', 5)
        self.env_init_height = cfg.get('env_init_height', False)

        # robot config
        # ⛔ Bug 74（系譜 9-281）: 以前は yml だけを読み、起動行の `+robot_param_scale=` を黙って捨てていた（R1 の凍結が効かなかった）
        self.robot_param_scale_src = 'flag' if FLAG.get('robot_param_scale', None) is not None else 'yml'
        self.robot_param_scale = FLAG.get('robot_param_scale', cfg.get('robot_param_scale', 0.1))
        self.robot_cfg = cfg.get('robot', dict())
        # ⭐ 系譜 9-235: `+fix_base=true` で根元の柱（台座）の形を設計変数から外す（xml_robot.Body.set_params）
        if FLAG.get('fix_base', False):
            self.robot_cfg = dict(self.robot_cfg, fix_base=True)

        self.restore_dir = FLAG.get('restore_dir', None)
        self.reset_epoch = FLAG.get('reset_epoch', False)
        self.reset_obs_norm = FLAG.get('reset_obs_norm', False)
        self.skel_uniform_prob = FLAG.get('skel_uniform_prob', 0.0)
        self.stack_follower_steps = int(FLAG.get('stack_follower_steps', 6))
        self.lamda = FLAG.get('lamda', 1e-2)
        self.fisher_correct = FLAG.get('fisher_correct', False)
        self.stabilize_fisher = FLAG.get('stabilize_fisher', False)
        self.s_gradient_limit = FLAG.get('s_gradient_limit', -1)
        self.gradient_ratio_limit = FLAG.get('gradient_ratio_limit', -1)
        self.morph_prior = FLAG.get('morph_prior', False)
        self.control_prior = FLAG.get('control_prior', False)
        self.fix_skeleton = FLAG.get('fix_skeleton', False)

        # ⭐ 系譜 9-283 ①・9-285: 読まれない設定を黙って捨てない（Bug 74 の型）。警告を出すだけで、効く値は変えない
        try:
            self.unread_warnings = self._unread_settings(FLAG, cfg)
        except Exception as e:      # 検査の失敗で学習を止めない
            self.unread_warnings = [f'読まれない設定の検査が例外で止まった: {e!r}']
        for w in self.unread_warnings:
            print(f'[Config] ⚠️ {w}', flush=True)

    # 上の __init__ が yml（cfg）だけから読む設定と、起動行・conf（FLAG）だけから読む設定。__init__ に足したらここにも足す
    _YML_ONLY = {'gamma': 'gamma', 'tau': 'tau', 'agent_specs': 'agent_specs', 'adv_clip': 'adv_clip',
                 'seed_method': 'seed_method', 'scheduled_params': 'scheduled_params', 'env_name': 'env_name',
                 'done_condition': 'done_condition', 'add_body_condition': 'add_body_condition',
                 'max_body_depth': 'max_body_depth', 'min_body_depth': 'min_body_depth',
                 'enable_remove': 'enable_remove', 'env_init_height': 'env_init_height', 'robot': 'robot_cfg'}
    _FLAG_ONLY = ('epoch', 'eval_batch_size', 'lr_decay', 'policy_optimizer', 'policy_lr', 'policy_momentum',
                  'policy_weightdecay', 'value_optimizer', 'value_lr', 'value_momentum', 'value_weightdecay',
                  'clip_epsilon', 'num_optim_epoch', 'min_batch_size', 'mini_batch_size', 'max_epoch_num', 'seed',
                  'save_model_interval', 'norm_advantage', 'max_grad_norm', 'uni_obs_norm', 'norm_return',
                  'reward_shift', 'xml_name', 'planner_demean', 'enable_wandb', 'group', 'skel_entropy_coef',
                  'attr_entropy_coef', 'control_entropy_coef', 'skel_transform_nsteps', 'reset_epoch',
                  'reset_obs_norm', 'skel_uniform_prob', 'stack_follower_steps', 'lamda', 'fisher_correct',
                  'stabilize_fisher', 's_gradient_limit', 'gradient_ratio_limit', 'morph_prior', 'control_prior',
                  'fix_skeleton')
    _BOTH = ('policy_specs', 'obs_specs', 'value_specs', 'env_specs', 'reward_specs', 'robot_param_scale', 'fix_base', 'restore_dir')
    # train.py が FLAG から直接読むもの・Hydra と conf の管理用（Config の外）
    _ELSEWHERE = ('render', 'show_noise', 'num_threads', 'gpu_index', 'project', 'job_name', 'cfg', 'hydra',
                  'n', 'tmp', 'base_dir', 'project_path')

    def _unread_settings(self, FLAG, cfg):
        from omegaconf import OmegaConf
        plain = lambda v: OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
        out = []
        for k, attr in self._YML_ONLY.items():
            if k in FLAG and plain(FLAG[k]) != plain(getattr(self, attr)):
                out.append(f'起動行・conf の {k}={plain(FLAG[k])} は読まれない（yml から読み {plain(getattr(self, attr))} が効く）')
        for k in self._FLAG_ONLY:
            if k in cfg and plain(cfg[k]) != plain(getattr(self, k)):
                out.append(f'yml の {k}={cfg[k]} は読まれない（起動行・conf から読み {plain(getattr(self, k))} が効く）')
        known = set(self._YML_ONLY) | set(self._FLAG_ONLY) | set(self._BOTH) | set(self._ELSEWHERE)
        for k in FLAG.keys():
            if k not in known:
                out.append(f'起動行の {k} はどこにも読まれない')
        return out


def replay_robot_param_scale(cfg, run_dir):
    """再生の道具が呼ぶ: 学習に効いた robot_param_scale に cfg を合わせる（学習では呼ばない）。

    ⛔ 系譜 9-283: Bug 74 の修正（c86c04e）より前の config.py は起動行の値を捨てて yml の値で学習した。
    その run を今の Config で組むと起動行の値が効き、学習しなかった形を再生する。
    ⭐ の行（train.py が出す「⭐ robot_param_scale = …」）が log_train.txt に無ければ修正前の run とみなす。
    """
    log = os.path.join(run_dir, 'log', 'log_train.txt')
    if not os.path.exists(log) or '⭐ robot_param_scale = ' in open(log, encoding='utf-8', errors='ignore').read():
        return cfg.robot_param_scale
    cfg_path = os.path.join(cfg.project_path, 'design_opt', 'cfg', f'{cfg.id}.yml')
    yml = yaml.safe_load(open(cfg_path, 'r')).get('robot_param_scale', 0.1)
    if float(yml) != float(cfg.robot_param_scale):
        print(f'[replay] ⛔ Bug 74 の修正前の run: 起動行の robot_param_scale {cfg.robot_param_scale} ではなく、'
              f'学習に効いた yml の {yml} で再生する', flush=True)
        cfg.robot_param_scale = yml
        cfg.robot_param_scale_src = 'yml（Bug 74 の修正前の run）'
    return cfg.robot_param_scale
