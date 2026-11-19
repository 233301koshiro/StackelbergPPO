#!/usr/bin/env python3
"""Stackelberg 勾配の間接の項を測るプローブ（評価専用。repo にも single_run にも書かない）。

使い方（repo の根で）:
  PROBE_RUN=single_run/<run> PROBE_CKPT=<int|best> PROBE_OUT=<scratch dir> \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 timeout -k 10 <sec> /choreonoid_ws/install/bin/choreonoid \
      --no-window --python <this file> > <log> 2>&1
"""
import os, sys, json, time, copy, traceback
sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'

RUN = os.environ['PROBE_RUN'].rstrip('/')
CKPT = os.environ.get('PROBE_CKPT', 'best')
OUT = os.environ['PROBE_OUT']
NB = int(os.environ.get('PROBE_NBATCH', '4'))
NEP = int(os.environ.get('PROBE_EPISODES', '52'))
NWIN = int(os.environ.get('PROBE_NWIN', '4'))
SEED = int(os.environ.get('PROBE_SEED', '777'))
HIDE = os.environ.get('PROBE_HIDE_REWARD') == '1'
DO_UPDATE = os.environ.get('PROBE_UPDATE', '1') == '1'
NTHREADS = int(os.environ.get('PROBE_TORCH_THREADS', '2'))
NDEC = int(os.environ.get('PROBE_NDECOMP', '0'))
CSCALES = [float(x) for x in os.environ.get('PROBE_CSCALES', '').split(',') if x]
os.makedirs(OUT, exist_ok=True)

_le = os.path.join(RUN, 'launch_env.json')
LAUNCH_ENV = json.load(open(_le)) if os.path.exists(_le) else {}
for _k, _v in LAUNCH_ENV.items():
    os.environ[_k] = _v
print(f'[probe] run={RUN} ckpt={CKPT} launch_env={LAUNCH_ENV}', flush=True)


def P(*a):
    print('[probe]', *a, flush=True)


def main():
    import numpy as np
    import yaml
    import torch
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config, replay_robot_param_scale
    from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
    from design_opt.utils.tools import set_global_seed, TrajBatchDisc
    from design_opt.utils.stackelberg import bilevel_leader_grad_correct, fisher_vector_product_selfkl
    from khrylib.utils.memory import Memory
    from khrylib.utils.torch import to_test, to_train, LongTensor
    from khrylib.utils.tools import index_select_list

    torch.set_num_threads(NTHREADS)
    project_path = os.getcwd()
    FLAGS = OmegaConf.create(yaml.safe_load(open(f'{RUN}/.hydra/config.yaml')))
    d = OmegaConf.to_container(FLAGS, resolve=True)
    d.pop('restore_dir', None)
    FLAGS = OmegaConf.create(d)
    cfg = Config(FLAGS, project_path, os.path.join(OUT, 'job'))     # job_dir は scratch（run のフォルダには書かない）
    cfg.restore_dir = RUN
    replay_robot_param_scale(cfg, RUN)
    cfg.control_prior = False
    cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    set_global_seed(SEED)
    ck = int(CKPT) if CKPT != 'best' else 'best'
    dev = torch.device('cpu')
    agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=dev, seed=SEED, num_threads=1, training=False, checkpoint=ck)
    env = agent.env
    P(f'cfg: lamda={agent.damping} ratio_limit={cfg.gradient_ratio_limit} s_grad_limit={cfg.s_gradient_limit} m={agent.ep_planner.m} '
      f'leader_step={agent.leader_step} max_grad_norm={cfg.max_grad_norm} clip_eps={agent.clip_epsilon} mb={agent.mini_batch_size} '
      f'robot_param_scale={cfg.robot_param_scale} fisher_correct={cfg.fisher_correct}')
    pm = agent.para_mgr
    pol0 = agent.policy_net
    names1 = set(pm.get_parameter_names(pol0, ['skel_trans', 'attr_trans']))
    th1_names = [n for n, p in pol0.named_parameters() if n in names1]
    names2 = set(pm.get_parameter_names(pol0, ['execution']))
    th2_names = [n for n, p in pol0.named_parameters() if n in names2]
    all_names = [n for n, p in pol0.named_parameters()]
    P(f'params: theta1={sum(p.numel() for n,p in pol0.named_parameters() if n in names1)} '
      f'theta2={sum(p.numel() for n,p in pol0.named_parameters() if n in names2)} '
      f'unassigned={[n for n in all_names if n not in names1 and n not in names2]}')
    sizes1 = [p.numel() for n, p in pol0.named_parameters() if n in names1]
    attr_mask = torch.cat([torch.full((s,), n.startswith('attr_'), dtype=torch.bool) for n, s in zip(th1_names, sizes1)])
    P(f'theta1 split: attr={int(attr_mask.sum())} skel={int((~attr_mask).sum())}')
    try:
        for b in env.robot.bodies:
            nm = []
            b.get_params(nm, get_name=True, pad_zeros=True)
            P(f'design names body={b.name}: {nm}')
    except Exception as e:
        P('design names: failed', repr(e))
    P(f'attr_design_dim={env.attr_design_dim} control_action_dim={env.control_action_dim} n_bodies={len(env.robot.bodies)}')
    P(f'attr_log_std={pol0.attr_action_log_std.detach().numpy().round(3).tolist()} control_log_std={pol0.control_action_log_std.detach().numpy().round(3).tolist()}')

    # ---------- sampling ----------
    def sample_batch(n_ep):
        memory = Memory()
        ep_ret = []
        with to_test(agent.policy_net):
            with torch.no_grad():
                agent.obs_norm.eval()
                for _ in range(n_ep):
                    state = env.reset()
                    R = 0.0
                    while True:
                        sv = tensorfy([state])
                        if cfg.uni_obs_norm:
                            sv = agent.normalize_observation(sv)
                        action = agent.policy_net.select_action(sv, False).numpy().astype(np.float64)
                        next_state, env_reward, termination, truncation, info = env.step(action)
                        reward = env_reward
                        c_reward = info.get('reward_ctrl', 0)
                        if info['stage'] == 'execution':
                            reward += cfg.reward_shift
                        done = (termination or truncation)
                        memory.push(state, action, termination, done, next_state, reward, 1, c_reward)
                        R += reward
                        if done:
                            break
                        state = next_state
                    ep_ret.append(R)
        return TrajBatchDisc([memory]), np.array(ep_ret)

    class D:  # one prepared batch
        pass

    def prep(batch):
        o = D()
        o.states = tensorfy(batch.states, dev)
        o.next_states = tensorfy(batch.next_states, dev)
        o.actions = tensorfy(batch.actions, dev)
        o.rewards = torch.from_numpy(batch.rewards).to(torch.float64)
        o.next_terminations = torch.from_numpy(batch.next_terminations).to(torch.float64)
        o.next_dones = torch.from_numpy(batch.next_dones).to(torch.float64)
        o.c_reward = torch.from_numpy(np.asarray(batch.c_reward, dtype=np.float64)).to(torch.float64)
        if cfg.uni_obs_norm:
            agent.obs_norm.eval()       # checkpoint の正規化のまま凍結（学習は 1 バッチぶん更新するが、差は小さい）
            o.states = agent.normalize_observation(o.states)
            o.next_states = agent.normalize_observation(o.next_states)
        with to_test(agent.policy_net, agent.value_net):
            with torch.no_grad():
                fl, vr = [], []
                for i in range(0, len(o.states), 10000):
                    fl.append(agent.policy_net.get_log_prob(o.states[i:i + 10000], o.actions[i:i + 10000]))
                    vr.append(agent.value_net(o.states[i:i + 10000]))
                o.fixlog = torch.cat(fl)
                o.v_raw = torch.cat(vr).reshape(-1)
        o.state_types = torch.from_numpy(np.array([it[2] for it in o.states])).to(torch.int)
        o.next_state_types = torch.from_numpy(np.array([it[2] for it in o.next_states])).to(torch.int)
        o.st_np = o.state_types.numpy().reshape(-1).astype(np.int64)
        o.episodes = agent.split_episodes_from_dones(o.next_dones)
        # MC returns for the scale fit
        r = o.rewards.numpy().reshape(-1)
        o.R_design = np.zeros_like(r)
        o.G_ctrl = np.zeros_like(r)
        o.remain = np.zeros_like(r)
        for ep in o.episodes:
            acc = 0.0
            g = 0.0
            for i in ep[::-1]:
                acc = r[i] + acc
                o.R_design[i] = acc
                if o.st_np[i] == 2:
                    g = r[i] + agent.gamma * g
                    o.G_ctrl[i] = g
                o.remain[i] = ep[-1] - i
        return o

    t0 = time.time()
    Ds = []
    ep_rets = []
    for b in range(NB):
        tb = time.time()
        batch, er = sample_batch(NEP)
        o = prep(batch)
        Ds.append(o)
        ep_rets.append(er)
        nL = int((o.st_np != 2).sum())
        P(f'batch {b}: steps={len(o.states)} episodes={len(o.episodes)} leader_steps={nL} follower_steps={len(o.states) - nL} '
          f'ep_len(min/med/max)={min(len(e) for e in o.episodes)}/{int(np.median([len(e) for e in o.episodes]))}/{max(len(e) for e in o.episodes)} '
          f'sample+prep {time.time() - tb:.0f}s' + ('' if HIDE else f' ep_ret mean={er.mean():.2f} std={er.std():.2f}'))
        del batch
    P(f'sampling done {time.time() - t0:.0f}s')

    # ---------- return-norm reconstruction (not stored in the checkpoint) ----------
    vd = np.concatenate([o.v_raw.numpy()[o.st_np != 2] for o in Ds])
    Rd = np.concatenate([o.R_design[o.st_np != 2] for o in Ds])
    selc = [(o.st_np == 2) & (o.remain >= 400) for o in Ds]
    if sum(int(s.sum()) for s in selc) < 1000:
        selc = [(o.st_np == 2) for o in Ds]
    vc = np.concatenate([o.v_raw.numpy()[s] for o, s in zip(Ds, selc)])
    Gc = np.concatenate([o.G_ctrl[s] for o, s in zip(Ds, selc)])
    c_d = float((vd * Rd).sum() / (vd * vd).sum())
    c_c = float((vc * Gc).sum() / (vc * vc).sum())
    def r2(pred, y):
        return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    fit = dict(corr_design=float(np.corrcoef(vd, Rd)[0, 1]) if vd.std() > 0 else float('nan'),
               corr_control=float(np.corrcoef(vc, Gc)[0, 1]),
               r2_design=r2(c_d * vd, Rd), r2_control=r2(c_c * vc, Gc))
    P(f'value-scale fit: corr(V,R) design={fit["corr_design"]:.3f} control={fit["corr_control"]:.3f} '
      f'R2 design={fit["r2_design"]:.3f} control={fit["r2_control"]:.3f}' + ('' if HIDE else f' scale design={c_d:.3f} control={c_c:.3f}'))
    for nrm, c in ((agent.design_ret_norm, c_d), (agent.control_ret_norm, c_c)):
        nrm.std[:] = abs(c)
        nrm.var[:] = c * c
        nrm.n += 10 ** 12
        nrm.eval()
    for o in Ds:
        o.adv, o.ret = agent.estimate_advantages(o.states, o.next_states, o.rewards, o.next_terminations, o.next_dones,
                                                 o.state_types, o.next_state_types, o.c_reward)
        L, F = agent.ep_planner.build_episode_indices(o.episodes, o.st_np)
        valid = [i for i, (l, f) in enumerate(zip(L, F)) if len(l) >= agent.leader_step and len(f) > 0]
        o.L = [L[i] for i in valid]
        o.F = [F[i] for i in valid]
        o.E = len(valid)
        o.Fmat = agent.ep_planner.make_Fmat(o.F)
        # leader advantage with no value baseline (sensitivity check)
        dm = (o.st_np != 2)
        Rn = torch.from_numpy(o.R_design[dm])
        o.adv_nobase = o.adv.clone()
        o.adv_nobase[torch.from_numpy(dm)] = ((Rn - Rn.mean()) / Rn.std()).reshape(-1, 1)
    P(f'valid episodes per batch: {[o.E for o in Ds]} (of {[len(o.episodes) for o in Ds]})')

    m = agent.ep_planner.m
    lam = float(agent.damping)

    def flat(vs):
        return torch.cat([v.reshape(-1) for v in vs])

    def get_th(pol):
        t1 = [p for n, p in pol.named_parameters() if n in names1]
        t2 = [p for n, p in pol.named_parameters() if n in names2]
        return t1, t2

    def mk_policy_exec(pol):
        def policy_exec(states_batch):
            out = pol(states_batch)
            if isinstance(out, tuple):
                out = out[0]
            return out
        return policy_exec

    def cg_inst(fvp_fn, b, max_iter=20, tol=1e-3, tol_abs=1e-8):
        x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rTr = torch.dot(r, r); r0 = r.norm() + 1e-12
        it = 0; rel = 1.0
        for k in range(max_iter):
            Ap = fvp_fn(p); pAp = torch.dot(p, Ap)
            if pAp <= 0:
                break
            alpha = rTr / (pAp + 1e-12); x = x + alpha * p; r = r - alpha * Ap
            rTr_new = torch.dot(r, r); rel = float(r.norm() / r0); it = k + 1
            if rel < tol or float(r.norm()) < tol_abs:
                break
            beta = rTr_new / (rTr + 1e-12); p = r + beta * p; rTr = rTr_new
        return x.detach(), it, rel

    def cosn(a, b):
        return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-300))

    def leader_terms(o, k, pol=None, adv=None, diag=False, decomp=False):
        """Exactly the code path of update_policy for one leader minibatch (all episodes), follower window k."""
        pol = pol or agent.policy_net
        adv = o.adv if adv is None else adv
        agent.policy_net = pol
        L_idx = np.concatenate(o.L, axis=0)
        F_idx = o.Fmat[:, k * m:(k + 1) * m].ravel()
        Ls, La = index_select_list(o.states, L_idx), index_select_list(o.actions, L_idx)
        Fs, Fa = index_select_list(o.states, F_idx), index_select_list(o.actions, F_idx)
        Lt, Ft = torch.from_numpy(L_idx), torch.from_numpy(F_idx)
        surr, Ja, Jc, _ = agent.ppo_loss_stackelberg(Ls, La, adv[Lt].clone(), o.fixlog[Lt].clone(), Fs, Fa, adv[Ft].clone(), o.fixlog[Ft].clone())
        t1, t2 = get_th(pol)
        pe = mk_policy_exec(pol)
        Jd = bilevel_leader_grad_correct(J_a=Ja, J_c=Jc, theta1_list=t1, theta2_list=t2, policy_exec=pe, F_states=Fs,
                                         damping=agent.damping, cg_max_iter=20, cg_tol=1e-3, verbose=False, fisher_correct=cfg.fisher_correct)
        gs = flat([g.detach() if g is not None else torch.zeros_like(p) for g, p in zip(torch.autograd.grad(surr, t1, retain_graph=True, allow_unused=True), t1)])
        gd = flat([g.detach() if g is not None else torch.zeros_like(p) for g, p in zip(torch.autograd.grad(Jd, t1, retain_graph=diag, allow_unused=True), t1)])
        res = dict(gs=gs, gd=gd)
        if diag:
            v = flat([g if g is not None else torch.zeros_like(p) for g, p in zip(torch.autograd.grad(Jc, t2, retain_graph=True, allow_unused=True), t2)]).detach()
            fvp = lambda x: fisher_vector_product_selfkl(policy_exec=pe, theta2_params=t2, states=Fs, v_flat=x, damping=agent.damping, J_c=Jc, fisher_correct=False)
            s, it, rel = cg_inst(fvp, v)
            Fv = (fvp(v) - lam * v).detach(); Fs_ = (fvp(s) - lam * s).detach()
            res.update(v_norm=float(v.norm()), s_norm=float(s.norm()), cos_sv=cosn(s, v), s_over_v_lam=float(s.norm() / (v.norm() / lam + 1e-300)),
                       rayleigh_v=float(torch.dot(v, Fv) / torch.dot(v, v)), rayleigh_s=float(torch.dot(s, Fs_) / torch.dot(s, s)), cg_iters=it, cg_rel=rel)
            # per-episode pseudo-advantage b_e = d/d(eps) f_bar_e(theta2 + eps * s_tilde), s_tilde = -s (follower's damped natural-gradient ascent step)
            st = -s
            u = st / (st.norm() + 1e-300)
            h = 1e-4
            def fbar():
                with torch.no_grad():
                    lp = pol.get_log_prob(Fs, Fa)
                    return (torch.exp(lp - o.fixlog[Ft]) * adv[Ft]).view(o.E, m).mean(dim=1)
            def shift(c):
                off = 0
                with torch.no_grad():
                    for p in t2:
                        n = p.numel(); p.add_(c * u[off:off + n].view_as(p)); off += n
            shift(+h); fp = fbar(); shift(-2 * h); fm = fbar(); shift(+h)
            b_e = float(st.norm()) * (fp - fm) / (2 * h)
            lp = pol.get_log_prob(Ls, La)
            lbar = torch.exp(lp - o.fixlog[Lt]).view(o.E, agent.leader_step).mean(dim=1)
            gfd = flat([g if g is not None else torch.zeros_like(p) for g, p in zip(torch.autograd.grad((lbar * b_e).mean(), t1, allow_unused=True), t1)]).detach()
            res.update(fd_relerr=float((gfd - gd).norm() / (gd.norm() + 1e-300)), fd_cos=cosn(gfd, gd), b_e=b_e.numpy().copy(),
                       advL_ep=adv[Lt].view(o.E, agent.leader_step).mean(dim=1).numpy().copy())
            if decomp:
                # b_e = g_e . M gbar  (g_e: per-episode follower gradient from its m sampled steps, M=(F+lam I)^-1, gbar=mean_e g_e)
                #     = self_e (g_e . M g_e / E)  +  cross_e (g_e . M sum_{e'!=e} g_e' / E)
                lpF = pol.get_log_prob(Fs, Fa)
                fb = (torch.exp(lpF - o.fixlog[Ft]) * adv[Ft]).view(o.E, m).mean(dim=1)
                G = torch.stack([flat([g if g is not None else torch.zeros_like(p) for g, p in zip(torch.autograd.grad(fb[e], t2, retain_graph=True, allow_unused=True), t2)]).detach() for e in range(o.E)])
                gbar = G.mean(0)
                b_lin = G @ st
                self_e = torch.zeros(o.E)
                for e in range(o.E):
                    x, _, _ = cg_inst(fvp, G[e])
                    self_e[e] = torch.dot(G[e], x) / o.E
                cross_e = b_lin - self_e
                lp2 = pol.get_log_prob(Ls, La)
                lbar2 = torch.exp(lp2 - o.fixlog[Lt]).view(o.E, agent.leader_step).mean(dim=1)
                g_self = flat([g if g is not None else torch.zeros_like(p) for g, p in zip(torch.autograd.grad((lbar2 * self_e).mean(), t1, retain_graph=True, allow_unused=True), t1)]).detach()
                g_cross = flat([g if g is not None else torch.zeros_like(p) for g, p in zip(torch.autograd.grad((lbar2 * cross_e).mean(), t1, allow_unused=True), t1)]).detach()
                aL = adv[Lt].view(o.E, agent.leader_step).mean(dim=1)
                en = (adv[Ft].view(o.E, m) ** 2).mean(dim=1)
                cc = lambda a, b: float(np.corrcoef(a.numpy(), b.numpy())[0, 1])
                res['decomp'] = dict(chk_gbar=float((gbar + v).norm() / (v.norm() + 1e-300)), chk_blin=float((b_lin - b_e).norm() / (b_e.norm() + 1e-300)),
                                     self_mean=float(self_e.mean()), self_std=float(self_e.std()), cross_mean=float(cross_e.mean()), cross_std=float(cross_e.std()),
                                     gd_self_over_gd=float(g_self.norm() / (gd.norm() + 1e-300)), gd_cross_over_gd=float(g_cross.norm() / (gd.norm() + 1e-300)),
                                     cos_self_gd=cosn(g_self, gd), cos_cross_gd=cosn(g_cross, gd), cos_self_gs=cosn(g_self, gs), cos_cross_gs=cosn(g_cross, gs),
                                     corr_self_advL=cc(self_e, aL), corr_cross_advL=cc(cross_e, aL), corr_self_energy=cc(self_e, en), corr_b_energy=cc(b_lin, en),
                                     corr_energy_advL=cc(en, aL), share_self_in_sum=float(self_e.sum() / (b_lin.sum() + 1e-300)))
        agent.policy_net = pol0
        return res

    def stats_pair(gs, gd):
        ratio = float(gd.norm() / (gs.norm() + 1e-12))
        lim = cfg.gradient_ratio_limit
        scale = (lim / (ratio + 1e-12)) if (lim > 0.0 and ratio > lim) else 1.0
        gc = gs - scale * gd
        return dict(gs_norm=float(gs.norm()), gd_norm=float(gd.norm()), ratio_raw=ratio, cos_gs_gd=cosn(gs, gd), scale=float(scale),
                    gs_attr=float(gs[attr_mask].norm()), gs_skel=float(gs[~attr_mask].norm()),
                    gd_attr=float(gd[attr_mask].norm()), gd_skel=float(gd[~attr_mask].norm()),
                    cos_attr=cosn(gs[attr_mask], gd[attr_mask]),
                    cos_dir_vs_corrected=cosn(-gs, -gc), cos_dir_vs_corrected_attr=cosn(-gs[attr_mask], -gc[attr_mask]),
                    gc_norm=float(gc.norm()), gradclip_factor=float(min(1.0, cfg.max_grad_norm / (gc.norm() + 1e-12))))

    # ---------- "ideal" (all follower steps) version of the indirect term ----------
    def full_terms(o):
        pol = agent.policy_net
        t1, t2 = get_th(pol)
        pe = mk_policy_exec(pol)
        F_all = np.concatenate(o.F, axis=0)
        Ft = torch.from_numpy(F_all)
        # v_full = grad_theta2 of  -mean(ratio*adv) over all follower steps (ratio=1 at this point, so clipping is inactive)
        gacc = [torch.zeros_like(p) for p in t2]
        n_all = len(F_all)
        for i in range(0, n_all, 4096):
            idx = F_all[i:i + 4096]
            lp = pol.get_log_prob(index_select_list(o.states, idx), index_select_list(o.actions, idx))
            it = torch.from_numpy(idx)
            loss = -(torch.exp(lp - o.fixlog[it]) * o.adv[it]).sum() / n_all
            gg = torch.autograd.grad(loss, t2, allow_unused=True)
            for a, g in zip(gacc, gg):
                if g is not None:
                    a.add_(g)
        v = flat(gacc).detach()
        sub = np.random.choice(F_all, size=min(2048, n_all), replace=False)
        Fsub = index_select_list(o.states, sub)
        fvp = lambda x: fisher_vector_product_selfkl(policy_exec=pe, theta2_params=t2, states=Fsub, v_flat=x, damping=agent.damping, J_c=None, fisher_correct=False)
        s, it_, rel = cg_inst(fvp, v)
        Fv = (fvp(v) - lam * v).detach()
        st = -s; u = st / (st.norm() + 1e-300); h = 1e-4
        ep_of = np.concatenate([np.full(len(f), e) for e, f in enumerate(o.F)])
        cnt = torch.from_numpy(np.array([len(f) for f in o.F], dtype=np.float64))
        def fbar():
            acc = torch.zeros(o.E)
            with torch.no_grad():
                for i in range(0, n_all, 8192):
                    idx = F_all[i:i + 8192]; it = torch.from_numpy(idx)
                    lp = pol.get_log_prob(index_select_list(o.states, idx), index_select_list(o.actions, idx))
                    val = (torch.exp(lp - o.fixlog[it]) * o.adv[it]).reshape(-1)
                    acc.index_add_(0, torch.from_numpy(ep_of[i:i + 8192]), val)
            return acc / cnt
        def shift(c):
            off = 0
            with torch.no_grad():
                for p in t2:
                    n = p.numel(); p.add_(c * u[off:off + n].view_as(p)); off += n
        shift(+h); fp = fbar(); shift(-2 * h); fm = fbar(); shift(+h)
        b_e = float(st.norm()) * (fp - fm) / (2 * h)
        L_idx = np.concatenate(o.L, axis=0); Lt = torch.from_numpy(L_idx)
        lp = pol.get_log_prob(index_select_list(o.states, L_idx), index_select_list(o.actions, L_idx))
        lbar = torch.exp(lp - o.fixlog[Lt]).view(o.E, agent.leader_step).mean(dim=1)
        gd_full = flat([g if g is not None else torch.zeros_like(p) for g, p in zip(torch.autograd.grad((lbar * b_e).mean(), t1, allow_unused=True), t1)]).detach()
        return dict(gd=gd_full, b_e=b_e.numpy().copy(), v_norm=float(v.norm()), s_norm=float(s.norm()), cos_sv=cosn(s, v),
                    s_over_v_lam=float(s.norm() / (v.norm() / lam + 1e-300)), rayleigh_v=float(torch.dot(v, Fv) / torch.dot(v, v)), cg_iters=it_, cg_rel=rel)

    # ---------- projection onto the design variables (attr mean output) ----------
    o0 = Ds[0]
    ref_state = [o0.states[int(o0.L[0][1])]]            # first attribute-edit state (same initial design in every episode)
    attr_states_all = [o0.states[int(i)] for ep in o0.L for i in ep[1:]]

    def attr_mean(pol, sts):
        with torch.no_grad():
            out = pol(sts)
            return out[1].loc.clone()                    # [nodes, attr_dim]

    def dmu(direction, sts):
        """first-order change of the attr mean when theta1 moves along `direction` (finite difference)."""
        pol = agent.policy_net
        t1, _ = get_th(pol)
        h = 1e-3 / (float(direction.norm()) + 1e-300)
        def shift(c):
            off = 0
            with torch.no_grad():
                for p in t1:
                    n = p.numel(); p.add_(c * direction[off:off + n].view_as(p)); off += n
        shift(+h); mp = attr_mean(pol, sts); shift(-2 * h); mm = attr_mean(pol, sts); shift(+h)
        return (mp - mm) / (2 * h)

    result = dict(run=RUN, ckpt=CKPT, nb=NB, nep=NEP, fit=fit, cfg=dict(lamda=lam, ratio_limit=cfg.gradient_ratio_limit, m=m, leader_step=agent.leader_step),
                  E=[o.E for o in Ds], batches=[], cross=dict())
    if not HIDE:
        result['ep_ret_mean'] = [float(e.mean()) for e in ep_rets]
        result['ep_ret_std'] = [float(e.std()) for e in ep_rets]

    to_train(agent.policy_net, agent.value_net)
    GS, GD0, GDF, GDW = [], [], [], []
    t1 = time.time()
    for bi, o in enumerate(Ds):
        rb = dict(windows=[])
        gdw = []
        for k in range(NWIN):
            r = leader_terms(o, k, diag=True, decomp=(k == 0 and bi < NDEC))
            sp = stats_pair(r['gs'], r['gd'])
            if 'decomp' in r:
                dc = r.pop('decomp'); rb['decomp'] = dc
                P(f'batch {bi} win0 DECOMP: checks gbar={dc["chk_gbar"]:.1e} blin={dc["chk_blin"]:.1e} | self mean={dc["self_mean"]:+.4g} std={dc["self_std"]:.4g} | cross mean={dc["cross_mean"]:+.4g} std={dc["cross_std"]:.4g} '
                  f'| share of sum(b) from self={dc["share_self_in_sum"]:.3f} | |gd_self|/|gd|={dc["gd_self_over_gd"]:.3f} cos(gd_self,gd)={dc["cos_self_gd"]:+.3f} |gd_cross|/|gd|={dc["gd_cross_over_gd"]:.3f} cos(gd_cross,gd)={dc["cos_cross_gd"]:+.3f} '
                  f'| cos with gs: self={dc["cos_self_gs"]:+.3f} cross={dc["cos_cross_gs"]:+.3f} | corr with leader adv: self={dc["corr_self_advL"]:+.3f} cross={dc["corr_cross_advL"]:+.3f} '
                  f'| corr(self, mean A_F^2)={dc["corr_self_energy"]:+.3f} corr(b, mean A_F^2)={dc["corr_b_energy"]:+.3f} corr(mean A_F^2, leader adv)={dc["corr_energy_advL"]:+.3f}')
            be, al = r.pop('b_e'), r.pop('advL_ep')
            sp.update({kk: vv for kk, vv in r.items() if kk not in ('gs', 'gd')})
            sp.update(b_mean=float(be.mean()), b_std=float(be.std()), advL_ep_std=float(al.std()), corr_b_advL=float(np.corrcoef(be, al)[0, 1]))
            rb['windows'].append(sp)
            gdw.append(r['gd'])
            if k == 0:
                GS.append(r['gs']); GD0.append(r['gd']); b0 = be
                # sensitivity: leader advantage without the value baseline
                r2_ = leader_terms(o, 0, adv=o.adv_nobase, diag=False)
                rb['nobase'] = stats_pair(r2_['gs'], r2_['gd'])
                rb['nobase']['cos_gs_with_base'] = cosn(r2_['gs'], r['gs'])
        ft = full_terms(o)
        gdf = ft.pop('gd'); bef = ft.pop('b_e')
        rb['full'] = dict(ft)
        rb['full'].update(stats_pair(GS[-1], gdf))
        rb['full'].update(b_mean=float(bef.mean()), b_std=float(bef.std()), corr_b_win0=float(np.corrcoef(bef, b0)[0, 1]),
                          cos_gdfull_gdwin=[cosn(gdf, g) for g in gdw])
        rb['cos_between_windows'] = [cosn(gdw[i], gdw[j]) for i in range(len(gdw)) for j in range(i + 1, len(gdw))]
        GDF.append(gdf); GDW.append(gdw)
        result['batches'].append(rb)
        w0 = rb['windows'][0]
        P(f'batch {bi} win0: |gs|={w0["gs_norm"]:.4g} |gd|={w0["gd_norm"]:.4g} ratio_raw={w0["ratio_raw"]:.3f} cos(gs,gd)={w0["cos_gs_gd"]:+.3f} scale={w0["scale"]:.3f} '
          f'cos(dir,corrected)={w0["cos_dir_vs_corrected"]:.3f} | attr: |gs|={w0["gs_attr"]:.4g} |gd|={w0["gd_attr"]:.4g} skel: |gs|={w0["gs_skel"]:.4g} |gd|={w0["gd_skel"]:.4g} '
          f'| |v|={w0["v_norm"]:.4g} cos(s,v)={w0["cos_sv"]:.4f} |s|/(|v|/lam)={w0["s_over_v_lam"]:.4f} rayleigh_v={w0["rayleigh_v"]:.4g} cg={w0["cg_iters"]}it rel={w0["cg_rel"]:.2e} '
          f'| FD check relerr={w0["fd_relerr"]:.2e} cos={w0["fd_cos"]:.6f} | b_e mean={w0["b_mean"]:+.4g} std={w0["b_std"]:.4g} corr(b,advL)={w0["corr_b_advL"]:+.3f} gradclip={w0["gradclip_factor"]:.3f}')
        P(f'batch {bi} windows ratio_raw={[round(w["ratio_raw"],3) for w in rb["windows"]]} cos={[round(w["cos_gs_gd"],3) for w in rb["windows"]]} cos_between_windows={[round(c,3) for c in rb["cos_between_windows"]]}')
        f_ = rb['full']
        P(f'batch {bi} full-follower version: ratio_raw={f_["ratio_raw"]:.4f} cos(gs,gd_full)={f_["cos_gs_gd"]:+.3f} |v|={f_["v_norm"]:.4g} cos(s,v)={f_["cos_sv"]:.4f} |s|/(|v|/lam)={f_["s_over_v_lam"]:.4f} '
          f'rayleigh_v={f_["rayleigh_v"]:.4g} cg={f_["cg_iters"]} | cos(gd_full, gd_win_k)={[round(c,3) for c in f_["cos_gdfull_gdwin"]]} corr(b_full,b_win0)={f_["corr_b_win0"]:+.3f} b_full std={f_["b_std"]:.4g}')
        P(f'batch {bi} no-baseline leader adv: ratio_raw={rb["nobase"]["ratio_raw"]:.3f} cos={rb["nobase"]["cos_gs_gd"]:+.3f} cos(gs_nobase,gs)={rb["nobase"]["cos_gs_with_base"]:.3f}')
    P(f'initial-point analysis {time.time() - t1:.0f}s')

    # cross-batch consistency (independent batches of NEP episodes at the same parameters)
    nb = len(Ds)
    pairs = [(i, j) for i in range(nb) for j in range(i + 1, nb)]
    result['cross'] = dict(cos_gs=[cosn(GS[i], GS[j]) for i, j in pairs], cos_gd=[cosn(GD0[i], GD0[j]) for i, j in pairs],
                           cos_gd_full=[cosn(GDF[i], GDF[j]) for i, j in pairs],
                           cos_gs_attr=[cosn(GS[i][attr_mask], GS[j][attr_mask]) for i, j in pairs],
                           cos_gd_attr=[cosn(GD0[i][attr_mask], GD0[j][attr_mask]) for i, j in pairs])
    gs_p = torch.stack(GS).mean(0); gd_p = torch.stack(GD0).mean(0); gdf_p = torch.stack(GDF).mean(0)
    gdw_p = torch.stack([g for gw in GDW for g in gw]).mean(0)
    result['pooled'] = dict(win0=stats_pair(gs_p, gd_p), allwin=stats_pair(gs_p, gdw_p), full=stats_pair(gs_p, gdf_p),
                            gs_snr=float(gs_p.norm() / np.mean([g.norm() for g in GS])), gd_snr=float(gd_p.norm() / np.mean([g.norm() for g in GD0])),
                            gdf_snr=float(gdf_p.norm() / np.mean([g.norm() for g in GDF])))
    P(f'cross-batch cos: gs={np.round(result["cross"]["cos_gs"],3).tolist()} gd(win0)={np.round(result["cross"]["cos_gd"],3).tolist()} gd_full={np.round(result["cross"]["cos_gd_full"],3).tolist()}')
    P(f'cross-batch cos (attr params only): gs={np.round(result["cross"]["cos_gs_attr"],3).tolist()} gd={np.round(result["cross"]["cos_gd_attr"],3).tolist()}')
    pp = result['pooled']
    P(f'pooled over {nb} batches: win0 ratio_raw={pp["win0"]["ratio_raw"]:.3f} cos={pp["win0"]["cos_gs_gd"]:+.3f} | all windows ratio_raw={pp["allwin"]["ratio_raw"]:.3f} cos={pp["allwin"]["cos_gs_gd"]:+.3f} '
      f'| full ratio_raw={pp["full"]["ratio_raw"]:.4f} cos={pp["full"]["cos_gs_gd"]:+.3f} | |mean|/mean|.|: gs={pp["gs_snr"]:.3f} gd={pp["gd_snr"]:.3f} gd_full={pp["gdf_snr"]:.3f} (1/sqrt(nb)={1/np.sqrt(nb):.3f} if pure noise)')

    # design-variable projection (per node x per design dim); update direction is -(gs - scale*gd)
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    proj = dict()
    for tag, gs_, gd_ in (('batch0_win0', GS[0], GD0[0]), ('pooled_win0', gs_p, gd_p), ('pooled_allwin', gs_p, gdw_p), ('pooled_full', gs_p, gdf_p)):
        sp = stats_pair(gs_, gd_)
        d_dir = dmu(-gs_, ref_state).numpy(); d_ind = dmu(sp['scale'] * gd_, ref_state).numpy()
        d_dir_all = dmu(-gs_, attr_states_all).numpy(); d_ind_all = dmu(sp['scale'] * gd_, attr_states_all).numpy()
        proj[tag] = dict(d_dir=d_dir.tolist(), d_ind=d_ind.tolist(), ratio=float(np.linalg.norm(d_ind) / (np.linalg.norm(d_dir) + 1e-300)),
                         cos=float((d_dir * d_ind).sum() / (np.linalg.norm(d_dir) * np.linalg.norm(d_ind) + 1e-300)),
                         ratio_all=float(np.linalg.norm(d_ind_all) / (np.linalg.norm(d_dir_all) + 1e-300)),
                         cos_all=float((d_dir_all * d_ind_all).sum() / (np.linalg.norm(d_dir_all) * np.linalg.norm(d_ind_all) + 1e-300)))
        P(f'design projection [{tag}] (rows=bodies, cols=design dims; per unit step along the negative gradient), scale={sp["scale"]:.3f}:')
        print('  direct  :\n' + str(d_dir), flush=True)
        print('  indirect:\n' + str(d_ind), flush=True)
        P(f'  |indirect|/|direct| at the initial design state={proj[tag]["ratio"]:.3f} cos={proj[tag]["cos"]:+.3f} ; over all attr states ratio={proj[tag]["ratio_all"]:.3f} cos={proj[tag]["cos_all"]:+.3f}')
    result['proj'] = proj
    result['attr_mean_ref'] = attr_mean(agent.policy_net, ref_state).numpy().tolist()
    vdes = np.concatenate([o.v_raw.numpy()[o.st_np != 2] for o in Ds])
    P(f'design value head: raw output std/|mean| over design states = {vdes.std() / (abs(vdes.mean()) + 1e-300):.3e} (n={len(vdes)})')
    result['design_value_rel_std'] = float(vdes.std() / (abs(vdes.mean()) + 1e-300))
    sens = dict()
    for cs in CSCALES:
        agent.control_ret_norm.std[:] = abs(c_c) * cs
        agent.control_ret_norm.var[:] = (c_c * cs) ** 2
        rr = []
        for o in Ds:
            keep = o.adv
            o.adv, _ = agent.estimate_advantages(o.states, o.next_states, o.rewards, o.next_terminations, o.next_dones, o.state_types, o.next_state_types, o.c_reward)
            for k in range(2):
                r = leader_terms(o, k, diag=True)
                sp = stats_pair(r['gs'], r['gd'])
                rr.append(dict(ratio_raw=sp['ratio_raw'], cos=sp['cos_gs_gd'], corr_b_advL=float(np.corrcoef(r['b_e'], r['advL_ep'])[0, 1]), cos_sv=r['cos_sv'], s_over_v_lam=r['s_over_v_lam']))
            o.adv = keep
        sens[str(cs)] = rr
        med = lambda k: float(np.median([x[k] for x in rr]))
        P(f'SENS follower critic scale x{cs}: ratio_raw median={med("ratio_raw"):.3f} cos(gs,gd) median={med("cos"):+.3f} corr(b,advL) median={med("corr_b_advL"):+.3f} cos(s,v)={med("cos_sv"):.3f} (n={len(rr)})')
    agent.control_ret_norm.std[:] = abs(c_c); agent.control_ret_norm.var[:] = c_c ** 2
    result['sens'] = sens
    json.dump(result, open(os.path.join(OUT, 'result_init.json'), 'w'), indent=1)

    # ---------- replicate one full update (10 optimisation rounds) with and without the indirect term ----------
    if DO_UPDATE:
        t2_ = time.time()
        o = Ds[0]
        np.random.seed(SEED + 1)
        E = o.E
        Fmat = agent.ep_planner.make_Fmat(o.F)
        MB = agent.mini_batch_size
        from design_opt.models.bodygen_policy import BodyGenPolicy
        def clone_policy():
            q = BodyGenPolicy(cfg.policy_specs, agent)
            q.load_state_dict(pol0.state_dict())
            q.train(pol0.training)
            return q
        polA = clone_policy(); polB = clone_policy()
        optA = torch.optim.Adam(polA.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
        optB = torch.optim.Adam(polB.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
        F_all = np.concatenate([ep[o.st_np[ep] == 2] for ep in o.episodes])       # as get_perm_follower_design (all control steps)
        series = []
        mu0 = attr_mean(pol0, ref_state); mu0_all = attr_mean(pol0, attr_states_all)
        for k in range(agent.opt_num_epochs):
            ep_order = np.random.permutation(E)
            L_idx = np.concatenate([o.L[e] for e in ep_order], axis=0)
            F_idx = Fmat[ep_order, k * m:(k + 1) * m].ravel()
            Ls, La = index_select_list(o.states, L_idx), index_select_list(o.actions, L_idx)
            Fs, Fa = index_select_list(o.states, F_idx), index_select_list(o.actions, F_idx)
            Lt, Ft = torch.from_numpy(L_idx), torch.from_numpy(F_idx)
            # A: the real Stackelberg leader step
            agent.policy_net = polA
            surr, Ja, Jc, lg = agent.ppo_loss_stackelberg(Ls, La, o.adv[Lt].clone(), o.fixlog[Lt].clone(), Fs, Fa, o.adv[Ft].clone(), o.fixlog[Ft].clone(), flag_log=True)
            t1A, t2A = get_th(polA)
            Jd = bilevel_leader_grad_correct(J_a=Ja, J_c=Jc, theta1_list=t1A, theta2_list=t2A, policy_exec=mk_policy_exec(polA), F_states=Fs,
                                             damping=agent.damping, cg_max_iter=20, cg_tol=1e-3, verbose=False, fisher_correct=cfg.fisher_correct)
            g_surr = [g.detach() if g is not None else torch.zeros_like(p) for g, p in zip(torch.autograd.grad(surr, t1A, retain_graph=True, allow_unused=True), t1A)]
            g_delta = [g.detach() if g is not None else torch.zeros_like(p) for g, p in zip(torch.autograd.grad(Jd, t1A, retain_graph=False, allow_unused=True), t1A)]
            sp = stats_pair(flat(g_surr), flat(g_delta))
            sp.update(k=k, L_kl=lg['L_approx_kl'], F_kl=lg['F_approx_kl'], L_clip=lg['L_clip_frac'], F_clip=lg['F_clip_frac'])
            g_delta = [g * sp['scale'] for g in g_delta]
            optA.zero_grad()
            for p, a, b in zip(t1A, g_surr, g_delta):
                p.grad = (a - b).clone()
            torch.nn.utils.clip_grad_norm_(polA.parameters(), cfg.max_grad_norm)
            optA.step()
            # B: plain PPO leader step (direct term only) on the same data
            agent.policy_net = polB
            lpB = polB.get_log_prob(Ls, La)
            rB = torch.exp(lpB - o.fixlog[Lt]); aB = o.adv[Lt]
            surrB = -torch.min(rB * aB, torch.clamp(rB, 1.0 - agent.clip_epsilon, 1.0 + agent.clip_epsilon) * aB).mean()
            t1B, _ = get_th(polB)
            gB = torch.autograd.grad(surrB, t1B, allow_unused=True)
            optB.zero_grad()
            for p, g in zip(t1B, gB):
                p.grad = (g.detach() if g is not None else torch.zeros_like(p)).clone()
            torch.nn.utils.clip_grad_norm_(polB.parameters(), cfg.max_grad_norm)
            optB.step()
            # follower: normal PPO update (policy only; value updates do not affect the quantities measured here)
            agent.policy_net = polA
            perm = np.random.permutation(F_all)
            for i in range(int(np.floor(len(perm) / MB))):
                idx = perm[i * MB:(i + 1) * MB]; it = torch.from_numpy(idx)
                loss = agent.ppo_loss(index_select_list(o.states, idx), index_select_list(o.actions, idx), o.adv[it].clone(), o.fixlog[it].clone())
                optA.zero_grad()
                pm.selective_backward(loss, polA, update_list=['execution'])
                torch.nn.utils.clip_grad_norm_(polA.parameters(), cfg.max_grad_norm)
                optA.step()
            dA = (attr_mean(polA, ref_state) - mu0); dB = (attr_mean(polB, ref_state) - mu0)
            sp.update(dmuA=float(dA.norm()), dmuB=float(dB.norm()), cos_dmu=float((dA * dB).sum() / (dA.norm() * dB.norm() + 1e-300)), diff_over_B=float((dA - dB).norm() / (dB.norm() + 1e-300)))
            series.append(sp)
            P(f'update k={k}: ratio_raw={sp["ratio_raw"]:.3f} cos(gs,gd)={sp["cos_gs_gd"]:+.3f} scale={sp["scale"]:.3f} gradclip={sp["gradclip_factor"]:.3f} L_kl={sp["L_kl"]:.2e} F_kl={sp["F_kl"]:.2e} F_clip={sp["F_clip"]:.3f} '
              f'| cumulative attr-mean change: with={sp["dmuA"]:.4g} without={sp["dmuB"]:.4g} cos={sp["cos_dmu"]:+.3f} |diff|/|without|={sp["diff_over_B"]:.3f}  ({time.time() - t2_:.0f}s)')
        agent.policy_net = pol0
        dA = (attr_mean(polA, ref_state) - mu0).numpy(); dB = (attr_mean(polB, ref_state) - mu0).numpy()
        dA_all = (attr_mean(polA, attr_states_all) - mu0_all).numpy(); dB_all = (attr_mean(polB, attr_states_all) - mu0_all).numpy()
        P('one-update change of the attr mean at the initial design state (rows=bodies, cols=design dims):')
        print('  with indirect term   :\n' + str(dA), flush=True)
        print('  direct term only     :\n' + str(dB), flush=True)
        upd = dict(series=series, dA=dA.tolist(), dB=dB.tolist(),
                   cos=float((dA * dB).sum() / (np.linalg.norm(dA) * np.linalg.norm(dB) + 1e-300)),
                   diff_over_B=float(np.linalg.norm(dA - dB) / (np.linalg.norm(dB) + 1e-300)),
                   cos_all=float((dA_all * dB_all).sum() / (np.linalg.norm(dA_all) * np.linalg.norm(dB_all) + 1e-300)),
                   diff_over_B_all=float(np.linalg.norm(dA_all - dB_all) / (np.linalg.norm(dB_all) + 1e-300)),
                   dlogstd_A=(polA.attr_action_log_std - pol0.attr_action_log_std).detach().numpy().tolist(),
                   dlogstd_B=(polB.attr_action_log_std - pol0.attr_action_log_std).detach().numpy().tolist())
        P(f'one update: cos(with, without)={upd["cos"]:+.3f} |with-without|/|without|={upd["diff_over_B"]:.3f} ; over all attr states cos={upd["cos_all"]:+.3f} rel diff={upd["diff_over_B_all"]:.3f}')
        json.dump(upd, open(os.path.join(OUT, 'result_update.json'), 'w'), indent=1)
    P('DONE')


try:
    main()
except Exception:
    traceback.print_exc()
    print('[probe] FAILED', flush=True)
finally:
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
