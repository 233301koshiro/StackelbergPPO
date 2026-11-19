import json, sys, numpy as np, os
def med(x): return float(np.median(x))
def rng(x): return f'{np.median(x):.3g} [{np.min(x):.3g}, {np.max(x):.3g}]'
for d in sys.argv[1:]:
    r = json.load(open(os.path.join(d, 'result_init.json')))
    W = [w for b in r['batches'] for w in b['windows']]
    print('=====', os.path.basename(d), 'E=', r['E'], 'fit', {k: round(v, 3) for k, v in r['fit'].items()}, ('ep_ret ' + str(np.round(r['ep_ret_mean'], 1).tolist())) if 'ep_ret_mean' in r else '')
    g = lambda k: [w[k] for w in W]
    print(' windows n=%d  ratio_raw %s | frac(ratio>1) %.2f | cos(gs,gd) %s | cos(dir,corrected) %s' % (len(W), rng(g('ratio_raw')), np.mean(np.array(g('ratio_raw')) > 1), rng(g('cos_gs_gd')), rng(g('cos_dir_vs_corrected'))))
    print('  attr-only: cos(gs,gd) %s cos(dir,corr) %s | |gs_skel|/|gs_attr| %s  |gd_skel|/|gd_attr| %s' % (rng(g('cos_attr')), rng(g('cos_dir_vs_corrected_attr')), rng(np.array(g('gs_skel')) / np.array(g('gs_attr'))), rng(np.array(g('gd_skel')) / np.array(g('gd_attr')))))
    print('  |gs| %s  gradclip %s | |v| %s cos(s,v) %s |s|/(|v|/lam) %s rayleigh_v %s cg_iters %s cg_rel %s' % (rng(g('gs_norm')), rng(g('gradclip_factor')), rng(g('v_norm')), rng(g('cos_sv')), rng(g('s_over_v_lam')), rng(g('rayleigh_v')), rng(g('cg_iters')), rng(g('cg_rel'))))
    print('  FD check relerr max %.2e | b_e std %s (advL per-episode std %s) corr(b,advL) %s' % (max(g('fd_relerr')), rng(g('b_std')), rng(g('advL_ep_std')), rng(g('corr_b_advL'))))
    print('  cos between follower windows (same episodes): %s' % rng([c for b in r['batches'] for c in b['cos_between_windows']]))
    F = [b['full'] for b in r['batches']]
    print('  full-follower: ratio_raw %s cos(gs,gd_full) %s | cos(gd_full, gd_window) %s | corr(b_full,b_win0) %s | cos(s,v) %s |s|/(|v|/lam) %s |v| %s' % (rng([f['ratio_raw'] for f in F]), rng([f['cos_gs_gd'] for f in F]), rng([c for f in F for c in f['cos_gdfull_gdwin']]), rng([f['corr_b_win0'] for f in F]), rng([f['cos_sv'] for f in F]), rng([f['s_over_v_lam'] for f in F]), rng([f['v_norm'] for f in F])))
    N = [b['nobase'] for b in r['batches']]
    print('  no-baseline: ratio_raw %s cos %s cos(gs_nobase,gs) %s' % (rng([n['ratio_raw'] for n in N]), rng([n['cos_gs_gd'] for n in N]), rng([n['cos_gs_with_base'] for n in N])))
    c = r['cross']
    print('  cross-batch cos: gs %s | gd(win0) %s | gd_full %s | attr-only gs %s gd %s' % (rng(c['cos_gs']), rng(c['cos_gd']), rng(c['cos_gd_full']), rng(c['cos_gs_attr']), rng(c['cos_gd_attr'])))
    p = r['pooled']
    print('  pooled: win0 ratio %.3f cos %+.3f | allwin ratio %.3f cos %+.3f | full ratio %.3f cos %+.3f | |mean|/mean|.| gs %.3f gd %.3f gdfull %.3f (noise floor %.3f)' % (p['win0']['ratio_raw'], p['win0']['cos_gs_gd'], p['allwin']['ratio_raw'], p['allwin']['cos_gs_gd'], p['full']['ratio_raw'], p['full']['cos_gs_gd'], p['gs_snr'], p['gd_snr'], p['gdf_snr'], 1 / np.sqrt(r['nb'])))
    for tag in ('batch0_win0', 'pooled_allwin', 'pooled_full'):
        q = r['proj'][tag]
        print('  proj[%s]: |ind|/|dir| %.3f cos %+.3f (all attr states: %.3f, %+.3f)' % (tag, q['ratio'], q['cos'], q['ratio_all'], q['cos_all']))
    q = r['proj']['pooled_allwin']
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    print('  pooled_allwin direct  :', np.array(q['d_dir']).round(2).tolist())
    print('  pooled_allwin indirect:', np.array(q['d_ind']).round(2).tolist())
    up = os.path.join(d, 'result_update.json')
    if os.path.exists(up):
        u = json.load(open(up))
        s = u['series']
        print('  UPDATE k: ratio_raw', [round(x['ratio_raw'], 2) for x in s])
        print('            cos(gs,gd)', [round(x['cos_gs_gd'], 2) for x in s])
        print('            scale', [round(x['scale'], 2) for x in s], 'L_kl', ['%.1e' % x['L_kl'] for x in s][-1], 'F_kl', ['%.1e' % x['F_kl'] for x in s][-1], 'F_clip', round(s[-1]['F_clip'], 3), 'L_clip', round(s[-1]['L_clip'], 3))
        print('            cum cos(with,without)', [round(x['cos_dmu'], 2) for x in s], ' |diff|/|without|', [round(x['diff_over_B'], 2) for x in s])
        print('   one update: cos %.3f reldiff %.3f | all attr states cos %.3f reldiff %.3f' % (u['cos'], u['diff_over_B'], u['cos_all'], u['diff_over_B_all']))
        print('   dA (with)   :', np.array(u['dA']).round(4).tolist())
        print('   dB (without):', np.array(u['dB']).round(4).tolist())
