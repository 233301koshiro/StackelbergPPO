#!/usr/bin/env python3
"""Compute simple comparisons between runs using per-epoch exec_R_eps series.

Usage: python3 analysis/stat_tests.py
"""
import re
import os
import numpy as np
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_exec_series(run_dir):
    path = os.path.join(ROOT, run_dir, 'log', 'log_train.txt')
    if not os.path.exists(path):
        return []
    series = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m_epoch = re.search(r'\]\s+(\d+)\s+', line)
            if not m_epoch:
                continue
            m_exec = re.search(r'exec_R_eps\s+([-\d\.eE]+)', line)
            if m_exec:
                try:
                    val = float(m_exec.group(1))
                    series.append(val)
                except:
                    continue
    return series

def summarize(series):
    if not series:
        return {'n':0,'mean':None,'std':None,'best':None}
    a = np.array(series)
    return {'n':len(a),'mean':float(a.mean()),'std':float(a.std(ddof=1)) if len(a)>1 else 0.0,'best':float(a.max())}

def bootstrap_diff(a,b,nsamples=2000,seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    na = len(a); nb = len(b)
    if na==0 or nb==0:
        return None
    for _ in range(nsamples):
        sa = rng.choice(a,size=na,replace=True)
        sb = rng.choice(b,size=nb,replace=True)
        diffs.append(sa.mean()-sb.mean())
    diffs = np.array(diffs)
    lo,hi = np.percentile(diffs,[2.5,97.5])
    p_approx = float((diffs<=0).mean()) if diffs.mean()>0 else float((diffs>=0).mean())
    return {'mean_diff':float(diffs.mean()),'ci':[float(lo),float(hi)],'p_approx':p_approx}

def compare(runA, runB, labelA=None, labelB=None):
    a = parse_exec_series(runA)
    b = parse_exec_series(runB)
    sa = summarize(a); sb = summarize(b)
    res = {'runA':runA,'runB':runB,'sumA':sa,'sumB':sb,'bs':None}
    if sa['n']>0 and sb['n']>0:
        bs = bootstrap_diff(np.array(a), np.array(b), nsamples=4000, seed=42)
        res['bs']=bs
    return res

def print_result(r):
    print('---')
    print(f"Compare: {r['runA']}  vs  {r['runB']}")
    a=r['sumA']; b=r['sumB']
    print(f"  {r['runA']}: n={a['n']} mean={a['mean']:.3f} std={a['std']:.3f} best={a['best']}")
    print(f"  {r['runB']}: n={b['n']} mean={b['mean']:.3f} std={b['std']:.3f} best={b['best']}")
    if r['bs']:
        print(f"  mean_diff={r['bs']['mean_diff']:.4f} 95%CI=[{r['bs']['ci'][0]:.4f},{r['bs']['ci'][1]:.4f}] approx_p={r['bs']['p_approx']:.3f}")
    else:
        print('  insufficient data for bootstrap')

def main():
    pairs = [
        ('single_run/rrbot_arm_reach_L1','single_run/rrbot_arm_reach_L1_s2'),
        ('single_run/rrbot_arm_pusher_L2','single_run/rrbot_arm_pusher_L2_s2'),
        ('single_run/rrbot_arm_targetpusher_TP1','single_run/rrbot_arm_targetpusher_TP2_s2'),
    ]
    results = []
    for a,b in pairs:
        r = compare(a,b)
        print_result(r)
        results.append(r)
    # write csv summary
    out = os.path.join(ROOT,'docs','stat_test_summary.csv')
    with open(out,'w',newline='',encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['runA','runB','nA','meanA','stdA','bestA','nB','meanB','stdB','bestB','mean_diff','ci_lo','ci_hi','p_approx'])
        for r in results:
            a=r['sumA']; b=r['sumB']; bs=r.get('bs')
            row=[r['runA'],r['runB'],a['n'],a['mean'],a['std'],a['best'],b['n'],b['mean'],b['std'],b['best']]
            if bs:
                row += [bs['mean_diff'], bs['ci'][0], bs['ci'][1], bs['p_approx']]
            else:
                row += ['','','','']
            w.writerow(row)
    print('\nWrote', out)

if __name__=='__main__':
    main()
