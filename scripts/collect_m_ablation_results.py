#!/usr/bin/env python3
"""Collect M-ablation morph & metric from runs and write CSV."""
import os, csv, pickle, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'm_ablation_results.csv')

RUN_PATTERN = [
    'single_run/rrbot_arm_pusher_M1_lenonly*',
    'single_run/rrbot_arm_pusher_M2b_gearonly*',
]

rows = []

for pattern in RUN_PATTERN:
    for run in sorted(glob.glob(os.path.join(ROOT, pattern))):
        run_name = os.path.basename(run)
        bestp = os.path.join(run, 'models', 'best.p')
        epoch = ''
        best = ''
        total_len = ''
        gear0 = ''
        gear1 = ''
        if os.path.exists(bestp):
            try:
                with open(bestp, 'rb') as f:
                    ck = pickle.load(f)
                epoch = ck.get('epoch','')
                best = ck.get('best_rewards','')
            except Exception:
                pass
        # try to read comparison dump
        comp = os.path.join(ROOT, 'single_run', 'comparison', f'{run_name}_morph_final.txt')
        if os.path.exists(comp):
            try:
                txt = open(comp,'r',encoding='utf-8',errors='ignore').read()
                import re
                m = re.search(r'total arm len \(m\)\s+(\d+\.\d+)', txt)
                if m:
                    total_len = m.group(1)
                # extract gear row
                for line in txt.splitlines():
                    if line.strip().startswith('body[1] gear'):
                        parts = line.split()
                        # parts example: ['body[1]', 'gear', '400', '181']
                        nums = [p for p in parts if p.isdigit()]
                        if len(nums)>=1:
                            gear0 = nums[0]
                        if len(nums)>=2:
                            gear1 = nums[1]
            except Exception:
                pass
        rows.append({'run':run_name,'epoch':epoch,'best':best,'total_len':total_len,'gear0':gear0,'gear1':gear1})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['run','epoch','best','total_len','gear0','gear1'])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print('Wrote', OUT)
