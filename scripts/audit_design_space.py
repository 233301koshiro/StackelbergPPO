#!/usr/bin/env python3
"""**設計空間と物理のパラメータを全部並べ、根拠の有無を突き合わせる**（実験系譜 9-149）。

⛔⛔ **「7 個中 4 個が根拠なし」→「7 個すべてに根拠」→「8 個中 6 個」と 3 回数え直した末に、
最初の数えが対象を取りこぼしていたと分かった**（9-148・9-149）。
⭐ **思い出して数えるのをやめ、設定ファイルと XML を機械的に全部並べる。**

判定軸は 2 つ。

1. ⭐ **元論文の移動ロボット cfg と同じ値か**（cheetah・walker・swimmer…）
   → 一致するほど「このタスクのために選ばれていない」疑いが強い
2. ⭐ **Choreonoid の実走に届くか**（変換器が読むか）
   → 届かない属性は、書いても根拠づけにならない（9-143 で実際に空振りした）

    python3 scripts/audit_design_space.py
    python3 scripts/audit_design_space.py --cfg pusher_tripo_v3_real --xml e2e_a1v_real
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOCO = ['cheetah', 'walker-regular', 'swimmer', 'glider-regular',
        'stepper', 'crawler', 'terraincrosser']

# ⭐ 根拠がついたもの（系譜の段を書く）。⚠️ ここを埋めるのが仕事である
JUSTIFIED = {
    'density': '9-126（アルミ A6061・肉厚 2 mm の薄肉パイプ）',
    'robot.actuator_params.gear.lb': '9-129（必要トルクの 1.25 倍）',
    'robot.actuator_params.gear.ub': '9-129（UR5e の関節トルク）',
    'joint_range': '9-134（実機の曲げ軸より保守側・第1層の判定に無影響）',
    'robot.body_params.offset.rel_frac': '9-150（下限＝反転しない幾何／上限＝UR5e 150 N·m）',
}
# ⛔ 変換器が読まない＝書いても実走に届かない（9-143 で実測）
UNREACHABLE = ['solref', 'solimp', 'friction', 'margin', 'condim', 'integrator', 'gravity']
# 学習アルゴリズムの HP（設計空間ではない）
RL_HP = ('gamma', 'tau', 'obs_specs', 'policy_specs', 'agent_specs', 'value_specs',
         'lr', 'eps', 'clip', 'num_optim_epoch', 'mini_batch', 'save_model_interval',
         'max_epoch_num', 'seed', 'enable_wandb', 'cfg', 'xml_name', 'num_threads')


def flat(d, pre=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f'{pre}.{k}' if pre else k))
    elif isinstance(d, list):
        out[pre] = str(d)
    else:
        out[pre] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='pusher_tripo_v3')
    ap.add_argument('--xml', default='e2e_a1v')
    a = ap.parse_args()

    main_cfg = flat(yaml.safe_load(open(ROOT / 'design_opt' / 'cfg' / f'{a.cfg}.yml')))
    loco = {}
    for n in LOCO:
        p = ROOT / 'design_opt' / 'cfg' / f'{n}.yml'
        if p.exists():
            loco[n] = flat(yaml.safe_load(open(p)))

    print(f'=== 設計空間（{a.cfg}.yml）')
    print(f'{"パラメータ":<40} {"値":<16} {"移動ロボと一致":>14}  根拠')
    n_no = 0
    for k, v in sorted(main_cfg.items()):
        if any(k.startswith(h) for h in RL_HP):
            continue
        hits = sum(1 for d in loco.values() if k in d and d[k] == v)
        j = JUSTIFIED.get(k, '')
        mark = f'{hits}/{len(loco)}' + (' ⛔' if hits >= 3 else '')
        if not j:
            n_no += 1
        print(f'{k:<40} {str(v)[:16]:<16} {mark:>14}  {j or "⛔ **未点検**"}')

    print(f'\n=== 物理（{a.xml}.xml）')
    conv = (ROOT / 'khrylib/rl/envs/common/mujoco_env_choreonoid.py').read_text(encoding='utf-8')
    t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{a.xml}.xml').getroot()
    attrs = {}
    for tag in ['option', 'default/geom', 'default/joint']:
        el = t.find(tag)
        if el is not None:
            attrs.update(el.attrib)
    print(f'{"属性":<18} {"値":<20} {"実走に届く":>12}  根拠')
    for k, v in sorted(attrs.items()):
        if k in ('rgba',):
            continue
        reach = ('⛔ **届かない**' if k in UNREACHABLE else
                 ('⭐ 届く' if f"'{k}'" in conv or f'"{k}"' in conv else '⚠️ 不明'))
        j = JUSTIFIED.get(k, '')
        if not j:
            n_no += 1
        print(f'{k:<18} {str(v)[:20]:<20} {reach:>12}  {j or "⛔ **未点検**"}')

    print(f'\n⛔ **根拠が書かれていない項目: {n_no} 個**')
    print('⭐ **JUSTIFIED に系譜の段を書いて減らすのが仕事である。**')
    print('⚠️ **「移動ロボと一致」が多いほど、このタスクのために選ばれていない疑いが強い**（9-149）。')


if __name__ == '__main__':
    main()
