"""**MuJoCo の XML を、学習と同じ変換（`mujoco_xml_to_body`）で Choreonoid の `.body` に書き出す**（2026-11-06、ユーザー「bodyにできる？」）。

なぜ: 学習の経路は XML を実行のたびに `.body` の YAML へ変えて読み込み、ファイルには残さない。
Choreonoid の画面で開いて形を見たり、他の道具へ渡したりするには `.body` のファイルが要る。
変換は `khrylib/rl/envs/common/mujoco_env_choreonoid.py` の `mujoco_xml_to_body()` をそのまま呼ぶので、
学習で Choreonoid に渡るものと同じ（別の変換器を書かない）。

⚠️ `.body` に入らないもの: アクチュエータ（ギア比・ctrlrange）・回転子の慣性（armature）・環境変数で入れる物理
（`CNOID_HINGE_DAMP_TORQUE` など）は、学習では env が実行中に掛ける。`.body` は形・質量・関節・当たり判定だけ。
⚠️ 変換器は `worldbody/body` しか読まない（`worldbody` 直下の geom は捨てる、系譜 9-98）。体ごとに 1 ファイル。

確かめ: 書き出した `.body` を `BodyLoader` で読み戻し、リンク数・関節数・質量の合計を XML と並べて出す（⑤-3-2）。

    USE_CHOREONOID=1 /choreonoid_ws/install/bin/choreonoid --no-window \\
        --python scripts/export_cnoid_body.py -- quad_4x2_ra quad_tripo_ra > /tmp/export_body.log 2>&1
    # → assets/choreonoid/bodies/<xml名>/<体の名前>.body
"""
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def xml_summary(xml_str):
    """XML の robot（worldbody の最初の body）の体の数・関節の数（free を除く）・`<inertial>` の質量の合計。"""
    top = ET.fromstring(xml_str).find('worldbody/body')
    bodies = list(top.iter('body'))
    joints = [j for b in bodies for j in b.findall('joint') if j.get('type', 'hinge') != 'free']
    masses = [float(i.get('mass')) for b in bodies for i in b.findall('inertial') if i.get('mass')]
    return len(bodies), len(joints), (sum(masses) if len(masses) == len(bodies) else None)


def main(names):
    from cnoid.Body import BodyLoader
    from khrylib.rl.envs.common.mujoco_env_choreonoid import mujoco_xml_to_body
    for name in names:
        src = ROOT / 'assets' / 'mujoco_envs' / f'{name}.xml'
        xml_str = src.read_text()
        body_defs = mujoco_xml_to_body(xml_str)[0]
        out_dir = ROOT / 'assets' / 'choreonoid' / 'bodies' / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (body_name, yaml_str) in enumerate(body_defs):
            path = out_dir / f'{body_name}.body'
            path.write_text(yaml_str)
            b = BodyLoader().load(str(path))
            assert b is not None, f'{path} を読み戻せない'
            links = [b.link(k) for k in range(b.numLinks)]
            mass = sum(l.m for l in links)
            root = b.rootLink
            print(f'[export] {name} → {path.relative_to(ROOT)}  リンク {b.numLinks}  関節 {b.numJoints}  '
                  f'質量 {mass:.4f} kg  根 {root.name}（{"free" if root.jointType == root.FreeJoint else "固定など"}）', flush=True)
            if i == 0:
                nb, nj, m = xml_summary(xml_str)
                ok = nb == b.numLinks and nj == b.numJoints and (m is None or abs(m - mass) < 1e-6)
                print(f'[export]   XML: 体 {nb}  関節 {nj}  質量 {m if m is None else round(m, 4)} kg'
                      f'  → {"✅ 一致" if ok else "⛔ 食い違う"}', flush=True)


if __name__ == '__main__':
    try:
        args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
        main(args or ['quad_4x2_ra', 'quad_tripo_ra'])
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
