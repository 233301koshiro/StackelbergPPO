#!/usr/bin/env python3
"""
check_mesh_interference.py: capsule近似で学習したcheckpointの形態(bone_offset)を、
元のGLBメッシュ(STL)に載せ替えたときに自己干渉(リンク同士の貫通)が起きないかを、
追加学習なしで安く確認する。

方針（メッシュXMLパイプライン.md「実メッシュ衝突を実装する際の検証方針」参照）:
  capsule近似は円形断面を仮定するため、実メッシュが非対称（チョップ問題の板状ヘラ等）
  だと近似で通っていた箇所が実メッシュでは干渉する可能性がある。本格的なメッシュ物理
  シミュレーション化の前に、今のcheckpointがそのまま使い物になるかを最安コストで見る。

スコープ（v1、意図的に限定）:
  - rest姿勢（全関節角0°）での静的干渉のみをチェックする。実行フェーズの関節角推移に
    沿った動的干渉（腕を振っている最中の自己衝突）は対象外 — 次の拡張候補。
  - tripo_arm_v2/v2b/v2c系のみ対応（data/tripo_arm_colorful2/topology.jsonに
    source_meshパスがあるrun）。tripo_arm_v3・rrbotは元メッシュのlink別STLが
    保存されていないため未対応。
  - python-fcl/rtreeが未インストールの環境でも動くよう、trimeshのCollisionManagerは
    使わず、リンクの凸包(scipy ConvexHull)のハーフスペース方程式で
    「相手の頂点が自分の内部にあるか」を判定する簡易チェック。凸包近似なので、
    非凸な凹み同士がすり抜ける干渉は検出できない（保守的ではなく楽観的な近似）。

使い方:
  EVAL_RESTORE_DIR=single_run/tripo_v2c_pusher EVAL_CHECKPOINT=best \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
      --no-window --python scripts/check_mesh_interference.py
"""
import os
import sys
import json

sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'

import numpy as np
import trimesh
import yaml
import torch
from omegaconf import OmegaConf
from scipy.spatial import ConvexHull

from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
from design_opt.utils.tools import set_global_seed

TOPOLOGY_PATH = 'data/tripo_arm_colorful2/topology.json'
INTERFERENCE_MARGIN = 1e-4  # [m] このマージン以上めり込んでいたら干渉とみなす


def load_topology_mesh_map(topology_path: str) -> dict:
    """body名(mesh_to_params側の名前) -> (STLパス, 元bone_offset長さ) を返す。"""
    topo = json.load(open(topology_path))
    out = {}
    for b in topo['bodies']:
        offset = np.asarray(b['bone_offset'], dtype=float)
        out[b['name']] = dict(
            mesh_path=b['geom']['source_mesh'],
            rest_length=float(np.linalg.norm(offset)),
        )
    return out


# tripo_arm_v2/v2b/v2c の choreonoid body 名 ("1","11","111") と
# topology.json 側の名前(upper_arm/forearm/hand)の対応。
# run_tripo_pipeline.sh の LINK_NAMES 順（base upper_arm forearm hand）に一致。
BODY_NAME_TO_TOPO_NAME = {'1': 'upper_arm', '11': 'forearm', '111': 'hand'}


def convex_hull_contains(hull_equations: np.ndarray, points: np.ndarray) -> np.ndarray:
    """点群のうち、凸包(ハーフスペース方程式 A x + b <= 0)の内側にあるものを返す真偽配列。"""
    A = hull_equations[:, :-1]
    b = hull_equations[:, -1]
    return np.all(points @ A.T + b <= INTERFERENCE_MARGIN, axis=1)


def check_pair(name_a, verts_a, name_b, verts_b):
    hull_a = ConvexHull(verts_a)
    hull_b = ConvexHull(verts_b)
    a_in_b = convex_hull_contains(hull_b.equations, verts_a)
    b_in_a = convex_hull_contains(hull_a.equations, verts_b)
    n_pen = int(a_in_b.sum() + b_in_a.sum())
    if n_pen > 0:
        print(f'[interference] {name_a} <-> {name_b}: '
              f'{int(a_in_b.sum())}/{len(verts_a)} 頂点({name_a})が{name_b}内部、'
              f'{int(b_in_a.sum())}/{len(verts_b)} 頂点({name_b})が{name_a}内部', flush=True)
        return True
    return False


def main():
    restore_dir = os.environ.get('EVAL_RESTORE_DIR', '')
    if not restore_dir:
        print('[check_mesh_interference] ERROR: EVAL_RESTORE_DIR を設定してください', flush=True)
        os._exit(1)
    epoch = os.environ.get('EVAL_CHECKPOINT', 'best')

    FLAGS = OmegaConf.create(yaml.safe_load(open(f'{restore_dir}/.hydra/config.yaml')))
    flags_dict = OmegaConf.to_container(FLAGS, resolve=True)
    xml_name = flags_dict.get('xml_name', '')
    if not xml_name.startswith('tripo_arm_v2'):
        print(f'[check_mesh_interference] ERROR: xml_name={xml_name} は未対応。'
              f'tripo_arm_v2/v2b/v2cのみ対応（{TOPOLOGY_PATH}のmesh mapping依存）', flush=True)
        os._exit(1)

    flags_dict.pop('restore_dir', None)
    FLAGS = OmegaConf.create(flags_dict)
    project_path = os.getcwd()
    cfg = Config(FLAGS, project_path, restore_dir)
    cfg.restore_dir = restore_dir
    cfg.control_prior = False
    cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    set_global_seed(cfg.seed)

    load_epoch = int(epoch) if epoch.isnumeric() else epoch
    agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                          seed=cfg.seed, num_threads=1, training=False, checkpoint=load_epoch)
    env = agent.env
    state = env.reset()
    for _ in range(cfg.skel_transform_nsteps + 2):
        if env.stage == 'execution':
            break
        sv = tensorfy([state])
        if agent.obs_norm is not None:
            sv = agent.normalize_observation(sv)
        with torch.no_grad():
            action = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        state, reward, done, _, info = env.step(action)

    topo_map = load_topology_mesh_map(TOPOLOGY_PATH)

    bodies = env.robot.bodies
    root_pos = np.zeros(3)
    world_meshes = {}  # body名 -> (world頂点, 干渉チェック対象か)
    cur_pos = root_pos.copy()
    print(f'[check_mesh_interference] restore_dir={restore_dir} epoch={epoch} xml_name={xml_name}',
          flush=True)
    for b in bodies[1:]:
        bname = b.name
        topo_name = BODY_NAME_TO_TOPO_NAME.get(bname)
        if topo_name is None or topo_name not in topo_map:
            print(f'[check_mesh_interference] WARNING: body "{bname}" のmesh mappingがなくスキップ',
                  flush=True)
            continue
        offset = np.asarray(b.bone_offset, dtype=float)
        length = float(np.linalg.norm(offset))
        rest_length = topo_map[topo_name]['rest_length']
        scale_x = length / rest_length if rest_length > 1e-9 else 1.0
        angle = np.arctan2(offset[1], offset[0])

        mesh = trimesh.load(topo_map[topo_name]['mesh_path'], force='mesh')
        verts = mesh.vertices.copy()
        # 元メッシュはtopology.json規約でローカル+X方向に伸びている前提でX軸のみ
        # bone_offset長さ比でスケール（capsule近似と同じく太さは固定・長さだけ可変）。
        verts[:, 0] *= scale_x
        # 学習後のoffset方向(atan2)へZ軸まわりに回転（v2系は全関節Z軸のため2D回転で足りる）。
        c, s = np.cos(angle), np.sin(angle)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        verts = verts @ Rz.T
        verts += cur_pos
        world_meshes[bname] = verts
        cur_pos = cur_pos + offset
        print(f'  body {bname} ({topo_name}): scale_x={scale_x:.3f} angle={np.degrees(angle):.1f}deg '
              f'world_origin={cur_pos}', flush=True)

    # 隣接しないリンク同士のペアだけをチェック（隣接ペアは関節で必ず接するため除外）
    names = list(world_meshes.keys())
    any_interference = False
    for i in range(len(names)):
        for j in range(i + 2, len(names)):  # i, i+1は隣接なのでスキップ
            hit = check_pair(names[i], world_meshes[names[i]], names[j], world_meshes[names[j]])
            any_interference = any_interference or hit

    if any_interference:
        print('[check_mesh_interference] 結果: 干渉あり（rest姿勢）。実メッシュではこの形態は成立しない可能性', flush=True)
    else:
        print('[check_mesh_interference] 結果: rest姿勢での自己干渉なし', flush=True)

    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
