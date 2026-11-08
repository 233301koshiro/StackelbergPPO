"""**再生した四脚の軌跡を、RobotAssembler の部品のメッシュ付きの `.body` に当てて Choreonoid の画面で動かす**（2026-11-08、ユーザー「choreonoidで結果をみたい」「部品付きで」）。

なぜ: `eval_cnoid_viewer.py` は学習と同じく XML から体を組むので、見えるのは当たり判定のカプセルだけ。しかも腕の run 専用で、
実行の毎 step で箱（cube）を読むため、箱の無い四脚では実行に移った最初の step で落ちる。
部品の見た目は `export_quad_ra_body.py`（ブランチ）が書いた `.body` にある（骨格は学習と同じ変換 `mujoco_xml_to_body`）。

やること: 物理は回さない。`record_arm_trace.py` の軌跡（毎 step の各リンクのワールドの位置 `xpos` と回転 `xmat`）から、
胴の位置・姿勢と 8 関節の角度を逆算して `.body` に入れ、順運動学で描く。⇒ **見えるのは記録した動きそのもので、物理の再現ではない。**
関節角は「関節 0 のときの親から子への回転 C」を先に求め、`C^T R_親^T R_子` を関節軸まわりの角度として読む。

確かめ（⑤-3-2）: 全話・全 step で、順運動学で出した各リンクの位置と軌跡の位置の差の最大と、関節軸以外の回転の残り（軸の取り違えで出る）を出す。
`.body` と軌跡のリンクの名前・本数も並べる。どれかが 1 mm・0.01 rad を超えたら ⛔ を出す（描いたものが記録と違う）。

    # 画面で見る（コンテナの中、/userdir/StackelbergPPO から。止めるときはウィンドウを閉じるか Ctrl+\）
    QUAD_TRACE=single_run/quad_walk_ra_s0/trace/arm_trace_wide_floor.npz DISPLAY=:1 \\
        /choreonoid_ws/install/bin/choreonoid --python scripts/play_quad_ra_trace.py
    # 確かめだけ（画面なし。差を出して終わる）
    QUAD_TRACE=... QUAD_CHECK_ONLY=1 /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/play_quad_ra_trace.py > ログ 2>&1

環境変数: QUAD_TRACE（必須）・QUAD_BODY（既定 quad_walk_ra235_rab2 の部品付き .body）・QUAD_FLOOR（既定 200 m 四方の floor_wide.body。
軌跡が内蔵の床のものなら x ≈ 5 m で床の端から落ちる動きがそのまま映る）・QUAD_EPISODES（例 `0,2`、既定 全話）・QUAD_FPS（既定 25 ＝ 実時間。
軌跡の 1 step は 0.04 s）・QUAD_FOLLOW（既定 1。カメラが胴を追う）・QUAD_CHECK_ONLY。
"""
import os
import sys
import time
import traceback

import numpy as np

MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(MAIN)


def _p(text):
    print(text, flush=True)
    if os.environ.get('QUAD_CHECK_ONLY') == '1':
        return   # ⛔ 画面なし（--no-window）でメッセージ欄に書くと 2 回目で Segmentation fault になった
    try:
        from cnoid.Base import MessageView
        MessageView.instance.putln(text)
    except Exception:
        pass


def _axis_angle(R, a):
    """R を軸 a まわりの回転として読んだ角と、軸以外の回転の残り [rad]。"""
    vee = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    q = np.arctan2(vee @ a, 0.5 * (np.trace(R) - 1.0))
    c, s = np.cos(q), np.sin(q)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    Ra = np.eye(3) + s * K + (1 - c) * K @ K
    off = np.linalg.norm(Ra.T @ R - np.eye(3))
    return q, off


try:
    from cnoid.Body import BodyLoader

    trace_path = os.environ['QUAD_TRACE']
    body_path = os.environ.get('QUAD_BODY', f'{MAIN}/assets/choreonoid/bodies/quad_walk_ra235_rab2/quad_walk_ra235_rab2.body')
    floor_path = os.environ.get('QUAD_FLOOR', f'{MAIN}/assets/choreonoid/bodies/floor_wide/floor_wide.body')
    fps = float(os.environ.get('QUAD_FPS', '25'))
    follow = os.environ.get('QUAD_FOLLOW', '1') == '1'
    check_only = os.environ.get('QUAD_CHECK_ONLY') == '1'

    z = np.load(trace_path, allow_pickle=True)
    names = [str(n) for n in z['body_names']]
    n_ep = int(z['ep_count'])
    eps = [int(e) for e in os.environ['QUAD_EPISODES'].split(',')] if os.environ.get('QUAD_EPISODES') else list(range(n_ep))
    _p(f'[quad] 軌跡: {trace_path}  {n_ep} 話（見せる: {eps}）  リンク {names}')
    _p(f'[quad] 学習時の物理: {z["launch_env"]}')

    # ---- 関節 0 のときの親から子への回転 C（胴を原点に置いて順運動学） ----
    body = BodyLoader().load(body_path)
    if body is None:
        raise RuntimeError(f'.body を読めない: {body_path}')
    links = {body.link(i).name: body.link(i) for i in range(body.numLinks)}
    _p(f'[quad] 部品付きの体: {body_path}  リンク {body.numLinks}・関節 {body.numJoints}')
    missing = [n for n in names if n not in links]
    if missing or len(names) != body.numLinks:
        raise RuntimeError(f'.body と軌跡のリンクが合わない: 軌跡に {len(names)} 本・.body に {body.numLinks} 本・.body に無い {missing}')
    root = body.rootLink
    if root.name != names[0]:
        raise RuntimeError(f'根のリンクが違う: .body {root.name} ／ 軌跡 {names[0]}')
    for n in names:
        links[n].q = 0.0
    root.setPosition(np.eye(4))
    body.calcForwardKinematics()
    R0 = {n: np.asarray(links[n].R, dtype=float).copy() for n in names}
    C = {n: R0[links[n].parent.name].T @ R0[n] for n in names if links[n].parent is not None}
    revolute = [n for n in names if links[n].isRotationalJoint()]
    _p(f'[quad] 回す関節 {len(revolute)}: {revolute}（軸 {[np.round(np.asarray(links[n].jointAxis), 3).tolist() for n in revolute]}）')

    def solve(xpos, xmat):
        """1 step 分の (胴の 4×4, {関節: 角}) と、関節軸以外の回転の残りの最大。"""
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = xmat[0], xpos[0]
        qs, off_max, off_at = {}, 0.0, None
        for n in names[1:]:
            i, ip = names.index(n), names.index(links[n].parent.name)
            Rrel = C[n].T @ xmat[ip].T @ xmat[i]
            if n in revolute:
                qs[n], off = _axis_angle(Rrel, np.asarray(links[n].jointAxis, dtype=float))
            else:
                off = np.linalg.norm(Rrel - np.eye(3))
            if off > off_max:
                off_max, off_at = off, n
        return T, qs, (off_max, off_at)

    def pose(T, qs):
        root.setPosition(T)
        for n, q in qs.items():
            links[n].q = q
        body.calcForwardKinematics()

    # ---- 確かめ: 全話・全 step で順運動学の位置と軌跡の位置を並べる ----
    frames = {}
    worst_p, worst_off, worst_at, n_off = 0.0, 0.0, None, {}
    for e in eps:
        xp, xm = np.asarray(z['ep_xpos'][e], dtype=float), np.asarray(z['ep_xmat'][e], dtype=float)
        fr = []
        for t in range(len(xp)):
            T, qs, (off, at) = solve(xp[t], xm[t])
            pose(T, qs)
            dp = max(np.linalg.norm(np.asarray(links[n].p) - xp[t, k]) for k, n in enumerate(names))
            worst_p = max(worst_p, dp)
            if off > worst_off:
                worst_off, worst_at = off, (e, t, at)
            if off >= 1e-2:
                n_off[e] = n_off.get(e, 0) + 1
            fr.append((T, qs))
        frames[e] = fr
        _p(f'[quad] 話 {e}: {len(xp)} step  胴の x {xp[0, 0, 0]:+.2f} → {xp[-1, 0, 0]:+.2f} m  y {xp[-1, 0, 1]:+.2f} m')
    ok = worst_p < 1e-3 and worst_off < 1e-2
    _p(f'[quad] {"✅" if ok else "⛔"} 順運動学と軌跡の差: 位置の最大 {worst_p * 1000:.3f} mm（1 mm 未満で可）・'
       f'関節軸以外の回転の残りの最大 {worst_off:.4f} rad（0.01 未満で可。話・step・リンク {worst_at}。0.01 以上の step の数 {n_off}）')
    if check_only:
        os._exit(0 if ok else 1)

    # ---- 画面: 床と部品付きの体を置いて、コマ送り ----
    import cnoid.IRSLUtil as IU
    from cnoid.Base import RootItem, SceneView
    from cnoid.BodyPlugin import WorldItem, BodyItem
    world = WorldItem()
    RootItem.instance.addChildItem(world)
    floor = BodyItem()
    floor.load(floor_path)
    world.addChildItem(floor)
    floor.setChecked(True)
    item = BodyItem()
    if not item.load(body_path):
        raise RuntimeError(f'BodyItem が .body を読めない: {body_path}')
    world.addChildItem(item)
    item.setChecked(True)
    body, root = item.body, item.body.rootLink
    links = {body.link(i).name: body.link(i) for i in range(body.numLinks)}
    _p(f'[quad] 床: {floor_path}')
    sw = SceneView.instance.sceneWidget
    dt = 1.0 / fps
    for e in eps:
        _p('=' * 50)
        _p(f'[quad] 第 {e} 話を再生（{len(frames[e])} step、{fps:g} fps）')
        for t, (T, qs) in enumerate(frames[e]):
            t0 = time.time()
            pose(T, qs)
            item.notifyKinematicStateChange()
            if follow:
                c = T[:3, 3]
                eye = c + np.array([-0.6, -0.9, 0.45])
                sw.setCameraPosition(eye, (c - eye) / np.linalg.norm(c - eye), np.array([0.0, 0.0, 1.0]))
            IU.processEvent()
            if t % 250 == 0 or t == len(frames[e]) - 1:
                _p(f'[quad]   step {t}  胴 x {T[0, 3]:+.2f} m・y {T[1, 3]:+.2f} m・高さ {T[2, 3]:.3f} m')
            time.sleep(max(0.0, dt - (time.time() - t0)))
    _p('[quad] 再生が終わった（ウィンドウを閉じると終わる）')
except Exception:
    traceback.print_exc()
    _p('[quad] ⛔ 例外で止まった（上の traceback）')
    if os.environ.get('QUAD_CHECK_ONLY') == '1':
        os._exit(1)
