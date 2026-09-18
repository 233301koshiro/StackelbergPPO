#!/usr/bin/env python3
"""Tripo3D API で画像から 3D モデルを生成する（実験系譜 9-115）。

**なぜ要るか**: M2（三次元化）は Web UI 経由だったため、
**同じ画像を入れても出力が ±0.7 % 揺れる**（9-39 実測）。
⭐ **API は `model_seed` を固定でき、公式は「同じ seed なら同一」と明記している。**
これが本当なら **M2 の揺れをゼロにでき、残る揺れを M1 に帰属できる**（研究方針 段 2〜4）。

⚠️⚠️ **公式の記述を検証せずに信じない**（9-80 で「公式が同じと言っている」を根拠にして外した）。
⭐ `--verify-seed` が**同じ seed で 2 回生成し、出力が同一かを実測する。**

    export TRIPO_API_KEY=...            # ⚠️ .env に置く（gitignore 済み）
    python3 scripts/tripo_api.py --balance
    python3 scripts/tripo_api.py --image data/test/B1/sketch/B1_m1.jpeg --seed 42 --out data/api/B1
    python3 scripts/tripo_api.py --verify-seed --image ... --seed 42

⚠️ **1 生成 = 30 クレジット（約 0.30 ドル）。** 残高は `--balance` で確認できる。
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

API = 'https://api.tripo3d.ai/v2/openapi'
ROOT = Path(__file__).resolve().parent.parent


def key():
    k = os.environ.get('TRIPO_API_KEY')
    if not k:
        env = ROOT / '.env'
        if env.exists():
            for ln in env.read_text().split('\n'):
                if ln.startswith('TRIPO_API_KEY='):
                    k = ln.split('=', 1)[1].strip()
    if not k:
        raise SystemExit('⛔ TRIPO_API_KEY が無い。.env に置くこと（gitignore 済み）')
    return k


def req(path, data=None, method=None, raw=None, ctype=None):
    r = urllib.request.Request(f'{API}/{path}', method=method or ('POST' if data or raw else 'GET'))
    r.add_header('Authorization', f'Bearer {key()}')
    body = None
    if data is not None:
        r.add_header('Content-Type', 'application/json')
        body = json.dumps(data).encode()
    elif raw is not None:
        r.add_header('Content-Type', ctype)
        body = raw
    with urllib.request.urlopen(r, body, timeout=120) as f:
        return json.loads(f.read())


def balance():
    d = req('user/balance')['data']
    return d['balance'], d['frozen']


def upload(path):
    """multipart/form-data を手で組む（外部依存を足さない）。"""
    p = Path(path)
    b = '----tripo' + hashlib.md5(p.name.encode()).hexdigest()[:16]
    ext = p.suffix.lstrip('.').lower()
    ext = 'jpeg' if ext == 'jpg' else ext
    body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{p.name}"\r\n'
            f'Content-Type: image/{ext}\r\n\r\n').encode() + p.read_bytes() + f'\r\n--{b}--\r\n'.encode()
    return req('upload/sts', raw=body, ctype=f'multipart/form-data; boundary={b}')['data']['image_token'], ext


def create(token, ext, seed):
    return req('task', data={'type': 'image_to_model',
                              'file': {'type': ext, 'file_token': token},
                              'model_seed': seed, 'texture_seed': seed})['data']['task_id']


def wait(tid, quiet=False):
    while True:
        d = req(f'task/{tid}')['data']
        if not quiet:
            print(f"  [{tid[:8]}] {d['status']} {d.get('progress', 0)}%", flush=True)
        if d['status'] in ('success', 'failed', 'banned', 'cancelled', 'expired'):
            return d
        time.sleep(15)


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as f:
        dest.write_bytes(f.read())
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--balance', action='store_true')
    ap.add_argument('--image')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out')
    ap.add_argument('--verify-seed', action='store_true',
                    help='⭐ 同じ seed で 2 回生成し、出力が同一かを実測する')
    a = ap.parse_args()

    if a.balance:
        b, f = balance()
        print(f'残高 {b} クレジット（凍結 {f}）／ 1 生成 = 30 クレジット ≒ {b // 30} 回ぶん')
        return 0
    if not a.image:
        ap.error('--image か --balance を指定してください')

    tok, ext = upload(a.image)
    print(f'アップロード完了 token={tok[:8]}… ({ext})')
    n = 2 if a.verify_seed else 1
    outs = []
    for i in range(n):
        tid = create(tok, ext, a.seed)
        print(f'生成 {i+1}/{n} task={tid}')
        d = wait(tid)
        if d['status'] != 'success':
            print(f"⛔ {d['status']}"); return 1
        url = (d.get('output') or {}).get('pbr_model') or (d.get('output') or {}).get('model')
        out = Path(a.out or 'data/api') / f'seed{a.seed}_{i+1}.glb'
        fetch(url, out)
        h = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f'  → {out} ({out.stat().st_size} bytes) sha256={h[:16]}…')
        outs.append((out, h))

    if a.verify_seed:
        same = outs[0][1] == outs[1][1]
        print(f"\n{'✅ 同一' if same else '⛔ 異なる'}: 同じ seed={a.seed} の 2 回")
        print(f"  1: {outs[0][1][:32]} ({outs[0][0].stat().st_size} B)")
        print(f"  2: {outs[1][1][:32]} ({outs[1][0].stat().st_size} B)")
        if not same:
            print('⚠️ **公式の「同じ seed なら同一」は成り立たない。**研究方針 段 2〜4 の前提を見直すこと')
    b, f = balance()
    print(f'\n残高 {b}（凍結 {f}）')
    return 0


if __name__ == '__main__':
    main()
