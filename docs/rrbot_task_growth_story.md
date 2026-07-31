# rrbot のタスク別成長比較

## 目的

同じ初期形状の rrbot を、Pusher / Reach / Target-Pusher で学習させたときに、
最終形態と成長の仕方がタスクごとに変わることを示す。

## 作成物

- 比較図: [docs/rrbot_task_growth_story.png](rrbot_task_growth_story.png)
- Pusher 動画:
  - [形態動画](../single_run/rrbot_arm_pusher_H1/videos/eval_morphology.mp4)
  - [実行動画](../single_run/rrbot_arm_pusher_H1/videos/eval_execution.mp4)
- Reach 動画:
  - [形態動画](../single_run/rrbot_arm_reach_L1/videos/eval_morphology.mp4)
  - [実行動画](../single_run/rrbot_arm_reach_L1/videos/eval_execution.mp4)
- Target-Pusher 動画:
  - [形態動画](../single_run/rrbot_arm_tp_TP1/videos/eval_morphology.mp4)
  - [実行動画](../single_run/rrbot_arm_tp_TP1/videos/eval_execution.mp4)
  - 本番 run（TP2, 1000ep, best 0.450）版もあり:
    [形態動画](../single_run/rrbot_arm_tp_TP2/videos/eval_morphology.mp4) /
    [実行動画](../single_run/rrbot_arm_tp_TP2/videos/eval_execution.mp4)

## 見方

- 3 run はいずれも rrbot 系の同一初期トポロジーを起点にしている。
- ただし、学習後の形態はタスクによって異なる。
  - Pusher: 長さとギアの強化が進み、押し切る方向に寄る。
  - Reach: 低ギア寄りで、到達精度と頑健性を優先する方向に寄る。
  - Target-Pusher: Reach と Pusher の中間的な形態に寄りつつ、行動は停止・保持を含む。

## 補足

- **2026-07-17 の注意**: 最初に生成した Target-Pusher の実行動画は2フレーム（1秒未満）で終わっていた。
  原因は学習の失敗ではなく Bug 10 の再発（動画スクリプトが転用 run の重みを読み込めず、
  乱数形態が初期接触ペナルティで即終了していた）。スクリプト修正後に TP1/TP2 とも
  201 フレームで正常再生成済み。詳細は [デバッグ戦記.md](デバッグ戦記.md) Bug 10 の再発項を参照。
- 比較図は各 run の `visualize_morph_changes.py` 出力をまとめたもの。
- 先生への説明では、「初期形状は同じだが、タスクに応じて成長先が分岐する」ことを
  主張の中心に置くと分かりやすい。
