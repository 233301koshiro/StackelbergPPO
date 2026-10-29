#!/bin/bash
# 系統（Hinge_Short_Plug＋Frame_Long_Socket とその鏡）× サーボを付ける向き 4 通りで A1v 1/5 を組み、可動域を出す
cd /userdir/StackelbergPPO_ra
OUTD=/userdir/StackelbergPPO/data/axis6/family_range
mkdir -p $OUTD; rm -f $OUTD/range.csv
for pair in "Hinge_Short_Plug Frame_Long_Socket" "Hinge_Short_Socket Frame_Long_Plug"; do
  set -- $pair
  for m in connect_point:right-side connect_point:left-side connect_point2:bottom-side2 connect_point2:left-side2; do
    tag="$1+$2+$m"
    PYTHONPATH=/choreonoid_ws/install/lib/choreonoid-2.3/python SCALE=1.0 TIP_LAYERS=0 TIP_HINGE=Hinge_Long_Socket \
      FAM_HINGE=$1 FAM_FRAME=$2 FAM_MOUNT=$m OUT=$OUTD/${tag//:/_}.json timeout -k 5 900 python3 scripts/build_arm3.py > $OUTD/${tag//:/_}.build.log 2>&1
    echo "$tag build exit=$?"; grep "^リンク" $OUTD/${tag//:/_}.build.log
    [ -f $OUTD/${tag//:/_}.json ] && IN=$OUTD/${tag//:/_}.json OUT=$OUTD/range.csv LABEL="$tag" timeout 900 python3 scripts/joint_range_a1v.py 2>&1 | grep "^関節"
  done
done
