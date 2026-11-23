#!/bin/sh
# A nucleus fired from INSIDE the trapezoid (z = 70 mm, its middle): every MeV it carries should be
# deposited there. ion_inside.sh <Z> <A> <total MeV> <tag>
L=/d/g4gpu/out/gpu.lock
D=/d/g4gpu/.claude/worktrees/agent-aadaae635fb6ac03e/out/p27/diag
M=$D/inside_$4.mac
printf '/B1/det/material G4_WATER\n/B1/gun/z 70 mm\n/run/initialize\n/gun/particle ion\n/gun/ion %s %s\n/gun/energy %s MeV\n/run/beamOn 100000\n' $1 $2 $3 > $M
while [ -e $L ]; do sleep 30; done
echo "P27 diag $M $(date '+%F %T')" > $L
export CUDA_VISIBLE_DEVICES=1 G4GPU_HADRONIC_STAGE=final
cd /d/g4gpu/.claude/worktrees/agent-aadaae635fb6ac03e/out/p27/bin
./exampleB1.exe "$(cygpath -w $M)" > $D/inside_$4_port.log 2>&1
head -1 $L | grep -q "^P27 " && rm -f $L
echo "port: $(grep -E 'Cumulated|edep|total track-steps' $D/inside_$4_port.log | tr '\n' ' ')"
cd /d/g4gpu/.claude/worktrees/agent-aadaae635fb6ac03e
G4FORCE_RUN_MANAGER_TYPE=Serial cmd //c "ref\\b1mat\\run.bat $(cygpath -w $M)" > $D/inside_$4_g4.log 2>&1
echo "g4:   $(grep -E '^ Cumulated' $D/inside_$4_g4.log)"
