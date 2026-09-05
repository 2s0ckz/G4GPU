#!/bin/sh
# Regenerates the NIST stopping tables from Geant4's sources, end to end, and checks them.
#
#   sh tools/regen_stopping.sh [<geant4 source root>]
#
# One script rather than four commands because the four have an order and a trap in it. The
# generator compiles the raw table header *in*, so regenerating the raw header and re-running
# the old generator binary produces the old tables from the new data, silently. That cost an
# hour: the extraction had dropped one material name in every ten - nameNIST is annotated
# `// 0 - 9` every tenth entry, and the flattener turned the annotation into a comment that
# swallowed the next name - the generated array was still 74 long, still compiled, and gave
# G4_WATER the stopping power of G4_TEFLON. Recompiling is step 2 here for that reason.
#
# The last step is the one that matters: check_pstar compares every point against Geant4's own
# G4PSTARStopping and G4ASTARStopping answers in ref/oracle/bragg.csv. A table that is the
# right shape and the wrong data passes every other check there is.
# The tree is resolved by tools/g4src.sh rather than named here, because naming it is exactly
# how this went wrong: for most of a day these tables were extracted from D:/g4gpu/reference-
# geant4, which is 11.5.0, while the oracle they are checked against is 11.1.1. The extracted
# arrays turned out to be byte-identical between the two versions, so the measurement passed
# and told me nothing. g4src.sh makes the next mismatch fail instead of pass.
set -e
G4="${1:-}"
G4=$(sh "$(dirname "$0")/g4src.sh" $G4) || exit 1
cd "$(dirname "$0")"

echo "1/4 extracting from $G4"
sh extract_stopping.sh "$G4" nist_stopping_raw.hh

echo "2/4 compiling the generator"
g++ -std=c++17 -O2 -I . -I ../src -o gen_stopping.exe gen_stopping.cc

echo "3/4 generating ../src/data/nist_stopping{,_names}.hh"
./gen_stopping.exe ../src/data/nist_stopping_names.hh ../src/data/nist_stopping.hh

echo "4/4 checking against Geant4's own answers"
g++ -std=c++17 -O2 -I . -I ../src -o check_pstar.exe check_pstar.cc
./check_pstar.exe
