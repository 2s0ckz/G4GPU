#!/bin/sh
# Regenerates the ICRU 90 stopping tables from Geant4's sources, end to end, and checks them.
#
#   sh tools/regen_icru90.sh [<geant4 source root>]
#
# The order is extract, **recompile**, generate, check - and the recompile is step 2 for the
# same reason it is in regen_stopping.sh: the generator compiles the raw header *in*, so
# regenerating the raw header and re-running the existing generator binary silently produces
# the old tables from the new data. See docs/RISK.md O6.
#
# The last step is the one that matters. tests/test_icru90.exe compares every point against
# G4ICRU90StoppingData's own answers in ref/oracle/icru90.csv, including the sqrt
# extrapolation below 1 keV and the saturation above the last grid point. It caught the one
# real trap in this table: Geant4 stores the stopping powers as `G4float` and widens them, so
# the number it splines is (double)(float)119.70 and not (double)119.70 - which put all six
# tables 6e-8 out, uniformly, at every energy.
set -e
G4="${1:-}"
G4=$(sh "$(dirname "$0")/g4src.sh" $G4) || exit 1
cd "$(dirname "$0")"

echo "1/4 extracting from $G4"
sh extract_icru90.sh "$G4"

echo "2/4 compiling the generator"
g++ -std=c++17 -O2 -I . -I ../src -o gen_icru90.exe gen_icru90.cc

echo "3/4 generating ../src/data/icru90.hh"
./gen_icru90.exe ../src/data/icru90.hh

echo "4/4 checking against Geant4's own answers"
cd ..
G4GPU_ORACLE="$(pwd)/ref/oracle" ./tests/test_icru90.exe
