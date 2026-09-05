#!/bin/sh
# Prints the path to the Geant4 source tree that MATCHES THE ORACLE, or fails.
#
# Why this exists: there are three Geant4 trees on this machine and I transcribed physics out
# of the wrong one for most of a day.
#
#   D:/g4gpu/reference-geant4                        a shallow clone of 11.5.0
#   D:/Documents/Geant4/Windows/geant4-v10.7.3-*     an older install, no source
#   D:/Documents/Geant4/Windows/geant4-v11.1.1       the source matching the oracle
#
# ref/oracle/*.csv is produced by linking the 11.1.1 *install*. A transcription taken from a
# different version is measured against numbers it was never meant to reproduce: it passes if
# the two versions happen to agree and fails mysteriously if they do not. Both happened.
#
# So the tree is not a constant here - it is checked. G4VERSION_NUMBER is read from the source
# tree and from the install's own header, and they must be equal. The install is the authority
# because the install is what the oracle links.
#
# Usage:  G4SRC=$(sh tools/g4src.sh) || exit 1
#         G4SRC=$(sh tools/g4src.sh /some/other/tree) || exit 1
set -e

SRC="${1:-D:/Documents/Geant4/Windows/geant4-v11.1.1}"
INSTALL="${G4GPU_ORACLE_INSTALL:-D:/Documents/Geant4/Windows/geant4-v11.1.1-install}"

src_hh="$SRC/source/global/management/include/G4Version.hh"
inst_hh="$INSTALL/include/Geant4/G4Version.hh"

ver() { grep -h "define G4VERSION_NUMBER" "$1" 2>/dev/null | awk '{print $NF}' | head -1; }

if [ ! -f "$src_hh" ]; then
  echo "g4src.sh: no G4Version.hh under $SRC - not a Geant4 source tree" >&2
  exit 1
fi
if [ ! -f "$inst_hh" ]; then
  echo "g4src.sh: no G4Version.hh under $INSTALL - cannot tell which version the oracle is" >&2
  echo "          set G4GPU_ORACLE_INSTALL to the install ref/dump links against" >&2
  exit 1
fi

sv=$(ver "$src_hh")
iv=$(ver "$inst_hh")
if [ -z "$sv" ] || [ -z "$iv" ]; then
  echo "g4src.sh: could not read G4VERSION_NUMBER (source='$sv' install='$iv')" >&2
  exit 1
fi
if [ "$sv" != "$iv" ]; then
  echo "g4src.sh: VERSION MISMATCH - refusing to transcribe from the wrong Geant4." >&2
  echo "          source  $SRC     G4VERSION_NUMBER $sv" >&2
  echo "          oracle  $INSTALL   G4VERSION_NUMBER $iv" >&2
  echo "          The oracle CSVs come from the install. Use the tree that matches it." >&2
  exit 1
fi

echo "$SRC"
