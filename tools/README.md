# tools

Code generators and build helpers. Nothing here is compiled into the transport; each is run by
hand when its input changes, and each writes something that is committed.

| | |
|---|---|
| `g4src.sh` | **run first, always.** Resolves the Geant4 source tree and refuses one whose `G4VERSION_NUMBER` differs from the install the oracle links. Every extractor goes through it; see `docs/RISK.md` O7 for the day this cost |
| `splice.sh` | replaces lines A..B of a file with another file, refusing an empty or out-of-range bound. Exists because the obvious `grep -n` + `sed` one-liner appends the whole file when the grep finds nothing, silently (`docs/RISK.md` S6) |
| `regen_stopping.sh` | the one to run for the NIST tables. Extracts, recompiles the generator, generates the headers, checks them against Geant4's own answers |
| `extract_stopping.sh` | step 1: `G4PSTARStopping.cc`, `G4ASTARStopping.cc` and `G4NISTStoppingData.hh` -> `nist_stopping_raw.hh` |
| `gen_stopping.cc` | step 2: raw tables + the not-a-knot spline -> `src/data/nist_stopping_names.hh` and `src/data/nist_stopping.hh` |
| `check_pstar.cc` | step 3: every generated point against `ref/oracle/bragg.csv` |
| `regen_icru90.sh` | the one to run for ICRU 90. Extract, recompile, generate, then `tests/test_icru90.exe` |
| `extract_icru90.sh` | `G4ICRU90StoppingData.cc` -> `icru90_raw.hh`, eight arrays with every count asserted |
| `gen_icru90.cc` | raw tables, **rounded through float** as Geant4 stores them, + the spline -> `src/data/icru90.hh` |
| `extract_fermi.sh` | `G4IonisParamElm.cc`'s `vFermi[92]` and `lFactor[92]` -> `src/data/fermi_velocity.hh` |
| `extract_mott.sh` | `G4MottData.hh`'s `fMottCoef[93][5][6]` -> `mott_raw.hh`, 2790 coefficients, count asserted |
| `gen_mott.cc` | those plus the 92 target nuclear masses from `ref/oracle/mott_target.csv` -> `src/data/mott.hh` |
| `freshness.ps1` | is an object older than any source it depends on? Used by `build_engine.bat` and `build_vis.bat` |

## Which Geant4 to read from

**Run `tools/g4src.sh` and use what it prints.** There are three Geant4 trees on this machine
and one of them is the wrong version:

| path | what it is |
|---|---|
| `D:/g4gpu/reference-geant4` | a shallow clone, **11.5.0** - not the oracle's version |
| `D:/Documents/Geant4/Windows/geant4-v10.7.3-install` | an older install, no source |
| `D:/Documents/Geant4/Windows/geant4-v11.1.1` + `-install` | the source **and** install the oracle links |

`g4src.sh` reads `G4VERSION_NUMBER` out of the source tree *and* out of the install
`ref/dump/` links against, and refuses if they differ:

```
$ sh tools/g4src.sh D:/g4gpu/reference-geant4
g4src.sh: VERSION MISMATCH - refusing to transcribe from the wrong Geant4.
          source  D:/g4gpu/reference-geant4     G4VERSION_NUMBER 1150
          oracle  ...geant4-v11.1.1-install     G4VERSION_NUMBER 1111
```

Every extractor here goes through it. For most of a day the stopping tables, the spline and the
Fermi velocities were read out of 11.5.0 and checked against 11.1.1, and no measurement could
tell: those particular things are byte-identical between the two versions, so the checks passed
at 1e-6 and said nothing about provenance. It surfaced only when a *call* failed to compile.
`docs/RISK.md` O7.

## Regenerating the stopping tables

```bash
sh tools/regen_stopping.sh                       # tree resolved by tools/g4src.sh (must match the oracle)
sh tools/regen_stopping.sh /path/to/geant4-src
```

Run it in that order and not by hand, for two reasons that each cost an hour:

**The generator compiles the raw tables in.** Regenerating `nist_stopping_raw.hh` and running
the existing `gen_stopping.exe` silently emits the old tables from the new data. Step 2 of the
script is a recompile for that reason. Same family as `docs/RISK.md` S4.

**A table can be the right shape and the wrong data.** The extraction once swallowed one
material name in every ten - `nameNIST` is annotated `// 0 - 9` every tenth entry, and the
flattener turned the annotation into a C++ comment that ate the following name. The array was
still 74 long, still compiled, and gave `G4_WATER` the stopping power of `G4_TEFLON`: 54% out
at 3 keV, where the Bragg peak is. Every structural check passed. Step 3 is the only thing in
the chain that could see it, and it works by comparing *answers* - each table read through
Geant4's own spline against `G4PSTARStopping::GetElectronicDEDX` - rather than tables.
`docs/RISK.md` O6.

## Regenerating the ICRU 90 tables

```bash
sh tools/regen_icru90.sh
```

Same shape and the same trap: extract, **recompile the generator**, generate, check. The check
is `tests/test_icru90.exe`, which compares every point against `G4ICRU90StoppingData`'s own
answers.

It earned its keep immediately. Geant4 stores these stopping powers as `G4float` and widens
them, so the number it splines is `(double)(float)119.70` and not `(double)119.70`. All six
tables were 6e-8 out - uniformly, at every energy, in every material, which is the fingerprint
of a type rather than of arithmetic. The generator now rounds through `float` before emitting
*and before computing the second derivatives*, because Geant4 splines the widened floats.
The energy grids are `G4double` there and are left alone. `docs/RISK.md` O11.

## Regenerating the Mott coefficients

```bash
sh tools/extract_mott.sh                 # G4MottData.hh -> mott_raw.hh, 2790 values
g++ -std=c++17 -O2 -I . -I ../src -o gen_mott.exe gen_mott.cc
./gen_mott.exe ../src/data/mott.hh ../ref/oracle/mott_target.csv
```

No wrapper script, because this one has a second input that is not a Geant4 source file: the 92
target nuclear masses in `ref/oracle/mott_target.csv`. Geant4 gets those from
`G4NucleiProperties` - an AME12 mass table with the Cameron mass-excess formula behind it - and
what the Mott ratio needs out of all that is 92 numbers, so the 92 numbers are dumped and read.

That would be circular if nothing checked them. `tests/test_mott.cu` does, through the ratio:
its beta depends on the target mass, so a wrong mass moves the answer at `fcost = 0`.

## The oracle dumper

Not here; it is `ref/dump/`, because it links against the real Geant4.

```
ref\dump\build.bat     rebuild it (cmake, MSVC, the installed Geant4 11.1.1)
ref\dump\run.bat       run it into out\oracle_tmp
```

`run.bat` writes into `out/oracle_tmp` rather than over `ref/oracle`, so that regenerating one
CSV cannot quietly change the reference every other test compares against. Copy across the
files you meant to change, and diff the rest to confirm they did not move - they should be
byte-identical, and they are.

It also sets every `G4*DATA` variable explicitly rather than trusting the install's
`geant4.bat`, which does not override one that is already set. On this machine `G4LEDATA`
points at a Geant4 **10.7.3** install whose G4EMLOW 7.13 lacks the photoelectric data 11.1.1
requires, and the dumper aborts inside `G4LivermorePhotoElectricModel`. That is the same stale
dataset that once left two physics processes silently disabled in this project's own runs;
`docs/RISK.md` S2.
