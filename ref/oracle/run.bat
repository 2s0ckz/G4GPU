@echo off
set G4BASE=D:\Documents\Geant4\Windows\geant4-v11.1.1-install
set PATH=%G4BASE%\bin;%PATH%
set G4DATA=%G4BASE%\share\Geant4\data
set G4LEDATA=%G4DATA%\G4EMLOW8.2
set G4LEVELGAMMADATA=%G4DATA%\PhotonEvaporation5.7
set G4RADIOACTIVEDATA=%G4DATA%\RadioactiveDecay5.6
set G4PARTICLEXSDATA=%G4DATA%\G4PARTICLEXS4.0
set G4PIIDATA=%G4DATA%\G4PII1.3
set G4REALSURFACEDATA=%G4DATA%\RealSurface2.2
set G4SAIDXSDATA=%G4DATA%\G4SAIDDATA2.0
set G4ABLADATA=%G4DATA%\G4ABLA3.1
set G4INCLDATA=%G4DATA%\G4INCL1.0
set G4ENSDFSTATEDATA=%G4DATA%\G4ENSDFSTATE2.3
set G4NEUTRONHPDATA=%G4DATA%\G4NDL4.7
rem Relative to this file, so a package developed in a git worktree regenerates ITS OWN oracle
rem from ITS OWN dump program rather than D:\g4gpu's. See docs/HADRONIC_PLAN.md section 6.
cd /d "%~dp0"
"%~dp0..\dumpbuild\Release\g4dump.exe" || exit /b 1

rem P21: G4GammaGeneralProcess, asked rather than read - ref\gammagp\gammagp.cc against the same
rem install. Two files, both read by the tests (docs/RISK.md V207-V209 and P21's entries):
rem
rem   gammagp_tables.csv   the general process's own tables for B1's four materials (all but table
rem                        5, which it never builds), node by node as BuildPhysicsTable filled
rem                        them, and LogVectorValue at a set of energies.
rem                        tests\test_emextra_wiring.cu holds the port's table 9 to it.
rem   gammagp_counts.csv   which sub-process took the first interaction of FORTY MILLION photons,
rem                        at each point P21 was asked about: Rayleigh at 1.5 MeV in water and
rem                        bone, table 9 at 20, 22 and 60 MeV in both and at every other point of
rem                        docs/RISK.md V209's table, and its zero-store edge at 11.2 MeV in
rem                        water. tests\test_emextra_transport.cu runs the port's stepper at the
rem                        same points - a million photons - against these counts.
rem
rem Forty million and not one, so that Geant4's side of each comparison is not its noise: at a
rem million the probe left its conversion count at 1.5 MeV in water 2.1 sigma over its own table,
rem and its photonNuclear counts at the twelve points under their own table 9 by a mean of 1.0
rem sigma; at forty million the twelve are their table's, z -1.84 to +1.20 and a mean of -0.02
rem (docs/RISK.md V223, V224).
rem One invocation per point, each from the probe's fixed seed, so that a point's count does not
rem depend on which points were asked before it; about a minute a point. Here and not after the
rem `tables` exit below: a package regenerating the tables needs these files as much as any.
call "%~dp0..\gammagp\build.bat" || exit /b 1
if exist "%~dp0gammagp_tables.csv" del /q "%~dp0gammagp_tables.csv"
for %%M in (G4_AIR G4_WATER G4_A-150_TISSUE G4_BONE_COMPACT_ICRU) do (
  call "%~dp0..\gammagp\run.bat" %%M tables 0.2 0.3 0.5 0.8 1.0 1.2 1.5 2 2.5 3 3.9 4.5 5.4 6 7 8 10 11.2 12.5 14 17 20 21 22 23 25 30 40 60 80 99.9 >> "%~dp0gammagp_tables.csv" || exit /b 1
)
if exist "%~dp0gammagp_counts.csv" del /q "%~dp0gammagp_counts.csv"
for %%P in ("G4_WATER 40000000 1.5" "G4_WATER 40000000 11.2" "G4_WATER 40000000 20" "G4_WATER 40000000 22" "G4_WATER 40000000 60" "G4_BONE_COMPACT_ICRU 40000000 1.5" "G4_BONE_COMPACT_ICRU 40000000 20" "G4_BONE_COMPACT_ICRU 40000000 22" "G4_BONE_COMPACT_ICRU 40000000 60" "G4_WATER 40000000 99.9" "G4_BONE_COMPACT_ICRU 40000000 99.9" "G4_AIR 40000000 17" "G4_AIR 40000000 6" "G4_A-150_TISSUE 40000000 60") do (
  call "%~dp0..\gammagp\run.bat" %%~P >> "%~dp0gammagp_counts.csv" || exit /b 1
)

rem `run.bat tables` stops here: every table dump above, none of the transport run below. The
rem tables take minutes; the proton depth-dose run below takes much longer, builds under
rem D:\g4gpu\ref\protonbuild rather than under this worktree, and is not what a package
rem developing a cross section or a model needs to regenerate.
if /I "%1"=="tables" exit /b 0

rem proton_depth.csv is an oracle file too, and it is produced by a different program: a real
rem Geant4 *transport* run rather than a table dump. It is here so that regenerating the
rem oracle after a Geant4 upgrade regenerates all of it - a stale depth-dose curve compared
rem against a freshly built port is the kind of disagreement that gets blamed on the port.
rem
rem 100,000 events so the reference itself is not the statistical limit: the pipeline runs the
rem port at 6,000 against it and the reference contributes about a sixth of the noise.
rem
rem WHAT THE REFERENCE IS, since P8c: QBBC with ONLY the processes this port lacks inactivated -
rem every `*Inelastic`, `hBrems`/`hPairProd`, the gamma-/electro-/muon-nuclear processes,
rem `ionElastic`, `NeutronGeneralProc` and the three at-rest captures. `hadElastic`,
rem `CoulombScat`, `msc`, `hIoni`, `ionIoni` and `Decay` are ACTIVE on both sides. The list is
rem in ref/proton/proton_depth.cc and the run prints `/particle/process/dump` for the proton and
rem for GenericIon, so the configuration is recorded by what ran. It was G4EmStandardPhysics and
rem nothing else until P8c, which is a reference with no elastic scattering in it - and that is
rem what made build_all.bat's plateau check fail by 1.32% once P8b wired hadElastic.
rem
rem RELATIVE, not D:\g4gpu: these two lines were the last place a worktree regenerated MAIN's
rem oracle from MAIN's source. See the note in ref/proton/build.bat and docs/RISK.md V45.
call "%~dp0..\proton\build.bat" || exit /b 1
call "%~dp0..\proton\run.bat" 100000 100 "%~dp0proton_depth.csv" 0.7 0.5 || exit /b 1

rem The same harness with an ELECTRON beam, added by P14c because nothing in this oracle had
rem ever asked Geant4 what a lepton above 100 MeV does to a depth-dose curve - which is the
rem question docs/RISK.md V64 turned out to be about. One source file, two builds, one
rem geometry: see the header of ref/proton/proton_depth.cc.
rem
rem 1 GeV in water is a SHOWER and not a track, so the phantom is not the proton's. X0 in water
rem is 360.8 mm, the shower maximum sits near 2 X0 and the tail runs for tens of radiation
rem lengths, so the depth is 4,000 mm in 20 mm slabs - 200 bins, the same count the proton run
rem uses - and the transverse half-width is 400 mm, which is 4.3 Moliere radii and holds the
rem 95% containment radius of 2 R_M with room. Both numbers are arguments and both are on the
rem CSV's header line, so a curve can never be read against the wrong phantom.
call "%~dp0..\proton\run.bat" 100000 1000 "%~dp0electron_depth.csv" 0.7 20 e- 4000 400 || exit /b 1
