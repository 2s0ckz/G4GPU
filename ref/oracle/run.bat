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
