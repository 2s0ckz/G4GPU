@echo off
rem Runs the real Geant4 11.1.1 build of example B1 on the macro given as %1.
rem
rem This install is a MULTITHREADED build, and G4RunManagerFactory hands B1 a
rem G4TaskRunManager that uses every core. Two things follow, and both have caught someone:
rem
rem   * Every worker prints its own "End of Local Run" block carrying only its slice of the
rem     events. Only the master s "End of Global Run" holds the merged accumulables. A script
rem     that reads the first dose line in this output reads one thread s share.
rem   * A wall-clock number from here is a whole machine, not a CPU thread. For a timing
rem     comparison against the port, set G4FORCE_RUN_MANAGER_TYPE=Serial first -
rem     tools/compare_b1_beams.ps1 does exactly that.
rem
rem The dose is unaffected either way, which is why the stored reference in
rem examples/B1/src/RunAction.cc came from a threaded run and is still right.
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
cd /d D:\g4gpu\ref\run
D:\g4gpu\ref\B1build\Release\exampleB1.exe %1
