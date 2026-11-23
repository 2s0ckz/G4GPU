@echo off
rem Runs ref/b1mat on a macro:  ref\b1mat\run.bat <absolute macro path>
rem
rem The Geant4 11.1.1 side of P27's acceptance sweep (docs/RISK.md V336): Geant4's own example B1
rem on QBBC as it ships, plus /B1/det/material. The environment is ref/run/runb1.bat's, data sets
rem and all. Unlike ref/b1ion/run.bat it does NOT force the run manager: the caller chooses, as the
rem sweep does for every row -
rem
rem   G4FORCENUMBEROFTHREADS=10           the box: one worker per physical core of the i9-10850K;
rem                                       the dose is the master's "End of Global Run" block and
rem                                       the master's User=/Real= line (no G4WT prefix) times it
rem   G4FORCE_RUN_MANAGER_TYPE=Serial     one core: events / User is the per-core rate
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
"%~dp0..\b1matbuild\Release\b1mat.exe" %1
