@echo off
rem Runs ref/b1ngp on a macro:  ref\b1ngp\run.bat <macro> <0|1>
rem
rem   0  G4NeutronGeneralProcess as 11.1.1 ships it - reads what ref/B1build's exampleB1 reads
rem   1  each step charged at the mean free path it was taken with (docs/RISK.md V219)
rem
rem SERIAL, forced, as ref/b1neutron/run.bat and for its reason: the install is multithreaded and
rem a task run manager's workers each print their own share of the dose.
set G4FORCE_RUN_MANAGER_TYPE=Serial
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
set "B1NGP_PATCH=%~2"
"%~dp0..\b1ngpbuild\Release\b1ngp.exe" %1
