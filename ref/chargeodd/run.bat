@echo off
rem Runs the V44 charge-split diagnostic for one charge pair and prints both rows.
rem
rem   run.bat <positive> <negative> [energy_MeV] [events] [cut_mm] [target_material]
rem
rem e.g. run.bat pi+ pi- 200 20000 0.7 G4_BONE_COMPACT_ICRU
rem
rem Both charges get the same seed, the same geometry and the same event count, in two
rem separate processes. V44's rule: the event count is chosen from the size of the effect
rem being tested. The split under test is 2-10%; the per-event quantities tallied here are
rem means of ~5 MeV with a per-event spread of order 10%, so 20,000 events gives a standard
rem error near 0.07% and can resolve a 2% split at ~30 sigma.
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
set EXE=%~dp0..\chargeoddbuild\Release\chargeodd.exe
set P=%1
if "%P%"=="" set P=pi+
set M=%2
if "%M%"=="" set M=pi-
set EN=%3
if "%EN%"=="" set EN=200
set EV=%4
if "%EV%"=="" set EV=20000
set CUT=%5
if "%CUT%"=="" set CUT=0.7
set TGT=%6
if "%TGT%"=="" set TGT=G4_BONE_COMPACT_ICRU
set MODE=%7
if "%MODE%"=="" set MODE=emonly
echo header,particle,energy_MeV,events,cut_mm,target,seed,mode,edep_primary,edep_secondary,edep_slab,ke_in,ke_out,primary_loss,primary_path_mm,esec_born,edep_world,frac_entered,steps_primary
"%EXE%" %P% %EN% %EV% %CUT% %TGT% 12345 %MODE% || exit /b 1
"%EXE%" %M% %EN% %EV% %CUT% %TGT% 12345 %MODE% || exit /b 1
