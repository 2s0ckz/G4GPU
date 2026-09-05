@echo off
rem Runs the oracle dumper against Geant4 11.1.1's own datasets, into out\oracle_tmp.
rem
rem Every G4*DATA variable is set explicitly rather than left to the install's geant4.bat,
rem which does not override one that is already set - and on this machine G4LEDATA points at a
rem Geant4 10.7.3 install whose G4EMLOW 7.13 lacks the photoelectric data 11.1.1 requires. The
rem dumper then aborts inside G4LivermorePhotoElectricModel. That is the same stale-dataset
rem trap that once left two physics processes silently disabled in this project's own runs;
rem see docs/RISK.md S2.
rem
rem It writes into out\oracle_tmp rather than over ref\oracle, so that regenerating one CSV
rem cannot quietly change the reference every other test compares against. Copy across the
rem files you meant to change, and only those.
setlocal
set G4DATA=D:\Documents\Geant4\Windows\geant4-v11.1.1-install\share\Geant4\data
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
set PATH=D:\Documents\Geant4\Windows\geant4-v11.1.1-install\bin;%PATH%
if not exist "D:\g4gpu\out\oracle_tmp" mkdir "D:\g4gpu\out\oracle_tmp"
cd /d "D:\g4gpu\out\oracle_tmp"
"D:\g4gpu\ref\dumpbuild\Release\g4dump.exe"
exit /b %errorlevel%
