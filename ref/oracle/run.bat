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
cd /d D:\g4gpu\ref\oracle
D:\g4gpu\ref\dumpbuild\Release\g4dump.exe

rem proton_depth.csv is an oracle file too, and it is produced by a different program: a real
rem Geant4 *transport* run rather than a table dump. It is here so that regenerating the
rem oracle after a Geant4 upgrade regenerates all of it - a stale depth-dose curve compared
rem against a freshly built port is the kind of disagreement that gets blamed on the port.
rem
rem 100,000 events so the reference itself is not the statistical limit: the pipeline runs the
rem port at 6,000 against it and the reference contributes about a sixth of the noise.
call D:\g4gpu\ref\proton\build.bat || exit /b 1
call D:\g4gpu\ref\proton\run.bat 100000 100 D:\g4gpu\ref\oracle\proton_depth.csv 0.7 0.5 || exit /b 1
