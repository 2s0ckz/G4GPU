@echo off
rem Runs the stage-1 neutron reference on the macro given as %1.
rem
rem   ref\b1neutron\run.bat ref\b1hadron\stage1_neutron.mac
rem
rem The eleven G4*DATA variables are the same set ref/run/runb1.bat and ref/proton/run.bat
rem export; Geant4 reads them at initialisation and fails unhelpfully if any is missing.
rem
rem SERIAL, forced, for the reason ref/b1hadron/stage1_compare.ps1 gives at length: the install
rem is a multithreaded build, G4RunManagerFactory would hand B1 a G4TaskRunManager, and every
rem worker then prints its own "End of Local Run" block carrying its slice of the events. A
rem script that reads the first dose line in that output reads one thread's share.
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
rem Relative to this file, not D:\g4gpu - a worktree runs the binary ITS build.bat built.
"%~dp0..\b1neutronbuild\Release\b1neutron.exe" %1
