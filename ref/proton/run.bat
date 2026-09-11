@echo off
rem Runs the real-Geant4 half of the proton depth-dose comparison.
rem
rem   run.bat [events] [energy_MeV] [output.csv]
rem
rem The data-file variables are the same set ref/run/runb1.bat exports; Geant4 reads them at
rem initialisation and fails with an unhelpful message if any is missing.
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
set EV=%1
if "%EV%"=="" set EV=20000
set EN=%2
if "%EN%"=="" set EN=100
set OUT=%3
if "%OUT%"=="" set OUT=%~dp0..\..\out\g4_depth.csv
set CUT=%4
set SLAB=%5
if "%SLAB%"=="" set SLAB=0.5
if "%CUT%"=="" set CUT=0.7
rem Relative to this file, not D:\g4gpu - see the note in build.bat beside it. A worktree runs
rem the binary ITS build.bat built, from ITS proton_depth.cc.
"%~dp0..\protonbuild\Release\g4proton.exe" %EV% %EN% "%OUT%" %CUT% %SLAB%
