# The gamma general-process counter behind docs/RISK.md V207 and V208: which sub-process of
# QBBC's G4GammaGeneralProcess a photon's first interaction is, in the running Geant4 11.1.1.
# One source file, ours; see gammagp.cc.
cmake_minimum_required(VERSION 3.16)
project(gammagp)

find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

add_executable(gammagp gammagp.cc)
target_link_libraries(gammagp ${Geant4_LIBRARIES})
