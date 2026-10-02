# Example B1 on QBBC with G4NeutronGeneralProcess's step charging as a switch - docs/RISK.md V219.
#
# The B1 sources come from the GEANT4 SOURCE TREE, as in ref/b1neutron and for the same reason:
# every Geant4 row of the B1 sweep is a build of exactly these files. What is ours is b1ngp.cc.
cmake_minimum_required(VERSION 3.16)
project(b1ngp)

find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

set(G4SRC "D:/Documents/Geant4/Windows/geant4-v11.1.1" CACHE PATH
    "Geant4 source tree whose examples/basic/B1 is compiled here")
set(B1DIR "${G4SRC}/examples/basic/B1")
if(NOT EXISTS "${B1DIR}/src/DetectorConstruction.cc")
  message(FATAL_ERROR
    "No examples/basic/B1 under G4SRC=${G4SRC}. Pass -DG4SRC=<Geant4 source tree>; "
    "tools/g4src.sh prints the one that matches the install the oracle links.")
endif()

file(GLOB B1_SOURCES "${B1DIR}/src/*.cc")
include_directories("${B1DIR}/include")

add_executable(b1ngp b1ngp.cc ${B1_SOURCES})
target_link_libraries(b1ngp ${Geant4_LIBRARIES})
