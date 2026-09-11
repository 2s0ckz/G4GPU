# The stage-1 neutron reference: Geant4's own example B1, with one line of ours.
#
# The sources come from the GEANT4 SOURCE TREE and not from this repository, deliberately:
# every other row of ref/b1hadron/stage1_README.md was measured with
# ref/B1build/Release/exampleB1.exe, which is a build of exactly these files, so compiling a
# copy here would make the neutron row a comparison against a different B1. What is ours is
# b1neutron.cc - the main, with `SetEnableNeutronGeneralProcess(false)` in it and no
# visualisation manager. See that file for why the line cannot be a macro command.
#
# G4SRC defaults to the tree tools/g4src.sh names, and g4src.sh is what checks that it matches
# the install the oracle links. Override on the command line for a different one.
cmake_minimum_required(VERSION 3.16)
project(b1neutron)

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

add_executable(b1neutron b1neutron.cc ${B1_SOURCES})
target_link_libraries(b1neutron ${Geant4_LIBRARIES})
