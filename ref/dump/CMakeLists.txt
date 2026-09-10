cmake_minimum_required(VERSION 3.16)
project(g4dump)
find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})
# One file per package, self-registering: see dump_registry.hh. CONFIGURE_DEPENDS re-globs at
# build time, so a new dump_<package>.cc is picked up by `cmake --build` with no reconfigure.
file(GLOB DUMP_MODULES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dump_*.cc)
add_executable(g4dump g4dump.cc ${DUMP_MODULES})
target_link_libraries(g4dump ${Geant4_LIBRARIES})
