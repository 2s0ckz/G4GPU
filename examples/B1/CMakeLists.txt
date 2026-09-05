#----------------------------------------------------------------------------
# Setup the project
#
# The same shape as Geant4's examples/basic/B1/CMakeLists.txt, with CUDA in place of a
# find_package(Geant4): there is no installed library to find, so the project compiles the
# transport engine and the visualisation manager from src/ alongside its own sources.
#
#   cmake -S . -B build -DG4GPU_DIR=D:/g4gpu
#   cmake --build build --config Release
#
# On a repeat build, or in the pipeline, point it at the objects build_engine.bat and
# build_vis.bat already produced and it links those instead of recompiling them:
#
#   cmake -S . -B build -DG4GPU_USE_PREBUILT_ENGINE=ON
#
# transport_run.cu takes about four minutes to compile on its own, which is the whole reason
# that option exists.
cmake_minimum_required(VERSION 3.18)
project(B1 LANGUAGES CXX CUDA)

#----------------------------------------------------------------------------
# Where the g4gpu sources are. Defaults to three directories up, which is where they are
# when this example sits in examples/B1 of a checkout.
#
set(G4GPU_DIR "${PROJECT_SOURCE_DIR}/../.." CACHE PATH "Root of the g4gpu source tree")
get_filename_component(G4GPU_DIR "${G4GPU_DIR}" ABSOLUTE)
if(NOT EXISTS "${G4GPU_DIR}/src/host/transport_run.cu")
  message(FATAL_ERROR
    "G4GPU_DIR=${G4GPU_DIR} does not contain src/host/transport_run.cu. "
    "Pass -DG4GPU_DIR=<root of the g4gpu tree>.")
endif()

option(G4GPU_USE_PREBUILT_ENGINE
       "Link out/transport_run.obj and out/vis_manager.obj instead of compiling them" OFF)

#----------------------------------------------------------------------------
# Compiler settings.
#
# sm_86 is this machine's GPU (RTX 3070). CMAKE_CUDA_ARCHITECTURES is the portable way to say
# it; override it on the command line for a different card.
#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${G4GPU_DIR}/src)
include_directories(${G4GPU_DIR}/src/g4)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs
#
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# The .cc files reach into device code through their headers, so they are CUDA translation
# units even though a Geant4 project would compile them as plain C++. This is what
# build.bat's -x cu does.
set_source_files_properties(${PROJECT_SOURCE_DIR}/exampleB1.cc ${sources}
                            PROPERTIES LANGUAGE CUDA)

#----------------------------------------------------------------------------
# Add the executable, and link it to the engine and the visualisation manager
#
add_executable(exampleB1 exampleB1.cc ${sources} ${headers})

if(G4GPU_USE_PREBUILT_ENGINE)
  set(_engine_obj "${G4GPU_DIR}/out/transport_run.obj")
  set(_vis_obj "${G4GPU_DIR}/out/vis_manager.obj")
  foreach(_obj ${_engine_obj} ${_vis_obj})
    if(NOT EXISTS "${_obj}")
      message(FATAL_ERROR
        "G4GPU_USE_PREBUILT_ENGINE is ON but ${_obj} does not exist. "
        "Run build_engine.bat and build_vis.bat first, or configure without the option.")
    endif()
  endforeach()
  # Marked EXTERNAL_OBJECT so CMake links them rather than trying to compile them.
  set_source_files_properties(${_engine_obj} ${_vis_obj}
                              PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  target_sources(exampleB1 PRIVATE ${_engine_obj} ${_vis_obj})
else()
  set(engine_sources
      ${G4GPU_DIR}/src/host/transport_run.cu
      ${G4GPU_DIR}/src/render/vis_manager.cu)
  target_sources(exampleB1 PRIVATE ${engine_sources})
endif()

# The visualisation manager is raw Win32 and WGL, which is why these three and no more.
if(WIN32)
  target_link_libraries(exampleB1 PRIVATE opengl32 user32 gdi32)
endif()

#----------------------------------------------------------------------------
# Copy all scripts to the build directory, i.e. the directory in which we
# build B1. This is so that we can run the executable directly because it
# relies on these scripts being in the current working directory.
#
set(EXAMPLEB1_SCRIPTS
  exampleB1.in
  init.mac
  init_vis.mac
  run1.mac
  run2.mac
  vis.mac
  )

foreach(_script ${EXAMPLEB1_SCRIPTS})
  configure_file(
    ${PROJECT_SOURCE_DIR}/${_script}
    ${PROJECT_BINARY_DIR}/${_script}
    COPYONLY
    )
endforeach()

#----------------------------------------------------------------------------
# For internal use - but has no effect if you build this example standalone
#
add_custom_target(B1 DEPENDS exampleB1)

#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
install(TARGETS exampleB1 DESTINATION bin)
