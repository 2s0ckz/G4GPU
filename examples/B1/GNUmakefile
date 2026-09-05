# --------------------------------------------------------------
# GNUmakefile for example B1, g4gpu edition.
#
# Geant4's own B1 GNUmakefile is four lines and an `include
# $(G4INSTALL)/config/binmake.gmk` - it delegates everything to the Geant4
# install's build system. There is no such thing here, so this makefile does
# the work itself, and it runs the same two nvcc invocations as build.bat.
#
#   make            build exampleB1
#   make run        build and run the 10000-event macro
#   make clean      remove the objects and the executable
#   make visclean   remove the pictures a vis macro leaves behind
#
# Two deliberate differences from Geant4's:
#
#  - Objects go to $(OBJDIR) = ../../out/B1, not next to the sources. A project
#    that imports Wavefront .obj meshes should not have a pile of files called
#    *.obj sitting beside its source code meaning something else entirely.
#
#  - The engine and viewer objects are *prebuilt* by the scripts at the
#    repository root and linked in. Compiling transport_run.cu here would add
#    twelve minutes to a build of seven small files.
#
# The engine and viewer objects are built by build_engine.bat and build_vis.bat,
# which are batch files: this makefile invokes them through cmd.exe rather than
# reimplementing their freshness check. On a shell without cmd.exe, run
# ../../build_all.bat build once and then `make` here.
# --------------------------------------------------------------

name     := exampleB1
G4GPUDIR := ../..
OBJDIR   := $(G4GPUDIR)/out/B1

# nvcc needs -x cu because these .cc files include headers that reach down into
# device code. -c and not -dc: they launch no kernels, and mixing relocatable
# device code with the engine object (compiled without it) fails at device link.
NVCC     := nvcc
NVFLAGS  := -std=c++17 -O2 -arch=sm_86
INCLUDES := -I include -I $(G4GPUDIR)/src -I $(G4GPUDIR)/src/g4
LIBS     := -lopengl32 -luser32 -lgdi32

ENGINE_OBJ := $(G4GPUDIR)/out/transport_run.obj
VIS_OBJ    := $(G4GPUDIR)/out/vis_manager.obj

# exampleB1.cc sits at the top, as in Geant4's layout; everything else is in src/.
SOURCES := exampleB1.cc $(wildcard src/*.cc)
OBJECTS := $(patsubst %.cc,$(OBJDIR)/%.obj,$(notdir $(SOURCES)))

.PHONY: all run clean visclean prebuilt

all: $(name).exe

# The prebuilt objects come from the root scripts, which skip the compile when
# the object is newer than every source it depends on (tools/freshness.ps1).
prebuilt:
	cmd //c "$(subst /,\,$(G4GPUDIR))\build_engine.bat"
	cmd //c "$(subst /,\,$(G4GPUDIR))\build_vis.bat"

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/exampleB1.obj: exampleB1.cc | $(OBJDIR)
	$(NVCC) $(NVFLAGS) $(INCLUDES) -x cu -c -o $@ $<

$(OBJDIR)/%.obj: src/%.cc | $(OBJDIR)
	$(NVCC) $(NVFLAGS) $(INCLUDES) -x cu -c -o $@ $<

$(name).exe: prebuilt $(OBJECTS)
	$(NVCC) $(NVFLAGS) -o $@ $(OBJECTS) $(ENGINE_OBJ) $(VIS_OBJ) $(LIBS) \
	  -Xlinker /IMPLIB:$(OBJDIR)/$(name).lib

run: $(name).exe
	./$(name).exe exampleB1.in

clean:
	rm -f $(OBJECTS) $(OBJDIR)/$(name).lib $(OBJDIR)/$(name).exp $(name).exe $(name).lib $(name).exp

visclean:
	rm -f g4gpu_*.png B1_*.png
