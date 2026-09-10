// Self-registration for oracle dumps, so a package adds one WITHOUT editing g4dump.cc.
//
// g4dump.cc was one main() of 1500 lines, and every package that wanted a Geant4 number in
// ref/oracle/ appended to it. With several packages in flight at once that is a merge conflict
// on every one of them, in the one file that has to compile and link against Geant4 to be
// tested. So: a dump lives in its own file, ref/dump/dump_<package>.cc, defines a function and
// registers it with the macro below at file scope. CMakeLists.txt globs dump_*.cc with
// CONFIGURE_DEPENDS, and main() runs whatever registered after the run manager is initialised
// and one event has been run - so every physics table exists when a dump asks for one.
//
// Nothing shared is touched when a dump is added. That is the whole point of the file.
#pragma once
#include <vector>

class G4Material;

/// What a dump gets. The run manager has been initialised with QBBC and one event has been
/// run, so cross sections and dE/dx tables exist. `materials` are the ones the dump program's
/// detector builds, in a fixed order - the materials every existing oracle CSV is keyed by.
struct DumpContext {
  const std::vector<G4Material*>& materials;
};

using DumpFn = void (*)(const DumpContext&);

struct DumpEntry {
  const char* name;    ///< the package, for the "wrote ..." line
  const char* files;   ///< the CSVs it writes, for the same line
  DumpFn fn;
};

/// Function-local static, so registration from another translation unit's static
/// initialiser cannot run before the vector exists.
std::vector<DumpEntry>& dump_registry();

struct DumpRegistrar {
  DumpRegistrar(const char* name, const char* files, DumpFn fn);
};

/// At file scope in dump_<package>.cc:
///
///   static void dump_hadronic_xs(const DumpContext& ctx) { ... }
///   G4GPU_REGISTER_DUMP("hadronic_xs", "hadronic_xs.csv", dump_hadronic_xs);
#define G4GPU_REGISTER_DUMP(name, files, fn) \
  static const DumpRegistrar g4gpu_dump_registrar_##fn(name, files, fn)
