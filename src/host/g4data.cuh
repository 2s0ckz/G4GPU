// Finds the Geant4 data files on this machine.
//
// Geant4 ships real measured data as files under share/Geant4/data - Livermore/EPICS2017
// photoelectric and Rayleigh cross sections, Seltzer-Berger bremsstrahlung DCS tables,
// nuclear level and decay data. Those are the numbers this port must read at runtime rather
// than carry copies of, and this header is the one place that decides where they live.
//
// Resolution order, per dataset:
//   1. the dataset's own Geant4 environment variable (G4LEDATA, G4LEVELGAMMADATA, ...) -
//      the same variable the real Geant4 reads, so a working Geant4 shell works here too;
//   2. <root>/<PREFIX><version>, picking the highest version present, where <root> comes from
//      G4GPU_DATA_DIR, GEANT4_DATA_DIR, or a short list of usual install locations.
//
// Note this covers data *files* only. Plenty of the empirical numbers in an EM physics list -
// Ziegler stopping coefficients, the Urban MSC coefficient grid, ICRU 73 oscillator shells,
// PSTAR/ASTAR curves - are not files at all; Geant4 compiles them into its libraries as C++
// arrays. Those are transcribed into src/data/*.cuh with the Geant4 source file and symbol
// named at the top of each, which is the same status they have in Geant4 itself.
#pragma once
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace g4gpu::host {

namespace detail {

inline std::string env_or_empty(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr && v[0] != '\0') ? std::string(v) : std::string();
}

/// Compares "8.2" style version suffixes numerically, so 8.10 sorts above 8.9.
inline bool version_less(const std::string& a, const std::string& b) {
  std::size_t ia = 0, ib = 0;
  while (ia < a.size() || ib < b.size()) {
    long va = 0, vb = 0;
    while (ia < a.size() && !std::isdigit(static_cast<unsigned char>(a[ia]))) { ++ia; }
    while (ib < b.size() && !std::isdigit(static_cast<unsigned char>(b[ib]))) { ++ib; }
    while (ia < a.size() && std::isdigit(static_cast<unsigned char>(a[ia]))) {
      va = va * 10 + (a[ia++] - '0');
    }
    while (ib < b.size() && std::isdigit(static_cast<unsigned char>(b[ib]))) {
      vb = vb * 10 + (b[ib++] - '0');
    }
    if (va != vb) { return va < vb; }
    if (ia >= a.size() && ib >= b.size()) { break; }
  }
  return false;
}

}  // namespace detail

/// Every directory that might hold G4* dataset folders, most-preferred first.
///
/// All of them are searched, rather than committing to the first one that exists. On a
/// machine with two Geant4 installs, G4LEDATA can name an old G4EMLOW whose parent contains
/// nothing usable; stopping there would report "not found" while a complete dataset sits in
/// the next candidate.
inline const std::vector<std::string>& g4_data_roots() {
  static const std::vector<std::string> roots = [] {
    namespace fs = std::filesystem;
    std::vector<std::string> c;
    for (const char* v : {"G4GPU_DATA_DIR", "GEANT4_DATA_DIR"}) {
      const std::string s = detail::env_or_empty(v);
      if (!s.empty()) { c.push_back(s); }
    }
    // A set G4LEDATA points *into* a dataset; its parent is a root.
    const std::string ledata = detail::env_or_empty("G4LEDATA");
    if (!ledata.empty()) { c.push_back(fs::path(ledata).parent_path().string()); }
    c.push_back("D:/Documents/Geant4/Windows/geant4-v11.1.1-install/share/Geant4/data");
    c.push_back("/usr/share/Geant4/data");
    c.push_back("/usr/local/share/Geant4/data");

    std::error_code ec;
    std::vector<std::string> out;
    for (const std::string& s : c) {
      if (fs::is_directory(s, ec)) { out.push_back(s); }
    }
    return out;
  }();
  return roots;
}

/// Locates one dataset. `env_var` is the variable real Geant4 reads for it; `prefix` is the
/// folder name without its version suffix. Returns an empty string if it cannot be found.
///
/// `must_contain`, when given, is a path relative to the dataset that has to exist for the
/// candidate to be accepted. It matters because a shell configured for an older Geant4 will
/// have G4LEDATA pointing at, say, G4EMLOW7.13, whose epics2017 directory holds only pair
/// data - no photoelectric, no Rayleigh. Without the check that directory is accepted and
/// every file under it then fails to open one at a time.
inline std::string g4_dataset_dir(const char* env_var, const char* prefix,
                                  const char* must_contain = nullptr) {
  namespace fs = std::filesystem;
  std::error_code ec;
  auto usable = [&](const fs::path& p) {
    if (!fs::is_directory(p, ec)) { return false; }
    return must_contain == nullptr || fs::exists(p / must_contain, ec);
  };

  const std::string from_env = detail::env_or_empty(env_var);
  if (!from_env.empty() && usable(from_env)) { return from_env; }

  const std::string pre(prefix);
  for (const std::string& root : g4_data_roots()) {
    std::string best;
    for (const auto& e : fs::directory_iterator(root, ec)) {
      if (!e.is_directory(ec)) { continue; }
      const std::string name = e.path().filename().string();
      if (name.rfind(pre, 0) != 0) { continue; }
      if (!usable(e.path())) { continue; }
      if (best.empty() || detail::version_less(best, name)) { best = name; }
    }
    if (!best.empty()) { return (fs::path(root) / best).string(); }
  }
  return {};
}

/// A subdirectory of G4EMLOW, e.g. "epics2017/phot" or "brem_SB". `sentinel` is a file that
/// has to be inside it, and it is checked per subdirectory rather than once for the whole
/// dataset: G4EMLOW 7.13 has an epics2017 directory holding only pair-production data, so
/// testing for epics2017 alone accepts a G4EMLOW with no photoelectric or Rayleigh in it.
inline std::string g4emlow_subdir(const char* sub, const char* sentinel) {
  const std::string rel = std::string(sub) + "/" + sentinel;
  const std::string base = g4_dataset_dir("G4LEDATA", "G4EMLOW", rel.c_str());
  if (base.empty()) { return {}; }
  return (std::filesystem::path(base) / sub).string();
}

inline const std::string& g4ensdfstate_dir() {
  static const std::string d = g4_dataset_dir("G4ENSDFSTATEDATA", "G4ENSDFSTATE");
  return d;
}

inline const std::string& g4photon_evaporation_dir() {
  static const std::string d = g4_dataset_dir("G4LEVELGAMMADATA", "PhotonEvaporation");
  return d;
}

inline const std::string& g4radioactive_decay_dir() {
  static const std::string d = g4_dataset_dir("G4RADIOACTIVEDATA", "RadioactiveDecay");
  return d;
}

// Deliberately here and not at the top of the file: the include block above is what every
// other package appending to this file will also touch, and one line added to it is one merge
// conflict per package (docs/HADRONIC_PLAN.md section 6 rule 5). A file-scope include is legal
// anywhere, and this keeps the G4PARTICLEXS addition contiguous.
#include <cstdio>

/// G4PARTICLEXS - the hadronic cross-section data set: per-element and per-isotope inelastic,
/// elastic and capture cross sections for n, p, d, t, He3, alpha and gamma.
///
/// THIS ONE CANNOT TRUST ITS ENVIRONMENT VARIABLE, AND THE REASON IS MEASURED
///
/// The other resolvers above take the dataset's own Geant4 variable first, on the argument
/// that "a working Geant4 shell works here too". On this machine that argument fails for
/// G4PARTICLEXS. The shell has
///
///     G4PARTICLEXSDATA = ...\geant4-v10.7.3-install\...\data\G4PARTICLEXS3.1.1
///
/// - every dataset variable points at the 10.7.3 install - while ref/oracle/run.bat sets
/// G4PARTICLEXS**4.0** from the 11.1.1 install, which is the version the oracle CSVs are
/// produced from. G4EMLOW got away with it: `must_contain` rejects 7.13 because its
/// epics2017 directory has no photoelectric data in it, so the search falls through to the
/// 11.1.1 root. G4PARTICLEXS3.1.1 has `neutron/el1` and every other file this port opens, so
/// no sentinel discriminates it. A test run outside run.bat's environment would read 3.1.1,
/// agree with itself, and disagree with the oracle - and the disagreement would look like a
/// transcription error in whichever class was being written.
///
/// So this resolver takes the HIGHEST version it can find anywhere, rather than the first
/// usable candidate, and says so when the environment variable is not it. That is the same
/// discipline tools/g4src.sh applies to the source tree, for the same reason and after the
/// same mistake. docs/RISK.md has the history.
inline const std::string& g4particlexs_dir() {
  static const std::string d = [] {
    namespace fs = std::filesystem;
    std::error_code ec;
    // `neutron/el1` - the neutron elastic cross section on hydrogen. Every reader in
    // src/physics/hadronic/xs/ opens Z = 1 first, so a dataset without it fails immediately
    // anyway; the check only moves the failure to where it can be explained.
    auto usable = [&](const fs::path& p) {
      return fs::is_directory(p, ec) && fs::exists(p / "neutron/el1", ec);
    };

    std::vector<std::string> cand;
    const std::string from_env = detail::env_or_empty("G4PARTICLEXSDATA");
    if (!from_env.empty() && usable(from_env)) { cand.push_back(from_env); }
    for (const std::string& root : g4_data_roots()) {
      for (const auto& e : fs::directory_iterator(root, ec)) {
        if (!e.is_directory(ec)) { continue; }
        const std::string name = e.path().filename().string();
        if (name.rfind("G4PARTICLEXS", 0) != 0) { continue; }
        if (!usable(e.path())) { continue; }
        cand.push_back(e.path().string());
      }
    }
    if (cand.empty()) { return std::string(); }

    std::string best = cand[0];
    for (const std::string& c : cand) {
      const std::string a = fs::path(best).filename().string();
      const std::string b = fs::path(c).filename().string();
      if (detail::version_less(a, b)) { best = c; }
    }
    if (!from_env.empty() && fs::path(best).filename() != fs::path(from_env).filename()) {
      std::printf("note: G4PARTICLEXSDATA names %s, using %s - the higher version, which is\n"
                  "      the one ref/oracle/run.bat sets and the oracle was produced from.\n",
                  fs::path(from_env).filename().string().c_str(),
                  fs::path(best).filename().string().c_str());
    }
    return best;
  }();
  return d;
}

/// One particle's subdirectory of G4PARTICLEXS: "proton", "neutron", "deuteron", "triton",
/// "He3", "alpha" or "gamma" - the names G4ParticleInelasticXS builds from
/// `particle->GetParticleName()`, so they are the Geant4 particle names and not a mapping.
///
/// Returns an empty string when the data set cannot be found at all, which the caller must
/// treat as fatal: a zero hadronic cross section is a particle that never interacts, which is
/// a wrong answer that looks like a physics result.
inline std::string g4particlexs_subdir(const char* particle) {
  const std::string& base = g4particlexs_dir();
  if (base.empty()) { return {}; }
  return (std::filesystem::path(base) / particle).string();
}

}  // namespace g4gpu::host
