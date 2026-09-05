// G4UImanager: the macro command interpreter.
//
// Geant4 users drive a run from a .mac file, and the commands are muscle memory:
//
//     /run/initialize
//     /gun/particle gamma
//     /gun/energy 6 MeV
//     /run/beamOn 10000
//
// The same commands work here. Anything under /vis/ is forwarded to whatever viewer has
// registered itself, so a visualisation macro reaches the renderer rather than being
// interpreted twice.
//
// Unknown commands are reported, not ignored. A macro that silently does nothing is worse than
// one that stops: the run still happens, with the default configuration, and the output looks
// like a physics result.
#pragma once
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include "g4/G4RunManager.hh"

class G4UImanager {
 public:
  static G4UImanager* GetUIpointer() {
    static G4UImanager inst;
    return &inst;
  }

  /// Installs a handler for /vis/ commands. The viewer calls this when it opens.
  void SetVisHandler(std::function<G4bool(const G4String&)> h) { vis_ = std::move(h); }

  /// Installs a sink for program output, so a GUI can show it in a panel. Everything the
  /// interpreter prints goes here as well as to stdout.
  void SetOutputSink(std::function<void(const G4String&)> s) { sink_ = std::move(s); }

  void Echo(const G4String& text) {
    std::printf("%s\n", text.c_str());
    if (sink_) { sink_(text); }
  }

  G4int GetVerbose() const { return verbose_; }

  /// Runs one command. Returns 0 on success, non-zero on an error the caller should surface.
  G4int ApplyCommand(const G4String& command_in);

  /// Runs every command in a macro file. Blank lines and `#` comments are skipped.
  G4int ExecuteMacroFile(const G4String& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
      Echo("cannot open macro " + path);
      return 1;
    }
    char line[1024];
    G4int errors = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
      std::string s(line);
      const std::size_t hash = s.find('#');
      if (hash != std::string::npos) { s = s.substr(0, hash); }
      Trim(s);
      if (s.empty()) { continue; }
      if (verbose_ > 0) { Echo(s); }
      errors += (ApplyCommand(s) != 0) ? 1 : 0;
    }
    std::fclose(f);
    return errors;
  }

  /// The commands executed so far, for a GUI to show or replay.
  const std::vector<G4String>& History() const { return history_; }

 private:
  static void Trim(std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) { ++a; }
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) { --b; }
    s = s.substr(a, b - a);
  }

  static std::vector<std::string> Split(const std::string& s) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
      while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) { ++i; }
      const std::size_t start = i;
      while (i < s.size() && !std::isspace(static_cast<unsigned char>(s[i]))) { ++i; }
      if (i > start) { out.push_back(s.substr(start, i - start)); }
    }
    return out;
  }

  /// Converts a Geant4 unit name to its internal factor. An unrecognised unit is an error
  /// rather than a silent factor of one - "/gun/energy 6 Mev" would otherwise run at 6 keV.
  static G4bool UnitFactor(const std::string& u, G4double& out) {
    struct Entry { const char* name; G4double value; };
    static const Entry kUnits[] = {
        {"mm", mm}, {"cm", cm}, {"m", m}, {"km", km}, {"um", um}, {"nm", nm},
        {"eV", eV}, {"keV", keV}, {"MeV", MeV}, {"GeV", GeV}, {"TeV", TeV},
        {"rad", rad}, {"mrad", mrad}, {"deg", deg}, {"degree", degree},
        {"ns", ns}, {"us", us}, {"ms", ms}, {"s", s},
        {"g/cm3", g / cm3}, {"mg/cm3", mg / cm3}, {"kg/m3", kg / m3},
        {"Bq", becquerel}, {"kBq", kilobecquerel}, {"MBq", megabecquerel},
        {"GBq", gigabecquerel}, {"Ci", curie}, {"mCi", millicurie}, {"uCi", microcurie},
    };
    for (const Entry& e : kUnits) {
      if (u == e.name) { out = e.value; return true; }
    }
    return false;
  }

  std::function<G4bool(const G4String&)> vis_;
  std::function<void(const G4String&)> sink_;
  std::vector<G4String> history_;
  G4int verbose_ = 0;
};

inline G4int G4UImanager::ApplyCommand(const G4String& command_in) {
  std::string cmd = command_in;
  Trim(cmd);
  if (cmd.empty() || cmd[0] == '#') { return 0; }
  history_.push_back(cmd);

  const std::vector<std::string> tok = Split(cmd);
  const std::string& head = tok[0];
  auto num = [&](std::size_t i, G4double def = 0) {
    return (i < tok.size()) ? std::atof(tok[i].c_str()) : def;
  };
  auto unit_at = [&](std::size_t i, G4double& factor) {
    if (i >= tok.size()) { factor = 1.0; return true; }
    return UnitFactor(tok[i], factor) == true;
  };

  G4RunManager* rm = G4RunManager::Instance();
  G4ParticleGun* gun = (rm != nullptr) ? rm->GetGun() : nullptr;

  // ------------------------------------------------------------------ /control
  if (head == "/control/execute") {
    if (tok.size() < 2) { Echo("/control/execute needs a filename"); return 1; }
    return ExecuteMacroFile(tok[1]);
  }
  if (head == "/control/verbose" || head == "/control/saveHistory") {
    if (head == "/control/verbose") { verbose_ = static_cast<G4int>(num(1)); }
    return 0;
  }
  if (head == "/control/echo") {
    std::string rest;
    for (std::size_t i = 1; i < tok.size(); ++i) { rest += (i > 1 ? " " : "") + tok[i]; }
    Echo(rest);
    return 0;
  }

  // ------------------------------------------------------------------ /run
  if (head == "/run/initialize") {
    if (rm == nullptr) { Echo("no run manager"); return 1; }
    rm->Initialize();
    return 0;
  }
  if (head == "/run/beamOn") {
    if (rm == nullptr) { Echo("no run manager"); return 1; }
    const G4int n = (tok.size() > 1) ? static_cast<G4int>(std::atol(tok[1].c_str())) : 1;
    rm->BeamOn(n);
    return 0;
  }
  if (head == "/run/setCut") {
    if (rm == nullptr) { Echo("no run manager"); return 1; }
    G4double f = 1.0;
    if (!unit_at(2, f)) { Echo("unknown unit in " + cmd); return 1; }
    rm->SetCutValue(num(1) * f);
    return 0;
  }
  if (head == "/run/setBatchSize") {
    if (rm != nullptr) { rm->SetBatchSize(static_cast<G4int>(num(1))); }
    return 0;
  }
  if (head == "/run/numberOfThreads" || head == "/run/verbose"
      || head == "/run/printProgress" || head == "/run/particle/verbose") {
    // Accepted and ignored: threading is the GPU's, and progress is reported per run.
    return 0;
  }
  if (head == "/event/verbose" || head == "/tracking/verbose"
      || head == "/process/verbose" || head == "/material/verbose") {
    // Accepted and ignored, and this one needs saying out loud rather than just returning 0.
    //
    // These raise the per-event and per-step printout, and there is nothing here to raise: a
    // step happens inside a device kernel where a G4cout cannot be reached, which is the same
    // reason G4UserTrackingAction is accepted and never called (see G4RunManager). Setting
    // them to 0, which is what every batch macro does, asks for silence and gets it; setting
    // them higher asks for output that does not exist.
    //
    // They are listed here rather than left to the unknown-command path because example B1's
    // own run1.mac uses both, and a shipped macro that this file rejects is a worse answer
    // than a shipped macro that runs and says what it could not do.
    if (num(1) > 0) {
      Echo(head + ": per-step and per-event verbosity is not available - tracking happens on"
                  " the device. Accepted and ignored.");
    }
    return 0;
  }

  // ------------------------------------------------------------------ /gun
  if (head.rfind("/gun/", 0) == 0) {
    if (gun == nullptr) {
      Echo("no particle gun registered; " + head + " ignored");
      return 1;
    }
    if (head == "/gun/particle") {
      if (tok.size() < 2) { Echo("/gun/particle needs a name"); return 1; }
      gun->SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle(tok[1]));
      return 0;
    }
    if (head == "/gun/energy") {
      G4double f = MeV;
      if (!unit_at(2, f)) { Echo("unknown unit in " + cmd); return 1; }
      gun->SetParticleEnergy(num(1) * f);
      return 0;
    }
    if (head == "/gun/position") {
      G4double f = mm;
      if (!unit_at(4, f)) { Echo("unknown unit in " + cmd); return 1; }
      gun->SetParticlePosition(G4ThreeVector(num(1) * f, num(2) * f, num(3) * f));
      return 0;
    }
    if (head == "/gun/direction") {
      gun->SetParticleMomentumDirection(G4ThreeVector(num(1), num(2), num(3)));
      return 0;
    }
    if (head == "/gun/number") {
      gun->SetNumberOfParticles(static_cast<G4int>(num(1)));
      return 0;
    }
    if (head == "/gun/beamRectangular") {
      G4double f = mm;
      if (!unit_at(3, f)) { Echo("unknown unit in " + cmd); return 1; }
      gun->SetBeamCrossSectionRectangular(num(1) * f, num(2) * f);
      return 0;
    }
    if (head == "/gun/beamElliptical") {
      G4double f = mm;
      if (!unit_at(3, f)) { Echo("unknown unit in " + cmd); return 1; }
      gun->SetBeamCrossSectionElliptical(num(1) * f, num(2) * f);
      return 0;
    }
    if (head == "/gun/angularSpread") {
      G4double f = deg;
      if (!unit_at(2, f)) { Echo("unknown unit in " + cmd); return 1; }
      gun->SetAngularSpread(num(1) * f);
      return 0;
    }
    if (head == "/gun/isotropic") {
      gun->SetIsotropic();
      return 0;
    }
    if (head == "/gun/isotropicShell") {
      G4double f = mm;
      if (!unit_at(2, f)) { Echo("unknown unit in " + cmd); return 1; }
      gun->SetIsotropicShell(num(1) * f);
      return 0;
    }
    if (head == "/gun/volumeBox") {
      G4double f = mm;
      if (!unit_at(4, f)) { Echo("unknown unit in " + cmd); return 1; }
      gun->SetSourceVolumeBox(num(1) * f, num(2) * f, num(3) * f);
      return 0;
    }
    if (head == "/gun/spectrum") {
      if (tok.size() < 2) { Echo("/gun/spectrum needs a CSV filename"); return 1; }
      return gun->SetEnergySpectrumFromCSV(tok[1]) ? 0 : 1;
    }
  }

  // ------------------------------------------------------------------ /vis
  if (head.rfind("/vis/", 0) == 0) {
    if (!vis_) {
      // Not an error: a batch run legitimately has no viewer, and Geant4 behaves the same way
      // when the visualisation manager is absent.
      if (verbose_ > 0) { Echo("no viewer; " + head + " ignored"); }
      return 0;
    }
    return vis_(cmd) ? 0 : 1;
  }

  Echo("unknown command: " + cmd);
  return 1;
}
