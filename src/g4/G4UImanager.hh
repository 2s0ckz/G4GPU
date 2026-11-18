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
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include "g4/G4RunManager.hh"
#include "g4/Randomize.hh"  // /random/setSeeds reseeds the host engine as well as the device one

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

  /// One /gun/ command on @p gun (G4ParticleGunMessenger). ApplyCommand calls it for the
  /// registered gun; see the definition for why it is separate.
  G4int ApplyGunCommand(G4ParticleGun* gun, const G4String& cmd);

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
  // ------------------------------------------------------------------ /random
  //
  // Restarts both random streams, which is what Geant4's one engine does when reseeded and
  // what a macro that wants a particular run again is asking for. Both, because this port has
  // two: CLHEP's engine on the host draws the primaries, and a counter-based device stream
  // keyed on (seed, position) draws the showers. Reseeding one and not the other would give a
  // run that is half a repeat, which is worse than either.
  //
  // Geant4 takes two integers and its engines use both; here the FIRST seeds both streams and
  // the second is accepted and ignored, because one number is what there is to set and
  // inventing a mix of the two would make the command's effect impossible to predict.
  //
  // WHAT THIS DOES NOT DO: return you to the state the process started in. Nothing can, and
  // that is worth knowing before relying on it. A freshly started process has the host engine
  // on CLHEP's own default seed and the device stream on G4RunManager's, which are different
  // numbers picked by different code; `/random/setSeeds n` puts BOTH on n, which is a
  // perfectly reproducible state and not that one. So a macro that reseeds mid-session repeats
  // itself exactly on the next process, which is what reproducibility means here - but it does
  // not reproduce the run it would have got without the command.
  if (head == "/random/setSeeds" || head == "/random/setSeed") {
    if (tok.size() < 2) { Echo("usage: /random/setSeeds <n> [<n>]"); return 1; }
    const long s1 = std::atol(tok[1].c_str());
    G4Random::setTheSeed(s1);
    if (rm != nullptr) { rm->SetRandomSeed(static_cast<unsigned int>(s1)); }
    return 0;
  }
  if (head == "/random/resetEngineFrom" || head == "/random/saveThisRun"
      || head == "/random/saveThisEvent" || head == "/random/setSavingFlag"
      || head == "/random/setDirectoryName") {
    // Accepted and ignored: there is no engine state file to write or read. A run here is
    // reproducible from its seed and its position in the stream, both of which are integers
    // the run manager already holds - see G4RunManager::SetRandomSeed.
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
    return ApplyGunCommand(gun, cmd);
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

/// The /gun/ commands, on one gun - G4ParticleGunMessenger::SetNewValue.
///
/// A function of its own rather than a branch of ApplyCommand so that a test can drive the
/// messenger without a run manager: ApplyCommand reaches /run/beamOn and with it the transport
/// engine, which a parser test has no reason to link (tests/test_ion_gun.cu).
inline G4int G4UImanager::ApplyGunCommand(G4ParticleGun* gun, const G4String& cmd) {
  const std::vector<std::string> tok = Split(cmd);
  if (tok.empty() || gun == nullptr) { return 1; }
  const std::string& head = tok[0];
  auto num = [&](std::size_t i, G4double def = 0) {
    return (i < tok.size()) ? std::atof(tok[i].c_str()) : def;
  };
  auto unit_at = [&](std::size_t i, G4double& factor) {
    if (i >= tok.size()) { factor = 1.0; return true; }
    return UnitFactor(tok[i], factor) == true;
  };
  {
    if (head == "/gun/particle") {
      if (tok.size() < 2) { Echo("/gun/particle needs a name"); return 1; }
      // G4ParticleGunMessenger: "ion" is not a particle but a MODE - `fShootIon = true` - and
      // the gun keeps its definition until `/gun/ion Z A` names the nucleus. Any other name
      // clears the mode and resolves in the particle table.
      if (tok[1] == "ion") {
        gun->SetShootIon(true);
        return 0;
      }
      gun->SetShootIon(false);
      gun->SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle(tok[1]));
      return 0;
    }
    if (head == "/gun/ion") {
      // G4ParticleGunMessenger::SetNewValue, then IonCommand (11.1.1). The guidance's usage
      // line is "/gun/ion Z A [Q E flb]": Z and A are non-omittable integers, Q an integer
      // defaulting to -1 (read as "Z"), E a double in keV defaulting to 0, flb one of
      // "noFloat X Y Z U V W R S T A B C D E".
      if (!gun->GetShootIon()) {
        Echo("Set /gun/particle to ion before using /gun/ion command");
        return 1;
      }
      long iv[3] = {0, 0, -1};
      for (std::size_t i = 1; i < tok.size() && i <= 3; ++i) {
        char* end = nullptr;
        iv[i - 1] = std::strtol(tok[i].c_str(), &end, 10);
        if (end == tok[i].c_str() || *end != '\0') {
          Echo("/gun/ion: parameter " + std::to_string(i) + " (\"" + tok[i]
               + "\") is not an integer. Usage: /gun/ion Z A [Q E flb]");
          return 1;
        }
      }
      if (tok.size() < 3) {
        Echo("/gun/ion needs Z and A. Usage: /gun/ion Z A [Q E flb]");
        return 1;
      }
      if (tok.size() > 6) {
        Echo("/gun/ion takes at most five parameters. Usage: /gun/ion Z A [Q E flb]");
        return 1;
      }
      const G4int z = static_cast<G4int>(iv[0]);
      const G4int a = static_cast<G4int>(iv[1]);
      // `fIonCharge = fAtomicNumber; ... if (StoI(sQ) >= 0) fIonCharge = StoI(sQ);`
      const G4int q = (tok.size() > 3 && iv[2] >= 0) ? static_cast<G4int>(iv[2]) : z;
      G4double e_kev = 0;
      if (tok.size() > 4) {
        char* end = nullptr;
        e_kev = std::strtod(tok[4].c_str(), &end);
        if (end == tok[4].c_str() || *end != '\0') {
          Echo("/gun/ion: parameter 4 (\"" + tok[4]
               + "\") is not a number. Usage: /gun/ion Z A [Q E flb]");
          return 1;
        }
      }
      if (tok.size() > 5 && tok[5] != "noFloat") {
        static const char* const kFlb[] = {"X", "Y", "Z", "U", "V", "W", "R",
                                           "S", "T", "A", "B", "C", "D", "E"};
        bool known = false;
        for (const char* f : kFlb) { known = known || (tok[5] == f); }
        if (!known) {
          Echo("/gun/ion: parameter 5 (\"" + tok[5] + "\") is not one of the candidates "
               "\"noFloat X Y Z U V W R S T A B C D E\"");
          return 1;
        }
        // A floating level base names an isomer's level scheme; with E = 0 it still asks
        // GetIon for a level ("C12[0.000X]") rather than the ground state.
        Echo("/gun/ion " + std::to_string(z) + " " + std::to_string(a)
             + " refused: a floating level base (\"" + tok[5] + "\") names an excited "
             "level, and this port transports ground states only - use /gun/ion Z A [Q].");
        return 1;
      }
      const G4double e = e_kev * keV;
      const char* why = G4IonTable::RefusalReason(z, a, e);
      if (why != nullptr) {
        Echo("/gun/ion " + std::to_string(z) + " " + std::to_string(a) + " refused: " + why
             + ". This port transports ground-state nuclei (E = 0) with a nuclear mass.");
        return 1;
      }
      G4ParticleDefinition* ion = G4IonTable::GetIonTable()->GetIon(z, a, e);
      if (ion == nullptr) {
        // Geant4's message, character for character - including its missing space.
        Echo("Ion with Z=" + std::to_string(z) + " A=" + std::to_string(a) + "is not defined");
        return 1;
      }
      gun->SetParticleDefinition(ion);
      gun->SetParticleCharge(q * eplus);
      if (q != z) {
        Echo("/gun/ion: " + ion->GetParticleName() + " with charge " + std::to_string(q)
             + " - the charge is held on the gun as Geant4's dynamic charge; nothing this "
             "port runs reads it (no magnetic field; G4ionIonisation recomputes the effective "
             "charge every step), so the transport is that of the bare nucleus either way.");
      }
      return 0;
    }
    if (head == "/gun/ionL") {
      // Deprecated in 11.1.1 ("THIS COMMAND IS DEPRECATED and will be removed in future
      // releases") and a LEVEL NUMBER where /gun/ion takes an energy. Refused rather than
      // mapped: the port transports ground states, and /gun/ion Z A [Q] is the form for one.
      Echo("/gun/ionL refused: the deprecated level form (Z A [Q I]) names an isomer level, "
           "and this port transports ground states only - use /gun/ion Z A [Q].");
      return 1;
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

  Echo("unknown command: " + cmd);
  return 1;
}
