// `/gun/particle ion` and `/gun/ion Z A [Q E flb]`: the macro commands and the gun, on the host.
//
// WHY THIS FILE EXISTS (P30)
//
// Every real nuclide has been TRANSPORTED since P8c - as a secondary, whose nuclide
// `BufferEmitter::push_nucleus` writes into `TrackState::ion_za`. Nothing could make one a
// PRIMARY: the port's G4UImanager had no `ion` and no `/gun/ion`, `G4ParticleTable::FindParticle
// ("ion")` exited with "unknown particle", and the seeding wrote `ion_za = 0`. This test is the
// parser and the gun - G4ParticleGunMessenger::SetNewValue and IonCommand, 11.1.1 - with every
// error path, the defaults, and what the port refuses by name where Geant4 would make a particle.
// The seeding and the step are tests/test_ion_primary.cu's.
//
// It drives `G4UImanager::ApplyGunCommand`, not `ApplyCommand`: the latter reaches /run/beamOn
// and so the transport engine, which a parser test has no reason to link.
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "g4/G4ParticleGun.hh"
#include "g4/G4SystemOfUnits.hh"
#include "g4/G4UImanager.hh"
#include "g4/G4UserActions.hh"

namespace {

int g_fails = 0;
void expect(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++g_fails;
  }
}

std::vector<std::string> g_echo;

/// Applies one /gun/ command, returns its code and leaves its echo in g_echo.
int gun_cmd(G4ParticleGun& gun, const std::string& cmd) {
  g_echo.clear();
  return G4UImanager::GetUIpointer()->ApplyGunCommand(&gun, cmd);
}

bool echoed(const std::string& needle) {
  for (const auto& s : g_echo) {
    if (s.find(needle) != std::string::npos) { return true; }
  }
  return false;
}

/// The gun's state, so a refused command can be shown to have changed nothing.
struct GunState {
  const G4ParticleDefinition* def;
  G4double charge;
  G4bool shoot_ion;
};
GunState state_of(const G4ParticleGun& g) {
  return {g.GetParticleDefinition(), g.GetParticleCharge(), g.GetShootIon()};
}
bool same(const GunState& a, const GunState& b) {
  return a.def == b.def && a.charge == b.charge && a.shoot_ion == b.shoot_ion;
}

/// A command that must fail, leave the gun as it was, and say @p needle.
void refused(G4ParticleGun& gun, const std::string& cmd, const std::string& needle) {
  const GunState before = state_of(gun);
  const int rc = gun_cmd(gun, cmd);
  expect(rc != 0, "\"" + cmd + "\" returned 0; it must fail");
  expect(same(before, state_of(gun)), "\"" + cmd + "\" changed the gun although it failed");
  expect(echoed(needle), "\"" + cmd + "\" did not say \"" + needle + "\"");
  std::printf("  %-40s -> %s\n", cmd.c_str(), g_echo.empty() ? "(silent)" : g_echo[0].c_str());
}

}  // namespace

int main() {
  std::printf("== /gun/particle ion and /gun/ion (G4ParticleGunMessenger, 11.1.1) ==\n\n");
  G4UImanager::GetUIpointer()->SetOutputSink([](const G4String& s) { g_echo.push_back(s); });

  // ---------------------------------------------------------------- the record
  //
  // The nuclide rides in padding: `particle` is an int at 56 and `t0` a double at 64.
  expect(sizeof(g4gpu::Primary<double>) == 72, "sizeof(Primary<double>) is not 72");
  expect(offsetof(g4gpu::Primary<double>, ion_za) == 60, "Primary::ion_za is not at offset 60");
  expect(offsetof(g4gpu::Primary<double>, t0) == 64, "Primary::t0 moved");
  std::printf("Primary<double>: %zu bytes, ion_za at %zu, t0 at %zu\n\n",
              sizeof(g4gpu::Primary<double>), offsetof(g4gpu::Primary<double>, ion_za),
              offsetof(g4gpu::Primary<double>, t0));

  G4ParticleGun gun(1);
  gun.SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle("gamma"));

  // ---------------------------------------------------------------- 1. the mode comes first
  std::printf("1. /gun/ion before /gun/particle ion - Geant4's own message\n");
  refused(gun, "/gun/ion 6 12", "Set /gun/particle to ion before using /gun/ion command");
  expect(gun_cmd(gun, "/gun/particle ion") == 0, "/gun/particle ion failed");
  expect(gun.GetShootIon(), "/gun/particle ion did not set the ion mode");
  expect(gun.GetParticleDefinition()->GetParticleName() == "gamma",
         "/gun/particle ion changed the definition; Geant4 keeps it until /gun/ion");

  // ---------------------------------------------------------------- 2. the defaults
  std::printf("\n2. /gun/ion 6 12 - Q = Z, E = 0, a ground state\n");
  expect(gun_cmd(gun, "/gun/ion 6 12") == 0, "/gun/ion 6 12 failed");
  const G4ParticleDefinition* c12 = gun.GetParticleDefinition();
  const double m_c12 = g4gpu::data::get_nuclear_mass(12, 6);
  expect(c12->GetParticleName() == "C12", "the name is not C12: " + c12->GetParticleName());
  expect(c12->GetType() == g4gpu::ParticleType::kGenericIon, "C12 is not kGenericIon");
  expect(c12->GetIonZA() == g4gpu::ion_za_of(6, 12), "C12's ion_za is not ion_za_of(6, 12)");
  expect(c12->GetAtomicNumber() == 6 && c12->GetAtomicMass() == 12, "C12's Z, A");
  expect(c12->GetPDGMass() == m_c12 * MeV, "C12's mass is not GetNuclearMass(12, 6)");
  expect(c12->GetPDGCharge() == 6.0, "C12's PDG charge is not 6");
  expect(gun.GetParticleCharge() == 6.0 * eplus, "the default charge is not Z");
  expect(g_echo.empty(), "a default /gun/ion printed something; Geant4's is silent");
  std::printf("  C12: %s, Z %d A %d, ion_za %u, mass %.6f MeV, charge %g\n",
              c12->GetParticleName().c_str(), c12->GetAtomicNumber(), c12->GetAtomicMass(),
              static_cast<unsigned>(c12->GetIonZA()), c12->GetPDGMass() / MeV,
              gun.GetParticleCharge());
  expect(gun_cmd(gun, "/gun/ion 6 12") == 0 && gun.GetParticleDefinition() == c12,
         "a second /gun/ion 6 12 made a second definition; G4IonTable finds the first");

  // Q: given, negative (Geant4's default -1, read as Z), and E = 0 / noFloat spelt out.
  expect(gun_cmd(gun, "/gun/ion 6 12 4") == 0 && gun.GetParticleCharge() == 4.0,
         "/gun/ion 6 12 4: charge is not 4");
  expect(echoed("nothing this port runs reads it"), "Q != Z was not said to be inert");
  expect(gun_cmd(gun, "/gun/ion 6 12 -1") == 0 && gun.GetParticleCharge() == 6.0,
         "/gun/ion 6 12 -1: a negative Q must mean Z");
  expect(gun_cmd(gun, "/gun/ion 6 12 6 0") == 0, "/gun/ion 6 12 6 0 (E = 0) failed");
  expect(gun_cmd(gun, "/gun/ion 6 12 6 0.0 noFloat") == 0, "/gun/ion 6 12 6 0.0 noFloat");
  expect(gun.GetParticleDefinition() == c12, "E = 0 / noFloat is not the ground state C12");

  // ---------------------------------------------------------------- 3. /gun/energy is TOTAL
  std::printf("\n3. /gun/energy is the ion's TOTAL kinetic energy, as in G4ParticleGun\n");
  expect(gun_cmd(gun, "/gun/energy 12000 MeV") == 0, "/gun/energy 12000 MeV");
  expect(gun.GetParticleEnergy() == 12000.0 * MeV, "12000 MeV is not 12000 MeV");
  expect(gun_cmd(gun, "/gun/energy 56 GeV") == 0 && gun.GetParticleEnergy() == 56000.0 * MeV,
         "56 GeV is not 56000 MeV");
  {
    G4Event ev(0);
    gun.GeneratePrimaryVertex(&ev);
    const G4PrimaryParticle& pp = ev.GetPrimaryVertex(0).GetPrimary(0);
    expect(pp.GetParticleDefinition() == c12, "the vertex's primary is not C12");
    expect(pp.GetParticleDefinition()->GetIonZA() == g4gpu::ion_za_of(6, 12),
           "the vertex's primary lost its nuclide");
    expect(pp.GetKineticEnergy() == 56000.0, "the vertex's kinetic energy is not the gun's");
    std::printf("  vertex: %s, %.1f MeV, ion_za %u\n",
                pp.GetParticleDefinition()->GetParticleName().c_str(), pp.GetKineticEnergy(),
                static_cast<unsigned>(pp.GetParticleDefinition()->GetIonZA()));
  }

  // ---------------------------------------------------------------- 4. other nuclides
  std::printf("\n4. other nuclides, and the five light ones G4IonTable::GetLightIon names\n");
  expect(gun_cmd(gun, "/gun/ion 26 56") == 0, "/gun/ion 26 56 failed");
  const G4ParticleDefinition* fe56 = gun.GetParticleDefinition();
  expect(fe56->GetParticleName() == "Fe56", "Fe56's name: " + fe56->GetParticleName());
  expect(fe56->GetIonZA() == g4gpu::ion_za_of(26, 56), "Fe56's ion_za");
  expect(fe56->GetPDGMass() == g4gpu::data::get_nuclear_mass(56, 26) * MeV, "Fe56's mass");
  expect(gun.GetParticleCharge() == 26.0, "Fe56's charge");
  std::printf("  Fe56: mass %.6f MeV, charge %g\n", fe56->GetPDGMass() / MeV,
              gun.GetParticleCharge());
  const struct { int z, a; const char* name; g4gpu::ParticleType t; } light[] = {
      {1, 1, "proton", g4gpu::ParticleType::kProton},
      {1, 2, "deuteron", g4gpu::ParticleType::kDeuteron},
      {1, 3, "triton", g4gpu::ParticleType::kTriton},
      {2, 3, "He3", g4gpu::ParticleType::kHe3},
      {2, 4, "alpha", g4gpu::ParticleType::kAlpha},
  };
  for (const auto& l : light) {
    const std::string c = "/gun/ion " + std::to_string(l.z) + " " + std::to_string(l.a);
    expect(gun_cmd(gun, c) == 0, c + " failed");
    expect(gun.GetParticleDefinition()->GetParticleName() == l.name
               && gun.GetParticleDefinition()->GetType() == l.t
               && gun.GetParticleDefinition()->GetIonZA() == 0,
           c + " is not the " + l.name + " definition");
  }
  // Past AME2012 with a mass Geant4 gives (P24: all four of GetNuclearMass's levels). C30 is
  // on the formula level - no AME row, no theoretical row (Z < 8) - and accepted, as Geant4
  // accepts it and as a secondary C30 is already transported.
  expect(gun_cmd(gun, "/gun/ion 6 30") == 0, "/gun/ion 6 30 (past AME2012, formula mass)");
  expect(g4gpu::data::nuclear_mass_level(30, 6) == g4gpu::data::NuclearMassLevel::kFormula,
         "C30 is not on the formula level; pick another nuclide for this check");
  expect(gun.GetParticleDefinition()->GetPDGMass()
             == g4gpu::data::get_nuclear_mass(30, 6) * MeV,
         "C30's mass is not GetNuclearMass's");
  expect(G4IonTable::GetIonName(119, 300) == "E119-300", "GetIonName past element 118");
  expect(G4IonTable::GetIonName(82, 208) == "Pb208", "GetIonName(82, 208)");

  // ---------------------------------------------------------------- 5. errors, Geant4's
  std::printf("\n5. every error path\n");
  refused(gun, "/gun/ion", "needs Z and A");
  refused(gun, "/gun/ion 6", "needs Z and A");
  refused(gun, "/gun/ion C 12", "is not an integer");
  refused(gun, "/gun/ion 6 12.5", "is not an integer");
  refused(gun, "/gun/ion 6 12 six", "is not an integer");
  refused(gun, "/gun/ion 6 12 6 zero", "is not a number");
  refused(gun, "/gun/ion 6 12 6 0 Q", "is not one of the candidates");
  refused(gun, "/gun/ion 6 12 6 0 noFloat extra", "at most five parameters");
  // G4IonTable::GetIon's own illegal arguments, and the messenger's message for them.
  refused(gun, "/gun/ion 0 12", "Ion with Z=0 A=12is not defined");
  refused(gun, "/gun/ion -6 12", "Ion with Z=-6 A=12is not defined");
  refused(gun, "/gun/ion 6 0", "Ion with Z=6 A=0is not defined");
  refused(gun, "/gun/ion 6 1000", "Ion with Z=6 A=1000is not defined");
  refused(gun, "/gun/ion 6 12 6 -5", "Ion with Z=6 A=12is not defined");

  // ---------------------------------------------------------------- 6. refused BY NAME
  std::printf("\n6. what Geant4 would make and this port refuses by name\n");
  refused(gun, "/gun/ion 6 12 6 4438.9", "an excited ion (E > 0)");
  refused(gun, "/gun/ion 6 12 6 0 X", "a floating level base");
  refused(gun, "/gun/ionL 6 12 6 0", "/gun/ionL refused");
  refused(gun, "/gun/ion 8 4", "GetNuclearMass(A, Z) is zero");
  refused(gun, "/gun/ion 130 300", "past TrackState::ion_za's encoding");
  refused(gun, "/gun/ion 26 600", "past TrackState::ion_za's encoding");

  // ---------------------------------------------------------------- 7. leaving the mode
  std::printf("\n7. /gun/particle <name> leaves the ion mode and the ion's charge\n");
  expect(gun_cmd(gun, "/gun/ion 6 12 3") == 0, "/gun/ion 6 12 3");
  expect(gun_cmd(gun, "/gun/particle proton") == 0, "/gun/particle proton");
  expect(!gun.GetShootIon(), "/gun/particle proton left the ion mode on");
  expect(gun.GetParticleCharge() == 1.0, "the proton kept the ion's charge (SetParticleDef)");
  expect(gun.GetParticleDefinition()->GetIonZA() == 0, "a proton carries a nuclide");
  refused(gun, "/gun/ion 6 12", "Set /gun/particle to ion before using /gun/ion command");

  // The bare GenericIon name still resolves - to the placeholder, with no nuclide, which the
  // run manager refuses at /run/beamOn (G4RunManager::RefuseBareGenericIon).
  expect(gun_cmd(gun, "/gun/particle GenericIon") == 0, "/gun/particle GenericIon");
  expect(gun.GetParticleDefinition()->GetType() == g4gpu::ParticleType::kGenericIon
             && gun.GetParticleDefinition()->GetIonZA() == 0,
         "the bare GenericIon definition carries a nuclide");

  std::printf("\n%s (%d failure%s)\n", g_fails == 0 ? "PASS" : "FAIL", g_fails,
              g_fails == 1 ? "" : "s");
  return g_fails == 0 ? 0 : 1;
}
