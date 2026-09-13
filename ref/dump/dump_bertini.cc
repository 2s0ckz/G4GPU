// The Bertini cascade's oracle: what Geant4 11.1.1's INUCL tree answers, so the port can be
// diffed against it rather than against a reading of the source.
//
// Files:
//
//   bertini_params.csv    every G4CascadeParameters value, read from the install - and the
//                         energy limits and de-excitation choice of the Bertini instances QBBC
//                         actually builds. The brief for this package said to dump the
//                         parameters before assuming any default; two of them are not what a
//                         reading of G4CascadeParameters.cc suggests. See the header comment on
//                         dump_params() below.
//   bertini_particles.csv G4InuclElementaryParticle::getParticleMass / getStrangeness and the
//                         G4InuclParticleNames predicates for every type code, so the port's
//                         particle table is checked rather than retyped from the PDG.
//   bertini_nuclei.csv    G4NucleiModel's whole zone structure for 13 nuclei: the zone radii
//                         and volumes, the proton and neutron densities, Fermi momenta and
//                         potentials per zone, the flat meson/hyperon potentials, the binding
//                         energy differences and the total radius and volume. Deterministic and
//                         exact - generateModel() draws no random numbers.
//   bertini_nucxsec.csv   G4NucleiModel::totalCrossSection and absorptionCrossSection on an
//                         energy grid for every initial state the model can look up, which is
//                         the channel tables' `tot` array seen through the crossSectionUnits
//                         scale factor.
//   bertini_channels.csv  per channel: initial state, energy bins, multiplicity structure, and
//                         getCrossSection / getCrossSectionSum on a grid that includes every
//                         bin edge and the midpoints between them.
//   bertini_chtables.csv  the whole of every channel data object - index[], multiplicities[][],
//                         sum[], tot[], inelastic[] and crossSections[][] - plus the
//                         final-state lists in bertini_chfinalstates.csv and the interpolator
//                         bin edges in bertini_chbins.csv. This is what
//                         tools/extract_bertini_channels.pl is checked against: the script
//                         parses the .cc source text, this reads what the compiler built from
//                         it, and they have to agree entry for entry. Read off printTable(),
//                         not off the data members - see the comment on dump_chtables().
//   bertini_angdst.csv    G4TwoBodyAngularDist: which distribution ChooseDist returns for every
//                         (initial state, final state, kw) the colliders ask for, and
//                         GetCosTheta under a prescribed uniform cycle for each of the fifteen
//                         distribution objects.
//   bertini_momdst.csv    G4MultiBodyMomentumDist: the same two things for the four momentum
//                         distributions.
//   bertini_epcollide.csv G4ElementaryParticleCollider::collide: the whole two-body chain -
//                         multiplicity, final state, momentum moduli, angles, rotations, the
//                         boost back to the lab and the descending-Ekin sort - for 28 particle
//                         pairs x 12 lab momenta x 2 target states x 8 phases x 2 generator
//                         settings, with the draw count beside each answer. The target is run
//                         at rest and with a Fermi momentum because the first makes the frame
//                         degenerate and `G4LorentzConvertor::rotate` the identity; the second
//                         pass turns `usePhaseSpace` on and needs a different engine. Both
//                         reasons are on dump_epcollide() below.
//   bertini_apply.csv     G4CascadeInterface::ApplyYourself, N events under a fixed seed, as
//                         QBBC configures it (PreCompound de-excitation) - multiplicity by
//                         species, kinetic-energy and angular moments per species, and the
//                         event-by-event energy/momentum balance.
//
// The cycle engine is the same eight values dump_elastic.cc and dump_precompound.cc use, for
// the same reason: it turns a sampler into a deterministic function of (inputs, phase) and the
// draw count is dumped beside the answer, because a transcription can get the right number out
// of the wrong number of deviates.
#include "dump_registry.hh"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "CLHEP/Random/RandomEngine.h"
#include "Randomize.hh"

#include "G4CascadeChannel.hh"
#include "G4CascadeChannelTables.hh"
#include "G4CascadeInterface.hh"
#include "G4CollisionOutput.hh"
#include "G4ElementaryParticleCollider.hh"
#include "G4CascadeParameters.hh"
#include "G4DynamicParticle.hh"
#include "G4HadProjectile.hh"
#include "G4HadFinalState.hh"
#include "G4HadSecondary.hh"
#include "G4HadronicParameters.hh"
#include "G4InuclElementaryParticle.hh"
#include "G4InuclParticleNames.hh"
#include "G4InuclSpecialFunctions.hh"
#include "G4MultiBodyMomentumDist.hh"
#include "G4Neutron.hh"
#include "G4InuclParamAngDst.hh"
#include "G4CascadParticle.hh"
#include "G4NucleiModel.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleus.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4Proton.hh"
#include "G4SystemOfUnits.hh"
#include "G4UImanager.hh"
#include "G4TwoBodyAngularDist.hh"
#include "G4VMultiBodyMomDst.hh"
#include "G4VTwoBodyAngDst.hh"

// The four-teen distribution objects ChooseDist dispatches to, constructed directly so that
// each one's own table is exercised rather than only the ones a collision happens to reach.
#include "G4GamP2NPipAngDst.hh"
#include "G4GamP2PPi0AngDst.hh"
#include "G4GammaNuclAngDst.hh"
#include "G4HadNElastic1AngDst.hh"
#include "G4HadNElastic2AngDst.hh"
#include "G4HadNucl3BodyAngDst.hh"
#include "G4NP2NPAngDst.hh"
#include "G4NuclNucl3BodyAngDst.hh"
#include "G4NuclNuclAngDst.hh"
#include "G4PP2PPAngDst.hh"
#include "G4Pi0P2Pi0PAngDst.hh"
#include "G4PiNInelasticAngDst.hh"
#include "G4PimP2Pi0NAngDst.hh"
#include "G4PimP2PimPAngDst.hh"
#include "G4PipP2PipPAngDst.hh"

#include "G4HadNucl3BodyMomDst.hh"
#include "G4HadNucl4BodyMomDst.hh"
#include "G4NuclNucl3BodyMomDst.hh"
#include "G4NuclNucl4BodyMomDst.hh"

using namespace G4InuclParticleNames;

namespace {

const double kSeq[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};

class CycleEngine : public CLHEP::HepRandomEngine {
 public:
  void reset(int phase) { phase_ = unsigned(phase); n_ = 0; }
  // `n_` is unsigned and the index is taken on the unsigned value: a rejection sampler driven by
  // this engine can draw more times than an int can count, and a signed overflow there is
  // undefined behaviour whose visible form is a NEGATIVE array index. See the comment on
  // dump_epcollide() - this is not hypothetical, it happened.
  long long draws() const { return static_cast<long long>(n_); }
  double flat() override {
    const double v = kSeq[(n_ + phase_) % 8u];
    ++n_;
    return v;
  }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "CycleEngine"; }

 private:
  unsigned phase_ = 0;
  unsigned n_ = 0;
};

// A 64-bit linear congruential generator, for the sampler the cycle engine cannot drive.
//
// `G4CascadeFinalStateAlgorithm::BetaKopylov` is an unbounded rejection loop - `do { chi = u1;
// F = sqrt(chi^N (1-chi)); } while (Fmax*u2 > F)` - and it draws TWO deviates per trial. Under a
// period-8 cycle that is four distinct trials and no more, forever. At N = 3k-5 = 19, which is
// the k = 8 term of an eight-body phase-space decay, F is below 0.007 for every chi in the cycle
// except one, and no pair passes: the loop does not terminate. (What it did instead, before the
// unsigned fix above, was run 2^31 times and then index kSeq out of bounds.)
//
// So the usePhaseSpace pass is driven by this instead. It is an LCG in exact 64-bit integer
// arithmetic - Knuth's MMIX multiplier and increment, the top 53 bits taken as the mantissa -
// so the port reproduces it bit for bit with six lines and no library, which is the whole
// requirement: the engine has to be deterministic and reproducible on both sides, and it has to
// have a period long enough that a rejection sampler terminates. `flat()` is in (0, 1) and never
// returns 0 or 1, because `randomInuclPowers` and the angular distributions divide by 1 - u.
class LcgEngine : public CLHEP::HepRandomEngine {
 public:
  void reset(int phase) {
    s_ = 88172645463325252ull + 1442695040888963407ull * static_cast<unsigned long long>(phase + 1);
    n_ = 0;
  }
  long long draws() const { return static_cast<long long>(n_); }
  double flat() override {
    s_ = 6364136223846793005ull * s_ + 1442695040888963407ull;
    ++n_;
    return (static_cast<double>(s_ >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "LcgEngine"; }

 private:
  unsigned long long s_ = 0;
  unsigned long long n_ = 0;
};

// An engine that returns one fixed value for every draw, for a branch the cycle cannot reach.
//
// `G4NucleiModel::choosePointAlongTraj` has a radial-incidence shortcut,
// `if (prang < 1e-6) posout = -pos;`, and the alternative is a rotation about `phat.cross(rhat)`
// - which at radial incidence is the ZERO vector, so CLHEP's `HepRotation::rotate` prints
// "zero axis" and leaves the vector alone. The two branches therefore disagree completely: one
// gives the antipode and a chord through the whole nucleus, the other gives the entry point back
// and a chord of zero length.
//
// The entry angle comes from `costh = sqrt(1 - inuclRndm())`, so the shortcut needs a deviate
// within about 1e-12 of zero, and the eight-value cycle's smallest is 0.05 - which puts `prang`
// at 0.2255 radians, four orders of magnitude above the cut. **Measured**: deleting the shortcut
// from the port left all 535,692 comparisons passing. One extra pass with this engine at
// 1e-13 reaches it; 0.5 is the control, on the same engine, so that a disagreement can be
// attributed to the branch rather than to the engine.
class ConstEngine : public CLHEP::HepRandomEngine {
 public:
  void reset(double v) { v_ = v; n_ = 0; }
  long long draws() const { return static_cast<long long>(n_); }
  double flat() override { ++n_; return v_; }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "ConstEngine"; }

 private:
  double v_ = 0.5;
  unsigned long long n_ = 0;
};

// The three constants, in the order the `phase` column encodes them as 8, 9 and 10. EXACTLY zero
// is the one that matters: it makes `costh` exactly 1, the entry point exactly (0, 0, -R) and
// `phat.cross(rhat)` the exact zero vector, where CLHEP does nothing at all and the shortcut
// gives the antipode. At 1e-13 the two branches differ by about 1e-13 relative - under the
// tolerance - so a near-radial case is not enough; it has to be radial.
const double kConstSeq[3] = {0.0, 1.0e-13, 0.5};

// ---------------------------------------------------------------------------------------------
// bertini_params.csv
//
// Two of these are not what reading G4CascadeParameters.cc alone suggests, which is why the
// file exists before any transcription:
//
//   DO_COALESCENCE is TRUE by default. Its initializer is
//   `(0==G4CASCADE_DO_COALESCENCE || G4CASCADE_DO_COALESCENCE[0]!='0')` - a NULL envvar pointer
//   makes the first disjunct true - where USE_PRECOMPOUND's is `(0!=... && ...[0]!='0')`, which
//   a null pointer makes false. The two lines are four characters apart in the same function
//   and they default opposite ways.
//
//   RADIUS_SCALE, RADIUS_TRAILING, FERMI_SCALE and XSEC_SCALE do not come from the initializer
//   at all when their envvar is unset: four `HDP.DeveloperGet("BERT_...")` calls overwrite them
//   from G4HadronicDeveloperParameters, whose defaults are registered by a file-scope object in
//   the same translation unit. The values agree here, but they are a second source and the
//   arithmetic differs - RADIUS_SMALL is `(8.0/OLD_RADIUS_UNITS)*RADIUS_SCALE`, which is 8.0
//   only up to rounding, so it is dumped at 17 digits rather than assumed.
//
// The energy limits are read off the objects QBBC builds, not off the constructor's constants:
// G4HadronInelasticQBBC gives nucleons [1, 6] GeV and pions [1, 12] GeV and calls
// usePreCompoundDeexcitation() on both, while G4HadronicBuilder::BuildFTFP_BERT - which is what
// kaons and hyperons get - calls SetMaxEnergy only, leaving G4HadronicInteraction's
// theMinEnergy(0.0), and does NOT call usePreCompoundDeexcitation. So QBBC contains Bertini in
// two configurations and only one of them uses PreCompound.
// ---------------------------------------------------------------------------------------------
void dump_params() {
  FILE* f = std::fopen("bertini_params.csv", "w");
  if (!f) return;
  std::fprintf(f, "name,value\n");

  std::fprintf(f, "verbose,%d\n", G4CascadeParameters::verbose());
  std::fprintf(f, "checkConservation,%d\n", G4CascadeParameters::checkConservation() ? 1 : 0);
  std::fprintf(f, "usePreCompound,%d\n", G4CascadeParameters::usePreCompound() ? 1 : 0);
  std::fprintf(f, "doCoalescence,%d\n", G4CascadeParameters::doCoalescence() ? 1 : 0);
  std::fprintf(f, "showHistory,%d\n", G4CascadeParameters::showHistory() ? 1 : 0);
  std::fprintf(f, "use3BodyMom,%d\n", G4CascadeParameters::use3BodyMom() ? 1 : 0);
  std::fprintf(f, "usePhaseSpace,%d\n", G4CascadeParameters::usePhaseSpace() ? 1 : 0);
  std::fprintf(f, "piNAbsorption,%.17g\n", G4CascadeParameters::piNAbsorption());
  std::fprintf(f, "randomFileEmpty,%d\n", G4CascadeParameters::randomFile().empty() ? 1 : 0);
  std::fprintf(f, "useTwoParam,%d\n", G4CascadeParameters::useTwoParam() ? 1 : 0);
  std::fprintf(f, "radiusScale,%.17g\n", G4CascadeParameters::radiusScale());
  std::fprintf(f, "radiusSmall,%.17g\n", G4CascadeParameters::radiusSmall());
  std::fprintf(f, "radiusAlpha,%.17g\n", G4CascadeParameters::radiusAlpha());
  std::fprintf(f, "radiusTrailing,%.17g\n", G4CascadeParameters::radiusTrailing());
  std::fprintf(f, "fermiScale,%.17g\n", G4CascadeParameters::fermiScale());
  std::fprintf(f, "xsecScale,%.17g\n", G4CascadeParameters::xsecScale());
  std::fprintf(f, "gammaQDScale,%.17g\n", G4CascadeParameters::gammaQDScale());
  std::fprintf(f, "dpMaxDoublet,%.17g\n", G4CascadeParameters::dpMaxDoublet());
  std::fprintf(f, "dpMaxTriplet,%.17g\n", G4CascadeParameters::dpMaxTriplet());
  std::fprintf(f, "dpMaxAlpha,%.17g\n", G4CascadeParameters::dpMaxAlpha());

  // G4HadronicParameters, for the two transition energies QBBC hands Bertini.
  G4HadronicParameters* hp = G4HadronicParameters::Instance();
  std::fprintf(f, "minEnergyTransitionFTF_Cascade_MeV,%.17g\n",
               hp->GetMinEnergyTransitionFTF_Cascade() / MeV);
  std::fprintf(f, "maxEnergyTransitionFTF_Cascade_MeV,%.17g\n",
               hp->GetMaxEnergyTransitionFTF_Cascade() / MeV);
  std::fprintf(f, "maxEnergy_MeV,%.17g\n", hp->GetMaxEnergy() / MeV);

  // The objects QBBC builds. A default-constructed G4CascadeInterface carries
  // G4HadronicInteraction's own limits; the three below are set exactly as the two
  // constructors set them.
  G4CascadeInterface bareBert;
  std::fprintf(f, "interface_default_minE_MeV,%.17g\n", bareBert.GetMinEnergy() / MeV);
  std::fprintf(f, "interface_default_maxE_MeV,%.17g\n", bareBert.GetMaxEnergy() / MeV);

  G4CascadeInterface nucleonBert;
  nucleonBert.SetMinEnergy(1.0 * CLHEP::GeV);
  nucleonBert.SetMaxEnergy(hp->GetMaxEnergyTransitionFTF_Cascade());
  std::fprintf(f, "qbbc_nucleon_minE_MeV,%.17g\n", nucleonBert.GetMinEnergy() / MeV);
  std::fprintf(f, "qbbc_nucleon_maxE_MeV,%.17g\n", nucleonBert.GetMaxEnergy() / MeV);

  G4CascadeInterface pionBert;
  pionBert.SetMinEnergy(1.0 * CLHEP::GeV);
  pionBert.SetMaxEnergy(12.0 * CLHEP::GeV);
  std::fprintf(f, "qbbc_pion_minE_MeV,%.17g\n", pionBert.GetMinEnergy() / MeV);
  std::fprintf(f, "qbbc_pion_maxE_MeV,%.17g\n", pionBert.GetMaxEnergy() / MeV);

  G4CascadeInterface kaonBert;   // G4HadronicBuilder::BuildFTFP_BERT, bert = true
  kaonBert.SetMaxEnergy(hp->GetMaxEnergyTransitionFTF_Cascade());
  std::fprintf(f, "qbbc_kaon_hyperon_minE_MeV,%.17g\n", kaonBert.GetMinEnergy() / MeV);
  std::fprintf(f, "qbbc_kaon_hyperon_maxE_MeV,%.17g\n", kaonBert.GetMaxEnergy() / MeV);

  // The interface's own constants: the retry limit and the two conservation tolerances it
  // installs on itself and on G4CascadeCheckBalance.
  std::fprintf(f, "epsilon_energy_rel,%.17g\n", 0.05);       // SetEnergyMomentumCheckLevels
  std::fprintf(f, "epsilon_energy_abs_MeV,%.17g\n", 10.0);
  std::fprintf(f, "balance_rel,%.17g\n", 0.05);              // balance->setLimits
  std::fprintf(f, "balance_abs_GeV,%.17g\n", 10.0 / 1000.0);
  std::fprintf(f, "maximumTries,%d\n", 20);

  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// bertini_particles.csv - the type table, through the two public static accessors.
// ---------------------------------------------------------------------------------------------
void dump_particles() {
  FILE* f = std::fopen("bertini_particles.csv", "w");
  if (!f) return;
  std::fprintf(f, "code,name,mass_GeV,strangeness,baryon,is_photon,is_muon,is_electron,"
                  "is_neutrino,is_pion,is_nucleon,is_antinucleon,is_quasideuteron,is_hyperon\n");

  static const int codes[] = {
      nuclei, proton, neutron, pionPlus, pionMinus, pionZero, photon,
      kaonPlus, kaonMinus, kaonZero, kaonZeroBar,
      lambda, sigmaPlus, sigmaZero, sigmaMinus, xiZero, xiMinus, omegaMinus,
      deuteron, triton, He3, alpha,
      antiProton, antiNeutron, antiDeuteron, antiTriton, antiHe3, antiAlpha,
      diproton, unboundPN, dineutron,
      electronNu, muonNu, tauNu, antiElectronNu, antiMuonNu, antiTauNu,
      electron, muonMinus, tauMinus, positron, muonPlus, tauPlus};

  for (int code : codes) {
    std::fprintf(f, "%d,%s,%.17g,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", code,
                 G4InuclParticleNames::nameShort(code),
                 G4InuclElementaryParticle::getParticleMass(code),
                 G4InuclElementaryParticle::getStrangeness(code),
                 G4InuclParticleNames::baryon(code),
                 G4InuclParticleNames::isPhoton(code) ? 1 : 0,
                 G4InuclParticleNames::isMuon(code) ? 1 : 0,
                 G4InuclParticleNames::isElectron(code) ? 1 : 0,
                 G4InuclParticleNames::isNeutrino(code) ? 1 : 0,
                 G4InuclParticleNames::pion(code) ? 1 : 0,
                 G4InuclParticleNames::nucleon(code) ? 1 : 0,
                 G4InuclParticleNames::antinucleon(code) ? 1 : 0,
                 G4InuclParticleNames::quasi_deutron(code) ? 1 : 0,
                 G4InuclParticleNames::hyperon(code) ? 1 : 0);
  }
  std::fclose(f);

  // G4NucleiModel::useQuasiDeuteron, the whole truth table: every projectile type against the
  // three dibaryons and the "any" case. It is a static, so no model is needed.
  FILE* q = std::fopen("bertini_quasideuteron.csv", "w");
  if (!q) return;
  std::fprintf(q, "ptype,qdtype,allowed\n");
  static const int qd[] = {0, pp, pn, nn};
  for (int code : codes) {
    for (int d : qd) {
      std::fprintf(q, "%d,%d,%d\n", code, d,
                   G4NucleiModel::useQuasiDeuteron(code, d) ? 1 : 0);
    }
  }
  std::fclose(q);
}

// ---------------------------------------------------------------------------------------------
// bertini_nuclei.csv - G4NucleiModel's zone structure, exactly.
//
// Thirteen nuclei chosen to cover every branch of fillZoneRadii and fillZoneVolumes: A < 5 is a
// single-zone ball with radiusForSmall (and the alpha's own radScaleAlpha), 5 <= A < 12 is a
// three-zone Gaussian, 12 <= A < 100 is a three-zone Woods-Saxon and A >= 100 is a six-zone
// Woods-Saxon. The two integrals are adaptive trapezoid loops with a 1e-3 relative convergence
// test, so their results depend on the iteration count as well as the integrand - which is
// exactly the kind of thing a transcription gets subtly wrong and an exact oracle catches.
// ---------------------------------------------------------------------------------------------
void dump_nuclei() {
  FILE* f = std::fopen("bertini_nuclei.csv", "w");
  if (!f) return;
  std::fprintf(f, "A,Z,zones,nuclei_radius,nuclei_volume,be_proton_GeV,be_neutron_GeV,"
                  "zone,zone_radius,zone_volume,dens_p,dens_n,pf_p,pf_n,vp_p,vp_n,"
                  "fermi_kin_p,fermi_kin_n,vp_pion,vp_kaon,vp_hyperon\n");

  struct AZ { int a, z; };
  static const AZ cases[] = {
      {1, 1}, {2, 1}, {3, 2}, {4, 2},          // A < 5: single-zone ball
      {6, 3}, {9, 4}, {11, 5},                 // 5 <= A < 12: Gaussian, three zones
      {12, 6}, {16, 8}, {27, 13}, {56, 26},    // 12 <= A < 100: Woods-Saxon, three zones
      {100, 42}, {207, 82}};                   // A >= 100: Woods-Saxon, six zones

  for (const AZ& c : cases) {
    G4NucleiModel model;
    model.generateModel(c.a, c.z);

    const double dm = G4InuclSpecialFunctions::bindingEnergy(c.a, c.z);
    const double bep = std::fabs(G4InuclSpecialFunctions::bindingEnergy(c.a - 1, c.z - 1) - dm) / GeV;
    const double ben = std::fabs(G4InuclSpecialFunctions::bindingEnergy(c.a - 1, c.z) - dm) / GeV;

    const int nz = model.getNumberOfZones();
    for (int iz = 0; iz < nz; ++iz) {
      std::fprintf(f,
                   "%d,%d,%d,%.17g,%.17g,%.17g,%.17g,"
                   "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                   "%.17g,%.17g,%.17g\n",
                   c.a, c.z, nz, model.getRadius(), model.getVolume(nz), bep, ben,
                   iz, model.getRadius(iz), model.getVolume(iz),
                   model.getDensity(proton, iz), model.getDensity(neutron, iz),
                   model.getFermiMomentum(proton, iz), model.getFermiMomentum(neutron, iz),
                   model.getPotential(proton, iz), model.getPotential(neutron, iz),
                   model.getFermiKinetic(proton, iz), model.getFermiKinetic(neutron, iz),
                   model.getPotential(pionPlus, iz), model.getPotential(kaonPlus, iz),
                   model.getPotential(lambda, iz));
    }
  }
  std::fclose(f);

  // getZone(r) and getPotential's type folding, separately: the first is the radius-to-zone
  // search every propagation step uses, the second is a five-way collapse of the type code
  // (p, n, pi/other, K, Y) with photons and leptons at zero, written as three ifs.
  FILE* z = std::fopen("bertini_zonelookup.csv", "w");
  if (!z) return;
  std::fprintf(z, "A,Z,r,zone\n");
  static const AZ zcases[] = {{4, 2}, {9, 4}, {27, 13}, {207, 82}};
  for (const AZ& c : zcases) {
    G4NucleiModel model;
    model.generateModel(c.a, c.z);
    const int nz = model.getNumberOfZones();
    for (int iz = -1; iz <= nz; ++iz) {
      const double rz = model.getRadius(iz);
      // Just inside, exactly on, and just outside each boundary radius.
      const double probes[] = {rz * 0.999, rz, rz * 1.001};
      for (double r : probes) std::fprintf(z, "%d,%d,%.17g,%d\n", c.a, c.z, r, model.getZone(r));
    }
  }
  std::fclose(z);

  FILE* p = std::fopen("bertini_potfold.csv", "w");
  if (!p) return;
  std::fprintf(p, "A,Z,ptype,zone,potential\n");
  G4NucleiModel fe;
  fe.generateModel(56, 26);
  static const int ptypes[] = {proton, neutron, pionPlus, pionMinus, pionZero, photon,
                               kaonPlus, kaonMinus, kaonZero, kaonZeroBar, lambda,
                               sigmaPlus, sigmaZero, sigmaMinus, xiZero, xiMinus, omegaMinus,
                               muonMinus, electron, muonNu, diproton, unboundPN, dineutron};
  for (int t : ptypes) {
    for (int iz = 0; iz <= fe.getNumberOfZones(); ++iz) {
      std::fprintf(p, "56,26,%d,%d,%.17g\n", t, iz, fe.getPotential(t, iz));
    }
  }
  std::fclose(p);
}

// ---------------------------------------------------------------------------------------------
// bertini_nucxsec.csv - the two cross sections G4NucleiModel looks up.
// ---------------------------------------------------------------------------------------------
void dump_nucxsec() {
  FILE* f = std::fopen("bertini_nucxsec.csv", "w");
  if (!f) return;
  std::fprintf(f, "kind,ptype,target,ke_GeV,xsec\n");

  G4NucleiModel model;
  model.generateModel(56, 26);

  // Energies: the pion/nucleon bin edges, the kaon-hyperon bin edges, and a scatter of
  // in-between values including below the first edge and above the last.
  static const double kes[] = {
      0.0,    1e-5,  1e-4, 0.0005, 0.001, 0.002, 0.005, 0.0075, 0.01,  0.011, 0.013,
      0.0155, 0.018, 0.021, 0.024,  0.028, 0.032, 0.037, 0.042, 0.049, 0.056, 0.065,
      0.075,  0.0875, 0.1,  0.115,  0.13,  0.155, 0.18,  0.21,  0.24,  0.28,  0.3,
      0.32,   0.37,  0.42,  0.49,   0.56,  0.65,  0.75,  0.875, 0.99,  1.0,   1.15,
      1.23,   1.3,   1.47,  1.55,   1.8,   1.9,   2.1,   2.4,   2.8,   3.2,   3.7,
      4.2,    4.9,   5.6,   6.5,    7.5,   8.75,  10.0,  11.5,  13.0,  15.0,  18.0,
      21.0,   24.0,  28.0,  32.0,   40.0};

  static const int bullets[] = {proton, neutron, pionPlus, pionMinus, pionZero, photon,
                                kaonPlus, kaonMinus, kaonZero, kaonZeroBar, lambda,
                                sigmaPlus, sigmaZero, sigmaMinus, xiZero, xiMinus, omegaMinus,
                                muonMinus};
  static const int targets[] = {proton, neutron};

  for (int b : bullets) {
    for (int t : targets) {
      if (!G4CascadeChannelTables::GetTable(b * t)) continue;   // no table, model refuses
      for (double ke : kes) {
        std::fprintf(f, "total,%d,%d,%.17g,%.17g\n", b, t, ke, model.totalCrossSection(ke, b * t));
      }
    }
    if (G4NucleiModel::useQuasiDeuteron(b)) {
      for (double ke : kes) {
        std::fprintf(f, "absorption,%d,0,%.17g,%.17g\n", b, ke, model.absorptionCrossSection(ke, b));
      }
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// bertini_channels.csv - getCrossSection and getCrossSectionSum through the public interface.
//
// The grid is every bin edge of both scales plus the midpoint of every interval plus points
// below the first edge and above the last, because G4CascadeInterpolator EXTRAPOLATES outside
// its range by default (doExtrapolation = true) rather than clamping, and the NN/NP/PP channels
// replace the whole 0-10 MeV bin with Stepanov's closed form - which their findCrossSection
// override selects by comparing the ARRAY CONTENTS against `tot` and against `crossSections[0]`,
// not by comparing pointers.
// ---------------------------------------------------------------------------------------------
void dump_channels() {
  FILE* f = std::fopen("bertini_channels.csv", "w");
  if (!f) return;
  std::fprintf(f, "initial_state,ke_GeV,xsec,xsec_sum\n");

  static const double kes[] = {
      -0.001, 0.0,   1e-6,  1e-5,  5e-5,  9.4e-5, 1e-4,  0.0005, 0.0009, 0.001,
      0.0011, 0.002, 0.005, 0.0075, 0.0095, 0.01,  0.0105, 0.011, 0.013, 0.0155,
      0.018,  0.02,  0.021, 0.024, 0.028, 0.03,  0.032, 0.037, 0.04,  0.042,
      0.049,  0.05,  0.056, 0.06,  0.065, 0.075, 0.08,  0.0875, 0.09, 0.1,
      0.105,  0.11,  0.115, 0.125, 0.13,  0.14,  0.15,  0.155, 0.175, 0.18,
      0.2,    0.21,  0.21,  0.24,  0.25,  0.28,  0.3,   0.32,  0.35,  0.36,
      0.37,   0.4,   0.42,  0.45,  0.49,  0.5,   0.53,  0.56,  0.6,   0.65,
      0.67,   0.7,   0.75,  0.8,   0.875, 0.9,   0.95,  0.99,  1.0,   1.1,
      1.15,   1.23,  1.3,   1.35,  1.47,  1.5,   1.55,  1.68,  1.8,   1.9,
      2.0,    2.1,   2.2,   2.4,   2.5,   2.8,   3.0,   3.2,   3.5,   3.7,
      4.0,    4.2,   4.5,   4.9,   5.0,   5.5,   5.6,   6.0,   6.5,   7.0,
      7.5,    8.0,   8.5,   8.75,  9.0,   9.5,   10.0,  11.5,  13.0,  15.0,
      15.5,   18.0,  21.0,  24.0,  28.0,  32.0,  40.0,  64.0};

  // Every initial state G4CascadeChannelTables registers, as the product of two type codes.
  static const int states[] = {
      gam * neu, gam * pro, k0 * neu,  k0 * pro,  k0b * neu, k0b * pro,
      kmi * neu, kmi * pro, kpl * neu, kpl * pro, lam * neu, lam * pro,
      neu * neu, neu * pro, pi0 * neu, pi0 * pro, pim * neu, pim * pro,
      pip * neu, pip * pro, pro * pro, s0 * neu,  s0 * pro,  sm * neu,
      sm * pro,  sp * neu,  sp * pro,  xi0 * neu, xi0 * pro, xim * neu,
      xim * pro, om * neu,  om * pro,  mum * pro};

  for (int is : states) {
    const G4CascadeChannel* ch = G4CascadeChannelTables::GetTable(is);
    if (!ch) continue;
    for (double ke : kes) {
      std::fprintf(f, "%d,%.17g,%.17g,%.17g\n", is, ke, ch->getCrossSection(ke),
                   ch->getCrossSectionSum(ke));
    }
  }
  std::fclose(f);

  // getMultiplicity and getOutgoingParticleTypes under the cycle, so the multiplicity table
  // and the final-state selection are deterministic functions of (state, ke, phase). The draw
  // count is dumped: getMultiplicity consumes one extra deviate exactly when the channel has a
  // measured inclusive cross section (tot != sum), and that is the only observable difference
  // between the two constructors.
  CycleEngine* eng = new CycleEngine;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);

  FILE* s = std::fopen("bertini_chsample.csv", "w");
  if (s) {
    std::fprintf(s, "initial_state,ke_GeV,phase,mult,draws_mult,fs_mult,fs_draws,fs_codes\n");
    static const double skes[] = {0.005, 0.05, 0.2, 0.5, 1.0, 2.0, 3.5, 6.0, 12.0, 20.0};
    for (int is : states) {
      const G4CascadeChannel* ch = G4CascadeChannelTables::GetTable(is);
      if (!ch) continue;
      for (double ke : skes) {
        for (int phase = 0; phase < 8; ++phase) {
          eng->reset(phase);
          const int mult = ch->getMultiplicity(ke);
          const int nmult = eng->draws();

          // Ask for every multiplicity the channel can produce, not only the sampled one, so
          // that each multiplicity's final-state block is reached.
          //
          // EXCEPT mu- p above multiplicity 2, which reads out of bounds in 11.1.1 and is not
          // exercised here for that reason. G4CascadeSampler::findFinalStateIndex returns the
          // ABSOLUTE index `start` when a multiplicity has one channel (`stop-start <= 1`) and
          // a RELATIVE index from sampleFlat() otherwise, and getOutgoingParticleTypes then
          // indexes the per-multiplicity array `xMbfs[channel]` with it. The two agree only
          // when start == 0, i.e. for multiplicity 2. G4CascadeMuMinusPChannel is
          // G4CascadeData<30,1,1,1,1,1,1,1,1> - every multiplicity has exactly one channel -
          // so for m = 3..9 it reads xMbfs[m-2] out of an [1][m] array. An oracle must not
          // record undefined behaviour, so the rows are absent and the port refuses the same
          // case by name. docs/RISK.md V119.
          const int max_m = (is == mum * pro) ? 2 : 9;
          for (int m = 2; m <= max_m; ++m) {
            std::vector<G4int> kinds;
            eng->reset(phase);
            ch->getOutgoingParticleTypes(kinds, m, ke);
            const int nfs = eng->draws();
            if (kinds.empty()) continue;
            std::fprintf(s, "%d,%.17g,%d,%d,%d,%d,%d,", is, ke, phase, mult, nmult, m, nfs);
            for (std::size_t i = 0; i < kinds.size(); ++i)
              std::fprintf(s, "%s%d", (i ? " " : ""), kinds[i]);
            std::fprintf(s, "\n");
          }
        }
      }
    }
    std::fclose(s);
  }

  CLHEP::HepRandom::setTheEngine(saved);
  delete eng;
}

// ---------------------------------------------------------------------------------------------
// bertini_chtables.csv - the complete contents of every G4CascadeData object, through the one
// public door this Geant4 build leaves open.
//
// This is what tools/extract_bertini_channels.pl is measured against: the script parses the .cc
// source text, this reads what the compiler built from it, and they have to agree entry for
// entry - including the four arrays G4CascadeData::initialize COMPUTES rather than reads
// (index[], multiplicities[][], sum[] and inelastic[]).
//
// **Why it is parsed out of a print stream instead of read as a member.** The obvious version of
// this function is `G4CascadeNPChannelData::data.crossSections[i][k]` for each of the 34 data
// structs. They are public static members of public structs, the headers are installed, and it
// compiles - and then the link fails with 34 unresolved externals, because this Windows Geant4
// does not export a class static from its DLLs. (That is the second time in this port: PORTED.md
// 2.1.3 records G4CameronTruranHilfShellCorrections' private static table failing the same way,
// and the same fix applied - reach the data through a function the library does export.)
//
// `G4CascadeChannel::printTable(std::ostream&)` is that function: a public virtual on the
// abstract base that `G4CascadeChannelTables::GetTable` returns, and it writes out the bin
// edges, `tot`, `sum`, `inelastic`, every multiplicity's summed cross section, and then every
// individual channel's final-state list and cross section. Everything the member access would
// have given. The stream is handed `setprecision(17)`, so each value round-trips exactly -
// printXsec applies `setw(6)` but never touches precision.
//
// The parser is a five-state machine over the lines, and the state is chosen by the header text
// G4CascadeData::print emits. The one thing it must not do is guess the value count: NE is read
// from the interpolator's own header line (`G4CascadeInterpolator<30>`), and every block's
// length is asserted against it.
// ---------------------------------------------------------------------------------------------

/// Invert G4InuclParticleNames::nameShort, which is what the final-state lists are printed as.
int code_from_short_name(const std::string& s) {
  static const int codes[] = {
      nuclei, proton, neutron, pionPlus, pionMinus, pionZero, photon,
      kaonPlus, kaonMinus, kaonZero, kaonZeroBar,
      lambda, sigmaPlus, sigmaZero, sigmaMinus, xiZero, xiMinus, omegaMinus,
      deuteron, triton, He3, alpha,
      antiProton, antiNeutron, antiDeuteron, antiTriton, antiHe3, antiAlpha,
      diproton, unboundPN, dineutron,
      electronNu, muonNu, tauNu, antiElectronNu, antiMuonNu, antiTauNu,
      electron, muonMinus, tauMinus, positron, muonPlus, tauPlus};
  for (int c : codes) {
    if (s == G4InuclParticleNames::nameShort(c)) return c;
  }
  return -9999;   // written to the CSV as-is, so an unmapped name is a visible failure
}

std::vector<std::string> split_ws(const std::string& s) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    std::size_t j = i;
    while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
    if (j > i) out.push_back(s.substr(i, j - i));
    i = j;
  }
  return out;
}

void dump_chtables() {
  FILE* f = std::fopen("bertini_chtables.csv", "w");
  FILE* g = std::fopen("bertini_chfinalstates.csv", "w");
  FILE* e = std::fopen("bertini_chbins.csv", "w");
  if (!f || !g || !e) { if (f) std::fclose(f); if (g) std::fclose(g); if (e) std::fclose(e); return; }
  std::fprintf(f, "initial_state,channel,block,mult,ebin,chan,value\n");
  std::fprintf(g, "initial_state,channel,mult,index,abs_index,codes\n");
  std::fprintf(e, "initial_state,channel,nbins,bin,edge\n");

  static const int states[] = {
      gam * neu, gam * pro, k0 * neu,  k0 * pro,  k0b * neu, k0b * pro,
      kmi * neu, kmi * pro, kpl * neu, kpl * pro, lam * neu, lam * pro,
      neu * neu, neu * pro, pi0 * neu, pi0 * pro, pim * neu, pim * pro,
      pip * neu, pip * pro, pro * pro, s0 * neu,  s0 * pro,  sm * neu,
      sm * pro,  sp * neu,  sp * pro,  xi0 * neu, xi0 * pro, xim * neu,
      xim * pro, om * neu,  om * pro,  mum * pro};

  int nchan_total = 0, nvalue_total = 0, nfs_total = 0;

  for (int is : states) {
    const G4CascadeChannel* ch = G4CascadeChannelTables::GetTable(is);
    if (!ch) continue;
    ++nchan_total;

    std::ostringstream oss;
    oss << std::setprecision(17);
    ch->printTable(oss);
    std::istringstream iss(oss.str());

    std::string name = "?";
    int ne = 0;
    // What the numbers on the following lines belong to.
    enum Block { kNone, kBins, kTot, kSum, kInel, kMult, kChan } block = kNone;
    int block_mult = -1;      // multiplicity, for kMult and kChan
    int block_index = -1;     // absolute channel index, for kChan
    std::vector<double> buf;

    auto flush = [&](void) {
      if (block == kNone || buf.empty()) { buf.clear(); return; }
      if (block == kBins) {
        std::fprintf(e, "%d,%s,%d,-1,-1\n", is, name.c_str(), static_cast<int>(buf.size()));
        for (std::size_t k = 0; k < buf.size(); ++k)
          std::fprintf(e, "%d,%s,%d,%d,%.17g\n", is, name.c_str(),
                       static_cast<int>(buf.size()), static_cast<int>(k), buf[k]);
      } else {
        const char* bn = (block == kTot) ? "tot" : (block == kSum) ? "sum"
                       : (block == kInel) ? "inelastic" : (block == kMult) ? "mult" : "chan";
        for (std::size_t k = 0; k < buf.size(); ++k) {
          std::fprintf(f, "%d,%s,%s,%d,%d,%d,%.17g\n", is, name.c_str(), bn, block_mult,
                       static_cast<int>(k), block_index, buf[k]);
          ++nvalue_total;
        }
        if (ne > 0 && static_cast<int>(buf.size()) != ne) {
          std::fprintf(f, "%d,%s,PARSE_ERROR,%d,%d,%d,%d\n", is, name.c_str(), block_mult,
                       -1, block_index, static_cast<int>(buf.size()));
        }
      }
      buf.clear();
    };

    std::string line;
    while (std::getline(iss, line)) {
      if (line.find(" ---------- ") != std::string::npos) {
        flush();
        const std::size_t a = line.find(" ---------- ") + 12;
        const std::size_t b = line.find(" ----------", a);
        name = line.substr(a, b - a);
        block = kNone;
        continue;
      }
      if (line.find("G4CascadeInterpolator<") != std::string::npos) {
        flush();
        const std::size_t a = line.find('<') + 1;
        ne = std::atoi(line.substr(a).c_str());
        block = kBins;
        continue;
      }
      if (line.find("Total cross section:") != std::string::npos) { flush(); block = kTot; block_mult = -1; block_index = -1; continue; }
      if (line.find("Summed cross section:") != std::string::npos) { flush(); block = kSum; block_mult = -1; block_index = -1; continue; }
      if (line.find("Inelastic cross section:") != std::string::npos) { flush(); block = kInel; block_mult = -1; block_index = -1; continue; }
      if (line.find("Individual channel cross sections") != std::string::npos) { flush(); block = kNone; continue; }
      if (line.find("Mulitplicity") != std::string::npos) {
        // " Mulitplicity M (indices A to B) summed cross section:"  (the typo is Geant4's)
        flush();
        const std::size_t a = line.find("Mulitplicity") + 12;
        block_mult = std::atoi(line.substr(a).c_str());
        const std::size_t ia = line.find("indices") + 7;
        const std::size_t it = line.find(" to ", ia);
        const int start = std::atoi(line.substr(ia, it - ia).c_str());
        const int stop = std::atoi(line.substr(it + 4).c_str());
        std::fprintf(f, "%d,%s,index,%d,-1,-1,%d\n", is, name.c_str(), block_mult, start);
        std::fprintf(f, "%d,%s,index_last,%d,-1,-1,%d\n", is, name.c_str(), block_mult, stop);
        block = kMult;
        block_index = -1;
        continue;
      }
      if (line.find("final state x") != std::string::npos) {
        // " final state xMbfs[i] :  a b c -- cross section [j]:"
        flush();
        const std::size_t a = line.find("final state x") + 13;
        const int m = std::atoi(line.substr(a).c_str());
        const std::size_t lb = line.find('[', a) + 1;
        const int ichan = std::atoi(line.substr(lb).c_str());
        const std::size_t colon = line.find(':', lb);
        const std::size_t dashes = line.find(" -- cross section [", colon);
        const int abs_index = std::atoi(line.substr(dashes + 19).c_str());
        const std::vector<std::string> names =
            split_ws(line.substr(colon + 1, dashes - colon - 1));
        std::fprintf(g, "%d,%s,%d,%d,%d,", is, name.c_str(), m, ichan, abs_index);
        for (std::size_t i = 0; i < names.size(); ++i) {
          std::fprintf(g, "%s%d", (i ? " " : ""), code_from_short_name(names[i]));
          ++nfs_total;
        }
        std::fprintf(g, "\n");
        if (static_cast<int>(names.size()) != m) {
          std::fprintf(g, "%d,%s,%d,%d,%d,PARSE_ERROR_%d\n", is, name.c_str(), m, ichan,
                       abs_index, static_cast<int>(names.size()));
        }
        block = kChan;
        block_mult = m;
        block_index = abs_index;
        continue;
      }
      if (line.find("------------------------------") != std::string::npos) { flush(); block = kNone; continue; }

      // Otherwise: a line of numbers belonging to the current block, or a blank line.
      const std::vector<std::string> tok = split_ws(line);
      bool numeric = !tok.empty();
      for (const std::string& t : tok) {
        char* end = nullptr;
        std::strtod(t.c_str(), &end);
        if (end == t.c_str() || *end != '\0') { numeric = false; break; }
      }
      if (!numeric) continue;
      for (const std::string& t : tok) buf.push_back(std::strtod(t.c_str(), nullptr));
    }
    flush();
  }

  std::fclose(f);
  std::fclose(g);
  std::fclose(e);
  std::printf("  bertini: %d channel tables, %d table values, %d final-state codes\n",
              nchan_total, nvalue_total, nfs_total);
}

// ---------------------------------------------------------------------------------------------
// bertini_angdst.csv and bertini_momdst.csv
// ---------------------------------------------------------------------------------------------
void dump_angdst() {
  // Which object ChooseDist picks. It is reachable only as a pointer, so the identity is
  // recorded by GetName() - which is the name string each subclass passes to its base.
  FILE* c = std::fopen("bertini_angchoice.csv", "w");
  if (c) {
    std::fprintf(c, "is,fs,kw,name\n");
    static const int hadrons[] = {pro, neu, pip, pim, pi0, gam, kpl, kmi, k0, k0b,
                                  lam, sp, s0, sm, xi0, xim, om, mum};
    static const int fss[] = {0, pro * pro, pro * neu, neu * neu, pro * pi0, neu * pi0,
                              neu * pip, pro * pim, pip * pro, pim * pro, pi0 * pro,
                              pi0 * neu, pim * neu, pip * neu, pro * kpl, neu * k0};
    for (int a : hadrons) {
      for (int b : hadrons) {
        for (int fs : fss) {
          for (int kw = 0; kw < 3; ++kw) {
            const G4VTwoBodyAngDst* d = G4TwoBodyAngularDist::GetDist(a * b, fs, kw);
            std::fprintf(c, "%d,%d,%d,%s\n", a * b, fs, kw, d ? d->GetName().c_str() : "none");
          }
        }
      }
    }
    std::fclose(c);
  }

  FILE* f = std::fopen("bertini_angdst.csv", "w");
  if (!f) return;
  std::fprintf(f, "name,ekin_GeV,pcm_GeV,phase,draws,costheta\n");

  G4GamP2NPipAngDst d0;   G4GamP2PPi0AngDst d1;   G4GammaNuclAngDst d2;
  G4HadNElastic1AngDst d3; G4HadNElastic2AngDst d4; G4NP2NPAngDst d5;
  G4NuclNuclAngDst d6;    G4PP2PPAngDst d7;       G4Pi0P2Pi0PAngDst d8;
  G4PiNInelasticAngDst d9; G4PimP2Pi0NAngDst d10; G4PimP2PimPAngDst d11;
  G4PipP2PipPAngDst d12;

  const G4VTwoBodyAngDst* dists[] = {&d0, &d1, &d2, &d3, &d4, &d5, &d6,
                                     &d7, &d8, &d9, &d10, &d11, &d12};

  // The three-body distributions take a particle type rather than a pcm, so they are dumped
  // through their own signature below.
  G4HadNucl3BodyAngDst h3;
  G4NuclNucl3BodyAngDst n3;

  static const double ekins[] = {0.0,   0.02,  0.031, 0.05,  0.09,  0.097, 0.11,  0.12,
                                 0.145, 0.1515, 0.169, 0.185, 0.2,  0.217, 0.24,  0.26,
                                 0.30,  0.31,  0.34,  0.35,  0.36,  0.40,  0.42,  0.44,
                                 0.45,  0.48,  0.50,  0.542, 0.55,  0.59,  0.603, 0.65,
                                 0.698, 0.70,  0.79,  0.793, 0.80,  0.802, 0.817, 0.873,
                                 0.902, 1.00,  1.05,  1.056, 1.162, 1.24,  1.238, 1.269,
                                 1.31,  1.34,  1.40,  1.48,  1.53,  1.575, 1.74,  1.77,
                                 1.825, 2.0,   2.24,  2.25,  2.45,  2.60,  2.75,  2.86,
                                 2.90,  3.5,   3.86,  4.25,  4.40,  5.0,   5.86,  5.90,
                                 6.15,  8.0,   10.0,  10.66, 12.0,  20.0};
  static const double pcms[] = {0.05, 0.2, 0.5, 1.0, 2.0};

  for (const G4VTwoBodyAngDst* d : dists) {
    for (double e : ekins) {
      for (double p : pcms) {
        for (int phase = 0; phase < 8; ++phase) {
          CycleEngine* eng = new CycleEngine;
          CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
          CLHEP::HepRandom::setTheEngine(eng);
          eng->reset(phase);
          const double ct = d->GetCosTheta(e, p);
          const int n = eng->draws();
          CLHEP::HepRandom::setTheEngine(saved);
          delete eng;
          std::fprintf(f, "%s,%.17g,%.17g,%d,%d,%.17g\n", d->GetName().c_str(), e, p, phase, n, ct);
        }
      }
    }
  }
  std::fclose(f);

  // The three-body angular and the four momentum distributions, by particle type.
  FILE* t = std::fopen("bertini_3bodydst.csv", "w");
  if (!t) return;
  std::fprintf(t, "kind,name,ptype,ekin_GeV,phase,draws,value\n");

  G4HadNucl3BodyMomDst m0; G4HadNucl4BodyMomDst m1;
  G4NuclNucl3BodyMomDst m2; G4NuclNucl4BodyMomDst m3;
  const G4VMultiBodyMomDst* moms[] = {&m0, &m1, &m2, &m3};

  static const int ptypes[] = {pro, neu, pip, pim, pi0, kpl, lam, gam};
  for (double e : ekins) {
    for (int pt : ptypes) {
      for (int phase = 0; phase < 8; ++phase) {
        for (const G4VMultiBodyMomDst* m : moms) {
          CycleEngine* eng = new CycleEngine;
          CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
          CLHEP::HepRandom::setTheEngine(eng);
          eng->reset(phase);
          const double v = m->GetMomentum(pt, e);
          const int n = eng->draws();
          CLHEP::HepRandom::setTheEngine(saved);
          delete eng;
          std::fprintf(t, "mom,%s,%d,%.17g,%d,%d,%.17g\n", m->GetName().c_str(), pt, e, phase, n, v);
        }
        // G4InuclParamAngDst::GetCosTheta(ptype, ekin) is the three-body angular form. It
        // rejects until Spow is in [0,1], up to 100 tries, so its draw count is the
        // interesting part.
        for (int which = 0; which < 2; ++which) {
          CycleEngine* eng = new CycleEngine;
          CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
          CLHEP::HepRandom::setTheEngine(eng);
          eng->reset(phase);
          const G4InuclParamAngDst* d = which ? static_cast<G4InuclParamAngDst*>(&n3)
                                              : static_cast<G4InuclParamAngDst*>(&h3);
          const double v = d->GetCosTheta(pt, e);
          const int n = eng->draws();
          CLHEP::HepRandom::setTheEngine(saved);
          delete eng;
          std::fprintf(t, "ang3,%s,%d,%.17g,%d,%d,%.17g\n", d->GetName().c_str(), pt, e, phase, n, v);
        }
      }
    }
  }
  std::fclose(t);

  // Which momentum distribution ChooseDist returns, including the use3BodyMom branch that the
  // dumped parameter says is off.
  FILE* mc = std::fopen("bertini_momchoice.csv", "w");
  if (!mc) return;
  std::fprintf(mc, "is,mult,name\n");
  static const int hh[] = {pro, neu, pip, pim, pi0, gam, kpl, kmi, lam, sm};
  for (int a : hh) {
    for (int b : hh) {
      for (int mult = 3; mult <= 9; ++mult) {
        const G4VMultiBodyMomDst* d = G4MultiBodyMomentumDist::GetDist(a * b, mult);
        std::fprintf(mc, "%d,%d,%s\n", a * b, mult, d ? d->GetName().c_str() : "none");
      }
    }
  }
  std::fclose(mc);
}

// ---------------------------------------------------------------------------------------------
// bertini_apply.csv - ApplyYourself, as QBBC configures it.
// ---------------------------------------------------------------------------------------------
void dump_apply() {
  FILE* f = std::fopen("bertini_apply.csv", "w");
  FILE* s = std::fopen("bertini_apply_species.csv", "w");
  if (!f || !s) { if (f) std::fclose(f); if (s) std::fclose(s); return; }
  std::fprintf(f, "projectile,ke_MeV,A,Z,events,mean_mult,mean_esum_MeV,mean_pz_MeV,"
                  "mean_de_MeV,mean_dp_MeV,max_abs_de_MeV,max_abs_dp_MeV,thrown\n");
  std::fprintf(s, "projectile,ke_MeV,A,Z,events,pdg,count,mean_ke_MeV,mean_ke2_MeV2,"
                  "mean_cos,mean_cos2\n");

  struct Case { const char* name; G4ParticleDefinition* pd; double ke_MeV; int a, z; };
  static const int kN = 2000;   // the port's campaigns run 20,000; the oracle is the cheap half

  std::vector<Case> cases;
  G4ParticleDefinition* pp = G4Proton::Proton();
  G4ParticleDefinition* nn = G4Neutron::Neutron();
  G4ParticleDefinition* pip_ = G4PionPlus::PionPlus();
  G4ParticleDefinition* pim_ = G4PionMinus::PionMinus();

  struct AZ { int a, z; };
  static const AZ targets[] = {{12, 6}, {16, 8}, {27, 13}, {56, 26}, {207, 82}};
  static const double nucleonKE[] = {1500., 3000., 5000.};
  static const double pionKE[] = {200., 1000., 3000., 8000.};

  for (const AZ& t : targets) {
    for (double ke : nucleonKE) {
      cases.push_back({"proton", pp, ke, t.a, t.z});
      cases.push_back({"neutron", nn, ke, t.a, t.z});
    }
    for (double ke : pionKE) {
      cases.push_back({"pi+", pip_, ke, t.a, t.z});
      cases.push_back({"pi-", pim_, ke, t.a, t.z});
    }
  }

  // QBBC's configuration: PreCompound de-excitation, as G4HadronInelasticQBBC sets it for
  // nucleons and pions.
  G4CascadeInterface bert;
  bert.usePreCompoundDeexcitation();

  for (const Case& c : cases) {
    CLHEP::HepRandom::setTheSeed(20260911);

    double sum_mult = 0., sum_e = 0., sum_pz = 0., sum_de = 0., sum_dp = 0.;
    double max_de = 0., max_dp = 0.;
    long thrown = 0;
    std::map<int, long> count;
    std::map<int, double> ke1, ke2, c1, c2;

    for (int ev = 0; ev < kN; ++ev) {
      G4DynamicParticle dp(c.pd, G4ThreeVector(0., 0., 1.), c.ke_MeV);
      G4HadProjectile proj(dp);
      G4Nucleus nucleus(c.a, c.z);

      // G4CascadeInterface::throwNonConservationFailure() ends the job with a
      // G4HadronicException after twenty failed attempts. That is a real outcome of the model
      // and the oracle must not die of it, so it is caught and counted: `thrown` in the output
      // row is the number of events Geant4 abandoned, and the port's equivalent refusal is
      // compared against it rather than against zero.
      G4HadFinalState* fs = nullptr;
      try {
        fs = bert.ApplyYourself(proj, nucleus);
      } catch (...) {
        ++thrown;
        continue;
      }
      if (!fs) continue;

      const int nsec = fs->GetNumberOfSecondaries();
      sum_mult += nsec;

      double etot = 0., pz = 0.;
      for (int i = 0; i < nsec; ++i) {
        G4DynamicParticle* d = fs->GetSecondary(i)->GetParticle();
        const int pdg = d->GetDefinition()->GetPDGEncoding();
        const double k = d->GetKineticEnergy();
        const double cz = d->GetMomentumDirection().z();
        ++count[pdg];
        ke1[pdg] += k;
        ke2[pdg] += k * k;
        c1[pdg] += cz;
        c2[pdg] += cz * cz;
        etot += d->GetTotalEnergy();
        pz += d->GetMomentum().z();
      }
      etot += fs->GetLocalEnergyDeposit();
      sum_e += etot;
      sum_pz += pz;

      // Balance against the initial state: projectile plus target nucleus at rest.
      const double m_target = G4NucleiProperties::GetNuclearMass(c.a, c.z);
      const double e_init = proj.GetTotalEnergy() + m_target;
      const double p_init = proj.GetTotalMomentum();
      const double de = etot - e_init;
      const double dpz = pz - p_init;
      sum_de += de;
      sum_dp += dpz;
      if (std::fabs(de) > std::fabs(max_de)) max_de = de;
      if (std::fabs(dpz) > std::fabs(max_dp)) max_dp = dpz;

      fs->Clear();
    }

    std::fprintf(f, "%s,%.17g,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%ld\n",
                 c.name, c.ke_MeV, c.a, c.z, kN, sum_mult / kN, sum_e / kN, sum_pz / kN,
                 sum_de / kN, sum_dp / kN, max_de, max_dp, thrown);
    for (const auto& kv : count) {
      const long n = kv.second;
      std::fprintf(s, "%s,%.17g,%d,%d,%d,%d,%ld,%.17g,%.17g,%.17g,%.17g\n", c.name, c.ke_MeV,
                   c.a, c.z, kN, kv.first, n, ke1[kv.first] / n, ke2[kv.first] / n,
                   c1[kv.first] / n, c2[kv.first] / n);
    }
  }
  std::fclose(f);
  std::fclose(s);
}

// ---------------------------------------------------------------------------------------------
// bertini_epcollide.csv - G4ElementaryParticleCollider::collide, exact under the cycle.
//
// `collide` and `setNucleusState` are the class's only public members, which is enough: under a
// prescribed uniform cycle the whole chain - multiplicity, final-state selection, momentum
// moduli, the three angular generators, the rotations and the boost back to the lab - becomes a
// deterministic function of (types, bullet momentum, phase). The draw count goes with it,
// because the number of deviates a final state costs is what a transcription of
// G4CascadeFinalStateAlgorithm's retry loops gets wrong first.
//
// Pairs are chosen to reach every arm of collide(): nucleon-nucleon (three charge states), all
// three pion charges on both nucleons, gamma and kaon on both, one hyperon, and a pion and a
// photon on each of the three DIBARYONS - which is the absorption arm and the only one that
// does not go through a channel table.
// ---------------------------------------------------------------------------------------------
//
// **The file is written twice, with `usePhaseSpace` off and then on.** That flag replaces the
// whole multi-body generator - the INUCL momentum parametrisations, the fixed-theta directions
// and the three-body triangle - with G4CascadeFinalStateAlgorithm::FillUsingKopylov, a chain of
// two-body decays. It is 0 in the install, so with one pass the Kopylov transcription would be
// a reading of the source that nothing measures. `G4CascadeParamMessenger` exposes it as
// `/process/had/cascade/usePhaseSpace` and its SetNewValue calls `Initialize()`, so the flag can
// be turned on for a second pass and turned off again; the `ps` column says which pass a row
// belongs to. Two-body final states are unaffected by it, and the file shows that too.
//
// **The second pass is driven by LcgEngine, not CycleEngine, and that is a property of Kopylov
// rather than a convenience.** `BetaKopylov` is an unbounded rejection loop that draws two
// deviates per trial, so a period-8 cycle gives it four distinct trials in total; at the
// multiplicities phase space reaches, no pair among the four is ever accepted and the loop does
// not terminate. Every other sampler in this package is bounded - `GenerateCosTheta` gives up
// after ten tries, `FillMagnitudes` after ten, `generateSCMfinalState` after ten - which is why
// the cycle works everywhere else. `ps` says which engine a row was produced with: 0 is
// CycleEngine, 1 is LcgEngine, and tests/test_bertini_collide.cu carries both.
// ---------------------------------------------------------------------------------------------
void dump_epcollide_pass(FILE* f, int ps);

void dump_epcollide() {
  FILE* f = std::fopen("bertini_epcollide.csv", "w");
  if (!f) return;
  std::fprintf(f, "ps,tm,type1,type2,plab_GeV,phase,draws,n,i,kind,px,py,pz,e\n");

  dump_epcollide_pass(f, 0);


  G4UImanager* ui = G4UImanager::GetUIpointer();
  ui->ApplyCommand("/process/had/cascade/usePhaseSpace true");
  if (G4CascadeParameters::usePhaseSpace()) {
    dump_epcollide_pass(f, 1);
  } else {
    std::fprintf(stderr, "dump_epcollide: could not turn usePhaseSpace on - no ps=1 rows\n");
  }
  ui->ApplyCommand("/process/had/cascade/usePhaseSpace false");

  std::fclose(f);
}

void dump_epcollide_pass(FILE* f, int ps) {
  struct Pair { int t1, t2; };
  static const Pair pairs[] = {
      {proton, proton}, {proton, neutron}, {neutron, neutron},
      {pionPlus, proton}, {pionMinus, proton}, {pionZero, proton},
      {pionPlus, neutron}, {pionMinus, neutron}, {pionZero, neutron},
      {photon, proton}, {photon, neutron},
      {kaonPlus, proton}, {kaonMinus, proton}, {kaonZero, neutron},
      {lambda, proton}, {sigmaMinus, proton}, {xiMinus, neutron}, {omegaMinus, proton},
      // The absorption arm: pion or photon on a dibaryon.
      {pionPlus, unboundPN}, {pionMinus, diproton}, {pionZero, unboundPN},
      {pionZero, diproton}, {pionZero, dineutron}, {pionPlus, dineutron},
      {pionMinus, unboundPN}, {photon, diproton}, {photon, unboundPN},
      {photon, dineutron}};

  // Lab momenta of the bullet, GeV/c. Covers the sub-threshold region, the Delta, the
  // multi-pion rise and the top of Bertini's range.
  static const double plabs[] = {0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0, 8.0, 12.0};

  // **The target is run twice: at rest, and with a Fermi momentum.** A head-on collision along
  // z against a stationary target has its CM velocity parallel to the SCM axis, so
  // `G4LorentzConvertor::degenerated` is TRUE and `rotate(mom)` returns its argument unchanged.
  // With only the first row, every two-body final state in this file is built about +z and the
  // rotation is never applied - which a perturbation campaign found by DELETING the call to
  // `rotate` in the port and watching 86,938 comparisons still agree. In the cascade the struck
  // nucleon always carries a Fermi momentum (`generateNucleon` samples one), so the second row
  // is the representative case and the first is the degenerate one; both are kept, because the
  // degenerate branch is a branch too. 0.2 GeV/c, off every axis, is a typical Fermi momentum.
  static const G4ThreeVector tmoms[] = {G4ThreeVector(0., 0., 0.),
                                        G4ThreeVector(0.15, -0.08, 0.11)};

  CycleEngine cyc;
  LcgEngine lcg;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(
      (ps == 0) ? static_cast<CLHEP::HepRandomEngine*>(&cyc)
                : static_cast<CLHEP::HepRandomEngine*>(&lcg));

  G4ElementaryParticleCollider coll;
  coll.setNucleusState(56, 26);     // for the pi-N absorption recoil, which is off by default
  G4CollisionOutput out;

  for (const Pair& p : pairs) {
    const double m1 = G4InuclElementaryParticle::getParticleMass(p.t1);
    const double m2 = G4InuclElementaryParticle::getParticleMass(p.t2);
    for (double plab : plabs) {
      for (int tm = 0; tm < 2; ++tm) {
        const G4ThreeVector& tp = tmoms[tm];
        for (int phase = 0; phase < 8; ++phase) {
          G4InuclElementaryParticle bullet(
              G4LorentzVector(0., 0., plab, std::sqrt(plab * plab + m1 * m1)), p.t1);
          G4LorentzVector tmom;
          tmom.setVectM(tp, m2);
          G4InuclElementaryParticle target(tmom, p.t2);
          out.reset();
          if (ps == 0) { cyc.reset(phase); } else { lcg.reset(phase); }
          coll.collide(&bullet, &target, out);
          const long long n = (ps == 0) ? cyc.draws() : lcg.draws();
          const std::vector<G4InuclElementaryParticle>& prods = out.getOutgoingParticles();
          if (prods.empty()) {
            std::fprintf(f, "%d,%d,%d,%d,%.17g,%d,%lld,0,-1,0,0,0,0,0\n", ps, tm, p.t1, p.t2,
                         plab, phase, n);
            continue;
          }
          for (std::size_t i = 0; i < prods.size(); ++i) {
            const G4LorentzVector& m = prods[i].getMomentum();
            std::fprintf(f, "%d,%d,%d,%d,%.17g,%d,%lld,%d,%d,%d,%.17g,%.17g,%.17g,%.17g\n", ps,
                         tm, p.t1, p.t2, plab, phase, n, static_cast<int>(prods.size()),
                         static_cast<int>(i), prods[i].type(), m.x(), m.y(), m.z(), m.e());
          }
        }
      }
    }
  }

  CLHEP::HepRandom::setTheEngine(saved);
}

// ---------------------------------------------------------------------------------------------
// bertini_initcascad.csv and bertini_fate.csv - G4NucleiModel's cascade half, exact.
//
// Both entry points are public members of G4NucleiModel and neither needs a cascader around it,
// so the same trick the collider dump uses works here: install the eight-value cycle engine and
// the whole chain - the entry point on the surface, the trajectory sampler for a photon, the
// partner list, the interaction lengths, the collision, Pauli blocking, the boundary transition
// - becomes a deterministic function of (nucleus, particle, position, direction, generation,
// phase), with the draw count beside the answer.
//
// **`generateParticleFate` mutates the model**, so every case calls `reset()` first. Without it
// the proton and neutron census carried by `protonNumberCurrent`/`neutronNumberCurrent` would
// drift down the grid and no row could be reproduced on its own; `reset()` also clears
// `collisionPts`, which is what `passTrailing` reads.
//
// **The grid has to move the particle off the axes and off the surface.** docs/RISK.md V124 is
// about exactly this file's failure mode one level up: a projectile along +z at a target at rest
// switched off `G4LorentzConvertor::rotate` for 2,688 collider cases and nothing noticed. Here
// the symmetries to break are three: a position on a coordinate axis (makes `pos.dot(mom)` a
// single component and `choosePointAlongTraj`'s rotation axis degenerate), a momentum parallel
// or antiparallel to the position (makes `prang < 1e-6` fire, which is the radial-incidence
// shortcut, and makes `pperp2` zero in `boundaryTransition` so the `qv+qperp` arm can never be
// the one that runs), and a generation of 0 for everything (which exempts every particle from
// the young-secondary veto through the `current_path < 1000` sentinel). So: four radii from deep
// inside to just under the surface along a non-axial unit vector, three directions of which one
// IS radial so that the shortcut is measured too, and both generations.
// ---------------------------------------------------------------------------------------------
void dump_initcascad() {
  FILE* f = std::fopen("bertini_initcascad.csv", "w");
  if (!f) return;
  std::fprintf(f, "a,z,type,plab,phase,draws,posx,posy,posz,zone,cpath,gen,px,py,pz,e\n");

  struct AZ { int a, z; };
  static const AZ nuclei[] = {{12, 6}, {27, 13}, {56, 26}, {207, 82}};
  static const int types[] = {proton, neutron, pionPlus, pionMinus, pionZero, photon};
  // The last entry is BELOW G4NucleiModel::small (1e-9 GeV): that is the capture-at-rest branch,
  // which starts the particle one zone INSIDE the surface instead of outside it.
  static const double plabs[] = {0.2, 1.0, 3.0, 1.0e-12};

  CycleEngine cyc;
  ConstEngine con;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();

  for (const AZ& n : nuclei) {
    G4NucleiModel model(n.a, n.z);
    for (int type : types) {
      const double m = G4InuclElementaryParticle::getParticleMass(type);
      for (double plab : plabs) {
        // Phases 0-7 are the eight-value cycle; 8 and 9 are the constant engine, which is the
        // only way to put a photon at radial incidence - see the comment on ConstEngine.
        for (int phase = 0; phase < 11; ++phase) {
          model.reset();
          if (phase < 8) {
            CLHEP::HepRandom::setTheEngine(&cyc);
            cyc.reset(phase);
          } else {
            CLHEP::HepRandom::setTheEngine(&con);
            con.reset(kConstSeq[phase - 8]);
          }
          G4InuclElementaryParticle bullet(
              G4LorentzVector(0., 0., plab, std::sqrt(plab * plab + m * m)), type);
          G4CascadParticle cp = model.initializeCascad(&bullet);
          const G4ThreeVector& p = cp.getPosition();
          const G4LorentzVector mom = cp.getMomentum();
          std::fprintf(f,
                       "%d,%d,%d,%.17g,%d,%lld,%.17g,%.17g,%.17g,%d,%.17g,%d,"
                       "%.17g,%.17g,%.17g,%.17g\n",
                       n.a, n.z, type, plab, phase,
                       (phase < 8) ? cyc.draws() : con.draws(), p.x(), p.y(), p.z(),
                       cp.getCurrentZone(), cp.getCurrentPath(), cp.getGeneration(), mom.x(),
                       mom.y(), mom.z(), mom.e());
        }
      }
    }
  }

  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

void dump_fate() {
  FILE* f = std::fopen("bertini_fate.csv", "w");
  if (!f) return;
  std::fprintf(f, "a,z,type,plab,dir,rad,gen,phase,draws,nout,i,otype,px,py,pz,e,"
                  "posx,posy,posz,zone,cpath,nrefl,movingin,"
                  "nucl1,nucl2,npcur,nncur,cposx,cposy,cposz,czone,ccpath,ogen\n");

  struct AZ { int a, z; };
  static const AZ nuclei[] = {{12, 6}, {27, 13}, {56, 26}, {207, 82}};
  static const int types[] = {proton, neutron, pionPlus, pionMinus, pionZero, photon};
  // The 0 is not decoration. `generateInteractionPartners` has a branch that fires only for a
  // particle with no momentum - `fabs(path) < small` with `mom.vect().mag() <= small` - and it
  // reaches two pieces of arithmetic nothing else does: the `path < small` disjunct that keeps a
  // partner whose sampled length is `large`, and `propagateAlongThePath` with
  // `Hep3Vector::unit()` of the ZERO vector, which CLHEP defines as zero and a port that
  // normalises to +z would turn into a thousand-unit jump along the beam axis.
  static const double plabs[] = {0.0, 0.3, 1.0, 3.0};

  // Three directions. [0] is the DEGENERATE one - parallel to the position vector, so the
  // particle is moving radially outward and `pperp2` is zero; [1] and [2] are off every axis and
  // off the radius, one outbound and one inbound.
  static const G4ThreeVector dirs[] = {
      G4ThreeVector(0.4242640687119285, 0.5656854249492380, 0.7071067811865476),
      G4ThreeVector(0.3713906763541037, -0.5570860145311556, 0.7427813527082074),
      G4ThreeVector(-0.5883484054145521, 0.1961161351381840, -0.7844645405527361)};
  // Fractions of the nuclear radius. 0.999 is inside the outermost zone and one step from the
  // surface; 0.15 is inside zone 0 of every nucleus in the list.
  static const double rads[] = {0.15, 0.45, 0.75, 0.999};
  // The position direction, chosen with all three components different and none zero.
  const G4ThreeVector rhat = G4ThreeVector(0.4242640687119285, 0.5656854249492380,
                                           0.7071067811865476);

  CycleEngine cyc;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&cyc);

  G4ElementaryParticleCollider coll;
  std::vector<G4CascadParticle> out;

  for (const AZ& n : nuclei) {
    G4NucleiModel model(n.a, n.z);
    for (int type : types) {
      const double m = G4InuclElementaryParticle::getParticleMass(type);
      for (double plab : plabs) {
        for (int idir = 0; idir < 3; ++idir) {
          for (int irad = 0; irad < 4; ++irad) {
            const G4ThreeVector pos = rhat * (rads[irad] * model.getRadius());
            for (int gen = 0; gen < 2; ++gen) {
              // generation 0 carries `large` in the path field, which is the sentinel that
              // exempts the projectile from the young-secondary veto; generation 1 carries 0,
              // which is what every real secondary carries.
              const double cpath = (gen == 0) ? 1000. : 0.;
              for (int phase = 0; phase < 8; ++phase) {
                model.reset();
                cyc.reset(phase);

                G4LorentzVector mom;
                mom.setVectM(dirs[idir] * plab, m);
                G4InuclElementaryParticle bullet(mom, type);
                G4CascadParticle cp(bullet, pos, model.getZone(pos.mag()), cpath, gen);

                out.clear();
                model.generateParticleFate(cp, &coll, out);

                const long long draws = cyc.draws();
                const std::pair<G4int, G4int> nucl = model.getTypesOfNucleonsInvolved();
                const G4ThreeVector& cpos = cp.getPosition();

                const int nout = static_cast<int>(out.size());
                for (int i = 0; i < (nout > 0 ? nout : 1); ++i) {
                  if (nout == 0) {
                    std::fprintf(f,
                                 "%d,%d,%d,%.17g,%d,%d,%d,%d,%lld,0,-1,0,0,0,0,0,"
                                 "0,0,0,-1,0,0,0,%d,%d,%d,%d,%.17g,%.17g,%.17g,%d,%.17g,-1\n",
                                 n.a, n.z, type, plab, idir, irad, gen, phase, draws,
                                 nucl.first, nucl.second, model.getNumberOfProtons(),
                                 model.getNumberOfNeutrons(), cpos.x(), cpos.y(), cpos.z(),
                                 cp.getCurrentZone(), cp.getCurrentPath());
                    continue;
                  }
                  const G4CascadParticle& o = out[i];
                  const G4LorentzVector om = o.getMomentum();
                  const G4ThreeVector& op = o.getPosition();
                  std::fprintf(f,
                               "%d,%d,%d,%.17g,%d,%d,%d,%d,%lld,%d,%d,%d,"
                               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%d,%d,"
                               "%d,%d,%d,%d,%.17g,%.17g,%.17g,%d,%.17g,%d\n",
                               n.a, n.z, type, plab, idir, irad, gen, phase, draws, nout, i,
                               o.getParticle().type(), om.x(), om.y(), om.z(), om.e(), op.x(),
                               op.y(), op.z(), o.getCurrentZone(), o.getCurrentPath(),
                               o.getNumberOfReflections(),
                               o.movingInsideNuclei() ? 1 : 0, nucl.first, nucl.second,
                               model.getNumberOfProtons(), model.getNumberOfNeutrons(),
                               cpos.x(), cpos.y(), cpos.z(), cp.getCurrentZone(),
                               cp.getCurrentPath(), o.getGeneration());
                }
              }
            }
          }
        }
      }
    }
  }

  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

void dump_bertini(const DumpContext&) {
  dump_params();
  dump_particles();
  dump_nuclei();
  dump_nucxsec();
  dump_channels();
  dump_chtables();
  dump_angdst();
  dump_epcollide();
  dump_initcascad();
  dump_fate();
  dump_apply();
}

}  // namespace

G4GPU_REGISTER_DUMP("bertini",
                    "bertini_params.csv bertini_particles.csv bertini_quasideuteron.csv "
                    "bertini_nuclei.csv bertini_zonelookup.csv bertini_potfold.csv "
                    "bertini_nucxsec.csv bertini_channels.csv bertini_chsample.csv "
                    "bertini_chtables.csv bertini_chfinalstates.csv bertini_chbins.csv bertini_angchoice.csv "
                    "bertini_angdst.csv bertini_3bodydst.csv bertini_momchoice.csv "
                    "bertini_epcollide.csv "
                    "bertini_initcascad.csv bertini_fate.csv "
                    "bertini_apply.csv bertini_apply_species.csv",
                    dump_bertini);
