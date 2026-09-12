// G4VSplitableHadron and G4DiffractiveSplitableHadron: the hadron that becomes a string.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/management/src/G4VSplitableHadron.cc
//     the G4ReactionProduct and the G4Nucleon constructors
//   .../management/include/G4VSplitableHadron.hh
//     Set4Momentum, IncrementCollisionCount, SetStatus, SetTimeOfCreation, SetPosition, IsSplit
//   .../diffraction/src/G4DiffractiveSplitableHadron.cc
//     SplitUp, GetNextParton, SetFirstParton, SetSecondParton, ChooseStringEnds, Diquark
//
// WHY THE PARTON MOMENTA ARE NOT HERE. `G4Parton` carries a four-momentum, and
// G4DiffractiveExcitation::CreateStrings is the only thing that ever writes one: it computes
// the two ends' momenta and hands the partons straight to a `G4ExcitedString`, which is what
// the fragmentation reads. Nothing reads a parton's momentum back off the splitable hadron. So
// the two PDG codes are stored and the momenta are locals in CreateStrings - which takes
// `sizeof(SplitableHadron)` from 160 bytes to 96, and the model's workspace holds one per
// nucleon of two nuclei.
//
// The one caveat is the KINKY string, which makes four partons out of two and needs the kink's
// own two momenta to survive into a second `G4ExcitedString`. That arm is unreachable
// (`Pt2Kink` is 0 in the G4FTFParameters constructor - `kKinkyStrings`), and CreateStrings
// refuses it by name rather than building half of it.
//
// `status` IS A SMALL INTEGER WITH SIX MEANINGS and no enum in Geant4:
//   0  took part in a non-diffractive interaction (a string is built from its partons)
//   1  takes part in the interaction (set by GetList) - diffractive if SoftCollisionCount != 0
//   2  took part in a quark exchange (ExciteParticipants_doChargeExchange)
//   3  involved by the reggeon cascade (ReggeonCascade)
//   4  used by G4FTFAnnihilation for an annihilated nucleon (this port refuses annihilation)
//   5  returned to the nucleus at low energy (BuildStrings' fourth target case)
// BuildStrings dispatches on it, so it is carried as the G4int it is.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/ftf_hadrons.hh"
#include "physics/hadronic/ftf/lorentz.cuh"
#include "physics/hadronic/ftf/refusal.cuh"

namespace g4gpu::hadronic::ftf {

/// G4VSplitableHadron with G4DiffractiveSplitableHadron's two partons.
struct SplitableHadron {
  int pdg = 0;                 ///< GetDefinition(), as the PDG code every consumer reads it as
  Vec4 momentum;               ///< the4Momentum
  double time_of_creation = 0.0;
  Vec3d position{0.0, 0.0, 0.0};
  int collision_count = 0;     ///< theCollisionCount, i.e. GetSoftCollisionCount()
  int status = 0;              ///< curStatus; see the file header for the six values
  bool is_split = false;       ///< isSplit
  int parton[2] = {0, 0};      ///< Parton[0] = string start, Parton[1] = string end
  int parton_index = -1;       ///< PartonIndex

  /// The two partons' four-momenta, and the ONE case in FTF that needs them stored.
  ///
  /// The file header says why they are not normally here: `CreateStrings` computes them and
  /// hands them straight to a `G4ExcitedString`, so nothing reads them back. G4FTFAnnihilation
  /// is the exception. Its four channels set the partons AND their momenta and then call
  /// `Splitting()` - so when `BuildStrings` later calls `CreateStrings` on the same hadron,
  /// `IsSplit()` is already true and the `HadronIsString` arm builds the string out of the
  /// parton objects as they are. Without these two vectors the annihilation strings would come
  /// out with zero momentum.
  Vec4 parton_mom[2];

  /// Not a Geant4 member: `alive` is the difference between a G4VSplitableHadron* that exists
  /// and a null one. G4FTFParticipants leaves `SetTarget(0)` on an interaction whose target
  /// nucleon was already hit, and G4FTFModel::GetResiduals' low-energy arm deletes a splitable
  /// hadron and calls `Hit(0)`. Both are pointer states, and a slot in an array has to say so.
  bool alive = false;
};

/// G4VSplitableHadron::G4VSplitableHadron( const G4ReactionProduct& ) - the primary. The
/// position is NOT set by this constructor (it is default-constructed to the origin) and
/// G4FTFParticipants::GetList sets it to the impact point immediately afterwards.
__host__ __device__ inline SplitableHadron splitable_from_primary(int pdg, const Vec4& p4) {
  SplitableHadron h;
  h.pdg = pdg;
  h.momentum = p4;
  h.time_of_creation = 0.0;
  h.position = Vec3d{0.0, 0.0, 0.0};
  h.collision_count = 0;
  h.status = 0;
  h.is_split = false;
  h.parton[0] = 0;
  h.parton[1] = 0;
  h.parton_index = -1;
  h.alive = true;
  return h;
}

/// G4VSplitableHadron::G4VSplitableHadron( const G4Nucleon& ) - a target or projectile nucleon.
/// This one DOES copy the position, and it copies `aNucleon.GetMomentum()`, which is the
/// nucleon's off-shell four-momentum (bic/nucleus/nucleus_model.cuh's second "not guaranteed").
__host__ __device__ inline SplitableHadron splitable_from_nucleon(int pdg, const Vec4& p4,
                                                                  const Vec3d& pos) {
  SplitableHadron h = splitable_from_primary(pdg, p4);
  h.position = pos;
  return h;
}

/// G4DiffractiveSplitableHadron::Diquark - `max*1000 + min*100 + 2*Spin + 1`, negative unless
/// both quarks are.
__host__ __device__ inline int splitable_diquark(int aquark, int bquark, int spin) {
  const int a = (aquark < 0) ? -aquark : aquark;
  const int b = (bquark < 0) ? -bquark : bquark;
  const int hi = (a > b) ? a : b;
  const int lo = (a < b) ? a : b;
  const int diquark_pdg = hi * 1000 + lo * 100 + 2 * spin + 1;
  return (aquark > 0 && bquark > 0) ? diquark_pdg : -diquark_pdg;
}

/// G4DiffractiveSplitableHadron::ChooseStringEnds.
///
/// THE MESON ARM'S `anti` IS AN INTEGER POWER OF -1 WRITTEN AS ARITHMETIC:
/// `1 - 2*(max(heavy,light) % 2)` is +1 for an even heaviest quark (u, c) and -1 for an odd one
/// (d, s, b), which is the quark-model rule that the heavier quark of a meson is the antiquark
/// when its flavour index is odd. The commented-out `std::pow(-1, max(heavy,light))` above it
/// is the same number through a floating-point pow; the integer form is what runs.
///
/// THE THREE-IDENTICAL-QUARK CASE CHANGES THE SUPPRESSION AND NOT THE BRANCH.
/// `SuppresUUDDSS` is 1/2 normally and 1 for a uuu/ddd/sss baryon, and it is consumed as a
/// rejection on `G4UniformRand() > SuppresUUDDSS` - so for Delta++ every draw is accepted and
/// the do-loop never `continue`s, while for a proton half of the same-flavour picks are thrown
/// away and the loop spends another deviate.
///
/// The 1000-iteration exhaustion answers `(j10, Diquark(j1000, j100, 1))` with Geant4's own
/// comment "just something acceptable, without any physics consideration".
template <typename Rng>
__host__ __device__ inline void splitable_choose_string_ends(int pdg_code, int* a_end,
                                                             int* b_end, Rng& rng) {
  const int abs_pdg = (pdg_code < 0) ? -pdg_code : pdg_code;

  if (abs_pdg < 1000) {  // -------------------- Meson -------------
    int heavy = 0, light = 0;
    if (!((abs_pdg == 111) || (abs_pdg == 221) || (abs_pdg == 331))) {
      // Ordinary mesons
      heavy = abs_pdg / 100;
      light = (abs_pdg % 100) / 10;
      const int mx = (heavy > light) ? heavy : light;
      int anti = 1 - 2 * (mx % 2);
      if (pdg_code < 0) { anti *= -1; }
      heavy *= anti;
      light *= -1 * anti;
    } else {
      // Pi0, Eta, Eta' - a coin toss between a u-ubar and a d-dbar pair. This deviate is spent
      // for a pi0 projectile and for nothing else in the meson arm.
      if (rng.uniform() < 0.5) {
        heavy = 1;
        light = -1;
      } else {
        heavy = 2;
        light = -2;
      }
    }
    if (rng.uniform() < 0.5) {
      *a_end = heavy;
      *b_end = light;
    } else {
      *a_end = light;
      *b_end = heavy;
    }
    return;
  }

  // -------------------- Baryon --------------
  // Note the DIVISIONS ARE ON THE SIGNED CODE, not on the absolute value: for an anti-baryon
  // j1000, j100 and j10 all come out negative, and `Diquark` then returns a positive diquark
  // code because its `aquark > 0 && bquark > 0` test fails on both. That is how an anti-proton
  // gets an anti-diquark end.
  const int j1000 = pdg_code / 1000;
  const int j100 = (pdg_code % 1000) / 100;
  const int j10 = (pdg_code % 100) / 10;

  if (abs_pdg > 4000) {  // A charmed or bottom baryon: the heavy quark is always the string end
    *a_end = j10;
    if (rng.uniform() > 0.25) {
      *b_end = splitable_diquark(j1000, j100, 0);
    } else {
      *b_end = splitable_diquark(j1000, j100, 1);
    }
    return;
  }

  double suppres_uuddss = 1.0 / 2.0;
  if ((j1000 == j100) && (j1000 == j10)) { suppres_uuddss = 1.0; }

  const int max_number_of_loops = 1000;
  int loop_counter = 0;
  do {
    const double random = rng.uniform();
    if (random < 0.33333) {
      if ((j100 == j10) && (rng.uniform() > suppres_uuddss)) { continue; }
      *a_end = j1000;
      if (j100 == j10) {
        *b_end = splitable_diquark(j100, j10, 1);
      } else if (rng.uniform() > 0.25) {
        *b_end = splitable_diquark(j100, j10, 0);
      } else {
        *b_end = splitable_diquark(j100, j10, 1);
      }
      break;
    } else if (random < 0.66667) {
      if ((j1000 == j10) && (rng.uniform() > suppres_uuddss)) { continue; }
      *a_end = j100;
      if (j1000 == j10) {
        *b_end = splitable_diquark(j1000, j10, 1);
      } else if (rng.uniform() > 0.25) {
        *b_end = splitable_diquark(j1000, j10, 0);
      } else {
        *b_end = splitable_diquark(j1000, j10, 1);
      }
      break;
    } else {
      if ((j1000 == j100) && (rng.uniform() > suppres_uuddss)) { continue; }
      *a_end = j10;
      if (j1000 == j100) {
        *b_end = splitable_diquark(j1000, j100, 1);
      } else if (rng.uniform() > 0.25) {
        *b_end = splitable_diquark(j1000, j100, 0);
      } else {
        *b_end = splitable_diquark(j1000, j100, 1);
      }
      break;
    }
  } while ((true) && ++loop_counter < max_number_of_loops);
  if (loop_counter >= max_number_of_loops) {
    *a_end = j10;
    *b_end = splitable_diquark(j1000, j100, 1);
  }
}

/// `G4Parton::G4Parton( G4int PDGcode )` - the deviates the constructor spends, and nothing
/// else.
///
/// A PARTON'S CONSTRUCTOR SAMPLES. It draws a colour - `(G4int)(3*G4UniformRand())+1`, signed
/// by the code, once for a quark, once for a diquark and TWICE for a gluon - and then a spin
/// projection, `(G4int)((iSpin+1)*G4UniformRand())`, only when `GetPDGiSpin()` is non-zero. So
/// building a proton's (u, ud_0) pair costs three deviates on top of ChooseStringEnds' two: two
/// for the quark (colour and spin-1/2) and one for the spin-0 diquark. Nothing in FTF ever
/// READS `theColour`, `theSpinZ` or `theIsoSpinZ` - the fragmentation uses only the PDG code -
/// so the values are discarded here and only the draws are reproduced.
///
/// It was found by the draw-count column and not by reading: ref/oracle/ftf_splitup.csv says a
/// proton at phase 0 costs five uniforms and the first transcription spent two, which would
/// have shifted every hadron of every event after the first string.
///
/// The gluon arm cannot be reached - `ChooseStringEnds` produces quarks and diquarks only, and
/// the two kinky-string gluon partons are behind `kKinkyStrings` - but it is transcribed,
/// because "cannot be reached" is a claim about the caller and this function is about G4Parton.
template <typename Rng>
__host__ __device__ inline void ftf_parton_construct_draws(int pdg, FtfRefusal* refused,
                                                           Rng& rng) {
  const data::FtfHadron* d = data::ftf_find_hadron(pdg);
  if (d == nullptr) {
    // Geant4 throws G4HadronicException "Encoding not in particle table" here.
    if (refused != nullptr) { *refused = FtfRefusal::kUnknownHadronCode; }
    return;
  }
  if (d->subtype == data::FtfSubType::kQuark || d->subtype == data::FtfSubType::kDiQuark) {
    (void)rng.uniform();  // the colour
  } else {
    // A gluon ("gluons" is a particle TYPE with sub-type "gluons" too) draws twice for its
    // two-index colour and then once more for its isospin projection when iIsospin != 0.
    (void)rng.uniform();
    (void)rng.uniform();
    if (d->iisospin != 0) { (void)rng.uniform(); }
  }
  if (d->iispin != 0) { (void)rng.uniform(); }  // the spin projection
}

/// G4DiffractiveSplitableHadron::SplitUp.
///
/// The `if (Parton[0] != nullptr) return;` after `Splitting()` is unreachable from FTF: the two
/// constructors FTF uses both null the parton pointers, and only the default constructor - which
/// no FTF class calls - pre-fills them with a q-qbar pair. Transcribed, as the `parton[0] != 0`
/// test, so that the file shows the guard the original has.
///
/// The two `new G4Parton(...)` calls after ChooseStringEnds each spend deviates; see
/// `ftf_parton_construct_draws`. The ORDER matters: start first, then end.
template <typename Rng>
__host__ __device__ inline void splitable_split_up(SplitableHadron* h, FtfRefusal* refused,
                                                   Rng& rng) {
  if (h->is_split) { return; }
  h->is_split = true;  // Splitting()
  if (h->parton[0] != 0) { return; }

  int string_start = 0, string_end = 0;
  splitable_choose_string_ends(h->pdg, &string_start, &string_end, rng);
  h->parton[0] = string_start;
  h->parton[1] = string_end;
  ftf_parton_construct_draws(string_start, refused, rng);
  ftf_parton_construct_draws(string_end, refused, rng);
  h->parton_index = -1;
}

/// G4DiffractiveSplitableHadron::GetNextParton, index arithmetic included.
///
/// The index cycles -1 -> 0 -> (returns 1 and resets to -1), so two consecutive calls give
/// Parton[0] then Parton[1] and a third starts over. CreateStrings makes exactly two calls.
/// A `false` in `*ok` is Geant4's null return, which CreateStrings answers with
/// "No start parton found" and two null strings.
__host__ __device__ inline int splitable_next_parton(SplitableHadron* h, bool* ok) {
  ++h->parton_index;
  if (h->parton_index > 1 || h->parton_index < 0) {
    *ok = false;
    return 0;
  }
  const int idx = h->parton_index;
  if (h->parton_index == 1) { h->parton_index = -1; }
  *ok = true;
  return h->parton[idx];
}

/// `GetDefinition()->GetParticleSubType() == "di_quark"` and `== "quark"`, which CreateStrings
/// uses to decide whether a kink is even allowed. data/ftf_hadrons.hh carries the sub-type as
/// an enum, so this is a table lookup and not a string compare.
__host__ __device__ inline data::FtfSubType parton_sub_type(int pdg, FtfRefusal* refused) {
  const data::FtfHadron* h = data::ftf_find_hadron(pdg);
  if (h == nullptr) {
    if (refused != nullptr) { *refused = FtfRefusal::kUnknownHadronCode; }
    return data::FtfSubType::kOther;
  }
  return h->subtype;
}

}  // namespace g4gpu::hadronic::ftf
