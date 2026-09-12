// G4FTFParameters: the tuned parameter set FTFP runs on, for every (projectile class, target
// Z and A, lab momentum per particle).
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/diffraction/src/G4FTFParameters.cc
//     G4FTFParameters::G4FTFParameters, InitForInteraction, GetMinMass, GetProcProb, Reset
//   .../include/G4FTFParameters.hh          (every Set*/Get* inline)
//   .../src/G4FTFTunings.cc
//     G4FTFSettingDefaultHDP (the HDP defaults), G4FTFParamCollection,
//     G4FTFParamCollBaryonProj, G4FTFParamCollMesonProj, G4FTFParamCollPionProj,
//     G4FTFTunings::GetIndexTune
//
// WHAT THIS IS. Ninety-odd numbers per interaction, of which about thirty are tuned constants,
// six come from a Glauber-Gribov hadron-nucleon cross section and the rest are arithmetic on
// those. `InitForInteraction` draws no random numbers at all, so ref/oracle/ftf_params.csv is
// an EXACT oracle with no phase column: 1716 rows over 22 projectiles x 6 targets x 13
// momenta, 66 columns each.
//
// THE TUNES ARE MEASURED TO BE UNREACHABLE, NOT ASSUMED TO BE. G4FTFTunings has ten tunes and
// GetIndexTune returns the first one switched on above index 0, else 0.
// ref/oracle/ftf_lund_params.csv dumps all ten applicability states: 1, 0, 0, 0, 0, 0, 0, 0,
// 0, 0. So QBBC runs tune 0 - which IS the tuned set, because every one of tune 0's values
// comes from G4HadronicDeveloperParameters::SetDefault in G4FTFTunings.cc - and tunes 1 to 9
// are refused by name (FtfRefusal::kFtfTuneNonDefault). The oracle also carries the index
// GetIndexTune returned on every row, so a future run with a tune switched on fails the test
// rather than silently comparing two different parameter sets.
//
// THREE THINGS THE REFERENCE DOES THAT LOOK LIKE BUGS AND ARE THE REFERENCE.
//
//  1. `Reset()` zeroes `ProcParams[i][j]` for i = 0..3, and the array is `[5][7]`. Row 4 -
//     "Qexchange with Exc. Additional multiplier" - is never cleared, so it holds whatever the
//     previous interaction left. It is harmless in 11.1.1 because every one of the five
//     projectile-class branches writes all five rows on every call, and that is checked here
//     rather than assumed: ftf_params.csv dumps the whole 5x7 array on every row of a grid
//     that interleaves projectile classes, so a row-4 latch would show up as a value from the
//     previous row. The port's Reset zeroes 0..3 exactly as Geant4's does, so that if a future
//     branch stops writing row 4 the two ports agree on what it then holds.
//
//  2. For a hadron projectile the "interaction on N" cross sections are taken at (Z, A) =
//     (0, 1), which is G4ComponentGGHadronNucleusXsc's A == 1 branch with Z = 0. That branch
//     sets `fTotalXsc = sigma` - correctly the hadron-NEUTRON total, since Z = 0 kills the
//     proton term - but `fInelasticXsc = hpInXsc`, the hadron-PROTON inelastic, which was
//     computed and then multiplied by Z = 0 in the total. So `Xelastic` on a neutron target is
//     `sigma_tot(hn) - sigma_inel(hp)`, mixing two different targets. See docs/RISK.md V86.
//     It propagates into FTFXelastic, the elastic slope, Gamma0 and the average Pt^2 of
//     elastic scattering, i.e. into the impact-parameter sampling of every FTF interaction on
//     a nucleus with neutrons in it.
//
//  3. In the nucleus-projectile arm the "PN" cross sections are
//     `GetTotalIsotopeCrossSection(Neutron, ..., 0, 1)` - a NEUTRON projectile on a NEUTRON
//     target - while the "PP" ones are a proton on a proton. So the mixed proton-neutron term
//     of a nucleus-nucleus average is built from n+n, not from p+n, and by (2) its elastic
//     part is n+n total minus n+p inelastic.
//
// G4Exp AND G4Log. Used for Ylab and for the four nuclear-destruction Fermi factors. This port
// uses std::log and std::exp, which is the doctrine data/g4pow.hh's header sets out and which
// docs/PORTED.md 2.1.x already relies on: the two agree to within an ulp, unlike G4Pow::powA,
// which is a Taylor expansion and differs at 1e-7. `powA` IS used - for
// `140*(MesonProdThreshold - SqrtS)^2.5` in the anti-baryon branch - and is taken from
// data/g4pow.hh, not from std::pow.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "physics/hadronic/ftf/lund_tables.cuh"
#include "physics/hadronic/ftf/refusal.cuh"
#include "physics/hadronic/xs/gg_hadron_nucleus_xsc.cuh"
#include "physics/hadronic/xs/projectile.cuh"

namespace g4gpu::hadronic::ftf {

/// One tune's worth of G4FTFParamCollection. The field names are Geant4's with the `f`
/// dropped, so a reader can put this next to G4FTFTunings.hh.
template <typename real_t>
struct FtfParamColl {
  real_t proc_a1[5], proc_b1[5], proc_a2[5], proc_b2[5], proc_a3[5], proc_atop[5],
      proc_ymin[5];
  bool proj_diff_dissociation;
  bool tgt_diff_dissociation;
  real_t delta_prob_at_quark_exchange;
  real_t prob_of_same_quark_exchange;
  real_t proj_min_diff_mass;
  real_t proj_min_non_diff_mass;
  real_t tgt_min_diff_mass;
  real_t tgt_min_non_diff_mass;
  real_t average_pt2;
  real_t prob_log_distr_prd;
  real_t prob_log_distr;
  real_t nuclear_proj_destruct_p1;
  bool nuclear_proj_destruct_p1_nbrndep;
  real_t nuclear_tgt_destruct_p1;
  bool nuclear_tgt_destruct_p1_adep;
  real_t nuclear_proj_destruct_p2, nuclear_proj_destruct_p3;
  real_t nuclear_tgt_destruct_p2, nuclear_tgt_destruct_p3;
  real_t pt2_nuclear_destruct_p1, pt2_nuclear_destruct_p2, pt2_nuclear_destruct_p3,
      pt2_nuclear_destruct_p4;
  real_t r2_of_nuclear_destruct;
  real_t exci_energy_per_wounded_nucleon;
  real_t dof_nuclear_destruct;
  real_t max_pt2_of_nuclear_destruct;
};

/// G4FTFParamCollection's constructor: everything zero, then the two projectile-destruction
/// constants the comment says are "kept fixed for now (i.e. do not take them from HDP)".
template <typename real_t>
__host__ __device__ inline void ftf_param_coll_zero(FtfParamColl<real_t>* c) {
  for (int i = 0; i < 5; ++i) {
    c->proc_a1[i] = real_t(0);
    c->proc_b1[i] = real_t(0);
    c->proc_a2[i] = real_t(0);
    c->proc_b2[i] = real_t(0);
    c->proc_a3[i] = real_t(0);
    c->proc_atop[i] = real_t(0);
    c->proc_ymin[i] = real_t(0);
  }
  c->proj_diff_dissociation = false;
  c->tgt_diff_dissociation = false;
  c->delta_prob_at_quark_exchange = real_t(0);
  c->prob_of_same_quark_exchange = real_t(0);
  c->proj_min_diff_mass = real_t(0);
  c->proj_min_non_diff_mass = real_t(0);
  c->tgt_min_diff_mass = real_t(0);
  c->tgt_min_non_diff_mass = real_t(0);
  c->average_pt2 = real_t(0);
  c->prob_log_distr_prd = real_t(0);
  c->prob_log_distr = real_t(0);
  c->nuclear_proj_destruct_p1 = real_t(0);
  c->nuclear_proj_destruct_p1_nbrndep = false;
  c->nuclear_tgt_destruct_p1 = real_t(0);
  c->nuclear_tgt_destruct_p1_adep = false;
  c->nuclear_tgt_destruct_p2 = real_t(0);
  c->nuclear_tgt_destruct_p3 = real_t(0);
  c->pt2_nuclear_destruct_p1 = real_t(0);
  c->pt2_nuclear_destruct_p2 = real_t(0);
  c->pt2_nuclear_destruct_p3 = real_t(0);
  c->pt2_nuclear_destruct_p4 = real_t(0);
  c->r2_of_nuclear_destruct = real_t(0);
  c->exci_energy_per_wounded_nucleon = real_t(0);
  c->dof_nuclear_destruct = real_t(0);
  c->max_pt2_of_nuclear_destruct = real_t(0);
  c->nuclear_proj_destruct_p2 = real_t(4.0);
  c->nuclear_proj_destruct_p3 = real_t(2.1);
}

/// G4FTFParamCollBaryonProj's constructor, i.e. the FTF_BARYON_* HDP defaults.
template <typename real_t>
__host__ __device__ inline FtfParamColl<real_t> ftf_param_coll_baryon() {
  FtfParamColl<real_t> c;
  ftf_param_coll_zero(&c);
  // Process 0 - quark exchange without excitation
  c.proc_a1[0] = real_t(13.71);
  c.proc_b1[0] = real_t(1.75);
  c.proc_a2[0] = real_t(-30.69);
  c.proc_b2[0] = real_t(3.0);
  c.proc_a3[0] = real_t(0.0);
  c.proc_atop[0] = real_t(1.0);
  c.proc_ymin[0] = real_t(0.93);
  // Process 1 - quark exchange with excitation
  c.proc_a1[1] = real_t(25.0);
  c.proc_b1[1] = real_t(1.0);
  c.proc_a2[1] = real_t(-50.34);
  c.proc_b2[1] = real_t(1.5);
  c.proc_a3[1] = real_t(0.0);
  c.proc_atop[1] = real_t(0.0);
  c.proc_ymin[1] = real_t(1.4);
  // Processes 2 and 3 are built from 6/Xinel inside InitForInteraction, not from HDP; the two
  // dissociation switches are.
  c.proj_diff_dissociation = false;
  c.tgt_diff_dissociation = false;
  // Process 4 - quark exchange with an additional multiplier
  c.proc_a1[4] = real_t(0.6);
  c.proc_b1[4] = real_t(0.0);
  c.proc_a2[4] = real_t(-1.2);
  c.proc_b2[4] = real_t(0.5);
  c.proc_a3[4] = real_t(0.0);
  c.proc_atop[4] = real_t(0.0);
  c.proc_ymin[4] = real_t(1.4);

  c.delta_prob_at_quark_exchange = real_t(0.0);
  c.prob_of_same_quark_exchange = real_t(0.0);
  c.proj_min_diff_mass = real_t(1.16);      // GeV, multiplied by GeV in the setter
  c.proj_min_non_diff_mass = real_t(1.16);
  c.tgt_min_diff_mass = real_t(1.16);
  c.tgt_min_non_diff_mass = real_t(1.16);
  c.average_pt2 = real_t(0.3);              // GeV^2, squared-multiplied in the setter
  // Kept fixed rather than taken from HDP, which is why the two commented-out DeveloperGet
  // lines are in G4FTFTunings.cc.
  c.prob_log_distr_prd = real_t(0.55);
  c.prob_log_distr = real_t(0.55);

  c.nuclear_proj_destruct_p1 = real_t(1.0);
  c.nuclear_proj_destruct_p1_nbrndep = false;
  c.nuclear_proj_destruct_p2 = real_t(4.0);
  c.nuclear_proj_destruct_p3 = real_t(2.1);
  c.nuclear_tgt_destruct_p1 = real_t(1.0);
  c.nuclear_tgt_destruct_p1_adep = false;
  c.nuclear_tgt_destruct_p2 = real_t(4.0);
  c.nuclear_tgt_destruct_p3 = real_t(2.1);

  c.pt2_nuclear_destruct_p1 = real_t(0.035);
  c.pt2_nuclear_destruct_p2 = real_t(0.04);
  c.pt2_nuclear_destruct_p3 = real_t(4.0);
  c.pt2_nuclear_destruct_p4 = real_t(2.5);

  c.r2_of_nuclear_destruct = real_t(1.5) * fermi<real_t>() * fermi<real_t>();
  c.exci_energy_per_wounded_nucleon = real_t(40.0) * units::MeV<real_t>();
  c.dof_nuclear_destruct = real_t(0.3);
  // NINE GeV^2, not one. The commented-out HDP line above it in G4FTFTunings.cc says the
  // parameter "has changed from 1. to 9. between 10.2 and 10.3.ref07 ... then it went back to
  // 1. for the 10.4-candidate"; the code that survived sets 9. The anti-baryon branch of
  // InitForInteraction hard-codes 1.0 GeV^2 and the meson collection sets 1.0, so the three
  // projectile classes genuinely differ by a factor of nine here.
  c.max_pt2_of_nuclear_destruct = real_t(9.0) * units::GeV<real_t>() * units::GeV<real_t>();
  return c;
}

/// G4FTFParamCollMesonProj's constructor - the FTF_MESON_* HDP defaults. Note that it sets
/// NO excitation parameters at all: a kaon or any other non-pion meson gets the hard-coded
/// SetParams block in InitForInteraction, and this collection is read only for the
/// nuclear-destruction half.
template <typename real_t>
__host__ __device__ inline FtfParamColl<real_t> ftf_param_coll_meson() {
  FtfParamColl<real_t> c;
  ftf_param_coll_zero(&c);
  c.nuclear_tgt_destruct_p1 = real_t(0.00481);
  c.nuclear_tgt_destruct_p1_adep = true;
  c.nuclear_tgt_destruct_p2 = real_t(4.0);
  c.nuclear_tgt_destruct_p3 = real_t(2.1);
  c.pt2_nuclear_destruct_p1 = real_t(0.035);
  c.pt2_nuclear_destruct_p2 = real_t(0.04);
  c.pt2_nuclear_destruct_p3 = real_t(4.0);
  c.pt2_nuclear_destruct_p4 = real_t(2.5);
  c.r2_of_nuclear_destruct = real_t(1.5) * fermi<real_t>() * fermi<real_t>();
  c.exci_energy_per_wounded_nucleon = real_t(40.0) * units::MeV<real_t>();
  c.dof_nuclear_destruct = real_t(0.3);
  c.max_pt2_of_nuclear_destruct = real_t(1.0) * units::GeV<real_t>() * units::GeV<real_t>();
  return c;
}

/// G4FTFParamCollPionProj's constructor: the meson collection, then the FTF_PION_* defaults
/// on top of it.
template <typename real_t>
__host__ __device__ inline FtfParamColl<real_t> ftf_param_coll_pion() {
  FtfParamColl<real_t> c = ftf_param_coll_meson<real_t>();
  c.proc_a1[0] = real_t(150.0);
  c.proc_b1[0] = real_t(1.8);
  c.proc_a2[0] = real_t(-247.3);
  c.proc_b2[0] = real_t(2.3);
  c.proc_a3[0] = real_t(0.0);
  c.proc_atop[0] = real_t(1.0);
  c.proc_ymin[0] = real_t(2.3);

  c.proc_a1[1] = real_t(5.77);
  c.proc_b1[1] = real_t(0.6);
  c.proc_a2[1] = real_t(-5.77);
  c.proc_b2[1] = real_t(0.8);
  c.proc_a3[1] = real_t(0.0);
  c.proc_atop[1] = real_t(0.0);
  c.proc_ymin[1] = real_t(0.0);

  // Process 2 is "kept fixed so far" - assigned directly rather than through HDP, with the
  // DeveloperGet block commented out above it.
  c.proc_a1[2] = real_t(2.27);
  c.proc_b1[2] = real_t(0.5);
  c.proc_a2[2] = real_t(-98052.0);
  c.proc_b2[2] = real_t(4.0);
  c.proc_a3[2] = real_t(0.0);
  c.proc_atop[2] = real_t(0.0);
  c.proc_ymin[2] = real_t(3.0);

  c.proc_a1[3] = real_t(7.0);
  c.proc_b1[3] = real_t(0.9);
  c.proc_a2[3] = real_t(-85.28);
  c.proc_b2[3] = real_t(1.9);
  c.proc_a3[3] = real_t(0.08);
  c.proc_atop[3] = real_t(0.0);
  c.proc_ymin[3] = real_t(2.2);

  c.proj_diff_dissociation = false;
  c.tgt_diff_dissociation = false;

  c.proc_a1[4] = real_t(1.0);
  c.proc_b1[4] = real_t(0.0);
  c.proc_a2[4] = real_t(-11.02);
  c.proc_b2[4] = real_t(1.0);
  c.proc_a3[4] = real_t(0.0);
  c.proc_atop[4] = real_t(0.0);
  c.proc_ymin[4] = real_t(2.4);

  c.delta_prob_at_quark_exchange = real_t(0.56);
  c.proj_min_diff_mass = real_t(1.0);
  c.proj_min_non_diff_mass = real_t(1.0);
  c.tgt_min_diff_mass = real_t(1.16);
  c.tgt_min_non_diff_mass = real_t(1.16);
  c.average_pt2 = real_t(0.3);
  c.prob_of_same_quark_exchange = real_t(0.0);
  c.prob_log_distr_prd = real_t(0.55);
  c.prob_log_distr = real_t(0.55);
  return c;
}

/// Everything G4FTFParameters holds. The data members are public in Geant4 too - "Is there any
/// reason for NOT making all the members data private ???" is a comment in the header - which
/// is why ref/oracle/ftf_params.csv can dump ProcParams directly.
template <typename real_t>
struct FtfParameters {
  real_t hn_cms_energy;   ///< FTFhNcmsEnergy - written by SethNcmsEnergy, which nothing calls
  real_t x_total;         ///< mb
  real_t x_elastic;       ///< mb
  real_t x_inelastic;     ///< mb
  real_t x_annihilation;  ///< mb - written only by Reset in 11.1.1; see the note below
  real_t prob_of_annihilation;
  real_t prob_of_elastic_scatt;
  real_t radius_of_hn_interactions2;  ///< fm^2
  real_t slope;                       ///< fm^-2
  real_t avarage_pt2_of_elastic_scattering;  ///< MeV^2 (Geant4's spelling)
  real_t gamma0;

  /// EVERY MEMBER THAT ONLY THE CONSTRUCTOR SETS CARRIES ITS CONSTRUCTOR VALUE HERE.
  /// `ftf_parameters_reset` is G4FTFParameters::Reset, and Reset does not touch row 4 of
  /// ProcParams, the kink switch, the gluon-splitting probabilities or the diffraction-
  /// dissociation switch - in Geant4 those are set once, by the constructor. A
  /// `FtfParameters<double> p;` on the stack followed by `ftf_init_for_interaction` therefore
  /// left them INDETERMINATE, and one of them decides physics: an indeterminate
  /// `enable_diff_dissociation_for_b_greater_10` that happened to be non-zero kept projectile
  /// and target diffraction ON for every target with A > 10, where Geant4 switches both off.
  /// docs/RISK.md V104.
  real_t proc_params[5][7] = {};

  real_t delta_prob_at_quark_exchange;
  real_t prob_of_same_quark_exchange;
  real_t proj_min_diff_mass;
  real_t proj_min_non_diff_mass;
  real_t prob_log_distr_prd;
  real_t tar_min_diff_mass;
  real_t tar_min_non_diff_mass;
  real_t average_pt2;
  real_t prob_log_distr;

  real_t pt2_kink = real_t(0.0) * units::GeV<real_t>() * units::GeV<real_t>();
  real_t quark_probabilities_at_gluon_split_up[3] = {
      real_t(1.0) / real_t(3.0), real_t(1.0) / real_t(3.0) + real_t(1.0) / real_t(3.0),
      real_t(1.0) / real_t(3.0) + real_t(1.0) / real_t(3.0) + real_t(1.0) / real_t(3.0)};

  real_t max_number_of_collisions;
  real_t prob_of_inel_interaction;
  real_t cof_nuclear_destruction_pr;
  real_t cof_nuclear_destruction;
  real_t r2_of_nuclear_destruction;
  real_t excitation_energy_per_wounded_nucleon;
  real_t dof_nuclear_destruction;
  real_t pt2_of_nuclear_destruction;
  real_t max_pt2_of_nuclear_destruction;

  bool enable_diff_dissociation_for_b_greater_10 = false;

  /// Not a Geant4 member. True when InitForInteraction's last `if (Xtotal == 0.0)` block ran,
  /// i.e. when the cross sections were recomputed for a PROTON because the projectile's came
  /// out zero. Geant4 does this silently. It is legitimate and reached - a proton on hydrogen
  /// at 20 MeV/c has a Glauber-Gribov total of exactly zero, so the block fires with a proton
  /// already as the projectile and changes nothing - but for a projectile that is not a
  /// nucleon it substitutes one, and that is a different physics. Reported rather than
  /// refused, because refusing would delete an interaction Geant4 performs.
  bool nucleon_assumed = false;

  /// The tune index GetIndexTune returned. 0 in every QBBC run; a non-zero value is
  /// kFtfTuneNonDefault.
  int index_tune = 0;

  FtfRefusal refused = FtfRefusal::kNone;
  /// Which of G4HadronNucleonXsc's unported branches the projectile needed, when `refused` is
  /// kHadronNucleonXscRefused.
  xs::XsRefusal xs_refused = xs::XsRefusal::kNone;
};

/// G4FTFParameters::Reset. Note the `i < 4` bound on a `[5][7]` array - see the file header.
template <typename real_t>
__host__ __device__ inline void ftf_parameters_reset(FtfParameters<real_t>* p) {
  p->hn_cms_energy = real_t(0);
  p->x_total = real_t(0);
  p->x_elastic = real_t(0);
  p->x_inelastic = real_t(0);
  p->x_annihilation = real_t(0);
  p->prob_of_annihilation = real_t(0);
  p->prob_of_elastic_scatt = real_t(0);
  p->radius_of_hn_interactions2 = real_t(0);
  p->slope = real_t(0);
  p->avarage_pt2_of_elastic_scattering = real_t(0);
  p->gamma0 = real_t(0);
  p->delta_prob_at_quark_exchange = real_t(0);
  p->prob_of_same_quark_exchange = real_t(0);
  p->proj_min_diff_mass = real_t(0);
  p->proj_min_non_diff_mass = real_t(0);
  p->prob_log_distr_prd = real_t(0);
  p->tar_min_diff_mass = real_t(0);
  p->tar_min_non_diff_mass = real_t(0);
  p->average_pt2 = real_t(0);
  p->prob_log_distr = real_t(0);
  p->pt2_kink = real_t(0);
  p->max_number_of_collisions = real_t(0);
  p->prob_of_inel_interaction = real_t(0);
  p->cof_nuclear_destruction_pr = real_t(0);
  p->cof_nuclear_destruction = real_t(0);
  p->r2_of_nuclear_destruction = real_t(0);
  p->excitation_energy_per_wounded_nucleon = real_t(0);
  p->dof_nuclear_destruction = real_t(0);
  p->pt2_of_nuclear_destruction = real_t(0);
  p->max_pt2_of_nuclear_destruction = real_t(0);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 7; ++j) { p->proc_params[i][j] = real_t(0); }
  }
}

/// G4FTFParameters::G4FTFParameters - the part that is not Reset: the kink switch-off and the
/// SU(3)-symmetric gluon-splitting probabilities, which are CUMULATIVE
/// (Puubar, Puubar+Pddbar, Puubar+Pddbar+Pssbar).
template <typename real_t>
__host__ __device__ inline void ftf_parameters_construct(FtfParameters<real_t>* p,
                                                         bool enable_diff_disso_b_gt_10) {
  ftf_parameters_reset(p);
  // Row 4 of ProcParams is not in Reset's loop, so the constructor is the only place it can be
  // given a defined value at all. Geant4 leaves it indeterminate; leaving it indeterminate
  // here would be undefined behaviour rather than a reproduction of it.
  for (int j = 0; j < 7; ++j) { p->proc_params[4][j] = real_t(0); }
  p->enable_diff_dissociation_for_b_greater_10 = enable_diff_disso_b_gt_10;
  p->pt2_kink = real_t(0.0) * units::GeV<real_t>() * units::GeV<real_t>();
  const real_t puubar = real_t(1.0) / real_t(3.0);
  const real_t pddbar = real_t(1.0) / real_t(3.0);
  const real_t pssbar = real_t(1.0) / real_t(3.0);
  p->quark_probabilities_at_gluon_split_up[0] = puubar;
  p->quark_probabilities_at_gluon_split_up[1] = puubar + pddbar;
  p->quark_probabilities_at_gluon_split_up[2] = puubar + pddbar + pssbar;
  p->nucleon_assumed = false;
  p->index_tune = 0;
  p->refused = FtfRefusal::kNone;
  p->xs_refused = xs::XsRefusal::kNone;
}

/// G4FTFParameters::SetParams.
template <typename real_t>
__host__ __device__ inline void ftf_set_params(FtfParameters<real_t>* p, int proc, real_t a1,
                                               real_t b1, real_t a2, real_t b2, real_t a3,
                                               real_t atop, real_t ymin) {
  p->proc_params[proc][0] = a1;
  p->proc_params[proc][1] = b1;
  p->proc_params[proc][2] = a2;
  p->proc_params[proc][3] = b2;
  p->proc_params[proc][4] = a3;
  p->proc_params[proc][5] = atop;
  p->proc_params[proc][6] = ymin;
}

/// G4FTFParameters::SetParams from a collection's process row - the eight-argument call the
/// nucleon and pion branches make.
template <typename real_t>
__host__ __device__ inline void ftf_set_params_from(FtfParameters<real_t>* p, int proc,
                                                    const FtfParamColl<real_t>& c, int from) {
  ftf_set_params(p, proc, c.proc_a1[from], c.proc_b1[from], c.proc_a2[from], c.proc_b2[from],
                 c.proc_a3[from], c.proc_atop[from], c.proc_ymin[from]);
}

/// G4FTFParameters::SetMaxNumberOfCollisions. The commented-out `MaxNumberOfCollisions = -1`
/// and `SetProbOfInteraction(G4Exp(0.25*(Plab-Pbound)))` above the live lines are the reason
/// the else-branch looks redundant: below the bound the answer is one collision at
/// probability -1, not a sampled number.
template <typename real_t>
__host__ __device__ inline void ftf_set_max_number_of_collisions(FtfParameters<real_t>* p,
                                                                 real_t plab, real_t pbound) {
  if (plab > pbound) {
    p->max_number_of_collisions = plab / pbound;
    p->prob_of_inel_interaction = real_t(-1.0);
  } else {
    p->max_number_of_collisions = real_t(1);
    p->prob_of_inel_interaction = real_t(-1.0);
  }
}

/// G4FTFParameters::GetProcProb - the process probability as a function of rapidity.
/// `Prob = A1 exp(-B1 y) + A2 exp(-B2 y) + A3`, clamped at zero, and `Atop` (also clamped)
/// below `Ymin`.
template <typename real_t>
__host__ __device__ inline real_t ftf_get_proc_prob(const FtfParameters<real_t>* p, int proc,
                                                    real_t y) {
  real_t prob = real_t(0.0);
  if (y < p->proc_params[proc][6]) {
    prob = p->proc_params[proc][5];
    if (prob < real_t(0.)) { prob = real_t(0.); }
    return prob;
  }
  prob = p->proc_params[proc][0] * exp(-p->proc_params[proc][1] * y) +
         p->proc_params[proc][2] * exp(-p->proc_params[proc][3] * y) +
         p->proc_params[proc][4];
  if (prob < real_t(0.)) { prob = real_t(0.); }
  return prob;
}

/// G4FTFParameters::GammaElastic - the elastic profile function at impact parameter squared.
template <typename real_t>
__host__ __device__ inline real_t ftf_gamma_elastic(const FtfParameters<real_t>* p,
                                                    real_t impact_square) {
  return p->gamma0 * exp(-p->slope * impact_square);
}

/// G4FTFParameters::GetInelasticProbability - `2 Gamma - Gamma^2`, which is NOT confined to
/// [0, 1]: Gamma0 is `Slope*Xtotal/(20 pi)` and exceeds 1 for every hadron on a heavy nucleus,
/// so at small impact parameter this returns a large negative number. Reproduced as written;
/// ref/oracle/ftf_geom.csv carries the negative values (a proton on hydrogen at 1 GeV/c gives
/// -6.32 at b = 0), so a port that clamped would fail.
template <typename real_t>
__host__ __device__ inline real_t ftf_get_inelastic_probability(const FtfParameters<real_t>* p,
                                                                real_t impact_square) {
  const real_t gamma = ftf_gamma_elastic(p, impact_square);
  return real_t(2) * gamma - gamma * gamma;
}

/// G4FTFParameters::GetProbabilityOfInteraction - a hard disc of area Xtotal/pi.
template <typename real_t>
__host__ __device__ inline real_t ftf_get_probability_of_interaction(
    const FtfParameters<real_t>* p, real_t impact_square) {
  return (p->radius_of_hn_interactions2 > impact_square) ? real_t(1.0) : real_t(0.0);
}

/// The nine annihilation weight sets of InitForInteraction's anti-baryon branch, keyed by the
/// projectile's PDG code. Returns false for a code the block has no arm for, which Geant4
/// answers with `G4cout << "Unknown anti-baryon for FTF annihilation"` and then uses the
/// zero-initialised Xann_on_P/N.
template <typename real_t>
__host__ __device__ inline bool ftf_annihilation_weights(int pdg, real_t x_a, real_t x_b,
                                                          real_t x_c, real_t x_d,
                                                          real_t* on_p, real_t* on_n) {
  switch (pdg) {
    case -2212:  // Pbar
      *on_p = x_a + x_b * real_t(5.0) + x_c * real_t(5.0) + x_d * real_t(6.0);
      *on_n = x_a + x_b * real_t(4.0) + x_c * real_t(4.0) + x_d * real_t(4.0);
      return true;
    case -2112:  // anti-neutron
      *on_p = x_a + x_b * real_t(4.0) + x_c * real_t(4.0) + x_d * real_t(4.0);
      *on_n = x_a + x_b * real_t(5.0) + x_c * real_t(5.0) + x_d * real_t(6.0);
      return true;
    case -3122:  // anti-Lambda
      *on_p = x_a + x_b * real_t(3.0) + x_c * real_t(3.0) + x_d * real_t(2.0);
      *on_n = x_a + x_b * real_t(3.0) + x_c * real_t(3.0) + x_d * real_t(2.0);
      return true;
    case -3112:  // anti-Sigma-
      *on_p = x_a + x_b * real_t(2.0) + x_c * real_t(2.0) + x_d * real_t(0.0);
      *on_n = x_a + x_b * real_t(4.0) + x_c * real_t(4.0) + x_d * real_t(2.0);
      return true;
    case -3212:  // anti-Sigma0
      *on_p = x_a + x_b * real_t(3.0) + x_c * real_t(3.0) + x_d * real_t(2.0);
      *on_n = x_a + x_b * real_t(3.0) + x_c * real_t(3.0) + x_d * real_t(2.0);
      return true;
    case -3222:  // anti-Sigma+
      *on_p = x_a + x_b * real_t(4.0) + x_c * real_t(4.0) + x_d * real_t(2.0);
      *on_n = x_a + x_b * real_t(2.0) + x_c * real_t(2.0) + x_d * real_t(0.0);
      return true;
    case -3312:  // anti-Xi-
      *on_p = x_a + x_b * real_t(1.0) + x_c * real_t(1.0) + x_d * real_t(0.0);
      *on_n = x_a + x_b * real_t(2.0) + x_c * real_t(2.0) + x_d * real_t(0.0);
      return true;
    case -3322:  // anti-Xi0
      *on_p = x_a + x_b * real_t(2.0) + x_c * real_t(2.0) + x_d * real_t(0.0);
      *on_n = x_a + x_b * real_t(1.0) + x_c * real_t(1.0) + x_d * real_t(0.0);
      return true;
    case -3334:  // anti-Omega-
      *on_p = x_a;
      *on_n = x_a;
      return true;
    default:
      *on_p = real_t(0.0);
      *on_n = real_t(0.0);
      return false;
  }
}

/// G4FTFParameters::InitForInteraction.
///
/// @param proj      the projectile, as xs/projectile.cuh carries a G4ParticleDefinition.
/// @param the_a     target mass number
/// @param the_z     target atomic number
/// @param plab_per_particle  the lab momentum per particle, MeV/c. Geant4's fourth argument is
///                  named `s` in the declaration and `PlabPerParticle` in the definition; it
///                  is a MOMENTUM, and the header's name is wrong.
/// @param lund      the Lund tables, for the `else` branch's GetMinMass calls.
template <typename real_t>
__host__ __device__ inline void ftf_init_for_interaction(FtfParameters<real_t>* p,
                                                         const xs::Projectile<real_t>& proj,
                                                         int the_a, int the_z,
                                                         real_t plab_per_particle,
                                                         const LundTables<real_t>* lund) {
  using namespace g4gpu::hadronic::xs;
  ftf_parameters_reset(p);
  p->nucleon_assumed = false;
  p->refused = FtfRefusal::kNone;
  p->xs_refused = xs::XsRefusal::kNone;

  int projectile_pdg = proj.pdg;
  int projectile_abs_pdg = (projectile_pdg < 0) ? -projectile_pdg : projectile_pdg;
  real_t projectile_mass = proj.mass;
  real_t projectile_mass2 = projectile_mass * projectile_mass;

  int projectile_baryon_number = 0;
  int abs_projectile_baryon_number = 0;
  int abs_projectile_charge = 0;
  bool projectile_is_nucleus = false;

  const int abs_b = (proj.baryon_number < 0) ? -proj.baryon_number : proj.baryon_number;
  if (abs_b > 1) {
    projectile_is_nucleus = true;
    projectile_baryon_number = proj.baryon_number;
    abs_projectile_baryon_number = abs_b;
    const int q = static_cast<int>(proj.charge);
    abs_projectile_charge = (q < 0) ? -q : q;
    if (projectile_baryon_number > 1) {
      projectile_pdg = 2212;
      projectile_abs_pdg = 2212;
    } else {
      projectile_pdg = -2212;
      projectile_abs_pdg = 2212;
    }
    projectile_mass = units::proton_mass_c2<real_t>();
    projectile_mass2 = projectile_mass * projectile_mass;
  }

  real_t target_mass = units::proton_mass_c2<real_t>();
  real_t target_mass2 = target_mass * target_mass;

  real_t plab = plab_per_particle;
  const real_t elab = sqrt(plab * plab + projectile_mass2);
  const real_t kinetic_energy = elab - projectile_mass;
  const real_t S = projectile_mass2 + target_mass2 + real_t(2.0) * target_mass * elab;

  real_t x_total = real_t(0.0), x_elastic = real_t(0.0), x_annihilation = real_t(0.0);

  const real_t ylab = real_t(0.5) * log((elab + plab) / (elab - plab));

  const real_t ecms_sqr = S / units::GeV<real_t>() / units::GeV<real_t>();
  const real_t sqrt_s = sqrt(S) / units::GeV<real_t>();

  target_mass /= units::GeV<real_t>();
  target_mass2 /= (units::GeV<real_t>() * units::GeV<real_t>());
  projectile_mass /= units::GeV<real_t>();
  projectile_mass2 /= (units::GeV<real_t>() * units::GeV<real_t>());

  plab /= units::GeV<real_t>();
  real_t xftf = real_t(0.0);

  const int number_of_target_protons = the_z;
  const int number_of_target_neutrons = the_a - the_z;
  const int number_of_target_nucleons = number_of_target_protons + number_of_target_neutrons;

  // ---------- hadron projectile ----------
  //
  // A refusal here is NOT returned immediately, and that is not laxity. For a projectile in
  // the anti-baryon PDG window below - an anti-Lambda, say - Geant4 runs this block and then
  // OVERWRITES both Xtotal and Xelastic in the Arkhipov block, so the hadron-nucleon cross
  // section this block needs does not reach the answer at all. Returning here would refuse a
  // case the port can do exactly. The flag is carried and acted on after the Arkhipov block,
  // before the `Xtotal == 0` nucleon substitution, because THAT block would otherwise turn a
  // refusal into a proton's cross section - the one outcome docs/HADRONIC_PLAN.md section 6
  // rule 4 forbids.
  bool hadron_xs_refused = false;
  if (abs_projectile_baryon_number <= 1) {
    const HadXs<real_t> on_p = ggh_compute_cross_sections<real_t>(proj, kinetic_energy, 1, 1);
    const HadXs<real_t> on_n = ggh_compute_cross_sections<real_t>(proj, kinetic_energy, 0, 1);
    if (!on_p.ok() || !on_n.ok()) {
      hadron_xs_refused = true;
      p->xs_refused = on_p.ok() ? on_n.refused : on_p.refused;
    } else {
      x_total = (static_cast<real_t>(number_of_target_protons) * on_p.total +
                 static_cast<real_t>(number_of_target_neutrons) * on_n.total) /
                static_cast<real_t>(number_of_target_nucleons);
      x_elastic = (static_cast<real_t>(number_of_target_protons) * on_p.elastic +
                   static_cast<real_t>(number_of_target_neutrons) * on_n.elastic) /
                  static_cast<real_t>(number_of_target_nucleons);
      x_annihilation = real_t(0.0);
      x_total /= millibarn<real_t>();
      x_elastic /= millibarn<real_t>();
    }
  }

  // ---------- nucleus projectile ----------
  if (projectile_is_nucleus && projectile_baryon_number > 1) {
    const xs::Projectile<real_t> the_proton = xs::proton<real_t>();
    const xs::Projectile<real_t> the_neutron = xs::neutron<real_t>();
    const HadXs<real_t> pp = ggh_compute_cross_sections<real_t>(the_proton, kinetic_energy, 1, 1);
    // A NEUTRON projectile on a NEUTRON target, which is what `(Neutron, ..., 0, 1)` is - see
    // the file header, point 3.
    const HadXs<real_t> pn =
        ggh_compute_cross_sections<real_t>(the_neutron, kinetic_energy, 0, 1);
    const real_t apc = static_cast<real_t>(abs_projectile_charge);
    const real_t apb = static_cast<real_t>(abs_projectile_baryon_number);
    const real_t ntp = static_cast<real_t>(number_of_target_protons);
    const real_t ntn = static_cast<real_t>(number_of_target_neutrons);
    const real_t ntnucl = static_cast<real_t>(number_of_target_nucleons);
    x_total = (apc * ntp * pp.total + (apb - apc) * ntn * pp.total +
               (apc * ntn + (apb - apc) * ntp) * pn.total) /
              (apb * ntnucl);
    x_elastic = (apc * ntp * pp.elastic + (apb - apc) * ntn * pp.elastic +
                 (apc * ntn + (apb - apc) * ntp) * pn.elastic) /
                (apb * ntnucl);
    x_annihilation = real_t(0.0);
    x_total /= millibarn<real_t>();
    x_elastic /= millibarn<real_t>();
  }

  // ---------- anti-baryon or anti-nucleus projectile ----------
  //              anti Sigma^0_c                 anti Delta^-
  const bool anti_baryon_branch = (projectile_pdg >= -4112 && projectile_pdg <= -1114);
  if (anti_baryon_branch) {
    real_t x_a = real_t(0.0), x_b = real_t(0.0), x_c = real_t(0.0), x_d = real_t(0.0);
    real_t meson_prod_threshold =
        projectile_mass + target_mass + (real_t(2.0) * real_t(0.14) + real_t(0.016));

    if (plab_per_particle < real_t(40.0) * units::MeV<real_t>()) {
      // The projectile is at rest: six measured constants, not a limit of the formula below.
      x_total = real_t(1512.9);
      x_elastic = real_t(473.2);
      x_a = real_t(625.1);
      x_b = real_t(9.780);
      x_c = real_t(49.989);
      x_d = real_t(6.614);
    } else {
      // Arkhipov's PbarP total and elastic parameterisation.
      real_t log_s = log(ecms_sqr / real_t(33.0625));
      real_t xasmpt = real_t(36.04) + real_t(0.304) * log_s * log_s;  // mb
      log_s = log(sqrt_s / real_t(20.74));
      const real_t basmpt = real_t(11.92) + real_t(0.3036) * log_s * log_s;  // GeV^-2
      const real_t r0 = sqrt(real_t(0.40874044) * xasmpt - basmpt);          // GeV^-1

      const real_t flow_f =
          sqrt_s / sqrt(ecms_sqr * ecms_sqr + projectile_mass2 * projectile_mass2 +
                        target_mass2 * target_mass2 - real_t(2.0) * ecms_sqr * projectile_mass2 -
                        real_t(2.0) * ecms_sqr * target_mass2 -
                        real_t(2.0) * projectile_mass2 * target_mass2);

      x_total = xasmpt * (real_t(1.0) +
                          real_t(13.55) * flow_f / r0 / r0 / r0 *
                              (real_t(1.0) - real_t(4.47) / sqrt_s + real_t(12.38) / ecms_sqr -
                               real_t(12.43) / sqrt_s / ecms_sqr));

      xasmpt = real_t(4.4) + real_t(0.101) * log_s * log_s;
      x_elastic = xasmpt * (real_t(1.0) +
                            real_t(59.27) * flow_f / r0 / r0 / r0 *
                                (real_t(1.0) - real_t(6.95) / sqrt_s +
                                 real_t(23.54) / ecms_sqr - real_t(25.34) / sqrt_s / ecms_sqr));

      x_a = real_t(25.0) * flow_f;  // the three-shirts diagram

      if (sqrt_s < meson_prod_threshold) {
        // G4Pow::powA, not std::pow: a Taylor expansion about a tabulated point that differs
        // from the exact function at 1e-7 (data/g4pow.hh).
        x_b = real_t(3.13) + real_t(140.0) * data::g4pow_pow_a<real_t>(
                                                 meson_prod_threshold - sqrt_s, real_t(2.5));
        x_elastic -= real_t(3.0) * x_b;
      } else {
        x_b = real_t(6.8) / sqrt_s;
        x_elastic -= real_t(3.0) * x_b;
      }

      const real_t sum_m = projectile_mass + target_mass;
      x_c = real_t(2.0) * flow_f * sum_m * sum_m / ecms_sqr;  // rearrangement
      x_d = real_t(23.3) / ecms_sqr;                          // anti-quark-quark string
    }

    real_t xann_on_p = real_t(0.0), xann_on_n = real_t(0.0);
    ftf_annihilation_weights<real_t>(projectile_pdg, x_a, x_b, x_c, x_d, &xann_on_p,
                                     &xann_on_n);

    const real_t ntp = static_cast<real_t>(number_of_target_protons);
    const real_t ntn = static_cast<real_t>(number_of_target_neutrons);
    const real_t ntnucl = static_cast<real_t>(number_of_target_nucleons);
    if (!projectile_is_nucleus) {
      x_annihilation = (ntp * xann_on_p + ntn * xann_on_n) / ntnucl;
    } else {
      const real_t apc = static_cast<real_t>(abs_projectile_charge);
      const real_t apb = static_cast<real_t>(abs_projectile_baryon_number);
      x_annihilation = ((apc * ntp + (apb - apc) * ntn) * xann_on_p +
                        (apc * ntn + (apb - apc) * ntp) * xann_on_n) /
                       (apb * ntnucl);
    }

    meson_prod_threshold =
        projectile_mass + target_mass + (real_t(0.14) + real_t(0.08));  // Mpi + DeltaE
    if (sqrt_s > meson_prod_threshold) {
      xftf = real_t(36.0) * (real_t(1.0) - meson_prod_threshold / sqrt_s);
    }
    x_total = x_elastic + x_annihilation + xftf;
  }

  // The hadron-nucleon refusal is acted on here: after the Arkhipov block, which discards
  // whatever the hadron branch computed, and before the nucleon substitution, which would
  // otherwise answer with a proton's numbers.
  if (hadron_xs_refused && !anti_baryon_branch) {
    p->refused = FtfRefusal::kHadronNucleonXscRefused;
    return;
  }

  if (x_total == real_t(0.0)) {  // Projectile undefined, nucleon assumed
    p->nucleon_assumed = true;
    const xs::Projectile<real_t> the_proton = xs::proton<real_t>();
    // Both of these are a PROTON projectile - unlike the nucleus arm above, which uses a
    // neutron for the second.
    const HadXs<real_t> pp = ggh_compute_cross_sections<real_t>(the_proton, kinetic_energy, 1, 1);
    const HadXs<real_t> pn = ggh_compute_cross_sections<real_t>(the_proton, kinetic_energy, 0, 1);
    const real_t ntp = static_cast<real_t>(number_of_target_protons);
    const real_t ntn = static_cast<real_t>(number_of_target_neutrons);
    const real_t ntnucl = static_cast<real_t>(number_of_target_nucleons);
    x_total = (ntp * pp.total + ntn * pn.total) / ntnucl;
    x_elastic = (ntp * pp.elastic + ntn * pn.elastic) / ntnucl;
    x_annihilation = real_t(0.0);
    x_total /= millibarn<real_t>();
    x_elastic /= millibarn<real_t>();
  }

  // ---------- geometrical parameters ----------
  p->x_total = x_total;
  p->x_elastic = x_elastic;
  p->x_inelastic = x_total - x_elastic;
  p->prob_of_elastic_scatt =
      (x_total == real_t(0.0)) ? real_t(0.0) : (x_elastic / x_total);
  p->radius_of_hn_interactions2 = x_total / units::pi<real_t>() / real_t(10.0);
  p->prob_of_annihilation = ((x_total - x_elastic) == real_t(0.0))
                                ? real_t(0.0)
                                : (x_annihilation / (x_total - x_elastic));

  if (x_elastic > real_t(0.0)) {
    // SetSlope's argument is in GeV^-2 and the member is in fm^-2: FTFSlope = 12.84/Slope.
    const real_t slope_gev2 = x_total * x_total / real_t(16.0) / units::pi<real_t>() /
                              x_elastic / real_t(0.3894);
    p->slope = real_t(12.84) / slope_gev2;
    // Geant4 recomputes the same expression here rather than reusing it. Written the same way
    // so that the two roundings match.
    p->avarage_pt2_of_elastic_scattering =
        real_t(1.0) / (x_total * x_total / real_t(16.0) / units::pi<real_t>() / x_elastic /
                       real_t(0.3894)) *
        units::GeV<real_t>() * units::GeV<real_t>();
  } else {
    p->slope = real_t(12.84) / real_t(1.0);
    p->avarage_pt2_of_elastic_scattering = real_t(0.0);
  }
  p->gamma0 = p->slope * x_total / real_t(10.0) / real_t(2.0) / units::pi<real_t>();

  const real_t xinel = x_total - x_elastic;

  // ---------- excitation parameters, by projectile class ----------
  const FtfParamColl<real_t> baryon = ftf_param_coll_baryon<real_t>();
  const FtfParamColl<real_t> meson = ftf_param_coll_meson<real_t>();
  const FtfParamColl<real_t> pion = ftf_param_coll_pion<real_t>();
  p->index_tune = 0;  // G4FTFTunings::GetIndexTune, measured - see the file header

  const real_t gev = units::GeV<real_t>();

  if (projectile_pdg == 2212 || projectile_pdg == 2112) {  // proton or neutron
    ftf_set_params_from(p, 0, baryon, 0);
    ftf_set_params_from(p, 1, baryon, 1);
    if (xinel > real_t(0.0)) {
      ftf_set_params(p, 2, real_t(6.0) / xinel, real_t(0.0),
                     -real_t(6.0) / xinel * real_t(16.28), real_t(3.0), real_t(0.0),
                     real_t(0.0), real_t(0.93));
      ftf_set_params(p, 3, real_t(6.0) / xinel, real_t(0.0),
                     -real_t(6.0) / xinel * real_t(16.28), real_t(3.0), real_t(0.0),
                     real_t(0.0), real_t(0.93));
      ftf_set_params_from(p, 4, baryon, 4);
    } else {
      ftf_set_params(p, 2, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                     real_t(0));
      ftf_set_params(p, 3, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                     real_t(0));
      ftf_set_params(p, 4, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                     real_t(0));
    }
    if ((abs_projectile_baryon_number > 10 || number_of_target_nucleons > 10) &&
        !p->enable_diff_dissociation_for_b_greater_10) {
      if (!baryon.proj_diff_dissociation) {
        ftf_set_params(p, 2, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                       real_t(-100.0));
      }
      if (!baryon.tgt_diff_dissociation) {
        ftf_set_params(p, 3, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                       real_t(-100.0));
      }
    }
    p->delta_prob_at_quark_exchange = baryon.delta_prob_at_quark_exchange;
    p->prob_of_same_quark_exchange =
        (number_of_target_nucleons > 26) ? real_t(1.0) : baryon.prob_of_same_quark_exchange;
    p->proj_min_diff_mass = baryon.proj_min_diff_mass * gev;
    p->proj_min_non_diff_mass = baryon.proj_min_non_diff_mass * gev;
    p->tar_min_diff_mass = baryon.tgt_min_diff_mass * gev;
    p->tar_min_non_diff_mass = baryon.tgt_min_non_diff_mass * gev;
    p->average_pt2 = baryon.average_pt2 * gev * gev;
    p->prob_log_distr_prd = baryon.prob_log_distr_prd;
    p->prob_log_distr = baryon.prob_log_distr;

  } else if (projectile_pdg == -2212 || projectile_pdg == -2112) {  // anti-nucleon
    ftf_set_params(p, 0, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                   real_t(1000.0));
    ftf_set_params(p, 1, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                   real_t(1000.0));
    if (xinel > real_t(0.)) {
      ftf_set_params(p, 2, real_t(6.0) / xinel, real_t(0.0),
                     -real_t(6.0) / xinel * real_t(16.28), real_t(3.0), real_t(0.0),
                     real_t(0.0), real_t(0.93));
      ftf_set_params(p, 3, real_t(6.0) / xinel, real_t(0.0),
                     -real_t(6.0) / xinel * real_t(16.28), real_t(3.0), real_t(0.0),
                     real_t(0.0), real_t(0.93));
      ftf_set_params(p, 4, real_t(1.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0),
                     real_t(0.0), real_t(0.93));
    } else {
      ftf_set_params(p, 2, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                     real_t(0));
      ftf_set_params(p, 3, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                     real_t(0));
      ftf_set_params(p, 4, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                     real_t(0));
    }
    // NOTE: no EnableDiffDissociationForBGreater10 test on this arm - the anti-nucleon branch
    // was not updated when the flag was introduced, so its diffraction is switched off for
    // A > 10 whatever the flag says. The baryon arm above reads the flag.
    if (abs_projectile_baryon_number > 10 || number_of_target_nucleons > 10) {
      if (!baryon.proj_diff_dissociation) {
        ftf_set_params(p, 2, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                       real_t(-100.0));
      }
      if (!baryon.tgt_diff_dissociation) {
        ftf_set_params(p, 3, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                       real_t(-100.0));
      }
    }
    p->delta_prob_at_quark_exchange = real_t(0.0);
    p->prob_of_same_quark_exchange = real_t(0.0);
    // projectile_mass and target_mass are in GeV by now, and the setters multiply by GeV.
    p->proj_min_diff_mass = (projectile_mass + real_t(0.22)) * gev;
    p->proj_min_non_diff_mass = (projectile_mass + real_t(0.22)) * gev;
    p->tar_min_diff_mass = (target_mass + real_t(0.22)) * gev;
    p->tar_min_non_diff_mass = (target_mass + real_t(0.22)) * gev;
    p->average_pt2 = real_t(0.3) * gev * gev;
    p->prob_log_distr_prd = real_t(0.55);
    p->prob_log_distr = real_t(0.55);

  } else if (projectile_abs_pdg == 211 || projectile_pdg == 111) {  // pion
    for (int i = 0; i < 5; ++i) { ftf_set_params_from(p, i, pion, i); }
    // `AbsProjectileBaryonNumber > 10` for a pion is always false - Geant4's own comment asks
    // "how can it be |ProjectileBaryonNumber| > 10 if projectile is a pion ???" - so this is
    // the target test only.
    if (abs_projectile_baryon_number > 10 || number_of_target_nucleons > 10) {
      if (!pion.proj_diff_dissociation) {
        ftf_set_params(p, 2, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                       real_t(-100.0));
      }
      if (!pion.tgt_diff_dissociation) {
        ftf_set_params(p, 3, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                       real_t(-100.0));
      }
    }
    p->delta_prob_at_quark_exchange = pion.delta_prob_at_quark_exchange;
    p->proj_min_diff_mass = pion.proj_min_diff_mass * gev;
    p->proj_min_non_diff_mass = pion.proj_min_non_diff_mass * gev;
    p->tar_min_diff_mass = pion.tgt_min_diff_mass * gev;
    p->tar_min_non_diff_mass = pion.tgt_min_non_diff_mass * gev;
    p->average_pt2 = pion.average_pt2 * gev * gev;
    p->prob_log_distr_prd = pion.prob_log_distr_prd;
    p->prob_log_distr = pion.prob_log_distr;
    // ProbOfSameQuarkExchange is NOT set on this arm: the pion collection zeroes it and
    // InitForInteraction never assigns it, so it keeps Reset's 0.

  } else if (projectile_abs_pdg == 321 || projectile_abs_pdg == 311 ||
             projectile_pdg == 130 || projectile_pdg == 310) {  // kaon
    ftf_set_params(p, 0, real_t(60.0), real_t(2.5), real_t(0.0), real_t(0.0), real_t(0.0),
                   real_t(0.0), real_t(-100.0));
    ftf_set_params(p, 1, real_t(6.0), real_t(1.0), real_t(-24.33), real_t(2.0), real_t(0.0),
                   real_t(0.0), real_t(1.40));
    ftf_set_params(p, 2, real_t(2.76), real_t(1.2), real_t(-22.5), real_t(2.7), real_t(0.04),
                   real_t(0.0), real_t(1.40));
    ftf_set_params(p, 3, real_t(1.09), real_t(0.5), real_t(-8.88), real_t(2.), real_t(0.05),
                   real_t(0.0), real_t(1.40));
    ftf_set_params(p, 4, real_t(1.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0),
                   real_t(0.0), real_t(0.93));
    if (abs_projectile_baryon_number > 10 || number_of_target_nucleons > 10) {
      ftf_set_params(p, 2, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                     real_t(-100.0));
      ftf_set_params(p, 3, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                     real_t(-100.0));
    }
    p->delta_prob_at_quark_exchange = real_t(0.6);
    p->proj_min_diff_mass = real_t(0.7) * gev;
    p->proj_min_non_diff_mass = real_t(0.7) * gev;
    p->tar_min_diff_mass = real_t(1.16) * gev;
    p->tar_min_non_diff_mass = real_t(1.16) * gev;
    p->average_pt2 = real_t(0.3) * gev * gev;
    p->prob_log_distr_prd = real_t(0.55);
    p->prob_log_distr = real_t(0.55);

  } else {  // any other baryon or meson
    if (projectile_abs_pdg > 1000) {  // a baryon, treated as p or n
      ftf_set_params(p, 0, real_t(13.71), real_t(1.75), real_t(-30.69), real_t(3.0),
                     real_t(0.0), real_t(1.0), real_t(0.93));
      ftf_set_params(p, 1, real_t(25.0), real_t(1.0), real_t(-50.34), real_t(1.5), real_t(0.0),
                     real_t(0.0), real_t(1.4));
      if (xinel > real_t(0.)) {
        ftf_set_params(p, 2, real_t(6.0) / xinel, real_t(0.0),
                       -real_t(6.0) / xinel * real_t(16.28), real_t(3.0), real_t(0.0),
                       real_t(0.0), real_t(0.93));
        ftf_set_params(p, 3, real_t(6.0) / xinel, real_t(0.0),
                       -real_t(6.0) / xinel * real_t(16.28), real_t(3.0), real_t(0.0),
                       real_t(0.0), real_t(0.93));
        ftf_set_params(p, 4, real_t(1.0), real_t(0.0), real_t(-2.01), real_t(0.5), real_t(0.0),
                       real_t(0.0), real_t(1.4));
      } else {
        ftf_set_params(p, 2, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                       real_t(0));
        ftf_set_params(p, 3, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                       real_t(0));
        ftf_set_params(p, 4, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                       real_t(0));
      }
    } else {  // a meson, treated as a kaon
      ftf_set_params(p, 0, real_t(60.0), real_t(2.5), real_t(0.0), real_t(0.0), real_t(0.0),
                     real_t(0.0), real_t(-100.0));
      ftf_set_params(p, 1, real_t(6.0), real_t(1.0), real_t(-24.33), real_t(2.0), real_t(0.0),
                     real_t(0.0), real_t(1.40));
      ftf_set_params(p, 2, real_t(2.76), real_t(1.2), real_t(-22.5), real_t(2.7),
                     real_t(0.04), real_t(0.0), real_t(1.40));
      ftf_set_params(p, 3, real_t(1.09), real_t(0.5), real_t(-8.88), real_t(2.), real_t(0.05),
                     real_t(0.0), real_t(1.40));
      ftf_set_params(p, 4, real_t(1.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0),
                     real_t(0.0), real_t(0.93));
    }
    if (abs_projectile_baryon_number > 10 || number_of_target_nucleons > 10) {
      ftf_set_params(p, 2, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                     real_t(-100.0));
      ftf_set_params(p, 3, real_t(0), real_t(0), real_t(0), real_t(0), real_t(0), real_t(0),
                     real_t(-100.0));
    }
    p->delta_prob_at_quark_exchange = real_t(0.0);
    p->prob_of_same_quark_exchange = real_t(0.0);
    // The minimal diffractive masses of a hyperon come out of the Lund minimal-mass tables,
    // by the projectile's own PDG code, and the TARGET's come from a NEUTRON's - not a
    // proton's, although TargetMass above is the proton's.
    p->proj_min_diff_mass = ftf_get_min_mass(lund, proj.pdg) / gev * gev;
    p->proj_min_non_diff_mass = ftf_get_min_mass(lund, proj.pdg) / gev * gev;
    p->tar_min_diff_mass = ftf_get_min_mass(lund, 2112) / gev * gev;
    p->tar_min_non_diff_mass = ftf_get_min_mass(lund, 2112) / gev * gev;
    p->average_pt2 = real_t(0.3) * gev * gev;
    p->prob_log_distr_prd = real_t(0.55);
    p->prob_log_distr = real_t(0.55);
  }

  // ---------- nuclear destruction, by projectile class ----------
  if (projectile_abs_pdg < 1000) {  // meson projectile
    ftf_set_max_number_of_collisions(p, plab, real_t(2.0));
    real_t coeff = meson.nuclear_tgt_destruct_p1;
    if (meson.nuclear_tgt_destruct_p1_adep) {
      coeff *= static_cast<real_t>(number_of_target_nucleons);
    }
    real_t exfactor =
        exp(meson.nuclear_tgt_destruct_p2 * (ylab - meson.nuclear_tgt_destruct_p3));
    coeff *= exfactor;
    coeff /= (real_t(1.) + exfactor);
    p->cof_nuclear_destruction = coeff;
    p->r2_of_nuclear_destruction = meson.r2_of_nuclear_destruct;
    p->dof_nuclear_destruction = meson.dof_nuclear_destruct;
    coeff = meson.pt2_nuclear_destruct_p2;
    exfactor = exp(meson.pt2_nuclear_destruct_p3 * (ylab - meson.pt2_nuclear_destruct_p4));
    coeff *= exfactor;
    coeff /= (real_t(1.) + exfactor);
    p->pt2_of_nuclear_destruction = (meson.pt2_nuclear_destruct_p1 + coeff) * gev * gev;
    p->max_pt2_of_nuclear_destruction = meson.max_pt2_of_nuclear_destruct;
    p->excitation_energy_per_wounded_nucleon = meson.exci_energy_per_wounded_nucleon;

  } else if (projectile_pdg == -2212 || projectile_pdg == -2112) {  // anti-baryon
    ftf_set_max_number_of_collisions(p, plab, real_t(2.0));
    p->cof_nuclear_destruction =
        real_t(0.00481) * static_cast<real_t>(number_of_target_nucleons) *
        exp(real_t(4.0) * (ylab - real_t(2.1))) /
        (real_t(1.0) + exp(real_t(4.0) * (ylab - real_t(2.1))));
    p->r2_of_nuclear_destruction = real_t(1.5) * fermi<real_t>() * fermi<real_t>();
    p->dof_nuclear_destruction = real_t(0.3);
    p->pt2_of_nuclear_destruction =
        (real_t(0.035) + real_t(0.04) * exp(real_t(4.0) * (ylab - real_t(2.5))) /
                             (real_t(1.0) + exp(real_t(4.0) * (ylab - real_t(2.5))))) *
        gev * gev;
    p->max_pt2_of_nuclear_destruction = real_t(1.0) * gev * gev;
    p->excitation_energy_per_wounded_nucleon = real_t(40.0) * units::MeV<real_t>();
    if (plab < real_t(2.0)) {  // GeV/c - "for slow anti-baryon we have to garanty putting on
                               // mass-shell"
      p->cof_nuclear_destruction = real_t(0.0);
      p->r2_of_nuclear_destruction = real_t(1.5) * fermi<real_t>() * fermi<real_t>();
      p->dof_nuclear_destruction = real_t(0.01);
      p->pt2_of_nuclear_destruction = real_t(0.035) * gev * gev;
      p->max_pt2_of_nuclear_destruction = real_t(0.04) * gev * gev;
    }

  } else {  // baryon projectile assumed
    ftf_set_max_number_of_collisions(p, plab, real_t(2.0));
    real_t coeff = baryon.nuclear_proj_destruct_p1;
    if (baryon.nuclear_proj_destruct_p1_nbrndep) {
      coeff *= static_cast<real_t>(abs_projectile_baryon_number);
    }
    real_t exfactor =
        exp(baryon.nuclear_proj_destruct_p2 * (ylab - baryon.nuclear_proj_destruct_p3));
    coeff *= exfactor;
    coeff /= (real_t(1.) + exfactor);
    p->cof_nuclear_destruction_pr = coeff;

    coeff = baryon.nuclear_tgt_destruct_p1;
    if (baryon.nuclear_tgt_destruct_p1_adep) {
      coeff *= static_cast<real_t>(number_of_target_nucleons);
    }
    exfactor = exp(baryon.nuclear_tgt_destruct_p2 * (ylab - baryon.nuclear_tgt_destruct_p3));
    coeff *= exfactor;
    coeff /= (real_t(1.) + exfactor);
    p->cof_nuclear_destruction = coeff;

    p->r2_of_nuclear_destruction = baryon.r2_of_nuclear_destruct;
    p->dof_nuclear_destruction = baryon.dof_nuclear_destruct;

    coeff = baryon.pt2_nuclear_destruct_p2;
    exfactor = exp(baryon.pt2_nuclear_destruct_p3 * (ylab - baryon.pt2_nuclear_destruct_p4));
    coeff *= exfactor;
    coeff /= (real_t(1.) + exfactor);
    p->pt2_of_nuclear_destruction = (baryon.pt2_nuclear_destruct_p1 + coeff) * gev * gev;

    p->max_pt2_of_nuclear_destruction = baryon.max_pt2_of_nuclear_destruct;
    p->excitation_energy_per_wounded_nucleon = baryon.exci_energy_per_wounded_nucleon;
  }
}

}  // namespace g4gpu::hadronic::ftf
