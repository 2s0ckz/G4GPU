// The nucleus model, as a contract. Include THIS file, not the four it pulls in.
//
// `G4Fancy3DNucleus` and the classes under it are shared between `G4BinaryCascade` (through
// `G4BinaryLightIonReaction`, which builds one for the projectile and one for the target) and
// `G4FTFModel` (which builds one for each participant and calls `SortNucleonsDecZ` on it). They
// live under `bic/nucleus/` because P9 transcribed them; this header is the surface P11 may
// depend on, and nothing below it will change shape without a note in docs/PORTED.md.
//
// ## What a caller has to provide
//
// Storage. A nucleus of mass number A needs A `Nucleon`s, and `Init` needs three A-sized
// scratch arrays plus a 600-double block:
//
//     bic::Nucleon        nucleons[kMaxA];
//     Vec3<double>        momentum[kMaxA];
//     double              fermi_p[kMaxA];
//     bic::NucleusSortEntry test_sums[kMaxA];
//     double              flat[bic::kFlatBlock];
//
//     bic::Nucleus3D nuc;  nuc.nucleons = nucleons;  nuc.capacity = kMaxA;
//     bic::Nucleus3DScratch sc;  sc.momentum = momentum;  sc.fermi_p = fermi_p;
//     sc.test_sums = test_sums;  sc.flat_block = flat;  sc.capacity = kMaxA;
//
//     bic::NucleusReport rep = bic::nucleus_init(nuc, sc, A, Z, rng);
//     if (rep.fatal()) { ... }   // nothing was built; the nucleus is empty
//
// Nothing here allocates and nothing here is a local array: a nucleus is up to 250 nucleons of
// 72 bytes each, which is 18 kB and does not belong in a kernel's frame. The scratch MAY be
// shared between successive `nucleus_init` calls and it does not matter either way: it carries
// no state between calls. It used to carry a Gaussian latch, because this package implemented
// `CLHEP::RandGauss` where Geant4 uses `CLHEP::RandGaussQ`, which has no such state; the two
// fields are gone and the struct is two doubles smaller. docs/RISK.md V180, and see note 5 in
// `fancy_3d_nucleus.cuh`.
//
// **What changed under this contract on 2026-09-19, for a caller that compares against an
// oracle event by event.** Nothing in the SHAPE of a call changed except those two fields, and
// nothing in any DISTRIBUTION changed. What changed is which random numbers come out:
//
//   * every C12 built anywhere now walks Geant4's stream (V180): a different number of uniforms
//     and different nucleon positions. No other nuclide has a Gaussian in its position sampler.
//   * every nucleon of EVERY nuclide now gets a different Fermi momentum vector (V181) - the
//     same number of draws, the same distribution, the three components reversed.
//
// A caller whose test is a distribution, a histogram or a moment sees no difference. A caller
// whose test replays a recorded Geant4 stream - which is the only kind of test that could ever
// have caught either bug - must re-measure. Nothing in `ref/oracle/` changes: the oracle is
// Geant4 and Geant4 did not move.
//
// `Rng` needs one member: `double uniform()`, in (0,1). `core/rng.cuh`'s `Philox<double>` has it.
//
// ## What the model guarantees, and what it does not
//
// GUARANTEED: A nucleons with types summing to Z protons, (A-Z-L) neutrons and L lambdas;
// positions inside the density's 0.001-relative-density radius with no pair closer than the
// hard-core distance; a total three-momentum of zero unless `rep.reduce_sum_failed`.
//
// NOT guaranteed, and both are Geant4's behaviour rather than an omission:
//   * **the nucleons are off their mass shells, downwards.** `Get4Momentum().mag() < GetPDGMass()`
//     for every one of them. A caller that hands them to `preco::propagate_residual` selects
//     that function's QGS arm and must pass a primary four-momentum or be refused.
//     docs/RISK.md V50. `G4FTFModel` puts each hit nucleon back on shell before handing over;
//     a caller that does not, must.
//   * **the total three-momentum is zero, the total ENERGY is not the nuclear mass.** Each
//     nucleon carries `m - BE/A`, so the sum is `sum(m) - BE` - which is
//     `Z m_p + (A-Z) m_n - BE`, i.e. exactly `Nucleus3D::mass()`, and NOT
//     `G4NucleiProperties::GetNuclearMass(A, Z)` nor `G4IonTable::GetIonMass(Z, A)`. All three
//     are used in 11.1.1 and they differ; `mass()`'s comment says where.
//
// ## Refused, by name
//
//   * `Nucleus3D::mass()` for `my_l > 0` - the `G4HyperNucleiProperties` branch. Sets
//     `hyper_nucleus`. `Init` itself accepts lambdas and places them, because `ChooseNucleons`
//     is the only method that knows about them and it is three lines.
//   * anti-nuclei: `G4Nucleon` has `SetParticleType` overloads for anti-proton, anti-neutron
//     and anti-lambda, and no code path in `G4Fancy3DNucleus` calls them. There is nothing to
//     port and nothing to refuse in `Init`; a caller that wants an anti-nucleus wants a
//     different class.
//   * the nucleon-array capacity, reported as `capacity` rather than truncated.
//   * `G4Fancy3DNucleus::Init(G4double, G4double, G4int)` - the `NON_INTEGER_A_Z` overload,
//     compiled out in 11.1.1.
#ifndef G4GPU_BIC_NUCLEUS_MODEL_CUH
#define G4GPU_BIC_NUCLEUS_MODEL_CUH

#include "physics/hadronic/bic/nucleus/fancy_3d_nucleus.cuh"
#include "physics/hadronic/bic/nucleus/fermi_momentum.cuh"
#include "physics/hadronic/bic/nucleus/nuclear_density.cuh"
#include "physics/hadronic/bic/nucleus/nucleon.cuh"

namespace g4gpu::bic {

/// The largest mass number this port's nucleus model accepts, and why it is a number.
///
/// `G4Fancy3DNucleus`'s vectors are constructed at 250 and resized to A, so Geant4 has no
/// limit. A device kernel does. 250 covers every nuclide in `data/natural_isotopes.hh` (the
/// heaviest is U238) and every fragment a cascade can produce from one; a caller that asks for
/// more gets `NucleusReport::capacity` and not a truncated nucleus.
inline constexpr int kMaxNucleons = 250;

}  // namespace g4gpu::bic

#endif
