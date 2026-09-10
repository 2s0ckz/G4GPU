// G4ComponentGGHadronNucleusXsc - hadron on nucleus in the Glauber-Gribov approximation.
//
// Transcribed from source/processes/hadronic/cross_sections/src/G4ComponentGGHadronNucleusXsc.cc
// and its header's inlines (11.1.1): ComputeCrossSections, GetParticleBarCorTot,
// GetParticleBarCorIn, GetElasticGlauberGribov, GetInelasticGlauberGribov, and the four pairs
// of 93-entry correction tables at the bottom of the .cc.
//
// This is the above-91-GeV branch of G4BGGNucleonElasticXS and G4BGGNucleonInelasticXS - the
// third of the three sub-cases docs/PORTED.md 2.1 lists as refused in barashenkov_xs.cuh -
// and the high-energy hand-over of G4ParticleInelasticXS (for a proton), G4NeutronInelasticXS
// and G4NeutronElasticXS above their data tables' 20 GeV top.
//
// THE MODEL, IN ONE PARAGRAPH
//
// Sum the hadron-nucleon cross section over the Z protons and N neutrons, divide by the
// geometric area 2*pi*R^2 to get an opacity `ratio`, and take a logarithm of it: the total is
// `area*log(1+ratio)` and the inelastic `area*log(1+2.4*ratio)/2.4`, with elastic the
// difference. `cofInelastic` is 2.4 for everything except kaons, where it is 2.2 and the
// radius is RadiusKNGG rather than RadiusHNGG. The two `BarCorrection` tables then multiply
// the result by a per-Z number near 1.08 - Geant4's own comment says they are the "correction
// at Tkin = 90 GeV GG -> Barashenkov", i.e. they exist to make this model agree with the
// Barashenkov table at the energy the BGG classes switch between them, and they are 1.0 for
// every particle that is not p, n, pi+ or pi-.
//
// A = 1 IS A DIFFERENT FORMULA, NOT A LIMIT
//
// `if (A > 1)` takes the Glauber-Gribov branch; A = 1 returns the hadron-nucleon cross section
// unmodified, with production equal to inelastic and diffraction a flat 0.2 of it. The
// nucleus area, the logarithm and the bar corrections are all skipped. That matters because
// the BGG classes call `GetInelasticGlauberGribov(dp, Z, theA[Z])` with theA[1] = 1.
//
// THE CACHE IS NOT PORTED, AND THAT IS A CORRECTNESS POINT
//
// ComputeCrossSections opens with a five-field cache check and returns early on a hit. It is
// pure memoisation of a pure function, so dropping it changes no number - but it is also the
// reason the real class is stateful, and every accessor (GetTotalElementCrossSection,
// GetElasticGlauberGribov, ...) is a call to ComputeCrossSections followed by a read of one
// member. Here the five results come back together in one struct, which is what the callers
// want anyway: G4ComponentGGNuclNuclXsc's hydrogen branch reads all five at once.
//
// The G4double A overloads (GetElasticElementCrossSection and friends) round A with G4lrint
// before calling ComputeCrossSections, so an element cross section evaluated at aeff[Z] is
// evaluated at the *rounded* aeff[Z]. That rounding is the caller's; ggh_compute_cross_sections
// takes an integer A and the callers below do the rounding where Geant4 does it.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/xs/g4pow_extra.cuh"
#include "physics/hadronic/xs/hadron_nucleon_xsc.cuh"
#include "physics/hadronic/xs/nuclear_radii.cuh"
#include "physics/hadronic/xs/projectile.cuh"
#include "physics/hadronic/xs/refusal.cuh"

namespace g4gpu::hadronic::xs {

/// G4ComponentGGHadronNucleusXsc::fNeutronBarCorrectionTot, indexed by Z (0..92).
__host__ __device__ inline const double* gg_neutron_bar_cor_tot() {
  static const double v[93] = {
    1.0, 1.0, 1.42517e+00, 1.082002e+00, 1.116171e+00, 1.078747e+00, 1.061315e+00,
    1.058205e+00, 1.082663e+00, 1.068500e+00, 1.076912e+00, 1.083475e+00, 1.079117e+00, 1.071856e+00,
    1.071990e+00, 1.073774e+00, 1.079356e+00, 1.081314e+00, 1.082056e+00, 1.090772e+00, 1.096776e+00,
    1.095828e+00, 1.097678e+00, 1.099157e+00, 1.103677e+00, 1.105132e+00, 1.109806e+00, 1.110816e+00,
    1.117378e+00, 1.115165e+00, 1.115710e+00, 1.111855e+00, 1.110482e+00, 1.110112e+00, 1.106676e+00,
    1.108706e+00, 1.105549e+00, 1.106318e+00, 1.106242e+00, 1.107672e+00, 1.107342e+00, 1.108119e+00,
    1.106655e+00, 1.102588e+00, 1.096657e+00, 1.092920e+00, 1.086629e+00, 1.083592e+00, 1.076030e+00,
    1.083777e+00, 1.089460e+00, 1.086545e+00, 1.079924e+00, 1.082218e+00, 1.077798e+00, 1.077062e+00,
    1.072825e+00, 1.072241e+00, 1.072104e+00, 1.072490e+00, 1.069829e+00, 1.070398e+00, 1.065458e+00,
    1.064968e+00, 1.060524e+00, 1.060048e+00, 1.057620e+00, 1.056428e+00, 1.055366e+00, 1.055017e+00,
    1.052304e+00, 1.051767e+00, 1.049728e+00, 1.048745e+00, 1.047399e+00, 1.045876e+00, 1.042972e+00,
    1.041824e+00, 1.039993e+00, 1.039021e+00, 1.036627e+00, 1.034176e+00, 1.032526e+00, 1.033633e+00,
    1.036107e+00, 1.037803e+00, 1.031266e+00, 1.032991e+00, 1.033284e+00, 1.035015e+00, 1.033945e+00,
    1.037075e+00, 1.034721e+00};
  return v;
}

/// G4ComponentGGHadronNucleusXsc::fNeutronBarCorrectionIn, indexed by Z (0..92).
__host__ __device__ inline const double* gg_neutron_bar_cor_in() {
  static const double v[93] = {
    1.0, 1.0, 1.167421e+00, 1.156250e+00, 1.205364e+00, 1.154225e+00, 1.120391e+00,
    1.124632e+00, 1.129460e+00, 1.107863e+00, 1.102152e+00, 1.104593e+00, 1.100285e+00, 1.098450e+00,
    1.092677e+00, 1.101124e+00, 1.106461e+00, 1.115049e+00, 1.123903e+00, 1.126661e+00, 1.131259e+00,
    1.133949e+00, 1.134185e+00, 1.133767e+00, 1.132813e+00, 1.131515e+00, 1.144338e+00, 1.134171e+00,
    1.139206e+00, 1.148474e+00, 1.142189e+00, 1.140725e+00, 1.140100e+00, 1.139848e+00, 1.137674e+00,
    1.138645e+00, 1.136339e+00, 1.136439e+00, 1.135946e+00, 1.136431e+00, 1.135702e+00, 1.135703e+00,
    1.134113e+00, 1.131935e+00, 1.128381e+00, 1.126373e+00, 1.122453e+00, 1.120908e+00, 1.115953e+00,
    1.115947e+00, 1.114426e+00, 1.111749e+00, 1.106207e+00, 1.107494e+00, 1.103622e+00, 1.102576e+00,
    1.098816e+00, 1.097889e+00, 1.097306e+00, 1.097130e+00, 1.094578e+00, 1.094552e+00, 1.090222e+00,
    1.089358e+00, 1.085409e+00, 1.084560e+00, 1.082182e+00, 1.080773e+00, 1.079464e+00, 1.078724e+00,
    1.076121e+00, 1.075235e+00, 1.073159e+00, 1.071920e+00, 1.070395e+00, 1.069503e+00, 1.067525e+00,
    1.066919e+00, 1.065779e+00, 1.065319e+00, 1.063730e+00, 1.062092e+00, 1.061085e+00, 1.059908e+00,
    1.059815e+00, 1.059109e+00, 1.051920e+00, 1.051258e+00, 1.049473e+00, 1.048823e+00, 1.045984e+00,
    1.046435e+00, 1.042614e+00};
  return v;
}

/// G4ComponentGGHadronNucleusXsc::fProtonBarCorrectionTot, indexed by Z (0..92).
__host__ __device__ inline const double* gg_proton_bar_cor_tot() {
  static const double v[93] = {
    1.0, 1.0, 1.118515e+00, 1.082000e+00, 1.116169e+00, 1.078745e+00, 1.061313e+00,
    1.058203e+00, 1.082661e+00, 1.068498e+00, 1.076910e+00, 1.083474e+00, 1.079115e+00, 1.071854e+00,
    1.071988e+00, 1.073772e+00, 1.079355e+00, 1.081312e+00, 1.082054e+00, 1.090770e+00, 1.096774e+00,
    1.095827e+00, 1.097677e+00, 1.099156e+00, 1.103676e+00, 1.105130e+00, 1.109805e+00, 1.110814e+00,
    1.117377e+00, 1.115163e+00, 1.115708e+00, 1.111853e+00, 1.110480e+00, 1.110111e+00, 1.106674e+00,
    1.108705e+00, 1.105548e+00, 1.106317e+00, 1.106241e+00, 1.107671e+00, 1.107341e+00, 1.108118e+00,
    1.106654e+00, 1.102586e+00, 1.096655e+00, 1.092918e+00, 1.086628e+00, 1.083590e+00, 1.076028e+00,
    1.083776e+00, 1.089458e+00, 1.086543e+00, 1.079923e+00, 1.082216e+00, 1.077797e+00, 1.077061e+00,
    1.072824e+00, 1.072239e+00, 1.072103e+00, 1.072488e+00, 1.069828e+00, 1.070396e+00, 1.065456e+00,
    1.064966e+00, 1.060523e+00, 1.060047e+00, 1.057618e+00, 1.056427e+00, 1.055365e+00, 1.055016e+00,
    1.052303e+00, 1.051766e+00, 1.049727e+00, 1.048743e+00, 1.047397e+00, 1.045875e+00, 1.042971e+00,
    1.041823e+00, 1.039992e+00, 1.039019e+00, 1.036626e+00, 1.034175e+00, 1.032525e+00, 1.033632e+00,
    1.036106e+00, 1.037802e+00, 1.031265e+00, 1.032990e+00, 1.033283e+00, 1.035014e+00, 1.033944e+00,
    1.037074e+00, 1.034720e+00};
  return v;
}

/// G4ComponentGGHadronNucleusXsc::fProtonBarCorrectionIn, indexed by Z (0..92).
__host__ __device__ inline const double* gg_proton_bar_cor_in() {
  static const double v[93] = {
    1.0, 1.0, 1.147419e+00, 1.156248e+00, 1.205362e+00, 1.154224e+00, 1.120390e+00,
    1.124630e+00, 1.129459e+00, 1.107861e+00, 1.102151e+00, 1.104591e+00, 1.100284e+00, 1.098449e+00,
    1.092675e+00, 1.101122e+00, 1.106460e+00, 1.115048e+00, 1.123902e+00, 1.126659e+00, 1.131258e+00,
    1.133948e+00, 1.134183e+00, 1.133766e+00, 1.132812e+00, 1.131514e+00, 1.140337e+00, 1.134170e+00,
    1.139205e+00, 1.151472e+00, 1.142188e+00, 1.140724e+00, 1.140099e+00, 1.139847e+00, 1.137672e+00,
    1.138644e+00, 1.136338e+00, 1.136438e+00, 1.135945e+00, 1.136429e+00, 1.135701e+00, 1.135702e+00,
    1.134112e+00, 1.131934e+00, 1.128380e+00, 1.126371e+00, 1.122452e+00, 1.120907e+00, 1.115952e+00,
    1.115946e+00, 1.114425e+00, 1.111748e+00, 1.106205e+00, 1.107493e+00, 1.103621e+00, 1.102575e+00,
    1.098815e+00, 1.097888e+00, 1.097305e+00, 1.097129e+00, 1.094577e+00, 1.094551e+00, 1.090221e+00,
    1.089357e+00, 1.085408e+00, 1.084559e+00, 1.082181e+00, 1.080772e+00, 1.079463e+00, 1.078723e+00,
    1.076120e+00, 1.075234e+00, 1.073158e+00, 1.071919e+00, 1.070394e+00, 1.069502e+00, 1.067524e+00,
    1.066918e+00, 1.065778e+00, 1.065318e+00, 1.063729e+00, 1.062091e+00, 1.061084e+00, 1.059907e+00,
    1.059814e+00, 1.059108e+00, 1.051919e+00, 1.051257e+00, 1.049472e+00, 1.048822e+00, 1.045983e+00,
    1.046434e+00, 1.042613e+00};
  return v;
}

/// G4ComponentGGHadronNucleusXsc::fPionPlusBarCorrectionTot, indexed by Z (0..92).
__host__ __device__ inline const double* gg_piplus_bar_cor_tot() {
  static const double v[93] = {
    1.0, 1.0, 1.075927e+00, 1.074407e+00, 1.126098e+00, 1.100127e+00, 1.089742e+00,
    1.083536e+00, 1.089988e+00, 1.103566e+00, 1.096922e+00, 1.126573e+00, 1.132734e+00, 1.136512e+00,
    1.136629e+00, 1.133086e+00, 1.132428e+00, 1.129299e+00, 1.125622e+00, 1.126992e+00, 1.127840e+00,
    1.162670e+00, 1.160392e+00, 1.157864e+00, 1.157227e+00, 1.154627e+00, 1.192555e+00, 1.197243e+00,
    1.197911e+00, 1.200326e+00, 1.220053e+00, 1.215019e+00, 1.211703e+00, 1.209080e+00, 1.204248e+00,
    1.203328e+00, 1.198671e+00, 1.196840e+00, 1.194392e+00, 1.193037e+00, 1.190408e+00, 1.188583e+00,
    1.206127e+00, 1.210028e+00, 1.206434e+00, 1.204456e+00, 1.200547e+00, 1.199058e+00, 1.200174e+00,
    1.200276e+00, 1.198912e+00, 1.213048e+00, 1.207160e+00, 1.208020e+00, 1.203814e+00, 1.202380e+00,
    1.198306e+00, 1.197002e+00, 1.196027e+00, 1.195449e+00, 1.192563e+00, 1.192135e+00, 1.187556e+00,
    1.186308e+00, 1.182124e+00, 1.180900e+00, 1.178224e+00, 1.176471e+00, 1.174811e+00, 1.173702e+00,
    1.170827e+00, 1.169581e+00, 1.167205e+00, 1.165626e+00, 1.180244e+00, 1.177626e+00, 1.175121e+00,
    1.173903e+00, 1.172192e+00, 1.171128e+00, 1.168997e+00, 1.166826e+00, 1.164130e+00, 1.165412e+00,
    1.165504e+00, 1.165020e+00, 1.158462e+00, 1.158014e+00, 1.156519e+00, 1.156081e+00, 1.153602e+00,
    1.154190e+00, 1.152974e+00};
  return v;
}

/// G4ComponentGGHadronNucleusXsc::fPionPlusBarCorrectionIn, indexed by Z (0..92).
__host__ __device__ inline const double* gg_piplus_bar_cor_in() {
  static const double v[93] = {
    1.0, 1.0, 1.140246e+00, 1.097872e+00, 1.104301e+00, 1.068722e+00, 1.056495e+00,
    1.062622e+00, 1.047987e+00, 1.037032e+00, 1.035686e+00, 1.042870e+00, 1.052222e+00, 1.075100e+00,
    1.084480e+00, 1.078286e+00, 1.081488e+00, 1.089713e+00, 1.099105e+00, 1.098003e+00, 1.102175e+00,
    1.117707e+00, 1.121734e+00, 1.125229e+00, 1.126457e+00, 1.128905e+00, 1.163312e+00, 1.126263e+00,
    1.126459e+00, 1.135191e+00, 1.116986e+00, 1.117184e+00, 1.117037e+00, 1.116777e+00, 1.115858e+00,
    1.115745e+00, 1.114489e+00, 1.113993e+00, 1.113226e+00, 1.112818e+00, 1.111890e+00, 1.111238e+00,
    1.111209e+00, 1.111775e+00, 1.110256e+00, 1.109414e+00, 1.107647e+00, 1.106980e+00, 1.106096e+00,
    1.107331e+00, 1.107849e+00, 1.106407e+00, 1.103426e+00, 1.103896e+00, 1.101756e+00, 1.101031e+00,
    1.098915e+00, 1.098260e+00, 1.097768e+00, 1.097487e+00, 1.095964e+00, 1.095773e+00, 1.093348e+00,
    1.092687e+00, 1.090465e+00, 1.089821e+00, 1.088394e+00, 1.087462e+00, 1.086571e+00, 1.085997e+00,
    1.084451e+00, 1.083798e+00, 1.082513e+00, 1.081670e+00, 1.080735e+00, 1.075659e+00, 1.074341e+00,
    1.073689e+00, 1.072787e+00, 1.072237e+00, 1.071107e+00, 1.069955e+00, 1.074856e+00, 1.065873e+00,
    1.065938e+00, 1.065694e+00, 1.062192e+00, 1.061967e+00, 1.061180e+00, 1.060960e+00, 1.059646e+00,
    1.059975e+00, 1.059658e+00};
  return v;
}

/// G4ComponentGGHadronNucleusXsc::fPionMinusBarCorrectionTot, indexed by Z (0..92).
__host__ __device__ inline const double* gg_piminus_bar_cor_tot() {
  static const double v[93] = {
    1.0, 1.0, 1.3956e+00, 1.077959e+00, 1.129145e+00, 1.102088e+00, 1.089765e+00,
    1.083542e+00, 1.089995e+00, 1.104895e+00, 1.097154e+00, 1.127663e+00, 1.133063e+00, 1.137425e+00,
    1.136724e+00, 1.133859e+00, 1.132498e+00, 1.130276e+00, 1.127896e+00, 1.127656e+00, 1.127905e+00,
    1.164210e+00, 1.162259e+00, 1.160075e+00, 1.158978e+00, 1.156649e+00, 1.194157e+00, 1.199177e+00,
    1.198983e+00, 1.202325e+00, 1.221967e+00, 1.217548e+00, 1.214389e+00, 1.211760e+00, 1.207335e+00,
    1.206081e+00, 1.201766e+00, 1.199779e+00, 1.197283e+00, 1.195706e+00, 1.193071e+00, 1.191115e+00,
    1.208838e+00, 1.212681e+00, 1.209235e+00, 1.207163e+00, 1.203451e+00, 1.201807e+00, 1.203283e+00,
    1.203388e+00, 1.202244e+00, 1.216509e+00, 1.211066e+00, 1.211504e+00, 1.207539e+00, 1.205991e+00,
    1.202143e+00, 1.200724e+00, 1.199595e+00, 1.198815e+00, 1.196025e+00, 1.195390e+00, 1.191137e+00,
    1.189791e+00, 1.185888e+00, 1.184575e+00, 1.181996e+00, 1.180229e+00, 1.178545e+00, 1.177355e+00,
    1.174616e+00, 1.173312e+00, 1.171016e+00, 1.169424e+00, 1.184120e+00, 1.181478e+00, 1.179085e+00,
    1.177817e+00, 1.176124e+00, 1.175003e+00, 1.172947e+00, 1.170858e+00, 1.168170e+00, 1.169397e+00,
    1.169304e+00, 1.168706e+00, 1.162774e+00, 1.162217e+00, 1.160740e+00, 1.160196e+00, 1.157857e+00,
    1.158220e+00, 1.157267e+00};
  return v;
}

/// G4ComponentGGHadronNucleusXsc::fPionMinusBarCorrectionIn, indexed by Z (0..92).
__host__ __device__ inline const double* gg_piminus_bar_cor_in() {
  static const double v[93] = {
    1.0, 1.0, 1.463e+00, 1.100898e+00, 1.106773e+00, 1.070289e+00, 1.040514e+00,
    1.062628e+00, 1.047992e+00, 1.038041e+00, 1.035862e+00, 1.043679e+00, 1.052466e+00, 1.065780e+00,
    1.070551e+00, 1.078869e+00, 1.081541e+00, 1.090455e+00, 1.100847e+00, 1.098511e+00, 1.102226e+00,
    1.118865e+00, 1.123143e+00, 1.126904e+00, 1.127785e+00, 1.130444e+00, 1.148502e+00, 1.127678e+00,
    1.127244e+00, 1.123634e+00, 1.118347e+00, 1.118988e+00, 1.118957e+00, 1.118696e+00, 1.118074e+00,
    1.117722e+00, 1.116717e+00, 1.116111e+00, 1.115311e+00, 1.114745e+00, 1.113814e+00, 1.113069e+00,
    1.113141e+00, 1.113660e+00, 1.112249e+00, 1.111343e+00, 1.109718e+00, 1.108942e+00, 1.108310e+00,
    1.109549e+00, 1.110227e+00, 1.108846e+00, 1.106183e+00, 1.106354e+00, 1.104388e+00, 1.103583e+00,
    1.101632e+00, 1.100896e+00, 1.100296e+00, 1.099873e+00, 1.098420e+00, 1.098082e+00, 1.095892e+00,
    1.095162e+00, 1.093144e+00, 1.092438e+00, 1.091083e+00, 1.090142e+00, 1.089236e+00, 1.088604e+00,
    1.087159e+00, 1.086465e+00, 1.085239e+00, 1.084388e+00, 1.083473e+00, 1.078373e+00, 1.077136e+00,
    1.076450e+00, 1.075561e+00, 1.074973e+00, 1.073898e+00, 1.072806e+00, 1.067706e+00, 1.068684e+00,
    1.068618e+00, 1.068294e+00, 1.065241e+00, 1.064939e+00, 1.064166e+00, 1.063872e+00, 1.062659e+00,
    1.062828e+00, 1.062699e+00};
  return v;
}

/// G4ComponentGGHadronNucleusXsc::GetParticleBarCorTot - 1.0 for anything that is not a
/// nucleon or a charged pion, and Z is clamped into 1..92 before the lookup.
template <typename real_t>
__host__ __device__ inline real_t ggh_bar_cor_tot(const Projectile<real_t>& p, int zz) {
  const int z = (zz > 92) ? 92 : ((zz < 1) ? 1 : zz);
  if (p.pdg == pdg::kProton) { return static_cast<real_t>(gg_proton_bar_cor_tot()[z]); }
  if (p.pdg == pdg::kNeutron) { return static_cast<real_t>(gg_neutron_bar_cor_tot()[z]); }
  if (p.pdg == pdg::kPiPlus) { return static_cast<real_t>(gg_piplus_bar_cor_tot()[z]); }
  if (p.pdg == pdg::kPiMinus) { return static_cast<real_t>(gg_piminus_bar_cor_tot()[z]); }
  return real_t(1.0);
}

/// G4ComponentGGHadronNucleusXsc::GetParticleBarCorIn.
template <typename real_t>
__host__ __device__ inline real_t ggh_bar_cor_in(const Projectile<real_t>& p, int zz) {
  const int z = (zz > 92) ? 92 : ((zz < 1) ? 1 : zz);
  if (p.pdg == pdg::kProton) { return static_cast<real_t>(gg_proton_bar_cor_in()[z]); }
  if (p.pdg == pdg::kNeutron) { return static_cast<real_t>(gg_neutron_bar_cor_in()[z]); }
  if (p.pdg == pdg::kPiPlus) { return static_cast<real_t>(gg_piplus_bar_cor_in()[z]); }
  if (p.pdg == pdg::kPiMinus) { return static_cast<real_t>(gg_piminus_bar_cor_in()[z]); }
  return real_t(1.0);
}

/// G4ComponentGGHadronNucleusXsc::ComputeCrossSections.
///
/// @param nl  the number of lambdas in a projectile hypernucleus, which only
///            G4ComponentGGNuclNuclXsc passes non-zero (and only when
///            G4HadronicParameters::EnableHyperNuclei is on, which it is not by default).
template <typename real_t>
__host__ __device__ inline HadXs<real_t> ggh_compute_cross_sections(
    const Projectile<real_t>& p, real_t kin_energy, int Z, int A, int nl = 0) {
  real_t cofInelastic = real_t(2.4);
  constexpr real_t cofTotal = real_t(2.0);
  real_t sigma = real_t(0.0), hpInXsc = real_t(0.0), hnInXsc = real_t(0.0), R = real_t(0.0);

  const int N = (A - Z > 0) ? (A - Z) : 0;  // number of neutrons

  const Projectile<real_t> thePr = proton<real_t>();
  const Projectile<real_t> theNe = neutron<real_t>();
  HadXs<real_t> out;

  const bool is_kaon = (p.pdg == pdg::kKaonPlus || p.pdg == pdg::kKaonMinus ||
                        p.pdg == pdg::kKaonZeroShort || p.pdg == pdg::kKaonZeroLong);
  if (is_kaon) {
    // Hydrogen gets the NS kaon fit, everything else the VG one - the same class, two
    // different parameterisations, chosen on Z and not on energy.
    HadXs<real_t> hp;
    if (Z == 1) {
      hp = hn_kaon_xsc_ns<real_t>(p, thePr, kin_energy);
      sigma = hp.total;
    } else {
      hp = hn_kaon_xsc_gg<real_t>(p, thePr, kin_energy);
      sigma = static_cast<real_t>(Z) * hp.total;
    }
    hpInXsc = hp.inelastic;
    if (N > 0) {
      const HadXs<real_t> hn = hn_kaon_xsc_gg<real_t>(p, theNe, kin_energy);
      sigma += static_cast<real_t>(N) * hn.total;
      hnInXsc = hn.inelastic;
    }
    R = nr_radius_kngg<real_t>(A);
    cofInelastic = real_t(2.2);
  } else {
    const HadXs<real_t> hp = hadron_nucleon_xsc<real_t>(p, thePr, kin_energy);
    if (!hp.ok()) {
      out.refused = hp.refused;
      return out;
    }
    sigma = static_cast<real_t>(Z) * hp.total;
    hpInXsc = hp.inelastic;

    if (N > 0) {
      const HadXs<real_t> hn = hadron_nucleon_xsc<real_t>(p, theNe, kin_energy);
      if (!hn.ok()) {
        out.refused = hn.refused;
        return out;
      }
      sigma += static_cast<real_t>(N) * hn.total;
      hnInXsc = hn.inelastic;
    }
    R = nr_radius_hngg<real_t>(A);
    if (nl > 0) {
      // The hypernucleus branch: a lambda projectile at a mass-scaled energy, and the radius
      // rescaled by sqrt(Z23(A-nL) + 0.88*Z23(nL))/Z13(A).
      const real_t mp = thePr.mass;
      const real_t ml = lambda<real_t>().mass;
      const real_t kinCof = ml / mp;  // moving hyperon - rest nucleon
      constexpr real_t cHN = real_t(0.88);
      const HadXs<real_t> hl =
          hadron_nucleon_xsc<real_t>(lambda<real_t>(), thePr, kin_energy * kinCof);
      if (!hl.ok()) {
        out.refused = hl.refused;
        return out;
      }
      sigma += static_cast<real_t>(nl) * hl.total;
      R *= sqrt(g4pow_z23<real_t>(A - nl) + cHN * g4pow_z23<real_t>(nl))
           / data::g4pow_z13<real_t>(A);
    }
  }

  const real_t nucleusSquare = cofTotal * units::pi<real_t>() * R * R;  // basically 2piRR
  const real_t ratio = sigma / nucleusSquare;
  const real_t difratio = ratio / (real_t(1.) + ratio);
  out.diffraction =
      real_t(0.5) * nucleusSquare * (difratio - log(real_t(1.) + difratio));

  if (A > 1) {
    out.total = nucleusSquare * log(real_t(1.) + ratio) * ggh_bar_cor_tot<real_t>(p, Z);

    const real_t axsc2piR2 = cofInelastic * ratio;
    const real_t modelInLog = log(real_t(1.) + axsc2piR2);
    out.inelastic = nucleusSquare * modelInLog / cofInelastic;
    const real_t barcorr = ggh_bar_cor_in<real_t>(p, Z);
    out.inelastic *= barcorr;
    out.elastic = (out.total - out.inelastic > real_t(0.))
                      ? out.total - out.inelastic
                      : real_t(0.);

    const real_t xratio =
        (static_cast<real_t>(Z) * hpInXsc + static_cast<real_t>(N) * hnInXsc) / nucleusSquare;
    out.production = nucleusSquare * log(real_t(1.) + cofInelastic * xratio) * barcorr
                     / cofInelastic;
    out.production = (out.production < out.inelastic) ? out.production : out.inelastic;
  } else {  // H
    out.total = sigma;
    out.inelastic = hpInXsc;
    out.elastic =
        (out.total - out.inelastic > real_t(0.)) ? out.total - out.inelastic : real_t(0.);
    out.production = out.inelastic;
    out.diffraction = real_t(0.2) * out.inelastic;
  }
  return out;
}

/// G4ComponentGGHadronNucleusXsc::GetInelasticGlauberGribov / GetElasticGlauberGribov, the two
/// inlines the BGG classes call above 91 GeV.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> ggh_inelastic(const Projectile<real_t>& p,
                                                         real_t ekin, int Z, int A) {
  const HadXs<real_t> r = ggh_compute_cross_sections<real_t>(p, ekin, Z, A);
  return {r.inelastic, r.refused};
}
template <typename real_t>
__host__ __device__ inline XsValue<real_t> ggh_elastic(const Projectile<real_t>& p,
                                                       real_t ekin, int Z, int A) {
  const HadXs<real_t> r = ggh_compute_cross_sections<real_t>(p, ekin, Z, A);
  return {r.elastic, r.refused};
}

/// The `G4double A` element-wise overloads: A is rounded with G4lrint before use.
///
/// G4lrint is `std::lrint`, i.e. round-half-to-even in the default rounding mode - not
/// `int(x+0.5)`. aeff[Z] never lands on a half so the two agree here, but the distinction is
/// what the isotope hand-over of G4ParticleInelasticXS depends on.
template <typename real_t> __host__ __device__ inline int g4lrint(real_t x) {
  return static_cast<int>(lrint(static_cast<double>(x)));
}

template <typename real_t>
__host__ __device__ inline XsValue<real_t> ggh_inelastic_element(const Projectile<real_t>& p,
                                                                 real_t ekin, int Z,
                                                                 real_t A) {
  return ggh_inelastic<real_t>(p, ekin, Z, g4lrint<real_t>(A));
}
template <typename real_t>
__host__ __device__ inline XsValue<real_t> ggh_elastic_element(const Projectile<real_t>& p,
                                                               real_t ekin, int Z, real_t A) {
  return ggh_elastic<real_t>(p, ekin, Z, g4lrint<real_t>(A));
}

}  // namespace g4gpu::hadronic::xs
