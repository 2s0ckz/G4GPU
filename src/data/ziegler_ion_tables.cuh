// Ziegler alpha-particle electronic stopping coefficients, 92 elements x 5, extracted
// programmatically from Geant4 11.1.1 G4BraggIonModel::ElectronicStoppingPower.
// ICRU Report 49 (1993), Ziegler-type parameterisation.
//
// A DIFFERENT table from the proton one in ziegler_tables.cuh, with a different formula
// around it. The extractor strips // comments first: this table contains two commented-out
// alternative rows (Be and Au from ICRU) whose numbers would otherwise be picked up,
// shifting everything after them - the same trap that corrupted a Rayleigh table earlier
// in this project. The element count (460 = 92 x 5) is asserted by the extractor.
#pragma once
#include "core/units.cuh"

namespace g4gpu::data {

/// a[Z-1][0..4]; all five entries are used, unlike the proton table.
template <typename real_t> __host__ __device__ inline const real_t* ziegler_ion_a() {
  static const real_t v[92 * 5] = {
    real_t(0.35485), real_t(0.6456), real_t(6.01525), real_t(20.8933), real_t(4.3515)
    , real_t(0.58), real_t(0.59), real_t(6.3), real_t(130.0), real_t(44.07)
    , real_t(1.42), real_t(0.49), real_t(12.25), real_t(32.0), real_t(9.161)
    , real_t(2.206), real_t(0.51), real_t(15.32), real_t(0.25), real_t(8.995)
    , real_t(3.691), real_t(0.4128), real_t(18.48), real_t(50.72), real_t(9.0)
    , real_t(3.83523), real_t(0.42993), real_t(12.6125), real_t(227.41), real_t(188.97)
    , real_t(1.9259), real_t(0.5550), real_t(27.1513), real_t(26.0665), real_t(6.2768)
    , real_t(2.81015), real_t(0.4759), real_t(50.0253), real_t(10.556), real_t(1.0382)
    , real_t(1.533), real_t(0.531), real_t(40.44), real_t(18.41), real_t(2.718)
    , real_t(2.303), real_t(0.4861), real_t(37.01), real_t(37.96), real_t(5.092)
    , real_t(9.894), real_t(0.3081), real_t(23.65), real_t(0.384), real_t(92.93)
    , real_t(4.3), real_t(0.47), real_t(34.3), real_t(3.3), real_t(12.74)
    , real_t(2.5), real_t(0.625), real_t(45.7), real_t(0.1), real_t(4.359)
    , real_t(2.1), real_t(0.65), real_t(49.34), real_t(1.788), real_t(4.133)
    , real_t(1.729), real_t(0.6562), real_t(53.41), real_t(2.405), real_t(3.845)
    , real_t(1.402), real_t(0.6791), real_t(58.98), real_t(3.528), real_t(3.211)
    , real_t(1.117), real_t(0.7044), real_t(69.69), real_t(3.705), real_t(2.156)
    , real_t(2.291), real_t(0.6284), real_t(73.88), real_t(4.478), real_t(2.066)
    , real_t(8.554), real_t(0.3817), real_t(83.61), real_t(11.84), real_t(1.875)
    , real_t(6.297), real_t(0.4622), real_t(65.39), real_t(10.14), real_t(5.036)
    , real_t(5.307), real_t(0.4918), real_t(61.74), real_t(12.4), real_t(6.665)
    , real_t(4.71), real_t(0.5087), real_t(65.28), real_t(8.806), real_t(5.948)
    , real_t(6.151), real_t(0.4524), real_t(83.0), real_t(18.31), real_t(2.71)
    , real_t(6.57), real_t(0.4322), real_t(84.76), real_t(15.53), real_t(2.779)
    , real_t(5.738), real_t(0.4492), real_t(84.6), real_t(14.18), real_t(3.101)
    , real_t(5.013), real_t(0.4707), real_t(85.8), real_t(16.55), real_t(3.211)
    , real_t(4.32), real_t(0.4947), real_t(76.14), real_t(10.85), real_t(5.441)
    , real_t(4.652), real_t(0.4571), real_t(80.73), real_t(22.0), real_t(4.952)
    , real_t(3.114), real_t(0.5236), real_t(76.67), real_t(7.62), real_t(6.385)
    , real_t(3.114), real_t(0.5236), real_t(76.67), real_t(7.62), real_t(7.502)
    , real_t(3.114), real_t(0.5236), real_t(76.67), real_t(7.62), real_t(8.514)
    , real_t(5.746), real_t(0.4662), real_t(79.24), real_t(1.185), real_t(7.993)
    , real_t(2.792), real_t(0.6346), real_t(106.1), real_t(0.2986), real_t(2.331)
    , real_t(4.667), real_t(0.5095), real_t(124.3), real_t(2.102), real_t(1.667)
    , real_t(2.44), real_t(0.6346), real_t(105.0), real_t(0.83), real_t(2.851)
    , real_t(1.413), real_t(0.7377), real_t(147.9), real_t(1.466), real_t(1.016)
    , real_t(11.72), real_t(0.3826), real_t(102.8), real_t(9.231), real_t(4.371)
    , real_t(7.126), real_t(0.4804), real_t(119.3), real_t(5.784), real_t(2.454)
    , real_t(11.61), real_t(0.3955), real_t(146.7), real_t(7.031), real_t(1.423)
    , real_t(10.99), real_t(0.41), real_t(163.9), real_t(7.1), real_t(1.052)
    , real_t(9.241), real_t(0.4275), real_t(163.1), real_t(7.954), real_t(1.102)
    , real_t(9.276), real_t(0.418), real_t(157.1), real_t(8.038), real_t(1.29)
    , real_t(3.999), real_t(0.6152), real_t(97.6), real_t(1.297), real_t(5.792)
    , real_t(4.306), real_t(0.5658), real_t(97.99), real_t(5.514), real_t(5.754)
    , real_t(3.615), real_t(0.6197), real_t(86.26), real_t(0.333), real_t(8.689)
    , real_t(5.8), real_t(0.49), real_t(147.2), real_t(6.903), real_t(1.289)
    , real_t(5.6), real_t(0.49), real_t(130.0), real_t(10.0), real_t(2.844)
    , real_t(3.55), real_t(0.6068), real_t(124.7), real_t(1.112), real_t(3.119)
    , real_t(3.6), real_t(0.62), real_t(105.8), real_t(0.1692), real_t(6.026)
    , real_t(5.4), real_t(0.53), real_t(103.1), real_t(3.931), real_t(7.767)
    , real_t(3.97), real_t(0.6459), real_t(131.8), real_t(0.2233), real_t(2.723)
    , real_t(3.65), real_t(0.64), real_t(126.8), real_t(0.6834), real_t(3.411)
    , real_t(3.118), real_t(0.6519), real_t(164.9), real_t(1.208), real_t(1.51)
    , real_t(3.949), real_t(0.6209), real_t(200.5), real_t(1.878), real_t(0.9126)
    , real_t(14.4), real_t(0.3923), real_t(152.5), real_t(8.354), real_t(2.597)
    , real_t(10.99), real_t(0.4599), real_t(138.4), real_t(4.811), real_t(3.726)
    , real_t(16.6), real_t(0.3773), real_t(224.1), real_t(6.28), real_t(0.9121)
    , real_t(10.54), real_t(0.4533), real_t(159.3), real_t(4.832), real_t(2.529)
    , real_t(10.33), real_t(0.4502), real_t(162.0), real_t(5.132), real_t(2.444)
    , real_t(10.15), real_t(0.4471), real_t(165.6), real_t(5.378), real_t(2.328)
    , real_t(9.976), real_t(0.4439), real_t(168.0), real_t(5.721), real_t(2.258)
    , real_t(9.804), real_t(0.4408), real_t(176.2), real_t(5.675), real_t(1.997)
    , real_t(14.22), real_t(0.363), real_t(228.4), real_t(7.024), real_t(1.016)
    , real_t(9.952), real_t(0.4318), real_t(233.5), real_t(5.065), real_t(0.9244)
    , real_t(9.272), real_t(0.4345), real_t(210.0), real_t(4.911), real_t(1.258)
    , real_t(10.13), real_t(0.4146), real_t(225.7), real_t(5.525), real_t(1.055)
    , real_t(8.949), real_t(0.4304), real_t(213.3), real_t(5.071), real_t(1.221)
    , real_t(11.94), real_t(0.3783), real_t(247.2), real_t(6.655), real_t(0.849)
    , real_t(8.472), real_t(0.4405), real_t(195.5), real_t(4.051), real_t(1.604)
    , real_t(8.301), real_t(0.4399), real_t(203.7), real_t(3.667), real_t(1.459)
    , real_t(6.567), real_t(0.4858), real_t(193.0), real_t(2.65), real_t(1.66)
    , real_t(5.951), real_t(0.5016), real_t(196.1), real_t(2.662), real_t(1.589)
    , real_t(7.495), real_t(0.4523), real_t(251.4), real_t(3.433), real_t(0.8619)
    , real_t(6.335), real_t(0.4825), real_t(255.1), real_t(2.834), real_t(0.8228)
    , real_t(4.314), real_t(0.5558), real_t(214.8), real_t(2.354), real_t(1.263)
    , real_t(4.02), real_t(0.5681), real_t(219.9), real_t(2.402), real_t(1.191)
    , real_t(3.836), real_t(0.5765), real_t(210.2), real_t(2.742), real_t(1.305)
    , real_t(4.68), real_t(0.5247), real_t(244.7), real_t(2.749), real_t(0.8962)
    , real_t(2.892), real_t(0.6204), real_t(208.6), real_t(2.415), real_t(1.416)
    , real_t(2.892), real_t(0.6204), real_t(208.6), real_t(2.415), real_t(1.416)
    , real_t(4.728), real_t(0.5522), real_t(217.0), real_t(3.091), real_t(1.386)
    , real_t(6.18), real_t(0.52), real_t(170.0), real_t(4.0), real_t(3.224)
    , real_t(9.0), real_t(0.47), real_t(198.0), real_t(3.8), real_t(2.032)
    , real_t(2.324), real_t(0.6997), real_t(216.0), real_t(1.599), real_t(1.399)
    , real_t(1.961), real_t(0.7286), real_t(223.0), real_t(1.621), real_t(1.296)
    , real_t(1.75), real_t(0.7427), real_t(350.1), real_t(0.9789), real_t(0.5507)
    , real_t(10.31), real_t(0.4613), real_t(261.2), real_t(4.738), real_t(0.9899)
    , real_t(7.962), real_t(0.519), real_t(235.7), real_t(4.347), real_t(1.313)
    , real_t(6.227), real_t(0.5645), real_t(231.9), real_t(3.961), real_t(1.379)
    , real_t(5.246), real_t(0.5947), real_t(228.6), real_t(4.027), real_t(1.432)
    , real_t(5.408), real_t(0.5811), real_t(235.7), real_t(3.961), real_t(1.358)
    , real_t(5.218), real_t(0.5828), real_t(245.0), real_t(3.838), real_t(1.25)
    };
  return v;
}

}  // namespace g4gpu::data
