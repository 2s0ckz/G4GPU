// Physical constants, in Geant4 internal units. Mirrors CLHEP/Units/PhysicalConstants.h.
#pragma once
#include "g4/G4SystemOfUnits.hh"

inline constexpr G4double pi = 3.14159265358979323846;
inline constexpr G4double twopi = 2 * pi;
inline constexpr G4double halfpi = pi / 2;
inline constexpr G4double pi2 = pi * pi;

inline constexpr G4double Avogadro = 6.02214076e+23 / mole;
inline constexpr G4double c_light = 2.99792458e+8 * m / s;
inline constexpr G4double c_squared = c_light * c_light;
inline constexpr G4double h_Planck = 6.62607015e-34 * joule * s;
inline constexpr G4double hbar_Planck = h_Planck / twopi;
inline constexpr G4double hbarc = hbar_Planck * c_light;
inline constexpr G4double electron_charge = -eplus;
inline constexpr G4double e_squared = eplus * eplus;
inline constexpr G4double electron_mass_c2 = 0.510998910 * MeV;
inline constexpr G4double proton_mass_c2 = 938.272013 * MeV;
inline constexpr G4double neutron_mass_c2 = 939.56536 * MeV;
inline constexpr G4double amu_c2 = 931.494028 * MeV;
inline constexpr G4double amu = amu_c2 / c_squared;
inline constexpr G4double k_Boltzmann = 8.617333e-11 * MeV / kelvin;
inline constexpr G4double fine_structure_const = 0.007297352565305215;
inline constexpr G4double classic_electr_radius = 2.817940545232519e-12 * mm;
inline constexpr G4double electron_Compton_length = 3.8615929818578764e-10 * mm;
