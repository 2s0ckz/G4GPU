// Geant4's unit system, with Geant4's internal base units.
//
// The whole point of writing `5.*cm` rather than `50.` is that the number carries its unit
// through the arithmetic and comes out right whatever the caller meant. Geant4's internal
// units are millimetre, nanosecond, MeV, positron charge, kelvin, mole, candela - and those
// are used here too, unchanged, so that a value copied from a Geant4 macro or source file
// means exactly the same thing.
//
// This mirrors CLHEP/Units/SystemOfUnits.h. Only the names an example is likely to use are
// present; add more as needed, but keep the base units fixed.
#pragma once
#include "g4/G4Types.hh"

// ---------------------------------------------------------------- length (base: mm)
inline constexpr G4double millimeter = 1.0;
inline constexpr G4double millimeter2 = millimeter * millimeter;
inline constexpr G4double millimeter3 = millimeter * millimeter * millimeter;

inline constexpr G4double centimeter = 10. * millimeter;
inline constexpr G4double centimeter2 = centimeter * centimeter;
inline constexpr G4double centimeter3 = centimeter * centimeter * centimeter;

inline constexpr G4double meter = 1000. * millimeter;
inline constexpr G4double meter2 = meter * meter;
inline constexpr G4double meter3 = meter * meter * meter;

inline constexpr G4double kilometer = 1000. * meter;
inline constexpr G4double micrometer = 1.e-6 * meter;
inline constexpr G4double nanometer = 1.e-9 * meter;
inline constexpr G4double angstrom = 1.e-10 * meter;
inline constexpr G4double fermi = 1.e-15 * meter;

inline constexpr G4double barn = 1.e-28 * meter2;
inline constexpr G4double millibarn = 1.e-3 * barn;
inline constexpr G4double microbarn = 1.e-6 * barn;

inline constexpr G4double mm = millimeter;
inline constexpr G4double mm2 = millimeter2;
inline constexpr G4double mm3 = millimeter3;
inline constexpr G4double cm = centimeter;
inline constexpr G4double cm2 = centimeter2;
inline constexpr G4double cm3 = centimeter3;
inline constexpr G4double m = meter;
inline constexpr G4double m2 = meter2;
inline constexpr G4double m3 = meter3;
inline constexpr G4double km = kilometer;
inline constexpr G4double um = micrometer;
inline constexpr G4double nm = nanometer;

// ---------------------------------------------------------------- angle (base: radian)
inline constexpr G4double radian = 1.0;
inline constexpr G4double milliradian = 1.e-3 * radian;
inline constexpr G4double degree = (3.14159265358979323846 / 180.0) * radian;
inline constexpr G4double steradian = 1.0;

inline constexpr G4double rad = radian;
inline constexpr G4double mrad = milliradian;
inline constexpr G4double deg = degree;
inline constexpr G4double sr = steradian;

// ---------------------------------------------------------------- time (base: ns)
inline constexpr G4double nanosecond = 1.0;
inline constexpr G4double second = 1.e+9 * nanosecond;
inline constexpr G4double millisecond = 1.e-3 * second;
inline constexpr G4double microsecond = 1.e-6 * second;
inline constexpr G4double picosecond = 1.e-12 * second;
inline constexpr G4double minute = 60 * second;
inline constexpr G4double hour = 60 * minute;
inline constexpr G4double day = 24 * hour;
inline constexpr G4double year = 365.25 * day;

inline constexpr G4double ns = nanosecond;
inline constexpr G4double s = second;
inline constexpr G4double ms = millisecond;
inline constexpr G4double us = microsecond;
inline constexpr G4double ps = picosecond;

inline constexpr G4double hertz = 1.0 / second;
inline constexpr G4double kilohertz = 1.e+3 * hertz;
inline constexpr G4double megahertz = 1.e+6 * hertz;

// ---------------------------------------------------------------- activity
inline constexpr G4double becquerel = 1.0 / second;
inline constexpr G4double curie = 3.7e+10 * becquerel;
inline constexpr G4double kilobecquerel = 1.e+3 * becquerel;
inline constexpr G4double megabecquerel = 1.e+6 * becquerel;
inline constexpr G4double gigabecquerel = 1.e+9 * becquerel;
inline constexpr G4double millicurie = 1.e-3 * curie;
inline constexpr G4double microcurie = 1.e-6 * curie;

// ---------------------------------------------------------------- energy (base: MeV)
inline constexpr G4double megaelectronvolt = 1.0;
inline constexpr G4double electronvolt = 1.e-6 * megaelectronvolt;
inline constexpr G4double kiloelectronvolt = 1.e-3 * megaelectronvolt;
inline constexpr G4double gigaelectronvolt = 1.e+3 * megaelectronvolt;
inline constexpr G4double teraelectronvolt = 1.e+6 * megaelectronvolt;
inline constexpr G4double petaelectronvolt = 1.e+9 * megaelectronvolt;

inline constexpr G4double MeV = megaelectronvolt;
inline constexpr G4double eV = electronvolt;
inline constexpr G4double keV = kiloelectronvolt;
inline constexpr G4double GeV = gigaelectronvolt;
inline constexpr G4double TeV = teraelectronvolt;
inline constexpr G4double PeV = petaelectronvolt;

inline constexpr G4double joule = electronvolt / 1.602176634e-19;

// ---------------------------------------------------------------- mass and density
inline constexpr G4double kilogram = joule * second * second / (meter * meter);
inline constexpr G4double gram = 1.e-3 * kilogram;
inline constexpr G4double milligram = 1.e-3 * gram;
inline constexpr G4double kg = kilogram;
inline constexpr G4double g = gram;
inline constexpr G4double mg = milligram;

// ---------------------------------------------------------------- electric charge, field
inline constexpr G4double eplus = 1.0;
inline constexpr G4double e_SI = 1.602176634e-19;
inline constexpr G4double coulomb = eplus / e_SI;
inline constexpr G4double volt = 1.e-6 * megaelectronvolt / eplus;
inline constexpr G4double kilovolt = 1.e+3 * volt;
inline constexpr G4double megavolt = 1.e+6 * volt;
inline constexpr G4double tesla = volt * second / meter2;
inline constexpr G4double gauss = 1.e-4 * tesla;
inline constexpr G4double kilogauss = 1.e-1 * tesla;

// ---------------------------------------------------------------- dose
inline constexpr G4double gray = joule / kilogram;
inline constexpr G4double milligray = 1.e-3 * gray;
inline constexpr G4double microgray = 1.e-6 * gray;
inline constexpr G4double nanogray = 1.e-9 * gray;
inline constexpr G4double picogray = 1.e-12 * gray;

// ---------------------------------------------------------------- amount, temperature
inline constexpr G4double mole = 1.0;
inline constexpr G4double kelvin = 1.0;
inline constexpr G4double atmosphere = 101325 * (1.e-6 * megaelectronvolt / (mm * mm * mm));

// `pascal` is a legacy calling-convention macro in windef.h (`#define pascal __stdcall`), so
// declaring a variable of that name fails to compile in any translation unit that has included
// windows.h - which the viewer and the GUI both do. Undefining it is what CLHEP does for the
// same reason; nothing has used that convention since 16-bit Windows.
#ifdef pascal
#undef pascal
#endif
inline constexpr G4double pascal = atmosphere / 101325;
inline constexpr G4double bar = 100000 * pascal;

inline constexpr G4double perCent = 0.01;
inline constexpr G4double perThousand = 0.001;
inline constexpr G4double perMillion = 0.000001;
