#!/usr/bin/perl
# Extracts the two CHIPS parameterisation tables of Geant4 11.1.1 - G4PhotoNuclearCrossSection
# and G4ElectroNuclearCrossSection - and writes them as src/data/chips_photonuclear.hh and
# src/data/chips_electronuclear.hh.
#
#   perl tools/extract_emextra_chips.pl
#
# Why a script. 8,281 doubles in the first file and 14,112 in the second, written as C array
# initialisers five to a line over 2,100 source lines. There is no way to retype them and no way
# to eyeball a diff of them. So they are parsed, the counts are asserted against the array
# dimensions declared in the same file (nL, nH, nLA, nHA, nE, nN), and the totals are printed
# for the commit message. A count that disagrees is a fatal error here rather than a wrong
# cross section later.
#
# WHAT THE TABLES ARE
#
#   G4PhotoNuclearCrossSection   SL0..SL48, 49 nuclei x nL = 105 GDR points, tabulated in E
#                                  from THmin = 2 MeV in steps of dE = 1 MeV;
#                                SH0..SH13, 14 nuclei x nH = 224 high-energy points, tabulated
#                                  in ln(E) from ln(Emin) to ln(50 GeV);
#                                LA[49] and HA[14], the A values those two sets are indexed by.
#
#   G4ElectroNuclearCrossSection P00..P013, P10..P113, P20..P213 - three J-functions
#                                  (J1, J2, J3) for each of nN = 14 nuclei, nE = 336 points
#                                  each, tabulated in ln(E) from ln(EMi) to ln(EMa);
#                                A[14], the A values, and LL[14], the low channel per nucleus.
#
# The two files spell their basic-A lists differently - the photonuclear one is A in amu with
# fractional entries (58.7, 63.5, 107.9, ...) compared against the NIST mean A with a 0.0005
# window, the electronuclear one rounds A to an integer first - and that difference is physics,
# not formatting: carbon matches the electronuclear table exactly and misses the photonuclear
# one by 0.0107, so the same element takes the tabulated branch in one class and the
# interpolating branch in the other. Both lists are emitted verbatim and neither is normalised.
#
# **This machine has two perls and they are not the same language** - see the header of
# tools/extract_bertini_channels.pl. Written to 5.8; nothing below needs anything newer.
use strict;
use warnings;
use 5.008;

my $g4 = $ENV{G4SRC} || 'D:/Documents/Geant4/Windows/geant4-v11.1.1';
my $xs = "$g4/source/processes/hadronic/cross_sections/src";

# ---------------------------------------------------------------------------------------
# Read a file with comments stripped, so a commented-out array cannot be parsed as live data
# (the lesson of G4CascadeMuMinusPChannel.cc; there is no such block in these two files, but
# the parser is the same shape and the guarantee costs nothing).
sub slurp {
  my ($path) = @_;
  open my $fh, '<', $path or die "cannot open $path: $!";
  local $/;
  my $s = <$fh>;
  close $fh;
  $s =~ s{/\*.*?\*/}{}gs;
  $s =~ s{//[^\n]*}{}g;
  return $s;
}

# Parse `static const G4double NAME[DIM]={ ... };` and return the list of numbers.
sub array_of {
  my ($src, $name) = @_;
  return () unless $src =~ /\bstatic\s+const\s+G4(?:double|int)\s+\Q$name\E\s*\[[^\]]*\]\s*=\s*\{(.*?)\}\s*;/s;
  my $body = $1;
  my @v = ($body =~ /(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)/g);
  return @v;
}

# A C++ initialiser line for `n` doubles, 17 significant digits, five per line.
#
# Everything below is emitted as a `__host__ __device__ inline const double*` accessor around a
# function-scope `static const` array, NOT as a namespace-scope `inline constexpr` array. The
# reason is nvcc's: a namespace-scope variable is a HOST variable and device code referencing it
# fails with "identifier is undefined in device code". The same pattern and the same reason as
# src/data/bertini_channels.hh and src/data/bertini_angdst.hh; the two-dimensional tables are
# flattened because the accessor's return type has to be a pointer.
sub emit_doubles {
  my ($fh, @v) = @_;
  for (my $i = 0; $i < @v; $i += 5) {
    my $end = $i + 4;  $end = $#v if $end > $#v;
    print $fh '      ', join(', ', map { sprintf('%.17g', $_) } @v[$i .. $end]);
    print $fh ($end == $#v) ? "\n" : ",\n";
  }
}

# One accessor: `__host__ __device__ inline const double* NAME() { static const double v[N] = {
# ... }; return v; }`, with the values already flattened.
sub emit_accessor {
  my ($fh, $name, $doc, @v) = @_;
  print $fh "$doc\n";
  printf $fh "__host__ __device__ inline const double* %s() {\n  static const double v[%d] = {\n",
             $name, scalar @v;
  emit_doubles($fh, @v);
  print $fh "  };\n  return v;\n}\n\n";
}

my $total = 0;

# =======================================================================================
# G4PhotoNuclearCrossSection
# =======================================================================================
{
  my $src = slurp("$xs/G4PhotoNuclearCrossSection.cc");

  my ($nL)  = $src =~ /static\s+const\s+G4int\s+nL\s*=\s*(\d+)/  or die 'nL not found';
  my ($nH)  = $src =~ /static\s+const\s+G4int\s+nH\s*=\s*(\d+)/  or die 'nH not found';
  my ($nLA) = $src =~ /static\s+const\s+G4int\s+nLA\s*=\s*(\d+)/ or die 'nLA not found';
  my ($nHA) = $src =~ /static\s+const\s+G4int\s+nHA\s*=\s*(\d+)/ or die 'nHA not found';

  my @LA = array_of($src, 'LA');
  my @HA = array_of($src, 'HA');
  die "LA has ${\ scalar @LA} entries, nLA is $nLA" unless @LA == $nLA;
  die "HA has ${\ scalar @HA} entries, nHA is $nHA" unless @HA == $nHA;

  my (@SL, @SH);
  for my $i (0 .. $nLA - 1) {
    my @v = array_of($src, "SL$i");
    die "SL$i has ${\ scalar @v} entries, nL is $nL" unless @v == $nL;
    push @SL, \@v;
  }
  for my $j (0 .. $nHA - 1) {
    my @v = array_of($src, "SH$j");
    die "SH$j has ${\ scalar @v} entries, nH is $nH" unless @v == $nH;
    push @SH, \@v;
  }

  open my $fh, '>', 'src/data/chips_photonuclear.hh' or die $!;
  print $fh <<"HDR";
// GENERATED by tools/extract_emextra_chips.pl from Geant4 11.1.1
// source/processes/hadronic/cross_sections/src/G4PhotoNuclearCrossSection.cc. Do not edit.
//
// The CHIPS photo-nuclear parameterisation M. Kossov fitted, in the two pieces the class keeps
// it in: a giant-dipole-resonance table linear in E and a high-energy table linear in ln(E).
//
//   kChipsGdrN   = nL  = $nL   points per GDR nucleus, at E = THmin + i*dE MeV
//   kChipsGdrA   = nLA = $nLA  GDR nuclei, at the A values in kChipsGdrAList
//   kChipsHenN   = nH  = $nH  points per high-energy nucleus, at lnE = milE + i*dlE
//   kChipsHenA   = nHA = $nHA  high-energy nuclei, at the A values in kChipsHenAList
//
// The A lists are NOT integers and are not rounded: G4PhotoNuclearCrossSection::GetFunctions
// compares `std::abs(a - LA[i]) < .0005` against the NIST mean atomic mass in amu, so 58.7 and
// 58.9 (the two nickel/cobalt entries) are distinct table rows and 12 does not match carbon's
// 12.0107. Rounding them here would move every element onto the tabulated branch and change
// the cross section.
#ifndef G4GPU_DATA_CHIPS_PHOTONUCLEAR_HH
#define G4GPU_DATA_CHIPS_PHOTONUCLEAR_HH

namespace g4gpu::data {

inline constexpr int kChipsGdrN = $nL;
inline constexpr int kChipsGdrA = $nLA;
inline constexpr int kChipsHenN = $nH;
inline constexpr int kChipsHenA = $nHA;

HDR
  emit_accessor($fh, 'chips_gdr_a_list',
    "/// LA[nLA] - the A values the GDR tables are tabulated at.", @LA);
  emit_accessor($fh, 'chips_hen_a_list',
    "/// HA[nHA] - the A values the high-energy tables are tabulated at.", @HA);
  my @flatL;
  for my $i (0 .. $nLA - 1) { push @flatL, @{ $SL[$i] }; $total += $nL; }
  emit_accessor($fh, 'chips_gdr',
    "/// SL[nLA][nL] flattened row-major - the GDR cross sections in mb, SL0..SL" . ($nLA-1) .
    ".\n/// Row i, column q is index i*kChipsGdrN + q.", @flatL);
  my @flatH;
  for my $j (0 .. $nHA - 1) { push @flatH, @{ $SH[$j] }; $total += $nH; }
  emit_accessor($fh, 'chips_hen',
    "/// SH[nHA][nH] flattened row-major - the high-energy cross sections in mb, SH0..SH" .
    ($nHA-1) . ".\n/// Row j, column q is index j*kChipsHenN + q.", @flatH);
  print $fh "}  // namespace g4gpu::data\n\n#endif\n";
  close $fh;
  printf "chips_photonuclear.hh: %d GDR nuclei x %d + %d HE nuclei x %d = %d values\n",
         $nLA, $nL, $nHA, $nH, $nLA * $nL + $nHA * $nH;
}

# =======================================================================================
# G4ElectroNuclearCrossSection
# =======================================================================================
{
  my $src = slurp("$xs/G4ElectroNuclearCrossSection.cc");

  my ($nE) = $src =~ /static\s+const\s+G4int\s+nE\s*=\s*(\d+)/ or die 'nE not found';
  my ($nN) = $src =~ /static\s+const\s+G4int\s+nN\s*=\s*(\d+)/ or die 'nN not found';

  my @A  = array_of($src, 'A');
  my @LL = array_of($src, 'LL');
  die "A has ${\ scalar @A} entries, nN is $nN"   unless @A == $nN;
  die "LL has ${\ scalar @LL} entries, nN is $nN" unless @LL == $nN;

  my @P;  # [j][i][k]  j = 0,1,2 (J1,J2,J3), i = nucleus
  for my $j (0 .. 2) {
    my @rows;
    for my $i (0 .. $nN - 1) {
      my @v = array_of($src, "P$j$i");
      die "P$j$i has ${\ scalar @v} entries, nE is $nE" unless @v == $nE;
      push @rows, \@v;
    }
    push @P, \@rows;
  }

  open my $fh, '>', 'src/data/chips_electronuclear.hh' or die $!;
  print $fh <<"HDR";
// GENERATED by tools/extract_emextra_chips.pl from Geant4 11.1.1
// source/processes/hadronic/cross_sections/src/G4ElectroNuclearCrossSection.cc. Do not edit.
//
// The three integrated J-functions of the CHIPS electro-nuclear parameterisation, per nucleus:
// P0 -> J1, P1 -> J2, P2 -> J3 in the source's own naming, which is off by one from the
// `lastUsedCacheEl->J1/J2/J3` they are copied into.
//
//   kChipsElnN = nE = $nE  points per nucleus, at lnE = lEMi + i*dlnE
//   kChipsElnA = nN = $nN   nuclei, at the A values in kChipsElnAList
//   kChipsElnLow[i]       LL[i], the lowest channel of nucleus i at which J is non-zero;
//                         GetEquivalentPhotonEnergy starts its cumulative walk there.
//
// Unlike the photo-nuclear table's A list, GetFunctions ROUNDS the NIST mean A to an integer
// (`iA = static_cast<G4int>(a + .499)`) before comparing against this list, so carbon matches
// A[7] = 12 exactly and takes the tabulated branch.
#ifndef G4GPU_DATA_CHIPS_ELECTRONUCLEAR_HH
#define G4GPU_DATA_CHIPS_ELECTRONUCLEAR_HH

namespace g4gpu::data {

inline constexpr int kChipsElnN = $nE;
inline constexpr int kChipsElnA = $nN;

HDR
  emit_accessor($fh, 'chips_eln_a_list',
    "/// A[nN] - the A values the J-function tables are tabulated at.", @A);
  print $fh "/// LL[nN] - the low channel per nucleus.\n";
  print $fh "__host__ __device__ inline const int* chips_eln_low() {\n";
  printf $fh "  static const int v[%d] = {\n      ", scalar @LL;
  print $fh join(', ', @LL), "\n  };\n  return v;\n}\n\n";
  my @jname = ('J1', 'J2', 'J3');
  for my $j (0 .. 2) {
    my @flat;
    for my $i (0 .. $nN - 1) { push @flat, @{ $P[$j][$i] }; $total += $nE; }
    emit_accessor($fh, 'chips_eln_' . lc($jname[$j]),
      "/// P$j" . "0..P$j" . ($nN - 1) . " flattened row-major - the $jname[$j] function per\n" .
      "/// nucleus. Row i, column k is index i*kChipsElnN + k.", @flat);
  }
  print $fh "}  // namespace g4gpu::data\n\n#endif\n";
  close $fh;
  printf "chips_electronuclear.hh: 3 x %d nuclei x %d = %d values\n",
         $nN, $nE, 3 * $nN * $nE;
}

printf "total extracted: %d doubles\n", $total;
