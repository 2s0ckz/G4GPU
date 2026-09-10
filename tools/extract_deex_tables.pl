#!/usr/bin/perl
# Extracts the compiled-in numeric tables the de-excitation module needs out of Geant4 11.1.1
# and writes them as src/data/*.hh headers.
#
#   perl tools/extract_deex_tables.pl
#
# Six sources, 25,000-odd numbers:
#
#   G4NucleiPropertiesTableAME12.cc          AME2012 mass excesses, 3353 nuclides
#   G4NucleiPropertiesTheoreticalTable{A,B}  the Moller-Nix theoretical masses, 8979 nuclides
#   G4Cameron*Corrections.cc, G4Cook*.cc     shell and pairing corrections, 7 tables
#   G4NuclearLevelData.cc                    AMIN/AMAX/LEVELIDX per Z and LEVELMAX per nuclide
#
# They are pulled out rather than retyped for the reason tools/extract_yang.sh gives: a
# transposed digit in a nuclear mass is invisible in every dose and fatal to a 1e-15 oracle
# comparison, and there is no way to eyeball 3353 of them. Every array's length is asserted
# against the enum Geant4 declares it with, so a reshaped table fails here loudly instead of
# writing a short one.
use strict;
use warnings;

my $g4 = $ENV{G4SRC} || 'D:/Documents/Geant4/Windows/geant4-v11.1.1';
my $part = "$g4/source/particles/management/src";
my $deex = "$g4/source/processes/hadronic/models/de_excitation";

# ---------------------------------------------------------------------------------------------
# Pulls the initialiser list of one C array out of a file. `decl` is a regex matching the line
# the declaration starts on; collection runs to the matching `};`. Comments are stripped, the
# `f` suffix on float literals is kept off, and every token that looks like a number is
# returned in order. Nested braces (indexArray[2][N]) are flattened, which is what the callers
# want - they slice the result themselves.
sub grab {
  my ($file, $decl, $want) = @_;
  open my $fh, '<', $file or die "cannot open $file: $!";
  my $on = 0;
  my $text = '';
  while (my $line = <$fh>) {
    if (!$on) { $on = 1 if $line =~ /$decl/; next if !$on; }
    $line =~ s{//.*}{};
    $line =~ s{/\*.*?\*/}{}g;
    if ($line =~ /\};/) { $line =~ s/\};.*//; $text .= $line; last; }
    $text .= $line;
  }
  close $fh;
  # Everything before the opening brace of the initialiser is the declaration, and it carries
  # digits of its own - "G4double", "indexArray[2]", "MaxA+1". Dropping it by brace rather than
  # by "=" is what makes a declaration that wraps onto a second line safe.
  die "$file: $decl - no opening brace found\n" if $text !~ /\{/;
  $text =~ s/^.*?\{//s;
  my @v = ($text =~ /(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)f?/g);
  die "$file: $decl gave " . scalar(@v) . " values, expected $want\n"
    if defined $want && scalar(@v) != $want;
  return @v;
}

sub emit {
  my ($fh, $type, $name, $per, @v) = @_;
  print $fh "__host__ __device__ inline const $type* $name() {\n";
  print $fh "  static const $type v[" . scalar(@v) . "] = {\n";
  for (my $i = 0; $i < @v; $i += $per) {
    my @row = @v[$i .. ($i + $per - 1 > $#v ? $#v : $i + $per - 1)];
    print $fh "    " . join(', ', @row) . ",\n";
  }
  print $fh "  };\n  return v;\n}\n\n";
}

# =============================================================================================
# 1. AME2012 masses.
# =============================================================================================
my $nAME = 3353;
my @excess = grab("$part/G4NucleiPropertiesTableAME12.cc",
                  qr/const G4double G4NucleiPropertiesTableAME12::MassExcess\b/, $nAME);
my @beta   = grab("$part/G4NucleiPropertiesTableAME12.cc",
                  qr/const G4double G4NucleiPropertiesTableAME12::BetaEnergy\b/, $nAME);
my @idx    = grab("$part/G4NucleiPropertiesTableAME12.cc",
                  qr/const G4int G4NucleiPropertiesTableAME12::indexArray\b/, 2 * $nAME);
# 295 initialisers for a 296-element array. Geant4 declares shortTable[MaxA+1] with MaxA=295
# and writes 295 values, so shortTable[295] is zero-filled by the language - reproduced here
# by appending the zero rather than by guessing 3353, because that zero is observable
# behaviour: see the assertion below.
my @short  = grab("$part/G4NucleiPropertiesTableAME12.cc",
                  qr/const G4int G4NucleiPropertiesTableAME12::shortTable\b/, 295);
push @short, 0;
my @ameZ = @idx[0 .. $nAME - 1];
my @ameA = @idx[$nAME .. 2 * $nAME - 1];

# The two-level index has to be self-consistent or every mass lookup is wrong in a way no
# tolerance would catch: shortTable[A-1] .. shortTable[A]-1 must be exactly the entries whose
# A is A.  Checked here because it is checkable, and because a mis-sliced flatten of
# indexArray[2][] would otherwise look like real data.
for my $A (1 .. 294) {
  for my $i ($short[$A - 1] .. $short[$A] - 1) {
    die "AME12 index inconsistent: entry $i has A=$ameA[$i], shortTable says $A\n"
      if $ameA[$i] != $A;
  }
}
# A = 295 is the case the zero above makes unreachable. The entry exists - it is the last one
# in the table - and G4NucleiPropertiesTableAME12::GetIndex scans
# shortTable[294] .. shortTable[295]-1, i.e. 3352 .. -1, which is empty. So the heaviest
# nuclide the AME2012 evaluation carries is never found through this table and
# G4NucleiProperties falls through to the theoretical one. Asserted so that the port
# reproducing it is a decision and not a coincidence.
die "expected the last AME12 entry to be A=295, got $ameA[$nAME - 1]\n"
  if $ameA[$nAME - 1] != 295;
die "expected shortTable[294]=3352, got $short[294]\n" if $short[294] != 3352;

open my $fh, '>', 'src/data/ame12_masses.hh' or die $!;
print $fh <<'HDR';
// AME2012 atomic mass excesses, transcribed from G4NucleiPropertiesTableAME12 (11.1.1).
//
// G.Audi, M.Wang, A.H.Wapstra, F.G.Kondev, M.MacCormick, X.Xu, B.Pfeiffer, "The Ame2012 atomic
// mass evaluation", Chinese Physics C36 (2012) 1287 and 1603. 3353 nuclides, keV, in Geant4's
// own order: sorted by A, and within one A by Z ascending.
//
// Written by tools/extract_deex_tables.pl - do not edit. The index is two-level exactly as
// Geant4's is: short_table()[A-1] is the first entry with mass number A, so a lookup is a scan
// over the (at most 21) entries of one A rather than a search over 3353. The extractor asserts
// that slicing is self-consistent, because a mass table that is off by one row returns a
// plausible mass for the wrong nuclide and nothing downstream can tell.
#ifndef G4GPU_DATA_AME12_MASSES_HH
#define G4GPU_DATA_AME12_MASSES_HH

namespace g4gpu::data {

constexpr int kAme12Entries = 3353;
constexpr int kAme12MaxA = 295;   ///< G4NucleiPropertiesTableAME12::MaxA
constexpr int kAme12ZMax = 120;   ///< G4NucleiPropertiesTableAME12::ZMax

HDR
emit($fh, 'double', 'ame12_mass_excess_keV', 6, @excess);
emit($fh, 'double', 'ame12_beta_energy_keV', 6, @beta);
emit($fh, 'short', 'ame12_z', 20, @ameZ);
emit($fh, 'short', 'ame12_a', 20, @ameA);
emit($fh, 'short', 'ame12_short_table', 20, @short);
print $fh "}  // namespace g4gpu::data\n#endif\n";
close $fh;
print "src/data/ame12_masses.hh: $nAME nuclides\n";

# =============================================================================================
# 2. Theoretical (Moller-Nix) masses. The declaration is in ...TableA.cc, the data in B.cc.
# =============================================================================================
my $nTh = 8979;
my @thExcess = grab("$part/G4NucleiPropertiesTheoreticalTableB.cc",
                    qr/AtomicMassExcess\s*$/, $nTh);
my @thIdx    = grab("$part/G4NucleiPropertiesTheoreticalTableB.cc",
                    qr/indexArray\[2\]/, 2 * $nTh);
my @thShort  = grab("$part/G4NucleiPropertiesTheoreticalTableB.cc",
                    qr/shortTable\[/, 137);
my @thZ = @thIdx[0 .. $nTh - 1];
my @thA = @thIdx[$nTh .. 2 * $nTh - 1];
# Here the short table is indexed by Z-8, not by A, and the inner scan is over A. Same check.
for my $Z (8 .. 136) {
  for my $i ($thShort[$Z - 8] .. $thShort[$Z - 8 + 1] - 1) {
    die "theoretical index inconsistent: entry $i has Z=$thZ[$i], shortTable says $Z\n"
      if $thZ[$i] != $Z;
  }
}

open $fh, '>', 'src/data/nuclei_theoretical.hh' or die $!;
print $fh <<'HDR';
// Theoretical nuclear mass excesses, transcribed from G4NucleiPropertiesTheoreticalTable
// (11.1.1, data in G4NucleiPropertiesTheoreticalTableB.cc).
//
// P.Moller, J.R.Nix, W.D.Myers, W.J.Swiatecki, "Nuclear ground-state masses and deformations",
// At. Data Nucl. Data Tables 59 (1995) 185. 8979 nuclides, MeV, Z = 8..136, A = 16..339.
//
// G4NucleiProperties reaches this table only for a nuclide the AME2012 evaluation does not
// carry, which for de-excitation means a fragment far off stability. It is here because
// "far off stability" is exactly what an evaporation chain walks into, and the alternative
// branch below it is the Weizsaecker formula - a different number, not a rounding of this one.
//
// Written by tools/extract_deex_tables.pl - do not edit. Indexed by Z: short_table()[Z-8] is
// the first entry with proton number Z, and the inner scan is over A.
#ifndef G4GPU_DATA_NUCLEI_THEORETICAL_HH
#define G4GPU_DATA_NUCLEI_THEORETICAL_HH

namespace g4gpu::data {

constexpr int kTheoEntries = 8979;
constexpr int kTheoShortTableSize = 137;

HDR
emit($fh, 'double', 'theo_mass_excess_MeV', 8, @thExcess);
emit($fh, 'short', 'theo_z', 20, @thZ);
emit($fh, 'short', 'theo_a', 20, @thA);
emit($fh, 'short', 'theo_short_table', 20, @thShort);
print $fh "}  // namespace g4gpu::data\n#endif\n";
close $fh;
print "src/data/nuclei_theoretical.hh: $nTh nuclides\n";

# =============================================================================================
# 3. Shell and pairing corrections. Seven pairs of (Z-table, N-table), all MeV.
# =============================================================================================
# [class, Z-table name, Z initialisers, Z declared size, N-table name, N initialisers, size].
# The two counts differ in exactly one place: G4CameronTruranHilfPairingCorrections declares
# PairingNTable[146] (N = 10..155) and writes 145 values, so N = 155 is a zero the language
# supplies. Recorded as two numbers rather than one so that any future short table fails the
# assertion instead of being silently padded.
my @sets = (
  ['G4CameronGilbertPairingCorrections',    'PairingZTable', 88, 88,  'PairingNTable', 140, 140],
  ['G4CameronGilbertShellCorrections',      'ShellZTable',   88, 88,  'ShellNTable',   140, 140],
  ['G4CookPairingCorrections',              'PairingZTable', 68, 68,  'PairingNTable', 118, 118],
  ['G4CookShellCorrections',                'ShellZTable',   68, 68,  'ShellNTable',   118, 118],
  ['G4CameronTruranHilfPairingCorrections', 'PairingZTable', 93, 93,  'PairingNTable', 145, 146],
  ['G4CameronTruranHilfShellCorrections',   'ShellZTable',   93, 93,  'ShellNTable',   146, 146],
  ['G4CameronShellPlusPairingCorrections',  'SPZTable',     200, 200, 'SPNTable',      200, 200],
);
open $fh, '>', 'src/data/shell_pairing.hh' or die $!;
print $fh <<'HDR';
// Shell and pairing corrections for the nuclear level density, transcribed from the seven
// G4Cameron*/G4Cook* classes under de_excitation/util (11.1.1).
//
// Each class holds two independent tables, one in Z and one in N, and the correction is their
// sum - so a nuclide's correction is separable and the tables are one-dimensional. Their
// validity windows differ and the windows are what selects between them:
//
//   G4ShellCorrection::GetShellCorrection     Cook (Z 28..95, N 33..150), else Cameron-Gilbert
//                                             (Z 11..98, N 11..150), else zero
//   G4PairingCorrection::GetPairingCorrection Cameron-Gilbert (Z 11..98, N 11..150), else
//                                             12 MeV/sqrt(A) times the odd-even count
//
// Cameron-Truran-Hilf and Cameron shell-plus-pairing are constructed by those two classes and
// exposed through accessors, but no default-configuration caller reads them; they are here so
// that a caller which does is not blocked, and so the windows above can be read as choices
// rather than as the only tables that exist.
//
// Sources: A.Gilbert, A.G.W.Cameron, Can. J. Phys. 43 (1965) 1446; J.L.Cook et al., Aust. J.
// Phys. 20 (1967) 477; A.G.W.Cameron, Can. J. Phys. 35 (1957) 1021; Cameron-Truran-Hilf as
// cited in G4CameronTruranHilfShellCorrections.cc.
//
// Written by tools/extract_deex_tables.pl - do not edit.
#ifndef G4GPU_DATA_SHELL_PAIRING_HH
#define G4GPU_DATA_SHELL_PAIRING_HH

namespace g4gpu::data {

HDR
for my $s (@sets) {
  my ($cls, $zn, $zc, $zsz, $nn, $nc, $nsz) = @$s;
  my @z = grab("$deex/util/src/$cls.cc", qr/\b\Q$zn\E\s*\[\s*\]/, $zc);
  my @n = grab("$deex/util/src/$cls.cc", qr/\b\Q$nn\E\s*\[\s*\]/, $nc);
  push @z, '0.0' while scalar(@z) < $zsz;
  push @n, '0.0' while scalar(@n) < $nsz;
  (my $tag = $cls) =~ s/^G4//;
  $tag =~ s/([a-z0-9])([A-Z])/$1_$2/g;
  $tag = lc $tag;
  print $fh "// $cls, $zc values in Z and $nc in N.\n";
  emit($fh, 'double', "${tag}_z", 10, @z);
  emit($fh, 'double', "${tag}_n", 10, @n);
  print "  $cls: $zc + $nc\n";
}
print $fh "}  // namespace g4gpu::data\n#endif\n";
close $fh;
print "src/data/shell_pairing.hh written\n";

# =============================================================================================
# 4. G4NuclearLevelData's per-Z index and the compiled maximum level energy per nuclide.
# =============================================================================================
my @amin = grab("$deex/management/src/G4NuclearLevelData.cc",
                qr/const G4int G4NuclearLevelData::AMIN\b/, 118);
my @amax = grab("$deex/management/src/G4NuclearLevelData.cc",
                qr/const G4int G4NuclearLevelData::AMAX\b/, 118);
my @lidx = grab("$deex/management/src/G4NuclearLevelData.cc",
                qr/const G4int G4NuclearLevelData::LEVELIDX\b/, 118);
my @lmax = grab("$deex/management/src/G4NuclearLevelData.cc",
                qr/static const G4float LEVELMAX\[3188\]/, 3188);
# LEVELIDX[Z] + (A - AMIN[Z]) has to stay inside LEVELMAX for every (Z, A) the AMIN/AMAX
# window admits, or GetMaxLevelEnergy reads another nuclide's number.
for my $Z (1 .. 117) {
  next if $amax[$Z] == 0;
  my $hi = $lidx[$Z] + $amax[$Z] - $amin[$Z];
  die "LEVELIDX overruns LEVELMAX at Z=$Z ($hi >= 3188)\n" if $hi >= 3188;
}

open $fh, '>', 'src/data/level_index.hh' or die $!;
print $fh <<'HDR';
// G4NuclearLevelData's compiled-in per-nuclide index, transcribed from G4NuclearLevelData.cc
// (11.1.1).
//
//   amin()/amax()   the mass-number window PhotonEvaporation data exists in, per Z. Outside it
//                   GetLevelManager returns null and every level-data question answers "no
//                   levels", which is a decision of Geant4's and not of the dataset's.
//   levelidx()      base of that Z's slice of levelmax().
//   levelmax()      the highest tabulated level energy, MeV, per nuclide.
//
// levelmax() is the one to be careful with. Geant4's comment above it says "obtained from
// PhotonEvaporation5.2", and the dataset installed here is PhotonEvaporation5.7 - so
// G4NuclearLevelData::GetMaxLevelEnergy and G4LevelManager::MaxLevelEnergy, which both answer
// "highest level of this nuclide", are two different numbers whenever the two dataset versions
// disagree about a nuclide. Both are load-bearing: the compiled table gates
// G4NuclearLevelData::GetLevelEnergy and decides which fragments enter the Fermi break-up pool
// (G4FermiFragmentsPoolVI::Initialise reads MaxLevelEnergy), while the read data drives the
// photon-evaporation cascade. So the compiled table is reproduced here as compiled, and
// tests/test_deex_levels.cu measures how far the two disagree rather than assuming they do not.
//
// Written by tools/extract_deex_tables.pl - do not edit.
#ifndef G4GPU_DATA_LEVEL_INDEX_HH
#define G4GPU_DATA_LEVEL_INDEX_HH

namespace g4gpu::data {

constexpr int kLevelZMax = 118;      ///< G4NuclearLevelData::ZMAX
constexpr int kLevelMaxEntries = 3188;

HDR
emit($fh, 'short', 'level_amin', 10, @amin);
emit($fh, 'short', 'level_amax', 10, @amax);
emit($fh, 'short', 'level_idx', 10, @lidx);
emit($fh, 'float', 'level_max_energy_MeV', 10, @lmax);
print $fh "}  // namespace g4gpu::data\n#endif\n";
close $fh;
print "src/data/level_index.hh written\n";
