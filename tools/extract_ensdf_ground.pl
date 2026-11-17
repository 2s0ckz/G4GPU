#!/usr/bin/perl
# Extracts every GROUND STATE G4NuclideTable keeps from G4ENSDFSTATE's ENSDFSTATE.dat - its 2J
# and its magnetic moment - and writes them as src/data/ensdf_ground_states.hh.
#
#   perl tools/extract_ensdf_ground.pl [path/to/ENSDFSTATE.dat]
#
# The default is the 11.1.1 install's G4ENSDFSTATE2.3 - by path, not by $G4ENSDFSTATEDATA, which on
# this machine points at the 10.7.3 install's copy (byte-identical today, but not by contract).
#
# WHAT GEANT4 READS, AND WHAT THIS KEEPS (P26b, docs/RISK.md V259-V260)
#
# `G4NuclideTable::GenerateNuclide` (particles/management/src/G4NuclideTable.cc, 11.1.1) reads
# the file as seven whitespace-separated fields a row - Z, A, E (keV), the floating-level flag,
# the mean life (ns), 2J (an int) and mu (J/T: the row is multiplied by `joule/tesla`) - and
# keeps a row when `ionE == 0 && flb == no_Float` or its mean life is at least 1 ns.
# `G4IonTable::CreateIon(Z, A, 0, no_Float)` then asks `GetIsotope(Z, A, 0, no_Float)`, which
# returns the first kept state within 0.5 eV of zero WITH the no_Float flag - the ground-state
# row, when there is one - and takes `J = GetiSpin()` and `mu = GetMagneticMoment()` from it.
# With no such row CreateIon keeps its defaults, J = 0 and mu = 0.
#
# So the ground-state rows are exactly the rows with E == 0 and flag "-": 2977 of them in
# G4ENSDFSTATE2.3, no (Z, A) twice (asserted below, because a duplicate would make "the first"
# an order question), and no other kept state of either flag lies within 0.5 eV of zero
# (asserted too). 2J is kept as the integer the file holds - including the 420 rows whose 2J is
# 99, ENSDF's "unknown", which Geant4 reads as spin 49.5 and so as "has spin"; that is what the
# ion's definition says in Geant4 and so what it says here. mu is kept as the TEXT the file holds,
# so the compiler parses the same decimal `ifs >> G4double` parses, to the same double.
#
# The rows are written in the file's order, which is ascending in 1000*Z + A (asserted), so the
# device can binary-search the code.
use strict;
use warnings;

my $path = $ARGV[0];
if (!$path) {
  $path = 'D:/Documents/Geant4/Windows/geant4-v11.1.1-install/share/Geant4/data/'
    . 'G4ENSDFSTATE2.3/ENSDFSTATE.dat';
}
open my $fh, '<', $path or die "cannot open $path: $!";

my (@code, @twoj, @mu, @zz, @aa);
my %seen;
my $nrows = 0;
my $prev = -1;
my $near_zero = 0;
while (my $l = <$fh>) {
  $l =~ s/^\s+|\s+$//g;
  next if $l eq '';
  my @f = split /\s+/, $l;
  die "row $.: " . scalar(@f) . " fields, not 7: $l\n" if @f != 7;
  my ($z, $a, $e, $fl, $life, $j, $m) = @f;
  ++$nrows;
  my $c = 1000 * $z + $a;
  die "row $.: code $c after $prev - the file is not in code order\n" if $c < $prev;
  $prev = $c;
  my $kept = (($e + 0) == 0 && $fl eq '-') || (($life + 0) >= 1.0);
  # A kept state other than the ground state within the 0.5 eV window would be a candidate for
  # GetIsotope's first match. None exists; say so if a future file adds one.
  if ($kept && ($e + 0) != 0 && abs($e + 0) < 0.0005) { ++$near_zero; }
  next unless (($e + 0) == 0 && $fl eq '-');
  die "row $.: ($z, $a) has a second ground-state row\n" if $seen{$c}++;
  die "row $.: 2J '$j' is not an integer\n" if $j !~ /^-?\d+$/;
  die "row $.: mu '$m' is not a number\n" if $m !~ /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;
  push @code, $c;
  push @twoj, $j;
  push @mu, $m;
  push @zz, $z;
  push @aa, $a;
}
close $fh;
die "kept states within 0.5 eV of a ground state: $near_zero\n" if $near_zero;
my $n = scalar @code;
die "no ground states read from $path\n" if !$n;
my $nspin = grep { $_ != 0 } @twoj;
my $n99 = grep { $_ == 99 } @twoj;
my $nmu = grep { ($_ + 0) != 0 } @mu;

my $out = 'src/data/ensdf_ground_states.hh';
open my $oh, '>', $out or die "cannot write $out: $!";
print $oh <<"EOT";
// G4ENSDFSTATE's ground states: 2J and the magnetic moment of every nuclide G4IonTable::CreateIon
// can find a ground-state row for. $n nuclides out of $nrows rows; $nspin with 2J > 0 ($n99 of
// them 2J = 99, ENSDF's "unknown", which Geant4 reads as spin 49.5), $nmu with a non-zero moment.
//
// Written by tools/extract_ensdf_ground.pl - do not edit. Its header says which rows
// G4NuclideTable::GenerateNuclide keeps and why these are the ones CreateIon reads; the extractor
// asserts the file's order, one ground-state row per nuclide, and no other kept state within
// GetIsotope's 0.5 eV of zero. Source: $path
//
// mu is in J/T, as the file writes it; G4NuclideTable multiplies it by `joule/tesla` and that
// conversion is done where it is used (em::ion_particle_def), with CLHEP's own constants.
#ifndef G4GPU_DATA_ENSDF_GROUND_STATES_HH
#define G4GPU_DATA_ENSDF_GROUND_STATES_HH

namespace g4gpu::data {

constexpr int kEnsdfGroundStates = $n;

/// 1000*Z + A, ascending.
__host__ __device__ inline const int* ensdf_ground_code() {
  static const int v[$n] = {
EOT
sub emit {
  my ($arr, $per) = @_;
  my @a = @$arr;
  for (my $i = 0; $i < @a; $i += $per) {
    my $hi = $i + $per - 1;
    $hi = $#a if $hi > $#a;
    print $oh '    ' . join(', ', @a[$i .. $hi]) . ($hi < $#a ? ',' : '') . "\n";
  }
}
emit(\@code, 10);
print $oh <<"EOT";
  };
  return v;
}

/// 2J - G4IsotopeProperty::GetiSpin(), the G4Ions constructor's iSpin.
__host__ __device__ inline const int* ensdf_ground_two_j() {
  static const int v[$n] = {
EOT
emit(\@twoj, 20);
print $oh <<"EOT";
  };
  return v;
}

/// The magnetic moment in J/T, as written in ENSDFSTATE.dat.
__host__ __device__ inline const double* ensdf_ground_mu() {
  static const double v[$n] = {
EOT
emit(\@mu, 6);
print $oh <<"EOT";
  };
  return v;
}

/// The row of (Z, A)'s ground state, or -1 when ENSDFSTATE has none - in which case
/// G4IonTable::CreateIon keeps J = 0 and mu = 0. A binary search over the code, 12 probes.
__host__ __device__ inline int ensdf_ground_index(int z, int a) {
  const int code = 1000 * z + a;
  const int* c = ensdf_ground_code();
  int lo = 0;
  int hi = kEnsdfGroundStates - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) / 2;
    if (c[mid] == code) { return mid; }
    if (c[mid] < code) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return -1;
}

}  // namespace g4gpu::data

#endif  // G4GPU_DATA_ENSDF_GROUND_STATES_HH
EOT
close $oh;
print "wrote $out: $n ground states of $nrows rows ($nspin with spin, $n99 with 2J = 99, $nmu with a moment)\n";
