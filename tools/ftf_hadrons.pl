#!/usr/bin/perl
# Turns ref/oracle/ftf_hadrons.csv into src/data/ftf_hadrons.hh.
#
# WHY A TABLE AT ALL. Every class in src/physics/hadronic/ftf/ turns a quark pair into a hadron
# by building a PDG code and calling G4ParticleTable::FindParticle on it - G4HadronBuilder's
# Meson() and Barion(), SetMinMasses' six tables, SplitLast's three last-splitting functions,
# G4FTFParameters::GetMinMass. A null answer is load-bearing in all of them: it is how the
# do-while loops over Meson[...] and Baryon[...] terminate, and how an illegal quark
# combination is rejected without an exception. So the port needs the same lookup, with the
# same membership, and with the PDG masses Geant4 11.1.1's own particle definitions carry.
#
# WHAT IS EXCLUDED, AND WHY IT HAS TO BE. The oracle dumps the WHOLE initialised particle
# table, which includes every nucleus G4IonTable happened to create while the dump program ran
# - Al27 and Xe132 are there because the dump program's detector is built out of them. That set
# is not a property of Geant4; it is a property of the run. Every one of them has |PDG| >= 1e9
# (the 10LZZZAAAI ion encoding) or PDG == 0 (the two geantinos), and no FTF class ever asks
# FindParticle for an ion code, so they are dropped here and the count of what is left is
# asserted. Codes like 100002210 (N(2220)+) and 9000211 (a0(980)+) are NOT ions and are kept:
# they are nine- and seven-digit MESON and BARYON codes, and G4HadronBuilder's heavy-flavour
# substitution list names several of the same shape (9000443, 9010553).
#
# Usage:  perl tools/ftf_hadrons.pl [oracle_dir] > src/data/ftf_hadrons.hh
use strict;
use warnings;

my $dir = $ARGV[0] || 'ref/oracle';
my $csv = "$dir/ftf_hadrons.csv";
open(my $fh, '<', $csv) or die "ftf_hadrons.pl: cannot read $csv: $!\n";

my $hdr = <$fh>;
chomp $hdr;
$hdr =~ s/\r$//;
my @cols = split /,/, $hdr;
my %ix;
$ix{$cols[$_]} = $_ for 0 .. $#cols;
for my $need (qw(pdg name subtype mass width charge baryon shortlived minmass
                 nq4 naq4 nq5 naq5)) {
  die "ftf_hadrons.pl: $csv has no column '$need'\n" unless exists $ix{$need};
}

my @rows;
my $dropped = 0;
my %seen;
while (my $line = <$fh>) {
  chomp $line;
  $line =~ s/\r$//;
  next unless length $line;
  my @f = split /,/, $line;
  die "ftf_hadrons.pl: row has " . scalar(@f) . " fields, expected " . scalar(@cols) .
      ": $line\n" unless @f == @cols;
  my $pdg = $f[$ix{pdg}] + 0;
  if ($pdg == 0 || $pdg >= 1000000000 || $pdg <= -1000000000) { $dropped++; next; }
  die "ftf_hadrons.pl: duplicate PDG code $pdg\n" if exists $seen{$pdg};
  $seen{$pdg} = 1;
  push @rows,
      { pdg   => $pdg,
        name  => $f[$ix{name}],
        sub   => $f[$ix{subtype}],
        mass  => $f[$ix{mass}],
        width => $f[$ix{width}],
        chg   => $f[$ix{charge}],
        bar   => $f[$ix{baryon}] + 0,
        sl    => $f[$ix{shortlived}] + 0,
        mmin  => $f[$ix{minmass}],
        nq4   => $f[$ix{nq4}] + 0,
        naq4  => $f[$ix{naq4}] + 0,
        nq5   => $f[$ix{nq5}] + 0,
        naq5  => $f[$ix{naq5}] + 0 };
}
close $fh;

@rows = sort { $a->{pdg} <=> $b->{pdg} } @rows;

# ---- the counts, asserted here and again as a static_assert in the output ----
#
# 485 hadrons, 12 quarks and 50 diquarks. The quark count is 12 and not 10 because the top
# quark and its antiquark are defined too (SampleQuarkFlavor cannot reach them: it returns
# 1 + int(u/StrangeSuppress) with StrangeSuppress near 0.44, so 1, 2 or 3, and its heavy branch
# returns 4 or 5). The diquark count is 50 = 2 * 25: fifteen flavour pairs, of which the five
# same-flavour ones have only the spin-1 state, so 10 mixed * 2 spins + 5 = 25.
my $EXPECT_ROWS = 485;
my $EXPECT_QUARKS = 12;
my $EXPECT_DIQUARKS = 50;
my $EXPECT_DROPPED = 31;

my $nq = grep { $_->{sub} eq 'quark' } @rows;
my $nd = grep { $_->{sub} eq 'di_quark' } @rows;
die "ftf_hadrons.pl: expected $EXPECT_ROWS rows, got " . scalar(@rows) . "\n"
    unless @rows == $EXPECT_ROWS;
die "ftf_hadrons.pl: expected $EXPECT_QUARKS quarks, got $nq\n" unless $nq == $EXPECT_QUARKS;
die "ftf_hadrons.pl: expected $EXPECT_DIQUARKS diquarks, got $nd\n"
    unless $nd == $EXPECT_DIQUARKS;
die "ftf_hadrons.pl: expected $EXPECT_DROPPED nuclei/geantinos dropped, got $dropped\n"
    unless $dropped == $EXPECT_DROPPED;

# Every code SetMinMasses and G4HadronBuilder reach must be present or absent deliberately.
# These eleven are the ones whose presence the tables depend on and whose absence would change
# an answer silently: the five q-qbar seeds of minMassQQbarStr, the pi0/eta/eta'/rho/omega/phi
# of the meson table.
for my $code (111, 211, 311, 411, 511, 113, 221, 223, 331, 333) {
  die "ftf_hadrons.pl: PDG $code is absent - a table SetMinMasses fills depends on it\n"
      unless exists $seen{$code};
}

sub subtype_enum {
  my ($s) = @_;
  return 'kQuark' if $s eq 'quark';
  return 'kDiQuark' if $s eq 'di_quark';
  return 'kOther';
}

print <<"HEADER";
// Geant4 11.1.1's particle table, as every FTF class reads it through
// G4ParticleTable::FindParticle(code).
//
// GENERATED by tools/ftf_hadrons.pl from ref/oracle/ftf_hadrons.csv. Do not edit by hand:
// re-run the oracle (ref/dump/build.bat then ref/oracle/run.bat tables) and the script.
//
// $EXPECT_ROWS entries, sorted by PDG code: every meson, baryon, lepton, boson, quark and
// diquark QBBC's ConstructParticle defines, less the nuclei and the two geantinos - see the
// script's header for why those cannot be part of a committed table.
//
// The fields are exactly the G4ParticleDefinition accessors the transcribed code calls.
// `minmass` is G4SampleResonance::GetMinimumMass, which G4ExcitedStringDecay needs to sample a
// resonance's Breit-Wigner mass; it is -1 where Geant4 cannot answer, i.e. for a particle that
// is not short-lived and for the diquarks, whose GetDecayTable() is null and whose minimum
// mass G4SampleResonance would read through that null pointer.
#pragma once

namespace g4gpu::data {

/// G4ParticleDefinition::GetParticleSubType(), reduced to the three values FTF branches on.
/// Everything that is not a quark or a diquark is `kOther`: no FTF class distinguishes a pion
/// from a Lambda by sub-type, only by PDG code and baryon number.
enum class FtfSubType : unsigned char { kOther = 0, kQuark = 1, kDiQuark = 2 };

struct FtfHadron {
  int pdg;
  double mass;    ///< GetPDGMass(), MeV
  double width;   ///< GetPDGWidth(), MeV
  double charge;  ///< GetPDGCharge()/eplus - a THIRD-integer for a quark or diquark
  int baryon;     ///< GetBaryonNumber()
  double minmass; ///< G4SampleResonance::GetMinimumMass, or -1 if Geant4 cannot answer
  FtfSubType subtype;
  bool shortlived;
  signed char nq4, naq4, nq5, naq5;  ///< charm / bottom (anti)quark content
  const char* name;
};

constexpr int kFtfHadronCount = $EXPECT_ROWS;
constexpr int kFtfQuarkCount = $EXPECT_QUARKS;
constexpr int kFtfDiQuarkCount = $EXPECT_DIQUARKS;

__host__ __device__ inline const FtfHadron* ftf_hadrons() {
  static const FtfHadron v[kFtfHadronCount] = {
HEADER

for my $r (@rows) {
  printf("    {%d, %s, %s, %s, %d, %s, FtfSubType::%s, %s, %d, %d, %d, %d, \"%s\"},\n",
         $r->{pdg}, $r->{mass}, $r->{width}, $r->{chg}, $r->{bar}, $r->{mmin},
         subtype_enum($r->{sub}), ($r->{sl} ? 'true' : 'false'), $r->{nq4}, $r->{naq4},
         $r->{nq5}, $r->{naq5}, $r->{name});
}

print <<'FOOTER';
  };
  return v;
}

/// G4ParticleTable::FindParticle(code), as a binary search over the sorted table.
///
/// Returns nullptr for a code that is not there, which is what Geant4 does and what every
/// caller in this package tests for: the loops in SplitLast terminate on it, the Meson table's
/// broken c-cbar entry (docs/RISK.md V85) resolves to it, and G4HadronBuilder returns it
/// unchanged for an illegal quark pair.
__host__ __device__ inline const FtfHadron* ftf_find_hadron(int pdg) {
  const FtfHadron* v = ftf_hadrons();
  int lo = 0, hi = kFtfHadronCount - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) >> 1;
    if (v[mid].pdg == pdg) { return &v[mid]; }
    if (v[mid].pdg < pdg) { lo = mid + 1; } else { hi = mid - 1; }
  }
  return nullptr;
}

}  // namespace g4gpu::data
FOOTER
