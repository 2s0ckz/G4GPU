#!/usr/bin/perl
# Extracts the 60 GEM evaporation channels of G4EvaporationDefaultGEMFactory out of
# Geant4 11.1.1 and writes them as src/data/gem_levels.hh.
#
#   perl tools/extract_gem_tables.pl
#
# Each channel is two objects and they carry different things:
#
#   G4XxGEMChannel.hh      the emitted nuclide's (A, Z) and its name. This is what
#                          G4GEMChannel uses for the mass, the Coulomb barrier object and
#                          the residual.
#   G4XxGEMProbability.cc  ANOTHER (A, Z), a spin, and a list of excited states of the
#                          emitted nuclide - energy, spin and lifetime. This is what
#                          G4GEMProbability uses for the width.
#
# The two (A, Z) pairs are supposed to be the same pair and for 58 of the 60 channels they
# are. They are extracted and emitted separately because for the other two they are not:
#
#   Be12   channel (12, 4), probability (9, 4)
#   O17    channel (17, 8), probability (17, 9)
#
# Both are defects in 11.1.1 and both change the emission probability of a channel that the
# default configuration uses - see src/physics/hadronic/deexcitation/gem.cuh. A port that
# assumed one pair would silently repair them.
#
# 1,617 excited states are pulled out rather than retyped for the reason
# tools/extract_deex_tables.pl gives: there is no way to eyeball them, and every lifetime is
# an expression in CLHEP units rather than a number - `fPlanck/(175.0*keV)` is a width
# converted to a lifetime, `800.0e-3*picosecond` is a lifetime, and three entries are written
# as `hbar_Planck*G4Log(2)/(x*keV)`, which is the same thing as the first form spelled out.
# The expression is evaluated here with CLHEP's own constants AND emitted as a comment beside
# the value, so the transcription can be audited against the source line it came from.
use strict;
use warnings;

my $g4 = $ENV{G4SRC} || 'D:/Documents/Geant4/Windows/geant4-v11.1.1';
my $gem = "$g4/source/processes/hadronic/models/de_excitation/gem_evaporation";

# CLHEP internal units, MeV/mm/ns, and hbar_Planck derived exactly as CLHEP derives it -
# `hbarc = hbar_Planck*c_light`, so hbar_Planck is hbarc/c_light and not an independent
# constant. Both factors are the ones src/core/units.cuh pins, so the value below is the one
# the port would compute.
my $hbarc = 1.9732698045930245e-10;      # MeV mm
my $c_light = 2.99792458e8 * 1000 / 1e9; # mm/ns = 299.792458
my $hbar_planck = $hbarc / $c_light;     # MeV ns
my $fplanck = $hbar_planck * log(2.0);   # G4GEMProbability::fPlanck

sub evaluate {
  my ($expr) = @_;
  my $e = $expr;
  # Longest identifier first: `second` contains `s`, `keV` contains `eV`.
  $e =~ s/hbar_Planck\s*\*\s*G4Log\s*\(\s*2(?:\.0*)?\s*\)/($fplanck)/g;
  $e =~ s/\bfPlanck\b/($fplanck)/g;
  $e =~ s/\bhbar_Planck\b/($hbar_planck)/g;
  $e =~ s/\bmillisecond\b/(1e6)/g;
  $e =~ s/\bmicrosecond\b/(1e3)/g;
  $e =~ s/\bnanosecond\b/(1.0)/g;
  $e =~ s/\bpicosecond\b/(1e-3)/g;
  $e =~ s/\bsecond\b/(1e9)/g;
  $e =~ s/\bkeV\b/(1e-3)/g;
  $e =~ s/\bMeV\b/(1.0)/g;
  $e =~ s/\bGeV\b/(1e3)/g;
  $e =~ s/\beV\b/(1e-6)/g;
  $e =~ s/(?<![\w.])s(?![\w])/(1e9)/g;
  $e =~ s/(?<![\w.])ns(?![\w])/(1.0)/g;
  die "unresolved identifier in '$expr' -> '$e'\n" if $e =~ /[A-DF-Za-df-z_]/;
  my $v = eval $e;
  die "cannot evaluate '$expr' -> '$e': $@\n" if !defined $v;
  return $v;
}

# The factory's order. Read from the factory rather than from a list retyped here, because
# G4Evaporation indexes `probabilities[]` by it and the order is therefore observable.
my @order;
{
  open my $fh, '<', "$gem/../evaporation/src/G4EvaporationDefaultGEMFactory.cc"
    or die "cannot open the factory: $!";
  while (my $l = <$fh>) {
    next if $l =~ m{^\s*//};
    push @order, $1 if $l =~ /theChannel->push_back\(\s*new\s+G4(\w+)GEMChannel\(\)\s*\)/;
  }
  close $fh;
}
die "factory gave " . scalar(@order) . " GEM channels, expected 60\n" if @order != 60;

my @chan;
my $nlev = 0;
for my $name (@order) {
  my %c = (name => $name);

  open my $fh, '<', "$gem/include/G4${name}GEMChannel.hh"
    or die "cannot open G4${name}GEMChannel.hh: $!";
  my $txt = do { local $/; <$fh> };
  close $fh;
  die "G4${name}GEMChannel.hh: no G4GEMChannel(A,Z,...)\n"
    if $txt !~ /G4GEMChannel\(\s*(\d+)\s*,\s*(\d+)\s*,/;
  ($c{ca}, $c{cz}) = ($1, $2);

  open $fh, '<', "$gem/src/G4${name}GEMProbability.cc"
    or die "cannot open G4${name}GEMProbability.cc: $!";
  my @lines = <$fh>;
  close $fh;
  my $body = join '', map { my $x = $_; $x =~ s{//.*}{}; $x } @lines;
  die "G4${name}GEMProbability.cc: no G4GEMProbability(A,Z,Spin)\n"
    if $body !~ /G4GEMProbability\(\s*(\d+)\s*,\s*(\d+)\s*,\s*([^)]+?)\s*\)/;
  ($c{pa}, $c{pz}) = ($1, $2);
  $c{spin_src} = $3;
  $c{spin} = evaluate($3);

  # The three vectors are filled in interleaved triples, and their ORDER is what pairs an
  # energy with its spin and its lifetime. Collected separately and their lengths compared,
  # so an interleaving mistake in the source shows up here rather than as a shifted table.
  my (@e, @sp, @lt, @esrc, @spsrc, @ltsrc);
  while ($body =~ /Excit(Energies|Spins|Lifetimes)\.push_back\(\s*(.*?)\s*\)\s*;/gs) {
    my ($which, $arg) = ($1, $2);
    my $v = evaluate($arg);
    if    ($which eq 'Energies')  { push @e, $v;  push @esrc, $arg; }
    elsif ($which eq 'Spins')     { push @sp, $v; push @spsrc, $arg; }
    else                          { push @lt, $v; push @ltsrc, $arg; }
  }
  die "G4${name}GEMProbability.cc: " . scalar(@e) . " energies, " . scalar(@sp)
    . " spins, " . scalar(@lt) . " lifetimes - must be equal\n"
    if @e != @sp || @e != @lt;
  $c{e} = \@e; $c{sp} = \@sp; $c{lt} = \@lt;
  $c{esrc} = \@esrc; $c{spsrc} = \@spsrc; $c{ltsrc} = \@ltsrc;
  $nlev += scalar(@e);
  push @chan, \%c;
}

# 1,617 across the 60 default-GEM channels. Asserted so that a Geant4 release that adds or
# removes an excited state fails here instead of being transcribed silently.
die "got $nlev excited states, expected 1617\n" if $nlev != 1617;

my $mismatch = join ', ', map { "$_->{name} ($_->{ca},$_->{cz}) vs ($_->{pa},$_->{pz})" }
                          grep { $_->{ca} != $_->{pa} || $_->{cz} != $_->{pz} } @chan;
die "expected exactly the Be12 and O17 (A,Z) mismatches, got: $mismatch\n"
  if $mismatch ne 'Be12 (12,4) vs (9,4), O17 (17,8) vs (17,9)';

my $out = 'src/data/gem_levels.hh';
open my $fh, '>', $out or die "cannot write $out: $!";
print $fh <<"HDR";
// The 60 GEM evaporation channels of G4EvaporationDefaultGEMFactory, and the excited states
// each one's G4GEMProbability sums a width over.
//
// GENERATED by tools/extract_gem_tables.pl from Geant4 11.1.1 - do not edit by hand.
// $nlev excited states over 60 channels, in the factory's own order (which is observable:
// G4Evaporation indexes its probability array by it).
//
// Two (A, Z) pairs per channel, because Geant4 has two and for two channels they disagree:
// `chan_a/chan_z` is G4GEMChannel's, `prob_a/prob_z` is G4GEMProbability's. Be12 is
// (12, 4) and (9, 4); O17 is (17, 8) and (17, 9). See gem.cuh for what each one drives.
//
// Energies and lifetimes are in CLHEP internal units (MeV, ns). A lifetime written in the
// source as `fPlanck/(E*keV)` is a level WIDTH converted to a lifetime with
// fPlanck = hbar_Planck*log(2); the source expression is beside every value.
#ifndef G4GPU_DATA_GEM_LEVELS_HH
#define G4GPU_DATA_GEM_LEVELS_HH

namespace g4gpu::data {

/// One GEM channel. `first_level` indexes gem_level_*() and `n_levels` is how many.
struct GemChannelEntry {
  int chan_a, chan_z;   ///< G4GEMChannel(A, Z) - the nuclide actually emitted
  int prob_a, prob_z;   ///< G4GEMProbability(A, Z) - what the width is computed for
  double spin;          ///< G4GEMProbability's ground-state spin
  int first_level, n_levels;
};

HDR

printf $fh "__host__ __device__ inline const GemChannelEntry* gem_channels() {\n";
printf $fh "  static const GemChannelEntry c[60] = {\n";
my $first = 0;
for my $c (@chan) {
  printf $fh "    {%3d, %2d, %3d, %2d, %-12s %4d, %2d},  // %s\n", $c->{ca}, $c->{cz},
             $c->{pa}, $c->{pz}, sprintf('%.17g,', $c->{spin}), $first,
             scalar(@{$c->{e}}), $c->{name} . ' spin ' . $c->{spin_src};
  $first += scalar(@{$c->{e}});
}
printf $fh "  };\n  return c;\n}\n\n";

for my $tri (['energy', 'e', 'esrc'], ['spin', 'sp', 'spsrc'],
             ['lifetime', 'lt', 'ltsrc']) {
  my ($label, $key, $src) = @$tri;
  printf $fh "__host__ __device__ inline const double* gem_level_%s() {\n", $label;
  printf $fh "  static const double v[%d] = {\n", $nlev;
  for my $c (@chan) {
    printf $fh "    // %s\n", $c->{name};
    for my $i (0 .. $#{$c->{$key}}) {
      printf $fh "    %.17g,  // %s\n", $c->{$key}[$i], $c->{$src}[$i];
    }
  }
  printf $fh "  };\n  return v;\n}\n\n";
}

print $fh "}  // namespace g4gpu::data\n\n#endif\n";
close $fh;
print "wrote $out: 60 channels, $nlev excited states\n";
