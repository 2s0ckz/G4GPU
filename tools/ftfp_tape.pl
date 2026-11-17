#!/usr/bin/perl
# P28: summarises the FTFP tape a run prints with G4GPU_FTFP_TAPE=N (src/physics/hadronic/
# interaction_apply.cuh, `ftfp_tape_row`). Reads the run's output on stdin or as arguments.
#
#   FTFPTAPE k kind pdg T Z A resZ resA resE* nDeex nSec KEchg KEneu KElight KEheavy KEnorow Ebal edep
#
# Prints, per kind (in flight / at rest) and per target: the number of interactions, the
# projectile's kinetic-energy distribution, the residual's mean E* (what the undecayed hand-over
# turned into an ion's kinetic energy before P28), the mean kinetic energy per class as the
# transport receives it, the energy balance, and the worst row - the first interaction whose final
# state carries more than the available energy.
use strict;
use warnings;

my (%n, %sum, %tb, %worst);
my @cols = qw(resE nDeex nSec KEchg KEneu KElight KEheavy KEnorow Ebal edep);
my $rows = 0;
while (my $l = <>) {
  next unless $l =~ /^FTFPTAPE\s+(.*)$/;
  my @f = split /\s+/, $1;
  next unless @f >= 18;
  my ($k, $kind, $pdg, $t, $z, $a, $rz, $ra, $re, $nd, $ns, $kc, $kn, $kl, $kh, $kr, $eb, $ed) = @f;
  ++$rows;
  my %v = (resE => $re, nDeex => $nd, nSec => $ns, KEchg => $kc, KEneu => $kn, KElight => $kl,
           KEheavy => $kh, KEnorow => $kr, Ebal => $eb, edep => $ed);
  for my $key ("kind$kind all", "kind$kind Z=$z") {
    ++$n{$key};
    $sum{$key}{$_} += $v{$_} for @cols;
    $sum{$key}{Ebal2} += $eb * $eb;
    if (!defined $worst{$key} || $eb > $worst{$key}[0]) { $worst{$key} = [$eb, $l]; }
  }
  my $bin = ($kind == 1) ? "rest" : sprintf("%4d-%4d", 100 * int($t / 100), 100 * int($t / 100) + 100);
  ++$tb{$bin};
}
print "$rows FTFPTAPE rows\n\nprojectile kinetic energy (MeV):\n";
printf "  %-10s %8d  %5.1f%%\n", $_, $tb{$_}, 100.0 * $tb{$_} / $rows for sort keys %tb;
print "\nper interaction, means (MeV):\n";
printf "  %-14s %8s %8s %6s %6s %9s %9s %8s %8s %8s %9s %9s\n", "", "N", "resE*", "nDeex",
  "nSec", "KEchg", "KEneu", "KElight", "KEheavy", "KEnorow", "Ebal", "rms Ebal";
for my $key (sort keys %n) {
  my $m = sub { $sum{$key}{$_[0]} / $n{$key} };
  my $rms = sqrt(abs($sum{$key}{Ebal2} / $n{$key} - $m->('Ebal') ** 2));
  printf "  %-14s %8d %8.3f %6.2f %6.2f %9.3f %9.3f %8.3f %8.3f %8.3f %9.4f %9.4f\n", $key,
    $n{$key}, $m->('resE'), $m->('nDeex'), $m->('nSec'), $m->('KEchg'), $m->('KEneu'),
    $m->('KElight'), $m->('KEheavy'), $m->('KEnorow'), $m->('Ebal'), $rms;
}
print "\nthe row with the largest energy excess, per kind:\n";
for my $key (sort grep { /all$/ } keys %worst) { print "  $key: $worst{$key}[1]"; }
