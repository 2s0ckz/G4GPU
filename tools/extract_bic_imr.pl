#!/usr/bin/perl
# Extracts the im_r_matrix collision tree's compiled-in tables out of Geant4 11.1.1 and writes
# them as src/physics/hadronic/bic/im_r/imr_tables.hh.
#
#   perl tools/extract_bic_imr.pl
#
# Why extracted and not retyped. The two angular-distribution tables are 39x180 and 40x180
# single-precision cumulative probabilities - 14,220 numbers whose only visible consequence is
# the shape of a scattering angle. A transposed digit in one of them moves one degree bin at one
# energy, which no histogram with fewer than a million samples would see, and there is no way to
# eyeball them. The same argument tools/extract_deex_tables.pl makes for nuclear masses.
#
# THE TABLES ARE FLOAT AND THAT IS LOAD-BEARING. G4AngularDistributionNP::sig and ::elab are
# declared `const G4float`, and `CosTheta` reads them into G4doubles and does its bisection and
# its two linear interpolations in double. So the values that enter the arithmetic are the
# float-rounded ones, and a port that stored them as double literals would interpolate between
# slightly different numbers. They are emitted as `float` here for that reason, and the port
# promotes on read exactly where Geant4 does.
#
# Every array's length is asserted against the enum Geant4 declares it with (NENERGY, NANGLE,
# tableSize, _tableSize, nFit, nPar), so a reshaped table fails here loudly instead of writing a
# short one and being interpolated off the end.
use strict;
use warnings;

my $g4 = $ENV{G4SRC} || 'D:/Documents/Geant4/Windows/geant4-v11.1.1';
my $imr = "$g4/source/processes/hadronic/models/im_r_matrix";

# ---------------------------------------------------------------------------------------------
# Pulls one C array's initialiser list out of a file, flattening nested braces. Same shape as
# tools/extract_deex_tables.pl's `grab`; repeated rather than shared because the two scripts are
# owned by different packages and a shared helper is a shared file to merge.
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
  die "$file: $decl - not found\n" if !$on;
  die "$file: $decl - no opening brace found\n" if $text !~ /\{/;
  $text =~ s/^.*?\{//s;
  my @v = ($text =~ /(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)f?/g);
  die "$file: $decl gave " . scalar(@v) . " values, expected $want\n"
    if defined $want && scalar(@v) != $want;
  return @v;
}

# Asserts that a declared enum/constant in a Geant4 header still says what the port assumes.
my $checks = 0;
sub want_const {
  my ($file, $re, $expect, $label) = @_;
  ++$checks;
  open my $fh, '<', $file or die "cannot open $file: $!";
  local $/;
  my $s = <$fh>;
  close $fh;
  # Comments are stripped first, because both angular-distribution headers carry a COMMENTED-OUT
  # `enum { NENERGY=22, NANGLE=180 }` - the pre-2010 table shape - immediately above the live
  # one, and it matches the same pattern. Reading that one would assert 22 energies against a
  # 39-row table and then read 17 rows off the end.
  $s =~ s{//[^\n]*}{}g;
  die "$label: pattern not found in $file\n" if $s !~ /$re/;
  die "$label: source says '$1', port assumes '$expect'\n" if $1 ne $expect;
  printf "  ok %-46s %s\n", $label, $expect;
}

sub emit {
  my ($fh, $type, $name, $per, @v) = @_;
  print $fh "__host__ __device__ inline const $type* $name() {\n";
  print $fh "  static const $type v[" . scalar(@v) . "] = {\n";
  for (my $i = 0; $i < @v; $i += $per) {
    my @row = @v[$i .. ($i + $per - 1 > $#v ? $#v : $i + $per - 1)];
    my $suffix = ($type eq 'float') ? 'f' : '';
    print $fh "    " . join(', ', map { $_ . $suffix } @row) . ",\n";
  }
  print $fh "  };\n  return v;\n}\n\n";
}

# =============================================================================================
# 1. The angular distributions. NP is 39 energies, PP is 40; both are 180 one-degree bins of
#    CUMULATIVE probability in cos(theta_cm), tabulated against a LAB kinetic energy in GeV.
# =============================================================================================
my ($NE_NP, $NE_PP, $NANG) = (39, 40, 180);
want_const("$imr/include/G4AngularDistributionNP.hh",
           qr/enum \{ NENERGY=(\d+), NANGLE=180 \}/, "$NE_NP", 'G4AngularDistributionNP NENERGY');
want_const("$imr/include/G4AngularDistributionPP.hh",
           qr/enum \{ NENERGY=(\d+), NENERGYC=22, NANGLE=180 \}/, "$NE_PP",
           'G4AngularDistributionPP NENERGY');

my @np_sig  = grab("$imr/include/G4AngularDistributionNPData.hh",
                   qr/G4AngularDistributionNP::sig\[NENERGY\]\[NANGLE\]/, $NE_NP * $NANG);
my @np_elab = grab("$imr/include/G4AngularDistributionNPData.hh",
                   qr/G4AngularDistributionNP::elab\[NENERGY\]/, $NE_NP);
my @pp_sig  = grab("$imr/include/G4AngularDistributionPPData.hh",
                   qr/G4AngularDistributionPP::sig\[NENERGY\]\[NANGLE\]/, $NE_PP * $NANG);
my @pp_elab = grab("$imr/include/G4AngularDistributionPPData.hh",
                   qr/G4AngularDistributionPP::elab\[NENERGY\]/, $NE_PP);

# The last angle bin of every energy row is the total, so it has to be 1 to within float
# rounding or the bisection in CosTheta has no root for a sample near 1. Asserted rather than
# assumed, because a truncated row would otherwise show up only as a rare forward-angle bias.
for my $j (0 .. $NE_NP - 1) {
  my $last = $np_sig[$j * $NANG + $NANG - 1];
  die "NP row $j ends at $last, not ~1\n" if abs($last - 1.0) > 2e-5;
}
for my $j (0 .. $NE_PP - 1) {
  my $last = $pp_sig[$j * $NANG + $NANG - 1];
  die "PP row $j ends at $last, not ~1\n" if abs($last - 1.0) > 2e-5;
}
# And they have to be non-decreasing along the row, which is what makes the bisection valid.
for my $j (0 .. $NE_NP - 1) {
  for my $k (1 .. $NANG - 1) {
    die "NP row $j not monotone at bin $k\n"
      if $np_sig[$j * $NANG + $k] < $np_sig[$j * $NANG + $k - 1];
  }
}
for my $j (0 .. $NE_PP - 1) {
  for my $k (1 .. $NANG - 1) {
    die "PP row $j not monotone at bin $k\n"
      if $pp_sig[$j * $NANG + $k] < $pp_sig[$j * $NANG + $k - 1];
  }
}
printf "  ok %-46s %d + %d values, rows end at 1, monotone\n", 'angular sig tables',
       scalar(@np_sig), scalar(@pp_sig);

# =============================================================================================
# 2. The low-energy NN cross-section tables.
# =============================================================================================
want_const("$imr/src/G4XNNTotalLowE.cc", qr/G4XNNTotalLowE::tableSize = (\d+)/, '29',
           'G4XNNTotalLowE tableSize');
# TWENTY-EIGHT values for a twenty-nine-element array. `ss[29]` is declared with 29 slots and
# initialised with 28 energies, so `ss[28]` is zero-filled by the language - and the constructor
# then pushes a 29th (sqrt(s), sigma) pair at sqrt(s) = 0 with the LAST cross section attached to
# it, into a table that G4LowEXsection interpolates in log(sqrt(s)). The zero is appended here
# rather than the count being relaxed to 28, because the pair it creates is part of the object
# and a release that filled the slot would change the table's top end; and asserting 28 means
# such a release fails here loudly. Whether the pair is ever READ is a separate question the
# port answers in xsec_nn.cuh: it is not, because G4XNNTotalLowE::IsValid stops at 3 GeV and the
# pair is only reachable above 3002.71 MeV.
my @nn_ss    = grab("$imr/src/G4XNNTotalLowE.cc", qr/G4XNNTotalLowE::ss\[29\]/, 28);
push @nn_ss, 0;
my @nn_pptot = grab("$imr/src/G4XNNTotalLowE.cc", qr/G4XNNTotalLowE::ppTot\[29\]/, 29);
my @nn_nptot = grab("$imr/src/G4XNNTotalLowE.cc", qr/G4XNNTotalLowE::npTot\[29\]/, 29);

want_const("$imr/src/G4XNNElasticLowE.cc", qr/G4XNNElasticLowE::tableSize = (\d+)/, '101',
           'G4XNNElasticLowE tableSize');
my @nnel_pp = grab("$imr/src/G4XNNElasticLowE.cc", qr/G4XNNElasticLowE::ppTable\[101\]/, 101);
my @nnel_np = grab("$imr/src/G4XNNElasticLowE.cc", qr/G4XNNElasticLowE::npTable\[101\]/, 101);

want_const("$imr/src/G4XnpElasticLowE.cc", qr/G4XnpElasticLowE::_tableSize = (\d+)/, '101',
           'G4XnpElasticLowE _tableSize');
my @npel = grab("$imr/src/G4XnpElasticLowE.cc", qr/G4XnpElasticLowE::_sigmaTable\[101\]/, 101);

want_const("$imr/src/G4XnpTotalLowE.cc", qr/G4XnpTotalLowE::_tableSize = (\d+)/, '101',
           'G4XnpTotalLowE _tableSize');
my @nptot = grab("$imr/src/G4XnpTotalLowE.cc", qr/G4XnpTotalLowE::_sigmaTable\[101\]/, 101);

# G4XNNElasticLowE::npTable and G4XnpElasticLowE::_sigmaTable are the same 101 numbers in two
# files, and so are G4XnpTotalLowE::_sigmaTable and the first eleven of them. Asserting the
# duplication means a release that edits one copy and not the other is caught here rather than
# producing two different np elastic cross sections in the same event.
for my $i (0 .. 100) {
  die "npTable[$i] = $nnel_np[$i] but _sigmaTable[$i] = $npel[$i]\n" if $nnel_np[$i] != $npel[$i];
}
++$checks;
printf "  ok %-46s all 101 agree\n", 'npTable == G4XnpElasticLowE table';

# The three shared grid constants. _eMinTable and _eStepLog decide where every one of those 101
# values sits in sqrt(s), and they are repeated verbatim in four files.
for my $f (qw(G4XNNElasticLowE G4XnpElasticLowE G4XnpTotalLowE)) {
  want_const("$imr/src/$f.cc", qr/${f}::_eMinTable = ([\d.]+)/, '1.8964808', "$f _eMinTable");
  want_const("$imr/src/$f.cc", qr/${f}::_eStepLog = ([\d.]+)/, '0.01', "$f _eStepLog");
}

# =============================================================================================
# 3. The PDG fits.
# =============================================================================================
want_const("$imr/src/G4XPDGTotal.cc", qr/G4XPDGTotal::nFit = (\d+)/, '5', 'G4XPDGTotal nFit');
want_const("$imr/src/G4XPDGElastic.cc", qr/G4XPDGElastic::nPar = (\d+)/, '7',
           'G4XPDGElastic nPar');
my @pdgt_pp  = grab("$imr/src/G4XPDGTotal.cc", qr/G4XPDGTotal::ppPDGFit\[5\]/, 5);
my @pdgt_np  = grab("$imr/src/G4XPDGTotal.cc", qr/G4XPDGTotal::npPDGFit\[5\]/, 5);
my @pdgt_pip = grab("$imr/src/G4XPDGTotal.cc", qr/G4XPDGTotal::pipPDGFit\[5\]/, 5);
my @pdge_pp   = grab("$imr/src/G4XPDGElastic.cc", qr/G4XPDGElastic::ppPDGFit\[7\]/, 7);
my @pdge_pip  = grab("$imr/src/G4XPDGElastic.cc", qr/G4XPDGElastic::pPiPlusPDGFit\[7\]/, 7);
my @pdge_pim  = grab("$imr/src/G4XPDGElastic.cc", qr/G4XPDGElastic::pPiMinusPDGFit\[7\]/, 7);

# The three PDG exponents are `static const` locals inside G4XPDGTotal::CrossSection, so they
# cannot be grabbed as an array; they are asserted in place instead.
want_const("$imr/src/G4XPDGTotal.cc", qr/G4double epsilon = ([\d.]+);/, '0.095', 'PDG epsilon');
want_const("$imr/src/G4XPDGTotal.cc", qr/G4double eta1 = (-[\d.]+);/, '-0.34', 'PDG eta1');
want_const("$imr/src/G4XPDGTotal.cc", qr/G4double eta2 = (-[\d.]+);/, '-0.55', 'PDG eta2');

# =============================================================================================
# 4. G4CollisionComposite's buffering grid, which decides every meson-baryon cross section:
#    the composite has no total cross-section source of its own, so it sums its components on
#    these 32 kinetic energies once and interpolates the sum forever after.
# =============================================================================================
want_const("$imr/src/G4CollisionComposite.cc", qr/G4CollisionComposite::nPoints = (\d+)/, '32',
           'G4CollisionComposite nPoints');
my @compT = grab("$imr/src/G4CollisionComposite.cc",
                 qr/G4CollisionComposite::theT\[nPoints\]/, 32);

# =============================================================================================
# Write the header.
# =============================================================================================
my $out = 'src/physics/hadronic/bic/im_r/imr_tables.hh';
open my $fh, '>', $out or die "cannot write $out: $!";
print $fh <<'HDR';
// The im_r_matrix collision tree's compiled-in tables, out of Geant4 11.1.1.
//
//   angular_np_sig()      G4AngularDistributionNP::sig   [39][180] cumulative P(theta_cm)
//   angular_np_elab()     G4AngularDistributionNP::elab  [39]      lab kinetic energy, GeV
//   angular_pp_sig()      G4AngularDistributionPP::sig   [40][180]
//   angular_pp_elab()     G4AngularDistributionPP::elab  [40]
//   nn_total_lowe_ss()    G4XNNTotalLowE::ss      [29]  sqrt(s), MeV
//   nn_total_lowe_pp()    G4XNNTotalLowE::ppTot   [29]  mb
//   nn_total_lowe_np()    G4XNNTotalLowE::npTot   [29]  mb
//   nn_elastic_lowe_pp()  G4XNNElasticLowE::ppTable [101] mb
//   nn_elastic_lowe_np()  G4XNNElasticLowE::npTable [101] mb, == G4XnpElasticLowE::_sigmaTable
//   np_total_lowe()       G4XnpTotalLowE::_sigmaTable [101] mb
//   pdg_total_*()         G4XPDGTotal's five-number fits
//   pdg_elastic_*()       G4XPDGElastic's seven-number fits
//   composite_T()         G4CollisionComposite::theT [32] GeV
//
// **The angular tables are float, and the port reads them as float.** Geant4 declares them
// `const G4float` and `CosTheta` promotes each element to G4double as it touches it, so the
// numbers the bisection interpolates between are the float-rounded ones. Stored as double
// literals they would be different numbers, and the sampled angle would differ in the seventh
// digit at every energy - small, systematic, and invisible to any histogram.
//
// **Two 101-value tables are the same numbers in two Geant4 files** - G4XNNElasticLowE::npTable
// and G4XnpElasticLowE::_sigmaTable - and the extractor asserts they still agree. They are
// emitted once.
//
// Written by tools/extract_bic_imr.pl - do not edit.
#ifndef G4GPU_BIC_IMR_TABLES_HH
#define G4GPU_BIC_IMR_TABLES_HH

namespace g4gpu::bic::imr {

constexpr int kAngularNpEnergies = 39;   ///< G4AngularDistributionNP::NENERGY
constexpr int kAngularPpEnergies = 40;   ///< G4AngularDistributionPP::NENERGY
constexpr int kAngularAngles = 180;      ///< NANGLE, one-degree bins
constexpr int kNNTotalLowESize = 29;     ///< G4XNNTotalLowE::tableSize
constexpr int kNNLowETableSize = 101;    ///< the four log-vector tables
constexpr int kCompositePoints = 32;     ///< G4CollisionComposite::nPoints

HDR
emit($fh, 'float', 'angular_np_sig', 8, @np_sig);
emit($fh, 'float', 'angular_np_elab', 8, @np_elab);
emit($fh, 'float', 'angular_pp_sig', 8, @pp_sig);
emit($fh, 'float', 'angular_pp_elab', 8, @pp_elab);
emit($fh, 'double', 'nn_total_lowe_ss', 8, @nn_ss);
emit($fh, 'double', 'nn_total_lowe_pp', 8, @nn_pptot);
emit($fh, 'double', 'nn_total_lowe_np', 8, @nn_nptot);
emit($fh, 'double', 'nn_elastic_lowe_pp', 8, @nnel_pp);
emit($fh, 'double', 'nn_elastic_lowe_np', 8, @nnel_np);
emit($fh, 'double', 'np_total_lowe', 8, @nptot);
emit($fh, 'double', 'pdg_total_pp', 5, @pdgt_pp);
emit($fh, 'double', 'pdg_total_np', 5, @pdgt_np);
emit($fh, 'double', 'pdg_total_pip', 5, @pdgt_pip);
emit($fh, 'double', 'pdg_elastic_pp', 7, @pdge_pp);
emit($fh, 'double', 'pdg_elastic_pip', 7, @pdge_pip);
emit($fh, 'double', 'pdg_elastic_pim', 7, @pdge_pim);
emit($fh, 'double', 'composite_T', 8, @compT);
print $fh "}  // namespace g4gpu::bic::imr\n#endif\n";
close $fh;
printf "%s written, %d assertions passed\n", $out, $checks;
