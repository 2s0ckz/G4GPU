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
# 5. The six resonance-production cross-section tables. Each is one 121-point energy grid in GeV
#    and between one and fifteen 121-point sigma columns in millibarn, one per resonance mass.
#
#    The columns are named by the resonance's nominal mass - sigmaND1600, sigmaNN1440 - and each
#    Geant4 table's constructor maps FOUR particle names onto each column, one per charge state
#    for the Deltas and (for the N* tables) two. The mapping is regular and is recorded here as
#    the list of masses per table; the port keys by that integer, because a kernel has no strings.
# =============================================================================================
my @res_tables = (
  ['G4XNDeltaTable',          'energyTable', ['ND1232'],                                   'nd'],
  ['G4XNDeltastarTable',      'energyTable',
   [qw(ND1600 ND1620 ND1700 ND1900 ND1905 ND1910 ND1920 ND1930 ND1950)],                   'ndstar'],
  ['G4XNNstarTable',          'energyTable',
   [qw(NN1440 NN1520 NN1535 NN1650 NN1675 NN1680 NN1700 NN1710 NN1720 NN1900 NN1990 NN2090
       NN2190 NN2220 NN2250)],                                                             'nnstar'],
  ['G4XDeltaDeltaTable',      'energyTable', ['DD1232'],                                   'dd'],
  ['G4XDeltaDeltastarTable',  'energyTable',
   [qw(DD1600 DD1620 DD1700 DD1900 DD1905 DD1910 DD1920 DD1930 DD1950)],                   'ddstar'],
  ['G4XDeltaNstarTable',      'energyTable',
   [qw(DN1440 DN1520 DN1535 DN1650 DN1675 DN1680 DN1700 DN1710 DN1720 DN1900 DN1990 DN2090
       DN2190 DN2220 DN2250)],                                                             'dnstar'],
);

my %res_energy;
my %res_sigma;
my %res_masses;
my %short;
my $n_res_cols = 0;for my $t (@res_tables) {
  my ($cls, $egrid, $cols, $tag) = @$t;
  $short{$tag} = [];
  my @e = grab("$imr/src/$cls.cc", qr/const G4double \Q$cls\E::\Q$egrid\E\[121\]/, 121);
  $res_energy{$tag} = \@e;
  my @flat;
  my @masses;
  for my $c (@$cols) {
    # NOT asserted at 121. Several of these columns are declared [121] and initialised with
    # FEWER values, leaving the tail zero-filled by the language - `G4XNNstarTable::sigmaNN1535`
    # has 113. The count is checked against the known set below instead, so a column that gets
    # longer or shorter fails, and the zero tail is written out explicitly rather than left to
    # whatever the port's compiler does.
    my @s = grab("$imr/src/$cls.cc", qr/const G4double \Q$cls\E::sigma\Q$c\E\[121\]/, undef);
    my $got = scalar(@s);
    # The `::` is escaped: `"$cls::sigma$c"` interpolates the package variable `$cls::sigma`,
    # which is empty and which perl warns about once per run.
    die "${cls}\:\:sigma$c has $got values, more than the 121 it is declared with\n"
      if $got > 121;
    push @{ $short{$tag} }, "$c:$got" if $got != 121;
    push @s, (0) x (121 - $got);
    push @flat, @s;
    ($masses[scalar(@masses)] = $c) =~ s/^\D+//;
    ++$n_res_cols;
  }
  $res_sigma{$tag} = \@flat;
  $res_masses{$tag} = \@masses;
}
# Every one of the six energy grids is the same 121 numbers. Asserted rather than assumed,
# because the port stores ONE of them: a release that moved one table's grid and not the others
# would then be read off the wrong energies with no other symptom.
for my $tag (keys %res_energy) {
  next if $tag eq 'nd';
  for my $i (0 .. 120) {
    die "$tag energy grid differs from G4XNDeltaTable's at $i: "
      . "$res_energy{$tag}[$i] vs $res_energy{'nd'}[$i]\n"
      if $res_energy{$tag}[$i] != $res_energy{'nd'}[$i];
  }
}
++$checks;
printf "  ok %-46s %d columns, all six energy grids identical\n", 'resonance tables',
       $n_res_cols;

# The columns that are declared [121] and initialised with fewer. Asserted as an exact SET, the
# form docs/RISK.md V41 recommends - a release that fills one of them, or truncates another,
# fails here rather than changing a cross section at 39 GeV that nobody would look at.
my $short_set = join(' ', map { "$_=[" . join(',', @{ $short{$_} }) . "]" }
                          sort keys %short);
my $short_want = 'dd=[] ddstar=[] dnstar=[] nd=[] ndstar=[] nnstar=[NN1535:113,NN2190:113]';
die "the set of short resonance columns changed:\n  got  $short_set\n  want $short_want\n"
  if $short_set ne $short_want;
++$checks;
printf "  ok %-46s %s\n", 'short resonance columns', 'NNstar 1535 and 2190, 113 of 121';

# G4XNDeltastarTable.hh carries the comment "40 is missing... @@@@@@@" against sigmaND1930, and
# G4XDeltaNstarTable's and G4XNNstarTable's mass lists skip 1940 likewise. Asserted so that the
# gap is a checked fact rather than a transcription that quietly dropped a column.
for my $tag (qw(ndstar ddstar)) {
  die "$tag has a 1940 column, which the port's list does not\n"
    if grep { $_ eq '1940' } @{ $res_masses{$tag} };
}
++$checks;
printf "  ok %-46s no 1940 column in either Delta* table\n", 'the missing delta(1940)';

# FIVE OF THE SIX TABLES HALVE THEIR CROSS SECTION AND THE SIXTH DOES NOT. Each
# `CrossSectionTable()` writes one line of the form `G4double value = <sigma> * 0.5 * millibarn`,
# and G4XNNstarTable's is `G4double value = *(sigmaPointer + i) * millibarn` - no 0.5. That
# doubles every NN -> N N* cross section relative to its five siblings. docs/RISK.md V94.
#
# Asserted per table, by reading the line, so a release that adds or removes the factor anywhere
# fails here. This is the one number in the six classes that is not data and not a formula - it
# is a convention, and it is applied inconsistently.
my %half_want = (
  'G4XNDeltaTable'         => 1, 'G4XDeltaDeltaTable'     => 1,
  'G4XNDeltastarTable'     => 1, 'G4XDeltaDeltastarTable' => 1,
  'G4XNNstarTable'         => 0, 'G4XDeltaNstarTable'     => 1,
);
for my $cls (sort keys %half_want) {
  open my $cfh, '<', "$imr/src/$cls.cc" or die "cannot open $cls.cc: $!";
  local $/;
  my $body = <$cfh>;
  close $cfh;
  die "$cls: no `G4double value = ... millibarn` line\n"
    if $body !~ /G4double value\s*=\s*([^;]*millibarn)\s*;/;
  my $expr = $1;
  my $has_half = ($expr =~ /0\.5/) ? 1 : 0;
  die "$cls: the 0.5 factor is " . ($has_half ? 'present' : 'absent')
    . ", the port assumes " . ($half_want{$cls} ? 'present' : 'absent') . "\n    $expr\n"
    if $has_half != $half_want{$cls};
  ++$checks;
}
printf "  ok %-46s %s\n", 'the 0.5 in CrossSectionTable',
       'present in five tables, absent in G4XNNstarTable';

# =============================================================================================
# 6. G4DetailedBalancePhaseSpaceIntegral's two tables.
# =============================================================================================
my $dbi = "$imr/src/G4DetailedBalancePhaseSpaceIntegral.cc";
my @dbi_cols = qw(delta delta1600 delta1620 delta1700 delta1900 delta1905 delta1910 delta1920
                  delta1930 delta1950 N1440 N1520 N1535 N1650 N1675 N1680 N1700 N1710 N1720
                  N1900 N1990 N2090 N2190 N2220 N2250);
my @dbi_e = grab($dbi, qr/G4DetailedBalancePhaseSpaceIntegral::sqrts\[120\]/, 120);
my @dbi_flat;
for my $c (@dbi_cols) {
  push @dbi_flat, grab($dbi, qr/G4DetailedBalancePhaseSpaceIntegral::\Q$c\E\[120\]/, 120);
}
# The energy grid is read with `sqrts[ie]*GeV > sqs`, and the loop stops at ie = 118 - so the
# LAST grid point is never a left edge and the function extrapolates past it using the last
# interval. Asserted increasing, which is what makes the linear search correct.
for my $i (1 .. 119) {
  die "G4DetailedBalancePhaseSpaceIntegral::sqrts not increasing at $i\n"
    if $dbi_e[$i] <= $dbi_e[$i - 1];
}
++$checks;
printf "  ok %-46s %d columns x 120, grid increasing\n", 'detailed-balance phase-space integral',
       scalar(@dbi_cols);

# =============================================================================================
# 7. The resonance masses and widths, which `G4VScatteringCollision::SampleResonanceMass` needs.
#
#    They live in three places: `G4ExcitedDeltaConstructor::mass/width` (nine Delta* multiplets),
#    `G4ExcitedNucleonConstructor::mass/width` (fifteen N*), and four inline literals in
#    `G4ShortLivedConstructor::ConstructResonances` for the ground-state Delta(1232).
#
#    TWO THINGS IN HERE ARE NOT WHAT THE NAMES SAY, and both are asserted rather than tidied.
#
#    * **delta- has a different width from its three partners.** 117 MeV against 120 - a
#      charge-state-dependent width inside one isospin multiplet, which `SampleResonanceMass`
#      reads, so the delta- mass spectrum is 2.5% narrower than the delta0's.
#    * **the names and the masses disagree, and for two multiplets they are SWAPPED.**
#      delta(1930) is built at 1950 MeV and delta(1950) at 1930; delta(1905) is at 1880 and
#      delta(1910) at 1890; N(1990) is at 1950, N(2220) at 2250 and N(2250) at 2275. The
#      cross-section table columns are keyed by NAME (`sigmaND1930`), so the column called 1930
#      is applied to a particle of mass 1950.
# =============================================================================================
my $sl = "$g4/source/particles/shortlived/src";
my @dstar_mass  = grab("$sl/G4ExcitedDeltaConstructor.cc",
                       qr/const G4double G4ExcitedDeltaConstructor::mass\[\]/, 9);
my @dstar_width = grab("$sl/G4ExcitedDeltaConstructor.cc",
                       qr/const G4double G4ExcitedDeltaConstructor::width\[\]/, 9);
my @nstar_mass  = grab("$sl/G4ExcitedNucleonConstructor.cc",
                       qr/const G4double G4ExcitedNucleonConstructor::mass\[\]/, 15);
my @nstar_width = grab("$sl/G4ExcitedNucleonConstructor.cc",
                       qr/const G4double G4ExcitedNucleonConstructor::width\[\]/, 15);
# The source writes them in GeV and MeV; the port stores MeV.
@dstar_mass = map { $_ * 1000 } @dstar_mass;
@nstar_mass = map { $_ * 1000 } @nstar_mass;

# The ground-state Delta, four inline literals. Asserted as an exact ordered set, because the
# odd one out is the whole point.
{
  open my $cfh, '<', "$sl/G4ShortLivedConstructor.cc" or die "cannot open G4ShortLivedConstructor.cc: $!";
  local $/;
  my $body = <$cfh>;
  close $cfh;
  my @got;
  while ($body =~ /"(delta(?:\+\+|\+|0|-))",\s*1\.232\*GeV,\s*([\d.]+)\*MeV/g) {
    push @got, "$1=$2";
  }
  my $want = 'delta++=120.0 delta+=120.0 delta0=120.0 delta-=117.0';
  die "the ground-state Delta widths changed:\n  got  @got\n  want $want\n"
    if join(' ', @got) ne $want;
  ++$checks;
  printf "  ok %-46s %s\n", 'ground-state Delta widths', 'three at 120 MeV, delta- at 117';
}
# The name/mass mismatches, asserted so a release that regularises them is visible.
{
  my @dnames = (1600, 1620, 1700, 1900, 1905, 1910, 1920, 1930, 1950);
  my @nnames = (1440, 1520, 1535, 1650, 1675, 1680, 1700, 1710, 1720, 1900, 1990, 2090, 2190,
                2220, 2250);
  my @mismatch;
  for my $i (0 .. 8) {
    push @mismatch, "delta($dnames[$i])=$dstar_mass[$i]" if abs($dstar_mass[$i] - $dnames[$i]) > 0.5;
  }
  for my $i (0 .. 14) {
    push @mismatch, "N($nnames[$i])=$nstar_mass[$i]" if abs($nstar_mass[$i] - $nnames[$i]) > 0.5;
  }
  my $want = 'delta(1620)=1630 delta(1905)=1880 delta(1910)=1890 delta(1930)=1950 '
           . 'delta(1950)=1930 N(1440)=1430 N(1520)=1515 N(1650)=1655 N(1680)=1685 '
           . 'N(1990)=1950 N(2090)=2080 N(2220)=2250 N(2250)=2275';
  die "the resonance name/mass mismatches changed:\n  got  @mismatch\n  want $want\n"
    if join(' ', @mismatch) ne $want;
  ++$checks;
  printf "  ok %-46s %d of 24 names differ from their mass\n", 'resonance name vs mass',
         scalar(@mismatch);
}

# =============================================================================================
# 8. The mass-dependent resonance widths G4XAnnihilationChannel divides one by the other.
#
#    `G4BaryonWidth` carries the TOTAL width of each resonance against sqrt(s), and
#    `G4BaryonPartialWidth` the partial width into N pi. `Branch()` is their ratio. Both are 120
#    points on their own energy grid, keyed by a G4String.
#
#    TWO OF THE KEYS ARE WRONG, and the port reproduces both:
#
#    * `G4BaryonPartialWidth`'s constructor writes `wMap["D1700_Npi"]` TWICE - once at line 939
#      with `pwN1700_Npi` (the N(1700) data under the Delta's label) and once at line 1007 with
#      `pwD1700_Npi`. The second overwrites the first, so there is no `N1700_Npi` key at all and
#      the `pwN1700_Npi` array is compiled in and unreachable.
#    * `G4BaryonWidth`'s map stops at `N(2220)`: there is no `N(2250)` entry, though the particle
#      exists and is produced.
#
#    In both cases `MassDependentWidth` returns 0 and `G4XAnnihilationChannel` silently falls back
#    to the constant `resonance->GetPDGWidth()`. docs/RISK.md V111.
# =============================================================================================
my @bw_names = qw(N1440 N1520 N1535 N1650 N1675 N1680 N1700 N1710 N1720 N1900 N1990 N2090
                  N2190 N2220 N2250 Delta D1600 D1620 D1700 D1900 D1905 D1910 D1920 D1930 D1950);
my @bw_arrays = qw(wN1440 wN1520 wN1535 wN1650 wN1675 wN1680 wN1700 wN1710 wN1720 wN1900 wN1990
                   wN2090 wN2190 wN2220 wN2250 wDelta wD1600 wD1620 wD1700 wD1900 wD1905 wD1910
                   wD1920 wD1930 wD1950);
my @bw_grid = grab("$imr/src/G4BaryonWidth.cc",
                   qr/const G4double G4BaryonWidth::baryonEnergyTable\[120\]/, 120);
my @bw_flat;
for my $a (@bw_arrays) {
  push @bw_flat, grab("$imr/src/G4BaryonWidth.cc",
                      qr/const G4double G4BaryonWidth::\Q$a\E\[120\]/, 120);
}
# The two classes name the ground-state Delta differently - `wDelta` for the total width and
# `pwD1232_Npi` for the partial - so the array lists are written out rather than derived from one
# name list. `pwN1700_Npi` IS extracted even though no map key reaches it: it is compiled into the
# program and the port carries it, so that a release which fixes the key finds the data already
# there and checked.
my @pw_arrays = qw(pwN1440_Npi pwN1520_Npi pwN1535_Npi pwN1650_Npi pwN1675_Npi pwN1680_Npi
                   pwN1700_Npi pwN1710_Npi pwN1720_Npi pwN1900_Npi pwN1990_Npi pwN2090_Npi
                   pwN2190_Npi pwN2220_Npi pwN2250_Npi pwD1232_Npi pwD1600_Npi pwD1620_Npi
                   pwD1700_Npi pwD1900_Npi pwD1905_Npi pwD1910_Npi pwD1920_Npi pwD1930_Npi
                   pwD1950_Npi);
my @pw_grid = grab("$imr/src/G4BaryonPartialWidth.cc",
                   qr/const G4double G4BaryonPartialWidth::energies\[120\]/, 120);
my @pw_flat;
for my $a (@pw_arrays) {
  push @pw_flat, grab("$imr/src/G4BaryonPartialWidth.cc",
                      qr/const G4double G4BaryonPartialWidth::\Q$a\E\[120\]/, 120);
}
# The two broken keys, asserted so a release that fixes either is visible here first.
{
  open my $cfh, '<', "$imr/src/G4BaryonPartialWidth.cc" or die "cannot open: $!";
  local $/;
  my $body = <$cfh>;
  close $cfh;
  my @d1700;
  while ($body =~ /wMap\["D1700_Npi"\]\s*=\s*\(G4double\*\)\s*(\w+);/g) { push @d1700, $1; }
  die "D1700_Npi is assigned " . scalar(@d1700) . " times (@d1700), expected twice "
    . "(pwN1700_Npi then pwD1700_Npi)\n"
    if join(',', @d1700) ne 'pwN1700_Npi,pwD1700_Npi';
  die "N1700_Npi has a key after all\n" if $body =~ /wMap\["N1700_Npi"\]/;
  ++$checks;
  printf "  ok %-46s %s\n", 'G4BaryonPartialWidth N1700_Npi',
         'no key; D1700_Npi assigned twice';
}
{
  open my $cfh, '<', "$imr/src/G4BaryonWidth.cc" or die "cannot open: $!";
  local $/;
  my $body = <$cfh>;
  close $cfh;
  die "N(2250) has a key in G4BaryonWidth after all\n" if $body =~ /wMap\["N\(2250\)"\]/;
  die "N(2220) is missing from G4BaryonWidth\n" if $body !~ /wMap\["N\(2220\)"\]/;
  ++$checks;
  printf "  ok %-46s %s\n", 'G4BaryonWidth N(2250)', 'no key; the map stops at N(2220)';
}

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
constexpr int kResonanceTableSize = 121;  ///< every G4X*Table column and its shared energy grid
constexpr int kDbiSize = 120;             ///< G4DetailedBalancePhaseSpaceIntegral
constexpr int kDbiColumns = 25;
constexpr int kBaryonWidthSize = 120;    ///< G4BaryonWidth::wSize and G4BaryonPartialWidth::wSize
constexpr int kBaryonWidthColumns = 25;  ///< the fifteen N* and the ten Deltas, in that order

/// How many sigma columns each of the six resonance tables has. `res_masses_*()` lists the
/// resonance masses in the order `res_sigma_*()` lays the columns out, 121 values each.
constexpr int kResColsNd = 1;
constexpr int kResColsNdstar = 9;
constexpr int kResColsNnstar = 15;
constexpr int kResColsDd = 1;
constexpr int kResColsDdstar = 9;
constexpr int kResColsDnstar = 15;

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
for my $t (@res_tables) {
  my ($cls, $egrid, $cols, $tag) = @$t;
  emit($fh, 'double', "res_sigma_$tag", 8, @{ $res_sigma{$tag} });
}
emit($fh, 'double', 'res_energy', 8, @{ $res_energy{'nd'} });
emit($fh, 'double', 'dbi_sqrts', 8, @dbi_e);
emit($fh, 'double', 'dbi_integral', 8, @dbi_flat);
emit($fh, 'double', 'deltastar_mass', 5, @dstar_mass);
emit($fh, 'double', 'deltastar_width', 5, @dstar_width);
emit($fh, 'double', 'nstar_mass', 5, @nstar_mass);
emit($fh, 'double', 'nstar_width', 5, @nstar_width);
emit($fh, 'double', 'baryon_width_grid', 8, @bw_grid);
emit($fh, 'double', 'baryon_width', 8, @bw_flat);
emit($fh, 'double', 'baryon_partial_width_grid', 8, @pw_grid);
emit($fh, 'double', 'baryon_partial_width', 8, @pw_flat);
for my $t (@res_tables) {
  my ($cls, $egrid, $cols, $tag) = @$t;
  print $fh "__host__ __device__ inline const int* res_masses_$tag() {
";
  print $fh "  static const int v[" . scalar(@{ $res_masses{$tag} }) . "] = {"
            . join(', ', @{ $res_masses{$tag} }) . "};
  return v;
}

";
}
print $fh "}  // namespace g4gpu::bic::imr\n#endif\n";
close $fh;
printf "%s written, %d assertions passed\n", $out, $checks;
