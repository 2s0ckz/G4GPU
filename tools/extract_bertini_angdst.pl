#!/usr/bin/perl
# Extracts Bertini's angular and momentum distribution tables out of Geant4 11.1.1 and writes
# them as src/data/bertini_angdst.hh.
#
#   perl tools/extract_bertini_angdst.pl
#
# Nineteen objects in four families - fifteen G4TwoBodyAngularDist constructs and four
# G4MultiBodyMomentumDist does, and the family decides what the numbers MEAN, which is why
# the generated header keeps them apart rather than in one flat block:
#
#   G4NumIntTwoBodyAngDst<NKE,NANG>   eight of them. A tabulated CUMULATIVE distribution in
#                                     cos(theta) at NKE lab energies: `integralTable[i][j]` is
#                                     the CDF at `angleBins[j]` for `labKE[i]`, and GetCosTheta
#                                     interpolates linearly between two energies and then
#                                     inverts the CDF. Above the last energy it switches to an
#                                     exponential in t with slope `2*tcoeff*pcm^2` - so tcoeff,
#                                     a single constructor argument, is as load-bearing as the
#                                     209 table entries and is extracted with them.
#   G4ParamExpTwoBodyAngDst<NKE>      five of them. Five parallel arrays over the same energy
#                                     scale: a small-angle fraction, two exponential slopes and
#                                     a cos(theta) cut.
#   G4InuclParamAngDst                two (the three-body angular ones). One [2][4][4] block of
#                                     power-series coefficients, indexed by whether the OUTGOING
#                                     particle is a nucleon.
#   G4InuclParamMomDst                four. A [2][4][4] block and a [2][3] block.
#
# Everything is asserted against the array dimensions in its own declaration and against the
# template arguments in the constructor call, and the totals are printed for the commit message.
# The one thing there is no second source for is the count of objects, so it is written here: 8,
# 5, 2, 4 - and 8 + 5 + 2 is the fifteen G4TwoBodyAngularDist owns.
#
# Written to perl 5.8: the perl on cmd.exe's and PowerShell's PATH here is
# C:\MinGW\msys\1.0\bin\perl.exe, which is 5.8.8, while Git Bash's is 5.34. See the note at the
# top of tools/extract_bertini_channels.pl - a `s///r` in that script ran from one shell and
# died with a bare syntax error from the other.
use strict;
use warnings;
use 5.008;

my $g4 = $ENV{G4SRC} || 'D:/Documents/Geant4/Windows/geant4-v11.1.1';
my $casc = "$g4/source/processes/hadronic/models/cascade/cascade";
my $out = 'src/data/bertini_angdst.hh';

sub slurp { my ($f) = @_; open my $h, '<', $f or die "cannot read $f: $!\n"; local $/; my $t = <$h>; close $h; $t }
sub trim { my ($s) = @_; $s =~ s/^\s+//; $s =~ s/\s+$//; $s }
sub strip_comments { my ($t) = @_; $t =~ s{/\*.*?\*/}{ }gs; $t =~ s{//[^\n]*}{}g; $t }

sub braced_body {
  my ($tref, $start, $open, $close) = @_;
  $open ||= '{'; $close ||= '}';
  my ($depth, $i, $n) = (0, $start, length $$tref);
  while ($i < $n) {
    my $ch = substr($$tref, $i, 1);
    $depth++ if $ch eq $open;
    if ($ch eq $close) { $depth--; return substr($$tref, $start + 1, $i - $start - 1) if $depth == 0; }
    $i++;
  }
  die "unbalanced $open at $start\n";
}

sub numbers {
  my ($body) = @_;
  (my $b = $body) =~ s/[{}]/ /g;
  return grep { /\S/ } split /\s*,\s*/, trim($b);
}

# Parse one .cc: every `static const G4double NAME[..][..] = {...};` plus the constructor call.
sub parse_file {
  my ($file) = @_;
  my $txt = strip_comments(slurp("$casc/src/$file.cc"));
  my %arr;
  while ($txt =~ /(?:static\s+)?const\s+G4double\s+(\w+)\s*((?:\[\s*\d+\s*\])+)\s*=\s*\{/g) {
    my ($name, $dims) = ($1, $2);
    my @d = $dims =~ /\[\s*(\d+)\s*\]/g;
    my $body = braced_body(\$txt, pos($txt) - 1);
    my @v = numbers($body);
    my $want = 1; $want *= $_ for @d;
    die "$file: $name declared [" . join('][', @d) . "] = $want values, parsed " . scalar(@v) . "\n"
      unless @v == $want;
    $arr{$name} = { dims => \@d, v => \@v };
  }
  # The constructor: `G4Xxx::G4Xxx(G4int verbose) : Base<...>("name", args...) {;}`
  $txt =~ /\Q$file\E::\Q$file\E\s*\(\s*G4int\s+\w+\s*\)\s*:\s*(\w+)\s*(<[^>]*>)?\s*\(([^;]*?)\)\s*\{/s
    or die "$file: cannot find constructor\n";
  my ($base, $targs, $args) = ($1, $2 || '', $3);
  my @a = map { trim($_) } split /\s*,\s*/, $args;
  return (\%arr, $base, $targs, \@a);
}

my (@numint, @paramexp, @paramang, @parammom);

# ---- G4NumIntTwoBodyAngDst<NKE,NANG>: (name, kebins, angles, dists, highKEscale, verbose)
for my $f (qw(G4GamP2NPipAngDst G4GamP2PPi0AngDst G4NP2NPAngDst G4PP2PPAngDst
              G4Pi0P2Pi0PAngDst G4PimP2Pi0NAngDst G4PimP2PimPAngDst G4PipP2PipPAngDst)) {
  next unless -f "$casc/src/$f.cc";
  my ($arr, $base, $targs, $a) = parse_file($f);
  next unless $base eq 'G4NumIntTwoBodyAngDst';
  my ($nke, $nang) = $targs =~ /<\s*(\d+)\s*,\s*(\d+)\s*>/
    or die "$f: cannot read template args from '$targs'\n";
  my ($name) = $a->[0] =~ /^"(.*)"$/ or die "$f: first ctor arg is not a name\n";
  my ($kb, $ab, $tb, $scale) = @{$a}[1, 2, 3, 4];
  die "$f: no array $kb\n" unless $arr->{$kb};
  die "$f: $kb is [@{$arr->{$kb}{dims}}], template says [$nke]\n"
    unless "@{$arr->{$kb}{dims}}" eq "$nke";
  die "$f: $ab is [@{$arr->{$ab}{dims}}], template says [$nang]\n"
    unless "@{$arr->{$ab}{dims}}" eq "$nang";
  die "$f: $tb is [@{$arr->{$tb}{dims}}], template says [$nke][$nang]\n"
    unless "@{$arr->{$tb}{dims}}" eq "$nke $nang";
  push @numint, { file => $f, name => $name, nke => $nke, nang => $nang,
                  ke => $arr->{$kb}{v}, ang => $arr->{$ab}{v}, tab => $arr->{$tb}{v},
                  tcoeff => $scale };
}

# ---- G4ParamExpTwoBodyAngDst<NKE>: (name, kebins, pFrac, pA, pC, pCos, verbose)
for my $f (qw(G4GammaNuclAngDst G4HadNElastic1AngDst G4HadNElastic2AngDst G4NuclNuclAngDst
              G4PiNInelasticAngDst)) {
  my ($arr, $base, $targs, $a) = parse_file($f);
  die "$f: base is $base, expected G4ParamExpTwoBodyAngDst\n"
    unless $base eq 'G4ParamExpTwoBodyAngDst';
  my ($nke) = $targs =~ /<\s*(\d+)\s*>/ or die "$f: cannot read template arg\n";
  my ($name) = $a->[0] =~ /^"(.*)"$/ or die "$f: first ctor arg is not a name\n";
  my @names = @{$a}[1 .. 5];    # kebins, pFrac, pA, pC, pCos
  for my $n (@names) {
    die "$f: no array $n\n" unless $arr->{$n};
    die "$f: $n is [@{$arr->{$n}{dims}}], template says [$nke]\n"
      unless "@{$arr->{$n}{dims}}" eq "$nke";
  }
  push @paramexp, { file => $f, name => $name, nke => $nke,
                    ke => $arr->{$names[0]}{v}, frac => $arr->{$names[1]}{v},
                    a => $arr->{$names[2]}{v}, c => $arr->{$names[3]}{v},
                    cos => $arr->{$names[4]}{v} };
}

# ---- G4InuclParamAngDst: (name, abnC[2][4][4], verbose)
for my $f (qw(G4HadNucl3BodyAngDst G4NuclNucl3BodyAngDst)) {
  my ($arr, $base, $targs, $a) = parse_file($f);
  die "$f: base is $base, expected G4InuclParamAngDst\n" unless $base eq 'G4InuclParamAngDst';
  my ($name) = $a->[0] =~ /^"(.*)"$/ or die "$f: first ctor arg is not a name\n";
  my $n = $a->[1];
  die "$f: $n is [@{$arr->{$n}{dims}}], want [2][4][4]\n"
    unless "@{$arr->{$n}{dims}}" eq "2 4 4";
  push @paramang, { file => $f, name => $name, ab => $arr->{$n}{v} };
}

# ---- G4InuclParamMomDst: (name, pqprC[2][4][4], psC[2][3], verbose)
for my $f (qw(G4HadNucl3BodyMomDst G4HadNucl4BodyMomDst G4NuclNucl3BodyMomDst
              G4NuclNucl4BodyMomDst)) {
  my ($arr, $base, $targs, $a) = parse_file($f);
  die "$f: base is $base, expected G4InuclParamMomDst\n" unless $base eq 'G4InuclParamMomDst';
  my ($name) = $a->[0] =~ /^"(.*)"$/ or die "$f: first ctor arg is not a name\n";
  my ($pq, $ps) = @{$a}[1, 2];
  die "$f: $pq is [@{$arr->{$pq}{dims}}], want [2][4][4]\n"
    unless "@{$arr->{$pq}{dims}}" eq "2 4 4";
  die "$f: $ps is [@{$arr->{$ps}{dims}}], want [2][3]\n"
    unless "@{$arr->{$ps}{dims}}" eq "2 3";
  push @parammom, { file => $f, name => $name, pqpr => $arr->{$pq}{v}, ps => $arr->{$ps}{v} };
}

die "expected 6 G4NumIntTwoBodyAngDst objects, found " . scalar(@numint) . "\n"
  unless @numint == 8;
die "expected 5 G4ParamExpTwoBodyAngDst objects, found " . scalar(@paramexp) . "\n"
  unless @paramexp == 5;
die "expected 2 G4InuclParamAngDst objects, found " . scalar(@paramang) . "\n"
  unless @paramang == 2;
die "expected 4 G4InuclParamMomDst objects, found " . scalar(@parammom) . "\n"
  unless @parammom == 4;

# ---------------------------------------------------------------------------------------------
sub wrap {
  my ($v, $per, $indent) = @_;
  my $s = '';
  for (my $i = 0; $i < @$v; $i += $per) {
    my @row = @{$v}[$i .. ($i + $per - 1 > $#$v ? $#$v : $i + $per - 1)];
    $s .= $indent . join(', ', @row) . (($i + $per < @$v) ? ",\n" : "\n");
  }
  $s;
}

my ($nvals) = (0);
$nvals += scalar(@{$_->{ke}}) + scalar(@{$_->{ang}}) + scalar(@{$_->{tab}}) + 1 for @numint;
$nvals += 5 * $_->{nke} for @paramexp;
$nvals += 32 for @paramang;
$nvals += 38 for @parammom;

open my $fh, '>', $out or die "cannot write $out: $!\n";
print $fh <<"HDR";
// Bertini's angular and momentum distribution tables, Geant4 11.1.1.
//
// GENERATED by tools/extract_bertini_angdst.pl - do not edit by hand.
// $nvals values over 8 numerically integrated angular distributions, 5 parametrised-exponential
// ones, 2 three-body angular parametrisations and 4 momentum parametrisations.
//
// The families are kept apart because the numbers mean different things. A NumInt table is a
// CUMULATIVE distribution in cos(theta) - `table[i][j]` is the CDF at `cos_bins[j]` for
// `ke_bins[i]` - and `tcoeff` is the exponential slope coefficient used ABOVE the last energy,
// where the table stops. A ParamExp object is five parallel arrays over one energy scale. The
// two ParamAng and four ParamMom objects are power-series coefficient blocks indexed by whether
// the outgoing particle is a nucleon.
//
// Names are G4VTwoBodyAngDst::GetName()'s strings, which is what ref/oracle/bertini_angchoice.csv
// identifies a chosen distribution by - the objects themselves are only reachable as pointers.
#ifndef G4GPU_DATA_BERTINI_ANGDST_HH
#define G4GPU_DATA_BERTINI_ANGDST_HH

namespace g4gpu::data {

// **Why these structs hold offsets and not pointers, and no name.** The obvious layout is a
// struct of `const double*` filled in from the per-object accessor functions. nvcc rejects it:
// a function-scope `static const` array in `__host__ __device__` code must be CONSTANT
// initialized, and an initializer that calls a function - even one that only returns the
// address of another static - is dynamic initialization ("dynamic initialization is not
// supported for a function-scope static __device__ variable"). The same applies to a
// `const char*` name, whose value is a link-time address rather than a compile-time constant.
// So every table is one flat array per family, the struct carries integer offsets into it, and
// the names live at namespace scope as a host-side `constexpr` array - which is where
// src/data/nist_stopping_names.hh keeps its names too, for the same reason.

/// One G4NumIntTwoBodyAngDst<NKE,NANG>. Offsets index bertini_numint_data().
struct BertiniNumIntAngDst {
  int nke;
  int nang;
  int ke_off;               ///< [nke] lab kinetic energies, GeV
  int cos_off;              ///< [nang] cos(theta) bin edges
  int tab_off;              ///< [nke][nang] cumulative distribution, row-major
  double tcoeff;            ///< slope coefficient above ke_bins[nke-1]
};

/// One G4ParamExpTwoBodyAngDst<NKE>. Offsets index bertini_paramexp_data().
struct BertiniParamExpAngDst {
  int nke;
  int ke_off;               ///< [nke], GeV
  int frac_off;             ///< [nke] small-angle fraction
  int a_off;                ///< [nke] small-angle slope
  int c_off;                ///< [nke] large-angle slope
  int cos_off;              ///< [nke] cos(theta) at the small/large boundary
};

/// One G4InuclParamAngDst: coefficients of Ekin^0..3 in blocks of S^0..3, for outgoing nucleon
/// (first 16) and outgoing meson/kaon/hyperon (second 16). Offset indexes
/// bertini_paramang_data().
struct BertiniParamAngDst {
  int ab_off;               ///< [2][4][4], flattened
};

/// One G4InuclParamMomDst. Offsets index bertini_parammom_data().
struct BertiniParamMomDst {
  int pqpr_off;             ///< [2][4][4], flattened
  int ps_off;               ///< [2][3], flattened
};

HDR

# One flat array per family, with the offsets recorded in the structs above.
my (@ni_flat, @pe_flat, @pa_flat, @pm_flat);
for my $d (@numint) {
  $d->{ke_off} = scalar @ni_flat;  push @ni_flat, @{$d->{ke}};
  $d->{cos_off} = scalar @ni_flat; push @ni_flat, @{$d->{ang}};
  $d->{tab_off} = scalar @ni_flat; push @ni_flat, @{$d->{tab}};
}
for my $d (@paramexp) {
  for my $k (qw(ke frac a c cos)) {
    $d->{"${k}_off"} = scalar @pe_flat;
    push @pe_flat, @{$d->{$k}};
  }
}
for my $d (@paramang) {
  $d->{ab_off} = scalar @pa_flat; push @pa_flat, @{$d->{ab}};
}
for my $d (@parammom) {
  $d->{pqpr_off} = scalar @pm_flat; push @pm_flat, @{$d->{pqpr}};
  $d->{ps_off} = scalar @pm_flat;   push @pm_flat, @{$d->{ps}};
}

printf $fh "constexpr int kBertiniNumIntAngDsts = %d;\n", scalar @numint;
printf $fh "constexpr int kBertiniParamExpAngDsts = %d;\n", scalar @paramexp;
printf $fh "constexpr int kBertiniParamAngDsts = %d;\n", scalar @paramang;
printf $fh "constexpr int kBertiniParamMomDsts = %d;\n\n", scalar @parammom;

# The names, host side only: they identify an object in ref/oracle/bertini_angchoice.csv and
# ref/oracle/bertini_angdst.csv, which is how the dispatcher's choice is checked at all, and
# they are not readable from device code (see the note on the structs above).
printf $fh "constexpr const char* const kBertiniNumIntNames[%d] = {\n", scalar @numint;
printf $fh "    \"%s\",\n", $_->{name} for @numint;
print $fh "};\n";
printf $fh "constexpr const char* const kBertiniParamExpNames[%d] = {\n", scalar @paramexp;
printf $fh "    \"%s\",\n", $_->{name} for @paramexp;
print $fh "};\n";
printf $fh "constexpr const char* const kBertiniParamAngNames[%d] = {\n", scalar @paramang;
printf $fh "    \"%s\",\n", $_->{name} for @paramang;
print $fh "};\n";
printf $fh "constexpr const char* const kBertiniParamMomNames[%d] = {\n", scalar @parammom;
printf $fh "    \"%s\",\n", $_->{name} for @parammom;
print $fh "};\n\n";

printf $fh "/// %d values: per object, ke_bins then cos_bins then the table.\n", scalar @ni_flat;
print $fh "__host__ __device__ inline const double* bertini_numint_data() {\n";
printf $fh "  static const double v[%d] = {\n%s  };\n  return v;\n}\n\n", scalar @ni_flat,
  wrap(\@ni_flat, 10, '    ');

printf $fh "/// %d values: per object, ke_bins, frac, a, c, cos_cut.\n", scalar @pe_flat;
print $fh "__host__ __device__ inline const double* bertini_paramexp_data() {\n";
printf $fh "  static const double v[%d] = {\n%s  };\n  return v;\n}\n\n", scalar @pe_flat,
  wrap(\@pe_flat, 10, '    ');

printf $fh "/// %d values: 32 per object.\n", scalar @pa_flat;
print $fh "__host__ __device__ inline const double* bertini_paramang_data() {\n";
printf $fh "  static const double v[%d] = {\n%s  };\n  return v;\n}\n\n", scalar @pa_flat,
  wrap(\@pa_flat, 8, '    ');

printf $fh "/// %d values: 32 + 6 per object.\n", scalar @pm_flat;
print $fh "__host__ __device__ inline const double* bertini_parammom_data() {\n";
printf $fh "  static const double v[%d] = {\n%s  };\n  return v;\n}\n\n", scalar @pm_flat,
  wrap(\@pm_flat, 8, '    ');

print $fh "__host__ __device__ inline const BertiniNumIntAngDst* bertini_numint_angdst() {\n";
printf $fh "  static const BertiniNumIntAngDst v[%d] = {\n", scalar @numint;
for my $d (@numint) {
  printf $fh "    {%3d, %3d, %6d, %6d, %6d, %s},   // %s\n",
    $d->{nke}, $d->{nang}, $d->{ke_off}, $d->{cos_off}, $d->{tab_off}, $d->{tcoeff},
    $d->{name};
}
print $fh "  };\n  return v;\n}\n\n";

print $fh "__host__ __device__ inline const BertiniParamExpAngDst* bertini_paramexp_angdst() {\n";
printf $fh "  static const BertiniParamExpAngDst v[%d] = {\n", scalar @paramexp;
for my $d (@paramexp) {
  printf $fh "    {%3d, %5d, %5d, %5d, %5d, %5d},   // %s\n", $d->{nke}, $d->{ke_off},
    $d->{frac_off}, $d->{a_off}, $d->{c_off}, $d->{cos_off}, $d->{name};
}
print $fh "  };\n  return v;\n}\n\n";

print $fh "__host__ __device__ inline const BertiniParamAngDst* bertini_paramang_angdst() {\n";
printf $fh "  static const BertiniParamAngDst v[%d] = {\n", scalar @paramang;
printf $fh "    {%3d},   // %s\n", $_->{ab_off}, $_->{name} for @paramang;
print $fh "  };\n  return v;\n}\n\n";

print $fh "__host__ __device__ inline const BertiniParamMomDst* bertini_parammom_momdst() {\n";
printf $fh "  static const BertiniParamMomDst v[%d] = {\n", scalar @parammom;
printf $fh "    {%3d, %3d},   // %s\n", $_->{pqpr_off}, $_->{ps_off}, $_->{name} for @parammom;
print $fh "  };\n  return v;\n}\n\n";
print $fh "}  // namespace g4gpu::data\n\n#endif  // G4GPU_DATA_BERTINI_ANGDST_HH\n";
close $fh;

printf "wrote %s: %d values; %d NumInt, %d ParamExp, %d ParamAng, %d ParamMom\n", $out,
  $nvals, scalar @numint, scalar @paramexp, scalar @paramang, scalar @parammom;
printf "  %-24s %-28s NKE %2d NANG %2d tcoeff %s\n", $_->{file}, $_->{name}, $_->{nke},
  $_->{nang}, $_->{tcoeff} for @numint;
printf "  %-24s %-28s NKE %2d\n", $_->{file}, $_->{name}, $_->{nke} for @paramexp;
printf "  %-24s %-28s [2][4][4]\n", $_->{file}, $_->{name} for @paramang;
printf "  %-24s %-28s [2][4][4] + [2][3]\n", $_->{file}, $_->{name} for @parammom;
