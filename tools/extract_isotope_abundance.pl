#!/usr/bin/perl
# Extracts G4NistElementBuilder's isotope table out of Geant4 11.1.1 and writes it as
# src/data/isotope_abundance.hh.
#
#   perl tools/extract_isotope_abundance.pl
#
# WHY THIS EXISTS AND WHY tools/extract_natural_isotopes.pl IS NOT ENOUGH
#
# That script writes the SET of naturally occurring isotopes, because its two callers in the
# de-excitation module only ask `GetIsotopeAbundance(Z, A) > 0.0` and use the sign. A target
# draw needs the WEIGHTS: G4CrossSectionDataStore::SampleZandA picks an element by macroscopic
# partial cross section and then an isotope by `abundance_j * isoXS_j` (or by abundance alone
# where the data set has no isotope file), and G4Element::GetRelativeAbundanceVector is where
# those abundances come from. No table in the port carried them, which is why P8 could wire
# neither hadElastic nor the neutron general process.
#
# THE THREE NORMALISATIONS, IN ORDER, BECAUSE THE ANSWER IS THE COMPOSITION OF THEM
#
# 1. G4NistElementBuilder::AddElement, over ALL nc tabulated isotopes of the element:
#        www = 0.01*W[i];  relAbundance[index] = www;  ww += www;
#    then `if (ww != 1.0) { relAbundance[idx+j] /= ww; }` - an EXACT float comparison against
#    1.0, so the division happens unless the percentages sum to exactly 1.0 after scaling.
#    This is what GetIsotopeAbundance(Z, A) returns.
# 2. G4NistElementBuilder::BuildElement drops every isotope with `relAbundance <= 0.0` and
#    calls G4Element::AddIsotope for the rest, in ascending mass number.
# 3. G4Element::AddIsotope, once the declared count is filled, re-normalises AGAIN:
#        wtSum += fRelativeAbundanceVector[i];
#        if (wtSum != 1.0) { fRelativeAbundanceVector[i] /= wtSum; }
#    The second division is not redundant: step 1 divided by the sum over all nc entries and
#    step 3 divides by the sum over the surviving ones, and the two sums differ in the last
#    places even when every dropped entry was exactly zero, because `sum of (a_i/ww)` is not
#    `(sum of a_i)/ww` in floating point.
#
# So the vector a G4Element hands SampleZandA is `(0.01*W_i/ww)/wtSum`, evaluated in double in
# that order. Reproduced here in the same order and printed with %.17g, and compared against
# G4Element::GetRelativeAbundanceVector itself by ref/dump/dump_isotopes.cc rather than trusted.
#
# The mass numbers are consecutive and that is load-bearing: GetIsotopeAbundance indexes
# `relAbundance[A - nFirstIsotope[Z] + idxIsotopes[Z]]`, which is only an isotope's slot if the
# XxN[] array runs N[0], N[0]+1, ... . Asserted below for all 107 elements rather than assumed;
# the parameter really is a mass number and not a neutron number (see extract_natural_isotopes.pl).
use strict;
use warnings;

my $g4  = $ENV{G4SRC} || 'D:/Documents/Geant4/Windows/geant4-v11.1.1';
my $src = "$g4/source/materials/src/G4NistElementBuilder.cc";
my $hh  = "$g4/source/materials/include/G4NistElementBuilder.hh";

my ($maxz, $maxab) = (0, 0);
{
  open my $fh, '<', $hh or die "cannot open $hh: $!";
  while (my $l = <$fh>) {
    $maxz  = $1 if $l =~ /const\s+G4int\s+maxNumElements\s*=\s*(\d+)/;
    $maxab = $1 if $l =~ /const\s+G4int\s+maxAbundance\s*=\s*(\d+)/;
  }
  close $fh;
}
die "no maxNumElements in $hh\n" if !$maxz;
die "no maxAbundance in $hh\n"   if !$maxab;

open my $fh, '<', $src or die "cannot open $src: $!";
my $text = do { local $/; <$fh> };
close $fh;
$text =~ s{//[^\n]*}{}g;

my %arr;
while ($text =~ /\b(?:G4int|G4double)\s+(\w+)\s*\[\s*\d*\s*\]\s*=\s*\{(.*?)\}\s*;/gs) {
  my ($name, $body) = ($1, $2);
  my @v = ($body =~ /(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)/g);
  $arr{$name} = \@v;
}

# Per Z, in the order AddElement is called - which is also the order `index` advances in.
my (@sym, @nc, @n0, @idx, @rel);   # rel[Z] = [ relAbundance after step 1 ], in slot order
my ($nelem, $niso, $nnat) = (0, 0, 0);
my $index = 0;
while ($text =~ /AddElement\(\s*"(\w+)"\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\*(\w+)\s*,\s*\*(\w+)\s*,\s*\*(\w+)\s*,\s*\*(\w+)\s*\)/g) {
  my ($s, $Z, $n, $nName, $aName, $sName, $wName) = ($1, $2, $3, $4, $5, $6, $7);
  ++$nelem;
  die "AddElement $s: no array $nName\n" if !$arr{$nName};
  die "AddElement $s: no array $wName\n" if !$arr{$wName};
  my @N = @{$arr{$nName}};
  my @W = @{$arr{$wName}};
  die "AddElement $s: $n isotopes but N has " . scalar(@N) . "\n" if @N < $n;
  die "AddElement $s: $n isotopes but W has " . scalar(@W) . "\n" if @W < $n;
  # The consecutive-mass-number claim GetIsotopeAbundance's indexing depends on.
  for my $i (0 .. $n - 1) {
    die "AddElement $s: N[$i] = $N[$i], expected " . ($N[0] + $i) . " - the XxN[] array is "
      . "not consecutive, so relAbundance[A - nFirstIsotope[Z]] is not an isotope's slot\n"
      if $N[$i] + 0 != $N[0] + $i;
  }

  # ---- step 1, in AddElement's own loop order.
  my @r;
  my $ww = 0.0;
  for my $i (0 .. $n - 1) {
    my $www = 0.01 * ($W[$i] + 0);
    $r[$i] = $www;
    $ww += $www;
  }
  if ($ww != 1.0) {
    for my $j (0 .. $n - 1) { $r[$j] /= $ww; }
  }

  $sym[$Z] = $s;
  $nc[$Z]  = $n;
  $n0[$Z]  = $N[0] + 0;
  $idx[$Z] = $index;
  $rel[$Z] = \@r;
  $index += $n;
  $niso  += $n;
  $nnat  += scalar(grep { $_ > 0.0 } @r);
}

die "found $nelem AddElement calls, expected 107\n"   if $nelem != 107;
die "found $niso tabulated isotopes, expected 2908\n"  if $niso != 2908;
die "found $nnat natural isotopes, expected 311\n"     if $nnat != 311;
die "index ran to $niso, above maxAbundance $maxab\n"  if $niso >= $maxab;

# ---- steps 2 and 3: the vector a G4Element built by BuildElement(Z) hands out.
my (@eoff, @ea, @ew);
my $off = 0;
for my $Z (0 .. $maxz - 1) {
  $eoff[$Z] = $off;
  next if !defined $rel[$Z];
  my @keep = grep { $rel[$Z][$_] > 0.0 } 0 .. $nc[$Z] - 1;   # BuildElement's `> 0.0`
  my @w = map { $rel[$Z][$_] } @keep;
  my $wtsum = 0.0;
  $wtsum += $_ for @w;                                        # AddIsotope's wtSum, in order
  if ($wtsum != 1.0) {
    for my $j (0 .. $#w) { $w[$j] /= $wtsum; }
  }
  my $check = 0.0;
  $check += $_ for @w;
  die "Z=$Z ($sym[$Z]): abundance vector sums to $check\n" if abs($check - 1.0) > 1e-15;
  for my $j (0 .. $#keep) {
    push @ea, $n0[$Z] + $keep[$j];
    push @ew, $w[$j];
    ++$off;
  }
}
$eoff[$maxz] = $off;
die "element vectors hold $off isotopes, expected $nnat\n" if $off != $nnat;

# ---------------------------------------------------------------------------------------------
my $out = 'src/data/isotope_abundance.hh';
open my $o, '>', $out or die "cannot write $out: $!";

print $o <<"HDR";
// G4NistElementBuilder's isotope table - the abundances, not only the set.
//
// GENERATED by tools/extract_isotope_abundance.pl from Geant4 11.1.1 - do not edit by hand.
// $niso tabulated isotopes over $nelem elements; $nnat of them have a non-zero abundance and so
// appear in a G4Element built by FindOrBuildElement(Z).
//
// WHAT READS THIS, AND WHY aeff[Z] IS NOT AN ANSWER
//
// G4CrossSectionDataStore::SampleZandA draws the target element from the material's macroscopic
// partial cross sections and then the target ISOTOPE from `abundance_j * isoXS(Z, A_j)` - or from
// the abundances alone, where the data set has no per-isotope file (G4NeutronElasticXS always,
// and the others outside the amin/amax window of data/isotope_list.hh). An elastic recoil is
// (Z, A)-resolved: G4ChipsElasticModel reads both and its tables are per isotope. So there is no
// version of a target draw that stops at the element, and substituting aeff[Z] - the mean mass
// number the ELEMENT cross section is defined at - would be a plausible number from different
// physics rather than a rounding of this one.
//
// data/natural_isotopes.hh is the same source data reduced to a PREDICATE, because its two
// callers in the de-excitation module only ask `GetIsotopeAbundance(Z, A) > 0.0`. It stays; this
// file carries the weights, and `nist_isotope_abundance` below is the value whose sign that
// predicate is.
//
// THE THREE NORMALISATIONS
//
// `nist_rel_abundance()` is relAbundance[] after G4NistElementBuilder::AddElement: `0.01*W[i]`,
// divided by the sum over all the element's tabulated isotopes unless that sum is exactly 1.0.
// That is what GetIsotopeAbundance returns.
//
// `nist_element_iso_w()` is what G4Element::GetRelativeAbundanceVector returns instead, and it
// is a SECOND normalisation of the first: BuildElement drops every isotope whose relAbundance is
// not > 0, and AddIsotope then divides the survivors by their own sum unless that is exactly
// 1.0. The two sums differ in the last places even though every dropped entry was exactly zero,
// because a sum of quotients is not the quotient of a sum. Both arrays are here because both are
// read: the first by anything asking Geant4's NIST manager, the second by a target draw.
//
// The mass numbers of an element's tabulated isotopes are consecutive from nFirstIsotope[Z],
// which is what makes `relAbundance[A - nFirstIsotope[Z] + idxIsotopes[Z]]` an isotope's slot.
// The extractor asserts it for all $nelem elements rather than assuming it.
//
// WHERE THE TABLE STOPS BEING USABLE: Z = 104, NOT $nelem.
//
// Every element above uranium carries a FABRICATED abundance - Db's W[] is
// `{0,0,0,0,0,0,0,100,0,0,0}` over A = 255..265, so Db-262 comes out at 1.0 - so
// G4NistElementBuilder::BuildElement creates a G4Element for them instead of skipping them for
// having no natural isotopes. G4Element::AddIsotope then calls
// `G4AtomicShells::GetNumberOfShells(Z)`, whose tables are declared `[105]`, and Z = 105 raises
// a FatalException that aborts the process. Measured, not read: ref/dump/dump_isotopes.cc
// looping to 107 killed g4dump.exe with "mat060 ... Atomic number out of range Z= 105".
//
// So `kNistBuildableMaxZ` below is 104 and the oracle compares Z = 1..104. The abundances for
// 105..107 are still transcribed, because they are what G4NistElementBuilder holds and a
// predicate over them (data/natural_isotopes.hh) already counts them; what cannot exist is a
// G4Element, and therefore a material, at those Z.
//
// MATERIALS BUILT FROM EXPLICIT ISOTOPES ARE A SECOND PATH AND ARE REFUSED BY NAME.
// A G4Element assembled with AddIsotope by a user carries its own abundances and
// SetNaturalAbundanceFlag(false), which switches G4CrossSectionDataStore::GetCrossSection from
// the element cross section to an abundance-weighted isotope sum. Everything this port builds
// goes through the NIST element database, so the composition is a function of Z alone and that
// is what the accessors below take. `nist_isotopes_of_z` therefore reports
// `natural_abundance = true` always, and a caller that has a custom composition must supply its
// own ElementIsotopes - see `IsotopeRefusal` and data/materials.cuh, which has no isotope field
// to hold one.
#ifndef G4GPU_DATA_ISOTOPE_ABUNDANCE_HH
#define G4GPU_DATA_ISOTOPE_ABUNDANCE_HH

namespace g4gpu::data {

/// G4NistElementBuilder's maxNumElements: Z runs 1..$nelem and every accessor here answers zero
/// outside `0 < Z < kNistMaxElements`, as GetIsotopeAbundance does.
constexpr int kNistMaxElements = $maxz;
/// The highest Z a G4Element can be BUILT for: G4AtomicShells' tables are `[105]` and
/// G4Element::AddIsotope indexes them, so FindOrBuildElement(105) aborts the process. See the
/// note in the file header - this is three below the table's own ceiling.
constexpr int kNistBuildableMaxZ = 104;
/// G4NistElementBuilder's maxAbundance - the size of the relAbundance[] array, of which
/// $niso slots are filled.
constexpr int kNistMaxAbundance = $maxab;
constexpr int kNistTabulatedIsotopes = $niso;
/// Isotopes with a non-zero abundance, summed over the elements. The length of the flat
/// per-element vectors below.
constexpr int kNistElementIsotopes = $nnat;

HDR

sub emit_int_array {
  my ($o, $name, $doc, $vals, $per) = @_;
  printf $o "%s\n", $doc;
  printf $o "__host__ __device__ inline const int* %s() {\n", $name;
  printf $o "  static const int v[%d] = {\n", scalar(@$vals);
  for (my $i = 0; $i < @$vals; $i += $per) {
    my @row = @{$vals}[$i .. ($i + $per - 1 > $#$vals ? $#$vals : $i + $per - 1)];
    printf $o "    %s,\n", join(', ', @row);
  }
  printf $o "  };\n  return v;\n}\n\n";
}

my @a_nc  = map { defined $nc[$_]  ? $nc[$_]  : 0 } 0 .. $maxz - 1;
my @a_n0  = map { defined $n0[$_]  ? $n0[$_]  : 0 } 0 .. $maxz - 1;
my @a_idx = map { defined $idx[$_] ? $idx[$_] : 0 } 0 .. $maxz - 1;

emit_int_array($o, 'nist_n_isotopes',
  "/// G4NistElementBuilder::nIsotopes - tabulated isotopes per Z, zero where no element exists.",
  \@a_nc, 12);
emit_int_array($o, 'nist_first_isotope_n',
  "/// G4NistElementBuilder::nFirstIsotope - the lowest tabulated MASS NUMBER per Z.",
  \@a_n0, 12);
emit_int_array($o, 'nist_isotope_index',
  "/// G4NistElementBuilder::idxIsotopes - base of this Z's slice of relAbundance[].",
  \@a_idx, 12);

# relAbundance, in slot order, with a comment per element.
printf $o "/// G4NistElementBuilder::relAbundance, slots 0..%d, after AddElement's\n", $niso - 1;
printf $o "/// normalisation. Indexed as `A - nist_first_isotope_n()[Z] + nist_isotope_index()[Z]`.\n";
printf $o "__host__ __device__ inline const double* nist_rel_abundance() {\n";
printf $o "  static const double v[%d] = {\n", $niso;
for my $Z (1 .. $maxz - 1) {
  next if !defined $rel[$Z];
  printf $o "    // %s (Z=%d), A = %d..%d\n", $sym[$Z], $Z, $n0[$Z], $n0[$Z] + $nc[$Z] - 1;
  for (my $i = 0; $i < $nc[$Z]; $i += 4) {
    my @row;
    for my $j ($i .. ($i + 3 > $nc[$Z] - 1 ? $nc[$Z] - 1 : $i + 3)) {
      push @row, sprintf('%.17g', $rel[$Z][$j]);
    }
    printf $o "    %s,\n", join(', ', @row);
  }
}
printf $o "  };\n  return v;\n}\n\n";

my @a_eoff = map { $eoff[$_] } 0 .. $maxz;
emit_int_array($o, 'nist_element_iso_offset',
  "/// Where each Z's slice of the two arrays below starts; [Z+1] - [Z] is its length. One entry\n"
  . "/// longer than the others so the last element's length needs no special case.",
  \@a_eoff, 12);

printf $o "/// The mass numbers a G4Element built from the NIST database carries, per Z, in the\n";
printf $o "/// ascending order BuildElement pushes them.\n";
printf $o "__host__ __device__ inline const int* nist_element_iso_a() {\n";
printf $o "  static const int v[%d] = {\n", $nnat;
for my $Z (1 .. $maxz - 1) {
  next if $eoff[$Z + 1] == $eoff[$Z];
  printf $o "    // %s (Z=%d)\n", $sym[$Z], $Z;
  my @row = @ea[$eoff[$Z] .. $eoff[$Z + 1] - 1];
  while (@row) {
    my @take = splice(@row, 0, 10);
    printf $o "    %s,\n", join(', ', @take);
  }
}
printf $o "  };\n  return v;\n}\n\n";

for my $t (['double', 'nist_element_iso_w', '%.17g', ''],
           ['float',  'nist_element_iso_w_f', '%.17e', 'f']) {
  my ($ty, $name, $fmt, $sfx) = @$t;
  printf $o "/// G4Element::GetRelativeAbundanceVector, per Z, matching nist_element_iso_a().\n";
  if ($sfx) {
    # %.17e rather than %.17g, because %.17g prints an exact 1.0 as "1" and "1f" is a
    # user-defined-literal syntax error rather than a float. Found by compiling.
    printf $o "/// The float mirror exists because xs::ElementIsotopes<real_t> takes a real_t\n";
    printf $o "/// array and this port is built for both precisions. Printed in exponential form\n";
    printf $o "/// with 17 digits, which round-trips a double and therefore rounds to the same\n";
    printf $o "/// float the double would convert to.\n";
  }
  printf $o "__host__ __device__ inline const %s* %s() {\n", $ty, $name;
  printf $o "  static const %s v[%d] = {\n", $ty, $nnat;
  for my $Z (1 .. $maxz - 1) {
    next if $eoff[$Z + 1] == $eoff[$Z];
    printf $o "    // %s (Z=%d)\n", $sym[$Z], $Z;
    my @row = map { sprintf("$fmt%s", $_, $sfx) } @ew[$eoff[$Z] .. $eoff[$Z + 1] - 1];
    while (@row) {
      my @take = splice(@row, 0, 4);
      printf $o "    %s,\n", join(', ', @take);
    }
  }
  printf $o "  };\n  return v;\n}\n\n";
}

print $o <<'TAIL';
/// G4NistElementBuilder::GetIsotopeAbundance(Z, N) - and `N` is a mass number, as
/// data/natural_isotopes.hh explains. Zero outside the table, which is the answer Geant4 gives
/// too rather than an error.
__host__ __device__ inline double nist_isotope_abundance(int Z, int A) {
  if (Z <= 0 || Z >= kNistMaxElements) { return 0.0; }
  const int i = A - nist_first_isotope_n()[Z];
  if (i < 0 || i >= nist_n_isotopes()[Z]) { return 0.0; }
  return nist_rel_abundance()[i + nist_isotope_index()[Z]];
}

/// How many isotopes a NIST-built G4Element of this Z has - the ones with a non-zero abundance.
__host__ __device__ inline int nist_element_n_isotopes(int Z) {
  if (Z <= 0 || Z >= kNistMaxElements) { return 0; }
  return nist_element_iso_offset()[Z + 1] - nist_element_iso_offset()[Z];
}

/// Which real_t array of abundances to read. A class template rather than a function template
/// specialisation because the arrays are function-local statics and a `__host__ __device__`
/// specialisation of a function template carrying one is not portable across nvcc versions.
template <typename real_t> struct NistIsotopeAbundance;
template <> struct NistIsotopeAbundance<double> {
  __host__ __device__ static const double* values() { return nist_element_iso_w(); }
};
template <> struct NistIsotopeAbundance<float> {
  __host__ __device__ static const float* values() { return nist_element_iso_w_f(); }
};

/// `G4EmUtility::SampleRandomIsotope(elm)` - the isotope draw every EM model makes through
/// `G4VEmModel::SelectIsotopeNumber`. Returns a MASS NUMBER, or 0 when Z has no element.
///
/// THREE THINGS IN FIVE LINES OF GEANT4, AND EACH ONE CHANGES AN ANSWER:
///
///   const G4Isotope* iso = elm->GetIsotope(0);
///   if(ni > 1) {
///     G4double x = G4UniformRand();
///     for(idx=0; idx<ni; ++idx) { x -= ab[idx]; if (x <= 0.0) { iso = ...; break; } }
///   }
///
///  1. A SINGLE-ISOTOPE ELEMENT DRAWS NO RANDOM NUMBER. Not an optimisation - it is the random
///     stream. An element like aluminium or sodium would consume a uniform that Geant4 does not.
///  2. The loop SUBTRACTS from the uniform rather than comparing against a running sum, and the
///     two are different in floating point. `G4CrossSectionDataStore::SampleZandA` does it the
///     other way (`sum += ab[j]; if (q <= sum)`), so the two isotope draws in this port cannot
///     share an implementation. Both are transcribed as written.
///  3. When the subtraction never reaches zero - which rounding can do on the last isotope -
///     `iso` keeps `GetIsotope(0)`, the LIGHTEST, not the last. The fallthrough is a real branch
///     and not an impossibility.
///
/// @param q one uniform in [0,1). Only read when the element has more than one isotope; a
///        caller must not draw it otherwise, for reason 1 above.
__host__ __device__ inline int nist_sample_isotope_n(int Z, double q) {
  const int n = nist_element_n_isotopes(Z);
  if (n <= 0) { return 0; }
  const int off = nist_element_iso_offset()[Z];
  int a = nist_element_iso_a()[off];
  if (n > 1) {
    double x = q;
    for (int j = 0; j < n; ++j) {
      x -= nist_element_iso_w()[off + j];
      if (x <= 0.0) {
        a = nist_element_iso_a()[off + j];
        break;
      }
    }
  }
  return a;
}

/// Why a caller could not be given an isotope composition. One value, because there is one
/// reason: the port's materials are NIST-built and a custom one has nowhere to live.
enum class IsotopeRefusal : int {
  kNone = 0,
  /// Z outside G4NistElementBuilder's table - there is no element there to have isotopes.
  kNoSuchElement,
};

__host__ __device__ inline const char* isotope_refusal_name(IsotopeRefusal r) {
  return (r == IsotopeRefusal::kNoSuchElement)
             ? "no NIST element at this Z (G4NistElementBuilder carries Z = 1..107)"
             : "none";
}

}  // namespace g4gpu::data

#endif  // G4GPU_DATA_ISOTOPE_ABUNDANCE_HH
TAIL
close $o;
printf "wrote %s: %d tabulated isotopes over %d elements, %d with abundance > 0\n",
       $out, $niso, $nelem, $nnat;
