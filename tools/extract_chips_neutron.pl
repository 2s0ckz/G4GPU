#!/usr/bin/perl
# Extract the per-isotope low-energy parameter table out of G4ChipsNeutronElasticXS::GetPTables.
#
# Why a script and not hand-typing: the table is 422 seven-number rows spread over 1200 lines of
# G4ChipsNeutronElasticXS.cc, declared as 422 `static const G4double pZ<z>N<n>[7]={...}` arrays,
# 422 `std::pair<G4int,const G4double*>` wrappers, and 99 per-Z arrays of those pairs whose
# initialiser lists wrap across lines. Typing 2954 numbers is not a thing a person does
# correctly, and tools/extract_mott.sh set the precedent: extract mechanically and assert the
# counts, so a silently truncated table cannot pass.
#
# What is asserted here, and why each one can fail:
#   - 422 parameter arrays and 422 pair wrappers, and every pair names an array that exists.
#     A wrapper pointing at a missing array means the regex missed a declaration.
#   - 99 per-Z lists (Z = 0..98), each of length equal to its own declared N<z>. Geant4 declares
#     the length separately from the list, and the model's loop trusts N<z>; a list longer than
#     its N would silently drop isotopes here exactly as it does there.
#   - the per-Z lengths sum to 422, so no row is in an array and out of every list, or in two.
#
# The pair NAME is NOT the neutron number: Geant4 has `Z3N1(3,pZ3N3)`, where the wrapper is
# called N1 and carries N=3. The number that matters is the pair's first element, which is what
# the model compares against tgN, so that is what is emitted. Reading the name instead would put
# Li-6's parameters at N=1.
#
# Usage: perl tools/extract_chips_neutron.pl <G4ChipsNeutronElasticXS.cc> > out.hh
use strict;
use warnings;

my $src = shift or die "usage: $0 <G4ChipsNeutronElasticXS.cc>\n";
open(my $fh, '<', $src) or die "cannot open $src: $!\n";
my $raw = do { local $/; <$fh> };
close($fh);

# Strip `//` to end of line BEFORE anything is matched. Five isotope rows in this table are
# commented out - Nb N=53 and N=54, Tb N=95, Ta N=109, U N=149 - and a regex that does not know
# what a comment is reads them as live rows. That is how the first version of this script got
# 422 rows out of a table with 417: the counts agreed with the number of declarations in the
# file and disagreed with the number of rows the model can reach, which is the only number that
# matters. The `appears in two lists` check is what caught it, because the padded slots of those
# same four elements had to be duplicates of something.
my $text = join("", map { my $l = $_; $l =~ s{//.*$}{}; $l } split(/(?<=\n)/, $raw));

# --- the 7-number parameter arrays -------------------------------------------------------
my %params;
my $n_empty = 0;
while ($text =~ /static\s+const\s+G4double\s+(p[A-Za-z0-9_]+)\s*\[\s*7\s*\]\s*=\s*\{([^}]*)\}/g) {
  my ($name, $body) = ($1, $2);
  my @v = split(/\s*,\s*/, $body);
  @v = map { my $t = $_; $t =~ s/^\s+|\s+$//g; $t } @v;
  # Geant4 has exactly one row written `[7]={}`, which is seven zeros by C++ aggregate
  # initialisation, not a missing row: pZ81N124, Tl-205. Zero-filling it here reproduces the
  # table; refusing it would drop an isotope, and hand-completing it would invent physics.
  if (@v == 0) { @v = ('0.', '0.', '0.', '0.', '0.', '0.', '0.'); ++$n_empty; }
  die "array $name has ".scalar(@v)." entries, expected 7\n" unless @v == 7;
  die "duplicate array $name\n" if exists $params{$name};
  $params{$name} = \@v;
}
my $n_arrays = scalar keys %params;
die "expected exactly 1 empty `[7]={}` row (pZ81N124), found $n_empty\n" unless $n_empty == 1;

# --- the pair wrappers: name -> (neutron number, array name) -----------------------------
my %pairs;
while ($text =~ /static\s+const\s+std::pair<G4int,\s*const\s+G4double\*>\s+([A-Za-z0-9_]+)\s*\(\s*(-?\d+)\s*,\s*(p[A-Za-z0-9_]+)\s*\)/g) {
  my ($pname, $nn, $aname) = ($1, $2, $3);
  die "duplicate pair $pname\n" if exists $pairs{$pname};
  die "pair $pname names array $aname which does not exist\n" unless exists $params{$aname};
  $pairs{$pname} = [ $nn, $aname ];
}
my $n_pairs = scalar keys %pairs;

# --- the declared per-Z lengths ----------------------------------------------------------
my %ndecl;
while ($text =~ /static\s+const\s+G4int\s+N(\d+)\s*=\s*(\d+)\s*;/g) { $ndecl{$1} = $2; }

# --- the per-Z ordered lists (initialisers wrap across lines) ----------------------------
my $flat = $text;
$flat =~ s/\r?\n/ /g;
my %lists;
while ($flat =~ /static\s+const\s+std::pair<G4int,\s*const\s+G4double\*>\s+Z(\d+)\s*\[\s*N(\d+)\s*\]\s*=\s*\{([^}]*)\}/g) {
  my ($z, $zn, $body) = ($1, $2, $3);
  die "Z$z list is dimensioned N$zn\n" unless $z == $zn;
  my @names = split(/\s*,\s*/, $body);
  @names = map { my $t = $_; $t =~ s/^\s+|\s+$//g; $t } @names;
  @names = grep { length $_ } @names;
  die "duplicate list for Z=$z\n" if exists $lists{$z};
  $lists{$z} = \@names;
}

# --- assertions --------------------------------------------------------------------------
die "expected 417 live parameter arrays, found $n_arrays\n" unless $n_arrays == 417;
die "expected 417 live pair wrappers, found $n_pairs\n"     unless $n_pairs  == 417;
die "expected 99 per-Z lists, found ".scalar(keys %lists)."\n" unless scalar(keys %lists) == 99;
my $total = 0;
my %used;
my $n_repeat = 0;
for my $z (0..98) {
  die "no list for Z=$z\n" unless exists $lists{$z};
  die "no declared N$z\n"  unless exists $ndecl{$z};
  my $l = scalar @{$lists{$z}};
  die "Z=$z: list has $l entries, N$z declares $ndecl{$z}\n" unless $l == $ndecl{$z};
  for my $pname (@{$lists{$z}}) {
    die "Z=$z references pair $pname which does not exist\n" unless exists $pairs{$pname};
    ++$n_repeat if $used{$pname}++;
  }
  $total += $l;
}
# Four elements pad their list with a repeat of an earlier entry, because the isotope rows that
# should have filled those slots are commented out in Geant4 while the declared N<z> still counts
# them: Nb (Z=41, N41=3, one row listed three times - the N=53 and N=54 rows are commented out),
# Tb (Z=65) and Ta (Z=73) the same with two slots and one row, and U (Z=92) whose tenth slot
# repeats N=146 in place of the commented-out N=149. The model's search compares each slot's
# neutron number against tgN, so the padded slots can never match anything the earlier slot did
# not already match: Nb-94, Nb-95, Tb-160, Ta-182 and U-241 fall to the default row. That is what
# Geant4 does, so it is what this table says. Five repeated slots is the whole difference between
# the 417 live rows and the 422 list slots, and both counts are asserted so that another padding -
# or a row silently dropped - cannot arrive unnoticed.
die "expected 5 repeated list entries (Z=41,65,73,92), found $n_repeat\n" unless $n_repeat == 5;
die "per-Z lengths sum to $total, expected 422\n" unless $total == 422;
for my $pname (keys %pairs) {
  die "pair $pname is declared and never listed\n" unless $used{$pname};
}

# --- emit --------------------------------------------------------------------------------
print <<'HDR';
// The per-isotope low-energy parameters of G4ChipsNeutronElasticXS::GetPTables (11.1.1).
//
// GENERATED by tools/extract_chips_neutron.pl - do not edit by hand. The script asserts the
// counts it read (422 isotope rows, 99 per-Z lists, each list's length equal to Geant4's own
// N<z>) and refuses to emit anything if one of them is off, because a table that is quietly
// short here reads as a physics disagreement later.
//
// Geant4 stores these as 422 `pZ<z>N<n>[7]` arrays wrapped in `std::pair<G4int,const G4double*>`
// and gathered into 99 per-Z arrays. GetPTables walks the array for tgZ in order and takes the
// FIRST entry whose first element equals tgN; the order is therefore part of the answer and is
// preserved here. Entries whose pair name disagrees with their neutron number (Geant4 has
// `Z3N1(3,pZ3N3)`) carry the number, not the name.
//
// The seven numbers become lastPAR[4], [7], [8], [9], [10], [11], [12] - the low-energy part of
// the nA elastic cross section. An isotope that is not in the list gets Geant4's default row
// {5.2e-7, 22., 0.00026, 1.3e-9, 2.7, 4.e-5, 0.005}, which is in the port beside this table.
//
// Z=0 and the Z=1 N=0 row are Geant4's own "not used (fake)" placeholders: the model treats a
// free proton or neutron target through the pp/np parameterisations before it ever indexes this
// table. They are kept so that the indices line up with Geant4's.
#pragma once

namespace g4gpu::physics::hadronic::chips_data {

/// One isotope row: the neutron number Geant4 matches on, then its seven parameters.
struct NeutronLowERow {
  int n;
  double p[7];
};

HDR

print "constexpr int kNeutronLowENZ = 99;\n\n";
print <<'ACC';
// The tables are reached through accessor functions returning a pointer to a function-local
// static, which is this port's idiom for a constant table (see data/barashenkov.hh): a
// `constexpr` array at namespace scope is not addressable from device code under nvcc, and the
// same header has to compile for the host tests and for a kernel.
ACC

for my $z (0..98) {
  my @names = @{$lists{$z}};
  printf("/// Z = %d, %d isotope%s.\n", $z, scalar(@names), (scalar(@names) == 1 ? "" : "s"));
  printf("__host__ __device__ inline const NeutronLowERow* neutron_lowe_z%d() {\n", $z);
  printf("  static const NeutronLowERow v[%d] = {\n", scalar(@names));
  for my $pname (@names) {
    my ($nn, $aname) = @{$pairs{$pname}};
    printf("    {%3d, {%s}},\n", $nn, join(", ", @{$params{$aname}}));
  }
  print "  };\n  return v;\n}\n";
}

print "\n/// Geant4's NIso[ZMAX], Z = 0..98: how many isotope rows each element has.\n";
print "__host__ __device__ inline const int* neutron_lowe_niso() {\n";
print "  static const int v[kNeutronLowENZ] = {\n   ";
for my $z (0..98) {
  printf(" %d,", scalar @{$lists{$z}});
  print "\n   " if ($z % 16) == 15;
}
print "\n  };\n  return v;\n}\n";

print "\n/// Geant4's Pars[ZMAX][*]: the per-Z row array. Indexing outside 0..98 is the caller's\n";
print "/// job to refuse - Geant4 would read past the end of its own array.\n";
print "__host__ __device__ inline const NeutronLowERow* neutron_lowe_pars(int z) {\n";
print "  switch (z) {\n";
for my $z (0..98) {
  printf("    case %d: return neutron_lowe_z%d();\n", $z, $z);
}
print "    default: return nullptr;\n  }\n}\n";
printf("\n/// Total isotope rows, asserted by the generator against Geant4's own array lengths.\n");
printf("constexpr int kNeutronLowETotalRows = %d;\n", $total);
print "\n}  // namespace g4gpu::physics::hadronic::chips_data\n";
