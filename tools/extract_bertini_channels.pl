#!/usr/bin/perl
# Extracts the 34 Bertini cascade channel tables of G4CascadeChannelTables out of Geant4
# 11.1.1 and writes them as src/data/bertini_channels.hh.
#
#   perl tools/extract_bertini_channels.pl
#
# Why a script. This is the bulk of the INUCL tree's data: 185,970 cross-section values and
# 12,036 final-state particle codes spread over thirty source files, each written as C array
# initialisers with the multiplicity structure carried only by the array DIMENSIONS in a
# template argument list in a different file. There is no way to eyeball it and no way to
# retype it, so it is parsed, the counts are asserted against the template arguments read
# from the .hh, and the totals are printed for the commit message.
#
# What the structure is, because the generated header has to carry it. A channel is a
# G4CascadeData<NE,N2,N3,N4,N5,N6,N7[,N8,N9]>: NE energy bins, and N_m exclusive final states
# of multiplicity m. crossSections[NXS][NE] is ONE array holding all NXS = sum(N_m) channels
# back to back, and `index[9]` (G4CascadeData::initialize) is the running offset that says
# where each multiplicity's block starts. The per-multiplicity summed cross section
# `multiplicities[m-2][k]` and the inclusive `sum[k]` are COMPUTED from it, not tabulated, so
# they are not extracted - the port recomputes them the same way and the oracle checks the
# result.
#
# Three things this found that a reading would not have:
#
#   * Isospin mirror pairs SHARE a cross-section table and do not share a final-state list.
#     G4CascadeT31piNChannel.cc defines pi-p and pi+n from one `pimPCrossSections` with two
#     different `*bfs` sets; T33 does pi+p/pi-n, T11 pi0p/pi0n, T1Gam gamma-p/gamma-n. Four
#     files, eight channels, four cross-section tables. Emitting per-channel copies would
#     quadruple the data and hide the fact that the two members of a pair cannot disagree.
#   * `tot` is sometimes the computed sum and sometimes a tabulated inclusive array, chosen
#     by WHICH CONSTRUCTOR the channel calls. The eight channels that pass an explicit
#     theTot (gamma, the three pion pairs, NN, NP, PP, mu-p) get a measured inclusive cross
#     section that is LARGER than the sum of their exclusive channels, and
#     G4CascadeFunctions::getMultiplicity uses the difference to return maxMultiplicity - so
#     whether the array is there changes the physics, and `tot_id = -1` records its absence.
#   * G4CascadeMuMinusPChannel.cc has two commented-out `/* ... */` blocks holding an older
#     100.0-millibarn cross section beside the 0.01 that is live. Comments are stripped
#     before parsing for that reason; a line-oriented parser would have read 60 of the 240
#     values from the dead block.
#
# **This machine has two perls and they are not the same language.** `/usr/bin/perl` under Git
# Bash is 5.34.0; the perl on the PATH that cmd.exe and PowerShell find is
# `C:\MinGW\msys\1.0\bin\perl.exe`, which is **5.8.8** and predates the `s///r` modifier (5.14),
# `//` (5.10) and `state` (5.10). A script that uses any of them runs from one shell and dies
# with a bare syntax error from the other - which is what happened here, and it looked like a
# corrupted file rather than a wrong interpreter. So this script is written to 5.8, `use 5.008`
# says so, and nothing below needs a feature newer than that.
use strict;
use warnings;
use 5.008;

my $g4 = $ENV{G4SRC} || 'D:/Documents/Geant4/Windows/geant4-v11.1.1';
my $casc = "$g4/source/processes/hadronic/models/cascade/cascade";
my $out = 'src/data/bertini_channels.hh';

# ---------------------------------------------------------------------------------------
# G4InuclParticleNames::Short, the enum the *bfs arrays are written in.
my %code = (
  nuc => 0, pro => 1, neu => 2, pip => 3, pim => 5, pi0 => 7, gam => 9,
  kpl => 11, kmi => 13, k0 => 15, k0b => 17,
  lam => 21, sp => 23, s0 => 25, sm => 27, xi0 => 29, xim => 31, om => 33,
  deu => 41, pp => 111, pn => 112, nn => 122,
  enu => -1, mnu => -3, tnu => -5, aenu => -7, amnu => -9, atnu => -11,
  ele => -21, mum => -23, pos => -27, mup => -29,
);
# The Long spellings too, in case a table ever uses them.
my %long = (
  nuclei => 0, proton => 1, neutron => 2, pionPlus => 3, pionMinus => 5,
  pionZero => 7, photon => 9, kaonPlus => 11, kaonMinus => 13, kaonZero => 15,
  kaonZeroBar => 17, lambda => 21, sigmaPlus => 23, sigmaZero => 25,
  sigmaMinus => 27, xiZero => 29, xiMinus => 31, omegaMinus => 33,
  muonNu => -3, muonMinus => -23,
);
%code = (%code, %long);

# A final-state entry is written either as an enum name or as the enum's VALUE: the eight
# hyperon channel files (G4CascadeLambdaNChannel.cc and friends) spell out {2, 21} where the
# pion and kaon files write {neu, lam}. Both forms are accepted and the numeric one is
# range-checked against the enum, so a typo in the source that is not a real type code is
# caught here rather than becoming a particle nothing can identify.
sub enumval {
  my ($t, $where) = @_;
  if ($t =~ /^-?\d+$/) {
    my %valid = map { $code{$_} => 1 } keys %code;
    die "extract_bertini_channels.pl: particle code $t is not a G4InuclParticleNames value" .
        ($where ? " in $where" : "") . "\n" unless $valid{0 + $t};
    return 0 + $t;
  }
  die "extract_bertini_channels.pl: unknown particle token '$t'" .
      ($where ? " in $where" : "") . "\n" unless exists $code{$t};
  return $code{$t};
}

# ---------------------------------------------------------------------------------------
# Read the template arguments from the channel headers, so every extracted count has an
# independent expected value. G4CascadeData<NE,N2,..,N7[,N8,N9]>.
my %tmpl;   # class name (e.g. PiMinusP) -> [NE, N2..N9]
my %sampler;  # class name -> sampler struct name
for my $hh (glob "$casc/include/G4Cascade*Channel.hh") {
  my $txt = slurp($hh);
  while ($txt =~ /struct\s+G4Cascade(\w+)ChannelData\s*\{\s*typedef\s+G4CascadeData<([-\d,\s]+)>\s*data_t;/g) {
    my ($cls, $args) = ($1, $2);
    my @a = map { 0 + $_ } grep { /\S/ } split /\s*,\s*/, $args;
    $tmpl{$cls} = \@a;
  }
  while ($txt =~ /G4CascadeFunctions<\s*G4Cascade(\w+)ChannelData\s*,\s*(\w+)\s*>/g) {
    $sampler{$1} = $2;
  }
}
die "no channel templates found under $casc/include\n" unless %tmpl;

# NN, NP and PP declare their class by hand rather than by typedef, so the sampler comes
# from the base-class clause instead.
for my $cls (keys %tmpl) {
  next if $sampler{$cls};
  my $hh = "$casc/include/G4Cascade${cls}Channel.hh";
  my $txt = slurp($hh);
  if ($txt =~ /public\s+G4CascadeFunctions<\s*G4Cascade\w+ChannelData\s*,\s*(\w+)\s*>/) {
    $sampler{$cls} = $1;
  } else {
    die "extract_bertini_channels.pl: no sampler for $cls\n";
  }
}

# ---------------------------------------------------------------------------------------
# The three energy-bin scales, from the sampler .cc files.
my %bins;
for my $s (qw(G4PionNucSampler G4KaonSampler G4KaonHypSampler)) {
  my $txt = strip_comments(slurp("$casc/src/$s.cc"));
  $txt =~ /static\s+const\s+G4double\s+bins\[(\d+)\]\s*=\s*\{([^}]*)\}/s
    or die "extract_bertini_channels.pl: no bins[] in $s.cc\n";
  my ($n, $body) = ($1, $2);
  my @v = grep { /\S/ } split /\s*,\s*/, trim($body);
  die "$s: bins[] declared $n, parsed " . scalar(@v) . "\n" unless @v == $n;
  $bins{$s} = \@v;
}

# ---------------------------------------------------------------------------------------
# Parse every channel .cc: the arrays, then the data() constructor calls that select them.
my (%fs, %xs, %tot);        # C array name -> parsed contents
my (%fs_dim, %xs_dim, %tot_dim);
my @chan;                   # one entry per G4CascadeData instance, in file order

for my $cc (sort glob "$casc/src/G4Cascade*.cc") {
  my $raw = slurp($cc);
  next unless $raw =~ /ChannelData::data\s*\(/;
  my $txt = strip_comments($raw);

  # Arrays. `static` is optional: T31's pimPCrossSections is declared without it.
  while ($txt =~ /(?:static\s+)?const\s+G4(int|double)\s+(\w+)\s*\[(\d+)\]\s*(?:\[(\d+)\]\s*)?=\s*\{/g) {
    my ($ty, $name, $n1, $n2) = ($1, $2, $3, $4);
    my $body = braced_body(\$txt, pos($txt) - 1);
    $body =~ s/[{}]/ /g;   # not s///r: 5.8.8 does not have it - see the note at the top
    my @v = grep { /\S/ } split /\s*,\s*/, trim($body);
    my $want = defined($n2) ? $n1 * $n2 : $n1;
    die "$cc: $name declared [$n1]" . (defined($n2) ? "[$n2]" : "") .
        " = $want values, parsed " . scalar(@v) . "\n" unless @v == $want;
    if ($ty eq 'int') {
      $fs{$name} = [ map { enumval($_) } @v ];
      $fs_dim{$name} = [ $n1, $n2 ];
    } elsif (defined $n2) {
      $xs{$name} = \@v;
      $xs_dim{$name} = [ $n1, $n2 ];
    } else {
      $tot{$name} = \@v;
      $tot_dim{$name} = $n1;
    }
  }

  # The constructor calls.
  while ($txt =~ /G4Cascade(\w+)ChannelData::data\s*\(/g) {
    my $cls = $1;
    my $args = braced_body(\$txt, pos($txt) - 1, '(', ')');
    my @a = split_args($args);
    my (@bfs, $xsname, $totname, $istate, $cname);
    for my $a (@a) {
      $a = trim($a);
      if ($a =~ /^"(.*)"$/)                 { $cname = $1; }
      elsif ($a =~ /bfs$/)                  { push @bfs, $a; }
      elsif ($a =~ /CrossSections$/)        { $xsname = $a; }
      elsif ($a =~ /[Tt]ot[A-Za-z]*$/)      { $totname = $a; }
      elsif ($a =~ /^(\w+)\s*\*\s*(\w+)$/)  { $istate = enumval($1) * enumval($2); }
      else { die "$cc: $cls: cannot classify constructor argument '$a'\n"; }
    }
    die "$cc: $cls: no cross-section array\n" unless $xsname;
    die "$cc: $cls: no initial state\n" unless defined $istate;
    die "$cc: $cls: " . scalar(@bfs) . " bfs arrays (want 6 or 8)\n"
      unless @bfs == 6 || @bfs == 8;
    push @chan, { cls => $cls, name => $cname, istate => $istate,
                  bfs => \@bfs, xs => $xsname, tot => $totname };
  }
}

# 34 G4CascadeData instances in 30 files - G4CascadeChannelTables' constructor registers
# exactly that many, and it is the one count with no second source in the tree to check it
# against, so it is written here.
my $expect_channels = 34;
die "extract_bertini_channels.pl: found " . scalar(@chan) .
    " channels, expected $expect_channels\n" unless @chan == $expect_channels;

# ---------------------------------------------------------------------------------------
# Cross-check every channel against its template arguments, and against the samplers.
my %sampler_id = (G4PionNucSampler => 0, G4KaonSampler => 1, G4KaonHypSampler => 2);
my %sampler_ne = (G4PionNucSampler => 30, G4KaonSampler => 30, G4KaonHypSampler => 31);
my %sampler_nm = (G4PionNucSampler => 8, G4KaonSampler => 8, G4KaonHypSampler => 6);

for my $c (@chan) {
  my $t = $tmpl{$c->{cls}} or die "no template for $c->{cls}\n";
  my ($ne, @nm) = @$t;
  my $samp = $sampler{$c->{cls}} or die "no sampler for $c->{cls}\n";
  die "$c->{cls}: NE $ne but $samp has $sampler_ne{$samp}\n" unless $ne == $sampler_ne{$samp};
  # NM is 6 when N8/N9 are absent, 8 when they are there. G4CascadeData: NM = N9?8:N8?7:6 -
  # and no 11.1.1 channel has N8 without N9, so NM is 6 or 8 and never 7.
  my $nmult = scalar(@nm);
  die "$c->{cls}: $nmult multiplicity counts, sampler says $sampler_nm{$samp}\n"
    unless $nmult == $sampler_nm{$samp};
  die "$c->{cls}: " . scalar(@{$c->{bfs}}) . " bfs arrays but $nmult counts\n"
    unless scalar(@{$c->{bfs}}) == $nmult;

  # Each bfs array's dimensions must be [N_m][m].
  for my $i (0 .. $#nm) {
    my $nm_ = $c->{bfs}[$i];
    my $d = $fs_dim{$nm_} or die "$c->{cls}: no array $nm_\n";
    die "$c->{cls}: $nm_ is [$d->[0]][$d->[1]], template says [$nm[$i]][" . ($i+2) . "]\n"
      unless $d->[0] == $nm[$i] && $d->[1] == $i + 2;
  }

  # The cross-section array must be [sum N_m][NE].
  my $nxs = 0; $nxs += $_ for @nm;
  my $d = $xs_dim{$c->{xs}} or die "$c->{cls}: no array $c->{xs}\n";
  my ($want_nxs, $want_ne) = ($nxs, $ne);
  die "$c->{cls}: $c->{xs} is [$d->[0]][$d->[1]], want [$want_nxs][$want_ne]\n"
    unless $d->[0] == $want_nxs && $d->[1] == $want_ne;
  if ($c->{tot}) {
    die "$c->{cls}: $c->{tot} is [$tot_dim{$c->{tot}}], want [$ne]\n"
      unless $tot_dim{$c->{tot}} == $ne;
  }
  $c->{ne} = $ne; $c->{nm} = \@nm; $c->{samp} = $sampler_id{$samp}; $c->{nxs} = $nxs;
}

# ---------------------------------------------------------------------------------------
# The four arrays G4CascadeData::initialize COMPUTES rather than reads.
#
# They are emitted rather than recomputed on the device for two reasons: the summation ORDER is
# observable (a multiplicity's sum runs over its channel block in index order, and
# floating-point addition is not associative), and recomputing them per lookup would put an
# O(NXS) loop inside the cascade's innermost sampling call. Perl's numbers are IEEE doubles and
# the loops below run in the same order as the .icc, so the results are bit for bit what the
# compiler's object holds - which ref/oracle/bertini_chtables.csv then confirms, because it
# dumps all four.
#
# `inelastic[]` is computed here and emitted, and **nothing in 11.1.1 reads it**: the only
# references in the whole tree are its declaration, the two lines that fill it, and a History
# entry. It is emitted anyway because the oracle dumps it, so agreeing on it is free evidence
# that the elastic-channel search below found the same row Geant4's initialize() did.
for my $c (@chan) {
  my $ne = $c->{ne};
  my @nm = @{$c->{nm}};
  my $xs = $xs{$c->{xs}};                # flat [nxs][ne], row-major
  my @index = (0);
  my $run = 0;
  for my $n (@nm) { $run += $n; push @index, $run; }

  my @mult;
  for my $im (0 .. $#nm) {
    my ($start, $stop) = ($index[$im], $index[$im + 1]);
    for my $k (0 .. $ne - 1) {
      my $s = 0.0;
      for my $i ($start .. $stop - 1) { $s += $xs->[$i * $ne + $k]; }
      push @mult, $s;
    }
  }
  my @sum;
  for my $k (0 .. $ne - 1) {
    my $s = 0.0;
    for my $im (0 .. $#nm) { $s += $mult[$im * $ne + $k]; }
    push @sum, $s;
  }
  # The elastic two-body row: the first multiplicity-2 channel whose two type codes multiply to
  # the initial state. initialize() scans only the multiplicity-2 block, and the FIXME beside
  # the `else` branch is Geant4's own note that some tables have no such row.
  my $elastic = -1;
  my $fs2 = $fs{$c->{bfs}[0]};
  for my $i (0 .. $nm[0] - 1) {
    if ($fs2->[$i * 2] * $fs2->[$i * 2 + 1] == $c->{istate}) { $elastic = $i; last; }
  }
  my $tot = $c->{tot} ? $tot{$c->{tot}} : \@sum;
  my @inel;
  for my $k (0 .. $ne - 1) {
    push @inel, ($elastic >= 0) ? $tot->[$k] - $xs->[$elastic * $ne + $k] : $tot->[$k];
  }
  $c->{mult_arr} = \@mult;
  $c->{sum_arr} = \@sum;
  $c->{inel_arr} = \@inel;
  $c->{elastic} = $elastic;
}

# ---------------------------------------------------------------------------------------
# Flatten. Distinct arrays are emitted once and referenced by offset, so a shared
# cross-section table is one copy and the sharing is visible in the generated table.
my (@xs_flat, %xs_off, @xs_order);
for my $c (@chan) {
  next if exists $xs_off{$c->{xs}};
  $xs_off{$c->{xs}} = scalar @xs_flat;
  push @xs_order, $c->{xs};
  push @xs_flat, @{$xs{$c->{xs}}};
}
my (@tot_flat, %tot_off, @tot_order);
for my $c (@chan) {
  next unless $c->{tot};
  next if exists $tot_off{$c->{tot}};
  $tot_off{$c->{tot}} = scalar @tot_flat;
  push @tot_order, $c->{tot};
  push @tot_flat, @{$tot{$c->{tot}}};
}
my (@fs_flat, %fs_off, @fs_order);
for my $c (@chan) {
  for my $a (@{$c->{bfs}}) {
    next if exists $fs_off{$a};
    $fs_off{$a} = scalar @fs_flat;
    push @fs_order, $a;
    push @fs_flat, @{$fs{$a}};
  }
}
# The derived arrays are NOT shared between the members of a mirror pair, even though two of
# the three would be: multiplicities[][] and sum[] depend only on the cross-section block, but
# inelastic[] subtracts the ELASTIC row, and pi-p's elastic row is not pi+n's. One block per
# channel, laid out [mult: n_mult*ne][sum: ne][inelastic: ne].
my @deriv_flat;
for my $c (@chan) {
  $c->{deriv_off} = scalar @deriv_flat;
  push @deriv_flat, @{$c->{mult_arr}}, @{$c->{sum_arr}}, @{$c->{inel_arr}};
}

# ---------------------------------------------------------------------------------------
open my $fh, '>', $out or die "cannot write $out: $!\n";
my $nxs_total = scalar @xs_flat;
my $nfs_total = scalar @fs_flat;

print $fh <<"HDR";
// The 34 Bertini cascade channel tables of G4CascadeChannelTables, Geant4 11.1.1.
//
// GENERATED by tools/extract_bertini_channels.pl - do not edit by hand.
// $nxs_total cross-section values, $nfs_total final-state particle codes, 34 channels.
//
// A channel is a G4CascadeData<NE,N2,...>: `nm[m-2]` exclusive final states of multiplicity
// m, whose cross sections live back to back in one xs block. `index[]` is the running offset
// G4CascadeData::initialize builds, so index[m-2] .. index[m-1] is multiplicity m's slice of
// the block. The per-multiplicity sums, the inclusive sum and the inelastic sum are COMPUTED
// from that block by G4CascadeData::initialize and are not stored here.
//
// xs_off and fs_off are offsets into the two flat arrays, and they are SHARED: the four
// isospin mirror pairs (pi-p/pi+n, pi+p/pi-n, pi0p/pi0n, gamma-p/gamma-n) have one
// cross-section table between them and two final-state lists, which is how Geant4 writes
// them (one .cc file per pair). tot_off is -1 for the channels whose constructor does not
// pass a measured inclusive cross section; for those, G4CascadeData's `tot` IS `sum` and
// G4CascadeFunctions::getMultiplicity skips the summed/total comparison entirely.
//
// Particle codes are G4InuclParticleNames::Short values, not PDG codes.
#ifndef G4GPU_DATA_BERTINI_CHANNELS_HH
#define G4GPU_DATA_BERTINI_CHANNELS_HH

namespace g4gpu::data {

/// One G4CascadeData instance, as G4CascadeChannelTables registers it.
struct BertiniChannel {
  int initial_state;    ///< product of the two G4InuclElementaryParticle type codes
  int sampler;          ///< 0 G4PionNucSampler, 1 G4KaonSampler, 2 G4KaonHypSampler
  int ne;               ///< energy bins, 30 or 31
  int n_mult;           ///< multiplicity bins: 6 (mult 2-7) or 8 (mult 2-9)
  int index[9];         ///< G4CascadeData::index - offsets into this channel's xs block
  int xs_off;           ///< start of [nxs][ne] cross sections in bertini_channel_xs()
  int tot_off;          ///< start of [ne] inclusive cross section, or -1 if there is none
  int deriv_off;        ///< start of this channel's block in bertini_channel_derived():
                        ///< [n_mult][ne] per-multiplicity sums, then [ne] sum, then
                        ///< [ne] inelastic
  int elastic_chan;     ///< index within the multiplicity-2 block of the elastic channel, or
                        ///< -1 when the table has none (G4CascadeData::initialize's FIXME)
  int fs_off[8];        ///< start of multiplicity m's [n_m][m] codes in bertini_channel_fs()
};

// The channel NAMES are at namespace scope and not in the struct above, in the same order.
// nvcc rejects a function-scope `static const` array in `__host__ __device__` code whose
// initializer is not a compile-time constant, and the value of a `const char*` is a link-time
// address: "dynamic initialization is not supported for a function-scope static __device__
// variable". They are needed on the host, because ref/oracle/bertini_chtables.csv keys every
// row by G4CascadeData::name, and never on the device.

HDR

printf $fh "constexpr int kBertiniChannels = %d;\n", scalar @chan;
printf $fh "constexpr int kBertiniChannelXsValues = %d;\n", $nxs_total;
printf $fh "constexpr int kBertiniChannelTotValues = %d;\n", scalar @tot_flat;
printf $fh "constexpr int kBertiniChannelFsCodes = %d;\n", $nfs_total;
printf $fh "constexpr int kBertiniChannelDerivedValues = %d;\n\n", scalar @deriv_flat;

printf $fh "constexpr const char* const kBertiniChannelNames[%d] = {\n", scalar @chan;
printf $fh "    \"%s\",\n", $_->{name} for @chan;
print $fh "};\n\n";

# The energy-bin scales.
for my $s (qw(G4PionNucSampler G4KaonSampler G4KaonHypSampler)) {
  (my $short = $s) =~ s/^G4//; $short =~ s/Sampler$//;
  my $v = $bins{$s};
  printf $fh "/// %s's bin edges, GeV. %d bins.\n", $s, scalar @$v;
  printf $fh "__host__ __device__ inline const double* bertini_bins_%s() {\n", lc $short;
  printf $fh "  static const double v[%d] = {\n", scalar @$v;
  print $fh wrap_values($v, 10, '    ');
  print $fh "  };\n  return v;\n}\n\n";
}

# The channel table itself.
print $fh "__host__ __device__ inline const BertiniChannel* bertini_channels() {\n";
printf $fh "  static const BertiniChannel c[%d] = {\n", scalar @chan;
for my $c (@chan) {
  my @nm = @{$c->{nm}};
  my @index = (0);
  my $run = 0;
  for my $n (@nm) { $run += $n; push @index, $run; }
  push @index, $index[-1] while @index < 9;
  my @fsoff = map { $fs_off{$_} } @{$c->{bfs}};
  push @fsoff, -1 while @fsoff < 8;
  printf $fh "    {%6d, %d, %2d, %d, {%s}, %6d, %5d, %6d, %3d, {%s}},   // %s\n",
    $c->{istate}, $c->{samp}, $c->{ne}, scalar(@nm),
    join(',', map { sprintf '%3d', $_ } @index),
    $xs_off{$c->{xs}},
    ($c->{tot} ? $tot_off{$c->{tot}} : -1),
    $c->{deriv_off}, $c->{elastic},
    join(',', map { sprintf '%5d', $_ } @fsoff),
    $c->{name};
}
print $fh "  };\n  return c;\n}\n\n";

# The two flat arrays.
printf $fh "/// Exclusive cross sections, millibarn, row-major [nxs][ne] per channel block.\n";
printf $fh "/// %d values over %d distinct blocks: %s.\n",
  $nxs_total, scalar @xs_order, join(', ', @xs_order);
print $fh "__host__ __device__ inline const double* bertini_channel_xs() {\n";
printf $fh "  static const double v[%d] = {\n", $nxs_total;
print $fh wrap_values(\@xs_flat, 10, '    ');
print $fh "  };\n  return v;\n}\n\n";

printf $fh "/// Measured inclusive cross sections, millibarn. %d values over %d blocks: %s.\n",
  scalar @tot_flat, scalar @tot_order, join(', ', @tot_order);
print $fh "__host__ __device__ inline const double* bertini_channel_tot() {\n";
printf $fh "  static const double v[%d] = {\n", scalar @tot_flat;
print $fh wrap_values(\@tot_flat, 10, '    ');
print $fh "  };\n  return v;\n}\n\n";

printf $fh "/// What G4CascadeData::initialize computes: per channel, [n_mult][ne] multiplicity\n";
printf $fh "/// sums, then [ne] summed and [ne] inelastic cross sections. %d values.\n",
  scalar @deriv_flat;
print $fh "__host__ __device__ inline const double* bertini_channel_derived() {\n";
printf $fh "  static const double v[%d] = {\n", scalar @deriv_flat;
print $fh wrap_values(\@deriv_flat, 6, '    ', '%.17g');
print $fh "  };\n  return v;\n}\n\n";

printf $fh "/// Final-state particle codes, %d of them, row-major [n_m][m] per block.\n",
  $nfs_total;
print $fh "__host__ __device__ inline const signed char* bertini_channel_fs() {\n";
printf $fh "  static const signed char v[%d] = {\n", $nfs_total;
print $fh wrap_values(\@fs_flat, 20, '    ');
print $fh "  };\n  return v;\n}\n\n";

print $fh "}  // namespace g4gpu::data\n\n#endif  // G4GPU_DATA_BERTINI_CHANNELS_HH\n";
close $fh;

printf "wrote %s: %d channels, %d cross sections (%d blocks), %d inclusive values (%d blocks), %d final-state codes (%d blocks)\n",
  $out, scalar @chan, $nxs_total, scalar @xs_order, scalar @tot_flat,
  scalar @tot_order, $nfs_total, scalar @fs_order;
for my $c (@chan) {
  printf "  %-16s state %6d sampler %d NE %2d mult %s nxs %4d tot %s\n",
    $c->{name}, $c->{istate}, $c->{samp}, $c->{ne},
    join('/', @{$c->{nm}}), $c->{nxs}, ($c->{tot} ? $c->{tot} : '(sum)');
}

# ---------------------------------------------------------------------------------------
sub slurp {
  my ($f) = @_;
  open my $h, '<', $f or die "cannot read $f: $!\n";
  local $/; my $t = <$h>; close $h; return $t;
}

sub trim { my ($s) = @_; $s =~ s/^\s+//; $s =~ s/\s+$//; return $s; }

# Strip C and C++ comments. G4CascadeMuMinusPChannel.cc keeps a dead cross-section table
# inside /* */, so this is load-bearing and not cosmetic.
sub strip_comments {
  my ($t) = @_;
  $t =~ s{/\*.*?\*/}{ }gs;
  $t =~ s{//[^\n]*}{}g;
  return $t;
}

# Given a position holding an opening delimiter, return the text inside its match.
sub braced_body {
  my ($tref, $start, $open, $close) = @_;
  $open ||= '{'; $close ||= '}';
  my $depth = 0;
  my $i = $start;
  my $n = length $$tref;
  while ($i < $n) {
    my $ch = substr($$tref, $i, 1);
    $depth++ if $ch eq $open;
    if ($ch eq $close) {
      $depth--;
      return substr($$tref, $start + 1, $i - $start - 1) if $depth == 0;
    }
    $i++;
  }
  die "unbalanced $open at offset $start\n";
}

# Split a constructor argument list at top-level commas.
sub split_args {
  my ($s) = @_;
  my (@out, $cur, $depth);
  $cur = ''; $depth = 0;
  for my $ch (split //, $s) {
    if ($ch eq '(' || $ch eq '[') { $depth++; }
    elsif ($ch eq ')' || $ch eq ']') { $depth--; }
    if ($ch eq ',' && $depth == 0) { push @out, $cur; $cur = ''; next; }
    $cur .= $ch;
  }
  push @out, $cur if $cur =~ /\S/;
  return @out;
}

# `$fmt` matters and defaults to Perl's own stringification, which is "%.15g".
#
# For the arrays read out of the Geant4 source that is exact: every literal in the .cc files has
# at most six significant digits, so 15 digits round-trips the double it parsed to. For the
# arrays this script COMPUTES it is not: a 352-term sum has all 17 digits, and printing 15 loses
# up to a few ulp. That is not academic - it is the whole of the first disagreement this test
# found. `ChannelSum` failed at 1.06e-15 with KzeroBarP's summed cross section reading 20.11
# where Geant4's is 20.109999999999978, and the culprit was this function, not the summation.
sub wrap_values {
  my ($v, $per, $indent, $fmt) = @_;
  my $s = '';
  for (my $i = 0; $i < @$v; $i += $per) {
    my @row = @{$v}[$i .. ($i + $per - 1 > $#$v ? $#$v : $i + $per - 1)];
    @row = map { sprintf $fmt, $_ } @row if $fmt;
    $s .= $indent . join(', ', @row) . (($i + $per < @$v) ? ",\n" : "\n");
  }
  return $s;
}
