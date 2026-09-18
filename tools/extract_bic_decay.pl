#!/usr/bin/perl
# Generates src/physics/hadronic/bic/im_r/decay_tables.hh from ref/oracle/bic_imr_decaytable.csv.
#
# WHY THIS ONE READS THE ORACLE AND NOT THE GEANT4 SOURCE. Every other table in this package is
# transcribed out of a literal array in a .cc file, and tools/extract_bic_imr.pl asserts each
# against its source. The decay tables are not literal arrays. G4ExcitedNucleonConstructor and
# G4ExcitedDeltaConstructor hold bRatio[NStates][NumberOfDecayModes] matrices - a branching ratio
# per MULTIPLET and per decay MODE - and G4ExcitedBaryonConstructor::CreateDecayTable turns each
# mode into one channel per charge state, weighting it by the isospin Clebsch-Gordan coefficient
# for that state; the ground-state Delta and the pions come from elsewhere again. The assembled
# G4DecayTable is what G4KineticTrack reads, and the only faithful way to get it is to ask Geant4
# for it. ref/dump/dump_bic.cc does exactly that, and the result has the same standing as
# bic_imr_species.csv: the port's copy is a CHECKED table, not a hand copy.
#
# The assertions below are therefore about structure rather than about matching a source array.
# A channel that does not conserve charge or baryon number, a species whose branching ratios do
# not sum to one, a daughter that is not itself in the file, or a change in the species count all
# fail here rather than as a wrong final state twenty thousand events later.
#
# Usage: perl tools/extract_bic_decay.pl
use strict;
use warnings;

my $csv = "ref/oracle/bic_imr_decaytable.csv";
my $out = "src/physics/hadronic/bic/im_r/decay_tables.hh";

open(my $fh, '<', $csv) or die "cannot read $csv: $!\nRun ref/oracle/run.bat bic first.\n";
my $header = <$fh>;
# The oracle writes CRLF on Windows and chomp leaves the CR on the last column name, after
# which that name carries a carriage return and the lookup below fails on a good file.
$header =~ s/\s+\z//;
my @cols = split /,/, $header;
my %ci;
$ci{ $cols[$_] } = $_ for 0 .. $#cols;
for my $need (qw(pdg name mass width shortlived charge baryon min_mass n_channels channel br
    n_daughters d0 d1 d2 d3)) {
    die "column $need missing from $csv\n" unless exists $ci{$need};
}

my (@order, %sp, %chan);
while (my $line = <$fh>) {
    $line =~ s/\s+\z//;
    next unless length $line;
    my @f = split /,/, $line;
    my $pdg = $f[ $ci{pdg} ] + 0;
    if (!exists $sp{$pdg}) {
        push @order, $pdg;
        $sp{$pdg} = {
            name       => $f[ $ci{name} ],
            mass       => $f[ $ci{mass} ],
            width      => $f[ $ci{width} ],
            shortlived => $f[ $ci{shortlived} ] + 0,
            charge     => $f[ $ci{charge} ] + 0,
            baryon     => $f[ $ci{baryon} ] + 0,
            min_mass   => $f[ $ci{min_mass} ],
            nchan      => $f[ $ci{n_channels} ] + 0,
        };
        $chan{$pdg} = [];
    }
    my $index = $f[ $ci{channel} ] + 0;
    next if $index < 0;    # a species with no decay table at all
    $chan{$pdg}[$index] = {
        br => $f[ $ci{br} ],
        nd => $f[ $ci{n_daughters} ] + 0,
        d  => [ $f[ $ci{d0} ] + 0, $f[ $ci{d1} ] + 0, $f[ $ci{d2} ] + 0, $f[ $ci{d3} ] + 0 ],
    };
}
close $fh;

# ---------------------------------------------------------------------------- assertions
my $nsp = scalar @order;
die "expected 126 species in the transitive closure, found $nsp\n" unless $nsp == 126;

my $nchan_total = 0;
my $max_chan    = 0;
my %offsum;
for my $pdg (@order) {
    my $s    = $sp{$pdg};
    my $list = $chan{$pdg};
    die "$s->{name}: n_channels says $s->{nchan} but " . scalar(@$list) . " rows\n"
        unless scalar(@$list) == $s->{nchan};
    $nchan_total += $s->{nchan};
    $max_chan = $s->{nchan} if $s->{nchan} > $max_chan;
    next unless $s->{nchan};
    my $sum = 0.0;
    for my $c (@$list) {
        die "$s->{name}: a channel index is missing\n" unless defined $c;
        die "$s->{name}: $c->{nd} daughters, this port carries at most 4\n" if $c->{nd} > 4;
        $sum += $c->{br};

        # Charge and baryon number, looked up in this same file - which is what makes the
        # transitive closure load-bearing rather than tidy.
        my ($q, $b) = (0, 0);
        for my $j (0 .. $c->{nd} - 1) {
            my $d = $c->{d}[$j];
            die "$s->{name}: daughter $d is not in the closure\n" unless exists $sp{$d};
            $q += $sp{$d}{charge};
            $b += $sp{$d}{baryon};
        }
        die "$s->{name}: channel charge $q against parent $s->{charge}\n" unless $q == $s->{charge};
        die "$s->{name}: channel baryon $b against parent $s->{baryon}\n" unless $b == $s->{baryon};
    }
    # FIFTEEN species' branching ratios do not sum to one, and the set is asserted exactly rather
    # than tolerated. Two of them are in the cascade's own production set: delta(1950)++ and
    # delta(1950)- come to 0.99, because G4ExcitedDeltaConstructor gives the multiplet an N gamma
    # mode with bRatio 0.01 and the two extreme charge states have no N gamma final state to put
    # it in - the branch is dropped and the remainder is NOT renormalised. N(1535)+ and N(1535)0
    # come to 1.001 the other way. The rest are the real particles' own truncated PDG tables.
    # docs/RISK.md V148. It does not bias the channel choice, which divides by the sum; it does
    # change the total actual width, and therefore the residual lifetime, by that per cent.
    $offsum{$pdg} = $sum if abs($sum - 1.0) > 1e-9;

}
die "expected 13 channels at most, found $max_chan\n" unless $max_chan == 13;

# The exact set of ten, with the sum each one reaches. A release that renormalises any of them,
# or that breaks a channel on one that is currently whole, fails here.
my %kOffSum = (
    221   => 0.9926,   # eta
    321   => 0.99981,  # kaon+
    -321  => 0.99981,  # kaon-
    130   => 0.9964,   # kaon0L
    310   => 0.99890,  # kaon0S
    223   => 0.997,    # omega
    333   => 0.985,    # phi          - the largest deficit in the set, 1.5%
    3122  => 0.997,    # lambda
    -3122 => 0.997,    # anti_lambda
    3222  => 0.999,    # sigma+
    -3222 => 0.999,    # anti_sigma+
    2228  => 0.99,     # delta(1950)++  - the dropped N gamma mode
    1118  => 0.99,     # delta(1950)-   - the same
    22212 => 1.001,    # N(1535)+
    22112 => 1.001,    # N(1535)0
);
for my $pdg (sort { $a <=> $b } keys %offsum) {
    die sprintf("%s (%d) branching ratios sum to %.17g and were expected to sum to one\n",
        $sp{$pdg}{name}, $pdg, $offsum{$pdg}) unless exists $kOffSum{$pdg};
    die sprintf("%s (%d) sums to %.17g, not the %.17g this file records\n", $sp{$pdg}{name}, $pdg,
        $offsum{$pdg}, $kOffSum{$pdg}) if abs($offsum{$pdg} - $kOffSum{$pdg}) > 1e-9;
}
for my $pdg (sort { $a <=> $b } keys %kOffSum) {
    die sprintf("%d was expected NOT to sum to one and now does\n", $pdg)
        unless exists $offsum{$pdg};
}


# The ground-state Delta asymmetry this package already records: delta- and delta++ have ONE
# channel each and delta0 and delta+ have three, because only the two neutral-pion-capable states
# have an N gamma mode. A release that adds one fails here and in the width assertion in
# tools/extract_bic_imr.pl at the same time.
die "delta- (1114) should have exactly 1 decay channel\n"  unless $sp{1114}{nchan} == 1;
die "delta++ (2224) should have exactly 1 decay channel\n" unless $sp{2224}{nchan} == 1;
for my $d (2114, 2214) {
    die "delta0/delta+ ($d) should have exactly 3 decay channels\n" unless $sp{$d}{nchan} == 3;
}

# A pion HAS a decay table - to mu nu - and that is what gives a cascade pion a finite but
# enormous residual lifetime rather than a division by zero in SampleResidualLifetime.
for my $p (211, -211, 111) {
    die "pion $p must have a decay table\n" unless $sp{$p}{nchan} >= 1;
}

# ---------------------------------------------------------------------------- emit
open(my $o, '>', $out) or die "cannot write $out: $!";
printf $o <<'HDR', $nsp, $nchan_total;
// The decay tables of every species the binary cascade can hold - GENERATED, do not edit.
//
// Written by tools/extract_bic_decay.pl from ref/oracle/bic_imr_decaytable.csv, which
// ref/dump/dump_bic.cc fills from Geant4 11.1.1's own G4DecayTable objects. See that script's
// header for why this one table comes from the oracle rather than from a literal array in the
// Geant4 source: G4ExcitedBaryonConstructor::CreateDecayTable assembles it at run time out of a
// per-multiplet branching-ratio matrix and an isospin weight per charge state, so there is no
// array to transcribe.
//
// The species set is the TRANSITIVE CLOSURE of the cascade's production set - both nucleons, the
// three pions and all 25 resonance multiplets in every charge state - under "is a decay daughter
// of". It closes at %d species and %d channels, and it pulls in the rho, the eta, the omega, the
// kaons, the lambda and the leptons a neutron or a pion decays to, because
// G4SampleResonance::GetMinimumMass recurses into every one of them.
HDR
print $o <<'HDR2';
#ifndef G4GPU_BIC_IMR_DECAY_TABLES_HH
#define G4GPU_BIC_IMR_DECAY_TABLES_HH

namespace g4gpu::bic::imr {

HDR2
printf $o "inline constexpr int kDecaySpeciesCount = %d;\n", $nsp;
printf $o "inline constexpr int kDecayChannelCount = %d;\n", $nchan_total;
printf $o "inline constexpr int kDecayMaxChannels = %d;\n", $max_chan;
print $o "inline constexpr int kDecayMaxDaughters = 4;\n\n";

sub emit_int {
    my ($name, $vals, $per) = @_;
    printf $o "__host__ __device__ inline const int* %s() {\n  static const int v[] = {", $name;
    for my $k (0 .. $#$vals) {
        print $o "\n     " if $k % $per == 0;
        printf $o "%d,", $vals->[$k];
    }
    print $o "};\n  return v;\n}\n\n";
}

sub emit_dbl {
    my ($name, $vals, $per) = @_;
    printf $o "__host__ __device__ inline const double* %s() {\n  static const double v[] = {",
        $name;
    for my $k (0 .. $#$vals) {
        print $o "\n     " if $k % $per == 0;
        printf $o "%.17g,", $vals->[$k];
    }
    print $o "};\n  return v;\n}\n\n";
}

my (@pdg, @mass, @width, @short, @charge, @baryon, @minm, @first, @nch);
my (@cbr, @cnd, @cd);
my $cursor = 0;
for my $pdg (@order) {
    my $s = $sp{$pdg};
    push @pdg,    $pdg;
    push @mass,   $s->{mass};
    push @width,  $s->{width};
    push @short,  $s->{shortlived};
    push @charge, $s->{charge};
    push @baryon, $s->{baryon};
    push @minm,   $s->{min_mass};
    push @first,  $cursor;
    push @nch,    $s->{nchan};
    for my $c (@{ $chan{$pdg} }) {
        push @cbr, $c->{br};
        push @cnd, $c->{nd};
        push @cd, $c->{d}[0], $c->{d}[1], $c->{d}[2], $c->{d}[3];
        ++$cursor;
    }
}
die "channel cursor $cursor against $nchan_total\n" unless $cursor == $nchan_total;

print $o "/// The PDG codes, in the order the closure found them: the seeds first, then each\n";
print $o "/// daughter the moment it is first named.\n";
emit_int("decay_species_pdg", \@pdg, 8);
emit_dbl("decay_species_mass", \@mass, 4);
emit_dbl("decay_species_width", \@width, 4);
print $o "/// G4ParticleDefinition::IsShortLived - what decides whether a daughter's mass is\n";
print $o "/// sampled or taken at its pole, and what the three width integrals branch on.\n";
emit_int("decay_species_shortlived", \@short, 16);
emit_int("decay_species_charge", \@charge, 16);
emit_int("decay_species_baryon", \@baryon, 16);
print $o "/// G4SampleResonance::GetMinimumMass, carried rather than recomputed at run time: it\n";
print $o "/// recurses over the daughters' own tables with a 0.10 branching-ratio threshold and a\n";
print $o "/// fallback to the single most probable channel. The port's own recursion is checked\n";
print $o "/// against this column in tests/test_bic_imr.cu rather than trusted.\n";
emit_dbl("decay_species_min_mass", \@minm, 4);
emit_int("decay_species_first_channel", \@first, 16);
emit_int("decay_species_n_channels", \@nch, 16);
emit_dbl("decay_channel_br", \@cbr, 4);
emit_int("decay_channel_n_daughters", \@cnd, 16);
print $o "/// FOUR slots per channel, zero where unused. Only f2(1270) uses the fourth, and it is\n";
print $o "/// in the closure because FTFP produces it - docs/HADRONIC_PLAN.md section 9.3.\n";
emit_int("decay_channel_daughters", \@cd, 12);

print $o "}  // namespace g4gpu::bic::imr\n\n#endif\n";
close $o;
printf "wrote %s: %d species, %d channels, max %d channels on one species\n", $out, $nsp,
    $nchan_total, $max_chan;
