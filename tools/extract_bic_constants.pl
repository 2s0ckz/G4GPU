#!/usr/bin/perl
# Checks the binary cascade's unobservable constants against the Geant4 11.1.1 SOURCE.
#
#   perl tools/extract_bic_constants.pl
#
# Every other number in this package is compared against a number Geant4 PRODUCES, through
# ref/oracle/bic_*.csv. These cannot be. `theBCminP`, `theCutOnP` and `theCutOnPAbsorb` are
# private members of G4BinaryCascade with no getters and no setters; what they decide - which
# nucleons `Capture()` moves into `theCapturedList` - is not exposed either. The nuclear
# fields' optical coefficients are default arguments to constructors that
# `G4RKPropagation::Init` calls without them, and `GetCoeff()` is only overridden on the
# classes this package refuses. The field table's 0.3 fm step and the driver's safety factor
# are observable only through a trajectory, where a wrong value would look like a wrong force.
#
# So a test that compared the port's copy of these against a literal in the test would be
# comparing a copy with itself, which is exactly the failure docs/RISK.md V52 is about. This
# script is the check instead: it reads the source lines and asserts the set is EXACTLY this
# one, so a Geant4 release that changes 90 MeV to 80 MeV fails here loudly rather than being
# transcribed as though nothing had happened. docs/RISK.md V41 is where the form comes from -
# "assert the known set is exactly this set", not "skip the rows you know about".
#
# It asserts, it does not generate. There is no table to emit: the port's copies live in
# src/physics/hadronic/bic/bic_params.cuh, nuclear_field.cuh, rk_propagation.cuh and
# nucleus/*.cuh with the Geant4 file named above each, and this script checks the other side of
# that naming.
use strict;
use warnings;

my $g4 = $ENV{G4SRC} || 'D:/Documents/Geant4/Windows/geant4-v11.1.1';
my $bc = "$g4/source/processes/hadronic/models/binary_cascade";
my $util = "$g4/source/processes/hadronic/util";
my $mf = "$g4/source/geometry/magneticfield";
my $pl = "$g4/source/physics_lists/constructors";

my $fails = 0;
my $checks = 0;

sub slurp {
  my ($path) = @_;
  open my $fh, '<', $path or die "cannot read $path: $!\n";
  local $/;
  my $s = <$fh>;
  close $fh;
  return $s;
}

# Asserts that $re matches $text exactly $want times, and reports the captures so that a
# release which keeps the pattern and changes the number is caught by the capture and not only
# by the count.
sub want_lines {
  my ($label, $text, $re, $want, @expect) = @_;
  ++$checks;
  my @got;
  while ($text =~ /$re/g) {
    push @got, defined($1) ? $1 : '(no capture)';
  }
  if (scalar(@got) != $want) {
    printf "FAIL %-52s %d matches, expected %d\n", $label, scalar(@got), $want;
    ++$fails;
    return;
  }
  if (@expect) {
    for my $i (0 .. $#expect) {
      if ($got[$i] ne $expect[$i]) {
        printf "FAIL %-52s match %d is '%s', expected '%s'\n", $label, $i, $got[$i],
               $expect[$i];
        ++$fails;
        return;
      }
    }
  }
  printf "  ok %-52s %s\n", $label, join(', ', @got);
}

# --------------------------------------------------------------------------------------------
# G4BinaryCascade: the three cuts, the energy window, the check levels
# --------------------------------------------------------------------------------------------
my $bic = slurp("$bc/src/G4BinaryCascade.cc");

want_lines('theBCminP', $bic, qr/theBCminP\s*=\s*([\d.]+)\*MeV/, 1, '45');
# Five assignments, not four: the constructor's 90 MeV and then Propagate's own 90, 70, 50, 45.
# Propagate re-sets the default before overriding it, so the constructor's value is dead for
# any nucleus that reaches Propagate - which is every nucleus, because ApplyYourself calls it.
want_lines('every theCutOnP assignment', $bic, qr/theCutOnP\s*=\s*([\d.]+)\*MeV;/, 5,
           '90', '90', '70', '50', '45');
want_lines('theCutOnPAbsorb', $bic, qr/theCutOnPAbsorb\s*=\s*([\d.]+)\*MeV/, 1, '0');
want_lines('SetMinEnergy / SetMaxEnergy', $bic,
           qr/Set(?:Min|Max)Energy\(\s*([\d.]+)\*GeV\s*\)/, 2, '0.0', '10.1');
want_lines('SetEnergyMomentumCheckLevels', $bic,
           qr/SetEnergyMomentumCheckLevels\(([\d.]+\*perCent, [\d.]+\*MeV)\)/, 1,
           '1.0*perCent, 1.0*MeV');

# The four theCutOnP lines in Propagate, and the thing that makes them a finding: the test is
# against GetMass(), a mass in MeV, and not GetMassNumber(). Every nucleus has a mass above
# 120 MeV, so the first three assignments are dead and theCutOnP is always 45 MeV.
# docs/RISK.md V72. Asserting BOTH the thresholds and the accessor, so that a release which
# corrects the accessor fails here rather than quietly changing the capture rate.
want_lines('theCutOnP by nucleus MASS, not mass number', $bic,
           qr/the3DNucleus->(GetMass)\(\)\s*>\s*(?:30|60|120)\)\s*theCutOnP\s*=/, 3,
           'GetMass', 'GetMass', 'GetMass');
want_lines('theCutOnP thresholds', $bic,
           qr/the3DNucleus->GetMass\(\)\s*>\s*(\d+)\)\s*theCutOnP\s*=/, 3, '30', '60', '120');
# Two matches: the live condition and the commented-out `particlesAboveCut==0 && ...` form
# above it, which is the shape the author first wrote. Both are pinned so a release that
# restores the commented one is visible here.
want_lines('Capture threshold is 0.2*theCutOnP', $bic,
           qr/capturedEnergy\/particlesBelowCut\s*<\s*([\d.]+)\*theCutOnP/, 2, '0.2', '0.2');
# `particlesAboveCut` is declared, initialised to 0, incremented only inside a commented-out
# block, and read only in a commented-out condition and a verbose printout. If a release brings
# it to life the capture condition changes, so its deadness is asserted rather than assumed.
want_lines('particlesAboveCut is never incremented', $bic,
           qr/^\s*\+\+particlesAboveCut;/m, 0);
want_lines('both collision loop budgets', $bic,
           qr/collisionLoopMaxCount\s*=\s*(\d+);/, 2, '200', '1000000');
want_lines('GetSpherePoint radius factor', $bic,
           qr/GetSpherePoint\(([\d.]+)\*radius, initial4Momentum\)/, 1, '1.1');
want_lines('GetSpherePoint 1.5 mom.unit()', $bic,
           qr/x2\*o2\.unit\(\) - ([\d.]+)\* mom\.unit\(\)/, 1, '1.5');
want_lines('the outer radius + 3 fermi in ApplyYourself', $bic,
           qr/GetOuterRadius\(\)\+(\d+)\*fermi/, 1, '3');

# --------------------------------------------------------------------------------------------
# G4BinaryLightIonReaction
# --------------------------------------------------------------------------------------------
my $blir = slurp("$bc/src/G4BinaryLightIonReaction.cc");
want_lines('the fusion threshold per nucleon', $blir,
           qr/\(mom\.t\(\)-mom\.mag\(\)\)\/pA\s*<\s*([\d.]+)\*MeV/, 1, '50');
want_lines('EnergyAndMomentumCorrector attempts', $blir,
           qr/nAttemptScale = (\d+)/, 1, '2500');
want_lines('EnergyAndMomentumCorrector ErrLimit', $blir,
           qr/ErrLimit = ([\dE.\-+]+)/, 1, '1.E-6');
want_lines('the 150-try Interact loop', $blir, qr/tryCount<\s*(\d+)\)/, 1, '150');
want_lines('the impact-parameter offset', $blir,
           qr/pos\(aX, aY, -([\d.]+)\*impactMax-([\d.]+)\*fermi\)/, 1, '2.');
want_lines('the 10 MeV momentum-balance loop', $blir,
           qr/std::abs\(momentum\.e\(\)-pspectators\.e\(\)\) > (\d+)\*MeV/, 1, '10');
want_lines('the 10 keV spectator momentum test', $blir,
           qr/momentum\.vect\(\)\.mag\(\) - momentum\.e\(\)\s*[<>]\s*(\d+)\*keV/, 2, '10', '10');

# --------------------------------------------------------------------------------------------
# The nuclear fields
# --------------------------------------------------------------------------------------------
want_lines('G4VNuclearField radius = OuterRadius + 4 fermi',
           slurp("$bc/src/G4VNuclearField.cc"),
           qr/radius\(aNucleus->GetOuterRadius\(\) \+ (\d+)\*fermi\)/, 1, '4');

for my $pair (['G4PionPlusField', '0.042'], ['G4PionMinusField', '0.042'],
              ['G4PionZeroField', '0.042'], ['G4AntiProtonField', '1.53'],
              ['G4KaonPlusField', '0.35'], ['G4KaonMinusField', '0.35'],
              ['G4KaonZeroField', '0.35'], ['G4SigmaPlusField', '0.36'],
              ['G4SigmaMinusField', '0.36'], ['G4SigmaZeroField', '0.36']) {
  my ($cls, $coeff) = @$pair;
  want_lines("$cls optical coefficient", slurp("$bc/include/$cls.hh"),
             qr/G4double coeff = ([\d.]+)\*CLHEP::fermi/, 1, $coeff);
}

# The pion fields' nucleus mass has the binding energy ADDED where a nuclear mass subtracts it.
# Asserted as written for all three plus G4KM_OpticalEqRhs, because it is reproduced rather
# than corrected and a release that fixes it changes four answers at once. docs/RISK.md V70.
for my $cls ('G4PionPlusField', 'G4PionMinusField', 'G4PionZeroField') {
  want_lines("$cls nucleusMass sign", slurp("$bc/src/$cls.cc"),
             qr/neutron_mass_c2\s*([+-])\s*bindingEnergy;/, 1, '+');
}
want_lines('G4KM_OpticalEqRhs nucleusMass sign', slurp("$bc/src/G4KM_OpticalEqRhs.cc"),
           qr/neutron_mass_c2\s*([+-])\s*bindingEnergy;/, 1, '+');

for my $cls ('G4ProtonField', 'G4NeutronField') {
  my $t = slurp("$bc/src/$cls.cc");
  want_lines("$cls table step", $t, qr/aR\+=([\d.]+)\*fermi/, 1, '0.3');
  want_lines("$cls table extent", $t, qr/=\s*([\d.]+)\*theNucleus->GetOuterRadius\(\)/, 1, '2.');
  want_lines("$cls GetField fallback bound", $t,
             qr/\(index\+(\d)\) > theFermiMomBuffer\.size\(\)/, 1, '2');
  want_lines("$cls GetField interpolation step", $t, qr/x1 = \(([\d.]+)\*fermi\)\*index/, 1,
             '0.3');
}
# The two `G4ThreeVector aPosition` locals that are built and never read in the last two blocks
# of both constructors. Their presence is what shows the two zeros were meant to be evaluated;
# if a release evaluates them the field's tail changes.
for my $cls ('G4ProtonField', 'G4NeutronField') {
  want_lines("$cls has two unread aPosition locals", slurp("$bc/src/$cls.cc"),
             qr/G4ThreeVector aPosition\(0,0,(?:theR(?:adius)?\+0\.001\*fermi|1\.\*m)\);\n\s*theFermiMomBuffer\.push_back\(0\)/,
             2);
}
want_lines('G4ProtonField barrier has a zero bindingEnergy', slurp("$bc/src/G4ProtonField.cc"),
           qr/G4double bindingEnergy\s*=(\d+);/, 1, '0');

# --------------------------------------------------------------------------------------------
# The two equations of motion and the propagator
# --------------------------------------------------------------------------------------------
my $neq = slurp("$bc/src/G4KM_NucleonEqRhs.cc");
want_lines('G4KM_NucleonEqRhs factor', $neq,
           qr/factor = hbarc\*hbarc\*G4Pow::GetInstance\(\)->A23\((\d+)\.\*pi2\*A\)\/(\d+)\./, 1,
           '3');
# Two matches, and the pair IS the finding: the commented-out `-deriv` the author first wrote
# and the live `+deriv` three lines below it, opposite in sign. Both are pinned.
want_lines('G4KM_NucleonEqRhs force sign', $neq, qr/dydx\[3\] = yMod == 0 \? 0 : ([+-]?)deriv/, 2,
           '-', '');
my $oeq = slurp("$bc/src/G4KM_OpticalEqRhs.cc");
want_lines('G4KM_OpticalEqRhs force sign', $oeq, qr/dydx\[3\] = yMod == 0 \? 0 : (-)deriv/, 1,
           '-');

my $rk = slurp("$bc/src/G4RKPropagation.cc");
want_lines('FieldTransport hMin', $rk, qr/hMin = ([\dEe.\-+]+)\*second/, 1, '1.0e-25');
want_lines('FieldTransport eps', $rk, qr/G4double eps = ([\d.]+);/, 1, '0.01');
want_lines('the stepper is G4ClassicalRK4', $rk, qr/new (G4ClassicalRK4)\(equation\)/, 1,
           'G4ClassicalRK4');
want_lines('GetSphereIntersectionTimes safety', $rk,
           qr/theOuterRadius \+ (\d+)\*fermi; \/\/ "safety"/, 1, '3');
want_lines('the free-transport overshoot factor', $rk, qr/currTimeStep = t_leave\*([\d.]+)/, 1,
           '1.05');
want_lines('the cannot-enter overshoot factor', $rk, qr/FreeTransport\(kt, ([\d.]+)\*t_leave\)/,
           1, '1.1');

# --------------------------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------------------------
my $drv = slurp("$mf/include/G4MagIntegratorDriver.hh");
want_lines('fMaxStepBase', $drv, qr/fMaxStepBase = (\d+);/, 1, '250');
want_lines('fSmallestFraction', $drv, qr/fSmallestFraction = ([\dEe.\-+]+);/, 1, '1.0e-12');
want_lines('fMinNoVars', $drv, qr/fMinNoVars = (\d+);/, 1, '12');
want_lines('ReSetParameters default safety',
           slurp("$mf/include/G4MagIntegratorDriver.icc"),
           qr/ReSetParameters\(G4double new_safety = ([\d.]+)\)/, 0);
want_lines('ReSetParameters default safety (header)', $drv,
           qr/ReSetParameters\(G4double new_safety = ([\d.]+)\)/, 1, '0.9');
want_lines('max_stepping_increase', slurp("$mf/include/G4VIntegrationDriver.hh"),
           qr/max_stepping_increase = (\d+)/, 1, '5');
want_lines('G4ClassicalRK4 order', slurp("$mf/include/G4ClassicalRK4.hh"),
           qr/IntegratorOrder\(\) const \{ return (\d+); \}/, 1, '4');
want_lines('G4FieldTrack::ncompSVEC', slurp("$mf/include/G4FieldTrack.hh"),
           qr/ncompSVEC = (\d+)/, 1, '12');

# --------------------------------------------------------------------------------------------
# The nucleus model
# --------------------------------------------------------------------------------------------
my $f3d = slurp("$util/src/G4Fancy3DNucleus.cc");
# Three: the constructor's initialiser `nucleondistance(0.8*fermi)`, Init's re-assignment of
# the same value, and the A == 12 override. The character class allows `=`, `(` and spaces so
# that the initialiser-list form is caught too.
want_lines('nucleondistance', $f3d, qr/nucleondistance[\s=(]*([\d.]+)\*fermi/, 3,
           '0.8', '0.8', '0.9');
want_lines('the shell-model / Fermi dispatch', $f3d, qr/if \( myA < (\d+) \)/, 1, '17');
want_lines('CenterNucleons is called for A == 12 only', $f3d,
           qr/if\( myA == (\d+) \) CenterNucleons\(\)/, 1, '12');
want_lines('ChoosePositions maxR relative density', $f3d,
           qr/maxR=GetNuclearRadius\(([\d.]+)\)/, 1, '0.001');
want_lines('ChoosePositions attempt budget', $f3d, qr/interationsLeft=(\d+)\*myA/, 1, '1000');
want_lines('the flat block size and refill threshold', $f3d,
           qr/jr=std::min\((\d+),(\d+)\*\(myA - i\)\)/, 1, '600');
want_lines('the C12 cluster base length', $f3d, qr/Lbase=([\d.]+)\*fermi/, 1, '3.05');
want_lines('the C12 cluster dispersion', $f3d, qr/Disp=([\d.]+);/, 1, '0.552');
want_lines('the C12 triangle 0.866', $f3d, qr/Lbase\*([\d.]+), 0\.\)/, 1, '0.866');
want_lines('the C12 cluster attempt budget', $f3d, qr/loopCounterLeft = (\d+);/, 3,
           '10000', '10000', '10000');
want_lines('the ChooseFermiMomenta retry loop runs once', $f3d,
           qr/for \(G4int ntry=0; ntry<(\d+) ; ntry \+\+ \)/, 1, '1');
want_lines('CoulombBarrier cfactor', $f3d, qr/cfactor = \((1\.44\/1\.14)\) \* MeV/, 1,
           '1.44/1.14');
want_lines('GetNuclearRadius default relative density', $f3d,
           qr/return GetNuclearRadius\(([\d.]+)\)/, 1, '0.5');
want_lines('GetMass subtracts the binding energy', $f3d,
           qr/\(myA-myZ\)\*G4Neutron::Neutron\(\)->GetPDGMass\(\) (-)\n\s*BindingEnergy/, 1, '-');

want_lines('G4NuclearFermiDensity surface thickness',
           slurp("$util/src/G4NuclearFermiDensity.cc"), qr/a\(([\d.]+) \* fermi\)/, 1, '0.545');
want_lines('G4NuclearFermiDensity r0', slurp("$util/src/G4NuclearFermiDensity.cc"),
           qr/r0 = ([\d.]+) \* \(1\. - ([\d.]+)\/\(a13\*a13\)\) \* fermi/, 1, '1.16');
want_lines('G4NuclearFermiDensity 40*theR cut-off',
           slurp("$util/include/G4NuclearFermiDensity.hh"),
           qr/currentR > (\d+)\*theR/, 1, '40');
want_lines('G4NuclearShellModelDensity r0sq',
           slurp("$util/src/G4NuclearShellModelDensity.cc"), qr/r0sq=([\d.]+)\*fermi\*fermi/, 1,
           '0.8133');

# The one line that makes every nucleon off-shell, and the commented-out alternative beside it.
want_lines('ChooseFermiMomenta energy = m - BE/A', $f3d,
           qr/energy = theNucleons\[i\]\.GetParticleType\(\)->GetPDGMass\(\)\n\s*(-) BindingEnergy\(\)\/myA;/,
           1, '-');
want_lines('the commented-out SetBindingEnergy alternative', $f3d,
           qr/\/\/theNucleons\[i\]\.SetBindingEnergy\(/, 1);

# G4KineticTrack's Fermi momentum is loaded and then discarded. docs/RISK.md V69.
my $kt = slurp("$util/src/G4KineticTrack.cc");
want_lines('the nucleon constructor loads theFermi3Momentum', $kt,
           qr/theFermi3Momentum\((nucleon->GetMomentum\(\))\)/, 1, 'nucleon->GetMomentum()');
want_lines('and its body calls Set4Momentum, which zeroes it', $kt,
           qr/theFermi3Momentum\.setE\(0\);\n\s*(Set4Momentum)\(a4Momentum\);/, 1,
           'Set4Momentum');
want_lines('Set4Momentum zeroes theFermi3Momentum', slurp("$util/include/G4KineticTrack.hh"),
           qr/theFermi3Momentum=(G4LorentzVector\(0\));/, 1, 'G4LorentzVector(0)');

# --------------------------------------------------------------------------------------------
# The physics lists' energy windows
# --------------------------------------------------------------------------------------------
my $qbbc = slurp("$pl/hadron_inelastic/src/G4HadronInelasticQBBC.cc");
want_lines('QBBC emaxBic', $qbbc, qr/emaxBic\s*=\s*([\d.]+)\*CLHEP::GeV/, 1, '1.5');
want_lines('QBBC eminBert', $qbbc, qr/eminBert\s*=\s*([\d.]+)\*CLHEP::GeV/, 1, '1.0');
want_lines('QBBC gives BIC no SetMinEnergy', $qbbc, qr/theBIC->SetMinEnergy/, 0);
want_lines('QBBC registers BIC four times', $qbbc, qr/(RegisterMe\(theBIC\));/, 4,
           'RegisterMe(theBIC)', 'RegisterMe(theBIC)', 'RegisterMe(theBIC)',
           'RegisterMe(theBIC)');
my $ion = slurp("$pl/ions/src/G4IonPhysics.cc");
want_lines('G4IonPhysics theIonBC SetMinEnergy', $ion,
           qr/theIonBC->SetMinEnergy\( ([\d.]+) \)/, 1, '0.0');
want_lines('G4IonPhysics theIonBC SetMaxEnergy', $ion,
           qr/theIonBC->SetMaxEnergy\( G4HadronicParameters::Instance\(\)->(GetMaxEnergyTransitionFTF_Cascade)\(\) \)/,
           1, 'GetMaxEnergyTransitionFTF_Cascade');
want_lines('G4EnergyRangeManager divides by the baryon number',
           slurp("$g4/source/processes/hadronic/management/src/G4EnergyRangeManager.cc"),
           qr/kineticEnergy \/= static_cast< G4double >\( std::abs\( aHadProjectile\.GetDefinition\(\)->(GetBaryonNumber)\(\) \) \)/,
           1, 'GetBaryonNumber');

printf "\nextract_bic_constants: %d checks, %d failed\n", $checks, $fails;
exit($fails == 0 ? 0 : 1);
