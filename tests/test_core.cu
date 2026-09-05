#include <cstdio>
#include <cmath>
#include <vector>
#include "core/rng.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "core/secondary_pool.cuh"
#include "physics/em/klein_nishina.cuh"

using namespace g4gpu;
using real_t = double;

int fails = 0;
void check(bool ok, const char* what) {
  if (!ok) { printf("  FAIL: %s\n", what); ++fails; } else { printf("  ok:   %s\n", what); }
}

int main() {
  printf("== RNG ==\n");
  Philox<real_t> rng(1, 7);
  double sum = 0; bool in_range = true;
  for (int i = 0; i < 100000; ++i) { double u = rng.uniform(); sum += u; if (u <= 0 || u >= 1) in_range = false; }
  check(in_range, "uniform() strictly inside (0,1)");
  check(std::fabs(sum / 100000.0 - 0.5) < 0.005, "uniform() mean ~0.5");
  Philox<real_t> a(3, 4), b(3, 4);
  check(a.uniform() == b.uniform(), "same (event,track) reproduces stream");
  Philox<real_t> c(3, 5);
  Philox<real_t> d(3, 4);
  check(c.uniform() != d.uniform(), "different track decorrelates stream");

  printf("== vec3 ==\n");
  Vec3<real_t> zhat{0, 0, 1};
  Vec3<real_t> v{0.3, 0.4, std::sqrt(1 - 0.25)};
  Vec3<real_t> r = rotate_uz(v, zhat);
  check(std::fabs(r.x - v.x) < 1e-15 && std::fabs(r.z - v.z) < 1e-15, "rotate_uz about +z is identity");
  Vec3<real_t> mz{0, 0, -1};
  Vec3<real_t> r2 = rotate_uz(v, mz);
  check(std::fabs(r2.x + v.x) < 1e-15 && std::fabs(r2.z + v.z) < 1e-15, "rotate_uz about -z flips x,z");
  check(std::fabs(mag(normalize(Vec3<real_t>{3, 4, 12})) - 1.0) < 1e-15, "normalize gives unit length");
  Vec3<real_t> u = normalize(Vec3<real_t>{1, 2, 3});
  check(std::fabs(mag(rotate_uz(v, u)) - mag(v)) < 1e-14, "rotate_uz preserves length");

  printf("== units ==\n");
  // water: 1 g/cm3, 18 g/mol -> 3.35e19 molecules/mm3
  double n = units::number_density<real_t>(1.0, 18.0);
  check(std::fabs(n / 3.3456e19 - 1.0) < 1e-3, "number_density(water) ~3.35e19 /mm^3");

  printf("== Klein-Nishina ==\n");
  const int cap = 4096;
  TrackSoA<real_t> st{};
  std::vector<real_t> X(cap), Y(cap), Z(cap), DX(cap), DY(cap), DZ(cap), EK(cap);
  std::vector<int> PT(cap), VO(cap), EV(cap), BI(cap);
  std::vector<uint8_t> AL(cap);
  int used = 0, ovf = 0;
  st.x=X.data(); st.y=Y.data(); st.z=Z.data(); st.dx=DX.data(); st.dy=DY.data(); st.dz=DZ.data();
  st.ekin=EK.data(); st.particle=PT.data(); st.volume=VO.data(); st.event=EV.data();
  st.birth=BI.data(); st.alive=AL.data(); st.capacity=cap; st.n_used=&used; st.n_overflow=&ovf;

  const real_t E0 = 6.0;  // MeV, the B1 primary energy
  bool energy_ok = true, dir_ok = true;
  real_t max_e1 = 0, min_e1 = 1e9;
  for (int i = 0; i < 20000; ++i) {
    used = 0;
    SecondaryEmitter<real_t> emitter{&st, {0,0,0}, 0, 0, 0};
    Philox<real_t> g(2, i);
    auto res = em::sample_klein_nishina<real_t>(E0, Vec3<real_t>{0,0,1}, g, emitter, 0, 1e-3, 1e-6);
    const real_t e_sec = (used > 0) ? EK[0] : real_t(0);
    const real_t total = res.gamma_energy + e_sec + res.local_deposit;
    if (std::fabs(total - E0) > 1e-9) energy_ok = false;
    if (std::fabs(mag(res.gamma_dir) - 1.0) > 1e-12) dir_ok = false;
    if (res.gamma_energy > max_e1) max_e1 = res.gamma_energy;
    if (res.gamma_energy < min_e1) min_e1 = res.gamma_energy;
  }
  check(energy_ok, "energy conserved exactly per interaction");
  check(dir_ok, "scattered gamma direction stays unit");
  // Backscatter limit: E1_min = E0/(1+2*E0/m_e)
  const real_t e1_min_analytic = E0 / (1 + 2 * E0 / units::electron_mass_c2<real_t>());
  printf("  E1 range sampled: [%.4f, %.4f] MeV; analytic min %.4f, max %.4f\n",
         min_e1, max_e1, e1_min_analytic, E0);
  check(min_e1 >= e1_min_analytic - 1e-6 && max_e1 <= E0 + 1e-9,
        "E1 within [E0/(1+2E0/me), E0] Compton bounds");

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails;
}
