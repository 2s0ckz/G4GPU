// A recorded trajectory must be the path the track actually took.
//
// WHY THIS EXISTS
//
// Every other check in this pipeline reads a number - a dose, a step count, a sigma, a ratio.
// None of them reads the SHAPE of anything, and that gap had something in it: charged tracks
// were recorded with a gap at every step, because the segment was written before multiple
// scattering displaced the track and the next step began from the displaced position. Segment N
// ended at A, segment N+1 began at A + d, and a charged trajectory drew as a row of dashes.
//
// It was found by someone looking at the viewer, not by anything here, and it could not have
// been found here: a segment carried two endpoints and a colour, so segments could not be
// attributed to tracks and "does this track's path join up" was not a question the data could
// answer. The track id exists so that it can be. See docs/RISK.md V13.
//
// WHY EXACT EQUALITY IS THE RIGHT TEST
//
// `pos_before` of one step is literally the same double as `p.pos` at the end of the previous
// step, so their float conversions are bit-identical. A chained path is therefore exactly
// chained, and a tolerance would only hide the class of error this is for.
#include <cuda_runtime.h>

#include <cstdio>
#include <map>
#include <vector>

#include "g4/G4RunManager.hh"
#include "render/trajectory.cuh"
#include "scenes/scene_registry.hh"

using namespace g4gpu;

namespace {

int g_fails = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what);
    ++g_fails;
  }
}

/// A point, compared bitwise. Float, because that is what the store holds.
struct Point {
  float x, y, z;
  bool operator<(const Point& o) const {
    if (x != o.x) { return x < o.x; }
    if (y != o.y) { return y < o.y; }
    return z < o.z;
  }
};

/// One track's segments, as the multiset of points they start and end at.
struct Path {
  std::map<Point, int> starts;
  std::map<Point, int> ends;
  int n = 0;
  unsigned char kind = 0;
};

}  // namespace

int main() {
  std::printf("== a recorded trajectory is a connected path ==\n\n");

  auto* rm = new G4RunManager;
  const auto it = scenes::Registry().find("B1");
  if (it == scenes::Registry().end()) {
    std::printf("  FAIL: no B1 scene registered\n");
    return 1;
  }
  it->second(rm);
  rm->Initialize();

  // Enough capacity that nothing is dropped for the events captured; a truncated sample would
  // make every unfinished track look like a broken one, so `dropped` is checked rather than
  // assumed.
  vis::TrajectoryBuffer traj{};
  const cudaError_t alloc = vis::allocate_trajectory(traj, 4 << 20, /*max_event=*/200);
  if (alloc != cudaSuccess) {
    std::printf("  FAIL: could not allocate the capture buffer: %s\n",
                cudaGetErrorString(alloc));
    return 1;
  }

  std::vector<double> sums, sums_sq;
  rm->RunEvents(2000, sums, sums_sq, traj);

  int n = 0, dropped = 0;
  cudaMemcpy(&n, traj.count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&dropped, traj.dropped, sizeof(int), cudaMemcpyDeviceToHost);
  Check(dropped == 0, "no segment was dropped, so the captured paths are complete");
  Check(n > 1000, "the run recorded a substantial number of segments");
  if (n > traj.capacity) { n = traj.capacity; }

  std::vector<float> x0(n), y0(n), z0(n), x1(n), y1(n), z1(n);
  std::vector<unsigned int> tid(n);
  std::vector<unsigned char> kind(n);
  cudaMemcpy(x0.data(), traj.x0, sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(y0.data(), traj.y0, sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(z0.data(), traj.z0, sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(x1.data(), traj.x1, sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(y1.data(), traj.y1, sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(z1.data(), traj.z1, sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(tid.data(), traj.track, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(kind.data(), traj.kind, n, cudaMemcpyDeviceToHost);

  // Segments arrive in whatever order threads reached the atomic, so the track id is the only
  // thing that groups them - which is the whole reason it is stored.
  std::map<unsigned int, Path> paths;
  for (int i = 0; i < n; ++i) {
    Path& p = paths[tid[i]];
    p.starts[Point{x0[i], y0[i], z0[i]}] += 1;
    p.ends[Point{x1[i], y1[i], z1[i]}] += 1;
    p.kind = kind[i];
    ++p.n;
  }

  // A path visits each interior point twice - once arriving, once leaving - so cancelling ends
  // against starts leaves exactly one unmatched start (the origin) and one unmatched end (where
  // the track stopped). Anything else is a discontinuity.
  //
  // Counted per CHARGE CLASS as well as in total, because the bug this is for hit exactly one
  // of them: neutral tracks have no multiple scattering and heavy charged ones apply no
  // lateral displacement, so a check reading only the total would have been dominated by the
  // classes that were fine.
  long long broken[3] = {0, 0, 0};
  long long total[3] = {0, 0, 0};
  long long worst_gap_track = 0;
  for (const auto& kv : paths) {
    const Path& p = kv.second;
    int unmatched_starts = 0;
    for (const auto& s : p.starts) {
      const auto e = p.ends.find(s.first);
      const int matched = (e == p.ends.end()) ? 0 : e->second;
      const int extra = s.second - matched;
      if (extra > 0) { unmatched_starts += extra; }
    }
    const int k = (p.kind < 3) ? p.kind : 0;
    ++total[k];
    // One unmatched start is the track's origin. More than one means the path is in pieces.
    if (unmatched_starts != 1) {
      ++broken[k];
      if (unmatched_starts > worst_gap_track) { worst_gap_track = unmatched_starts; }
    }
  }

  const char* name[3] = {"neutral", "negative", "positive"};
  std::printf("  %d segments over %zu tracks, %d dropped\n", n, paths.size(), dropped);
  for (int k = 0; k < 3; ++k) {
    std::printf("    %-9s %lld tracks, %lld with a break\n", name[k], total[k], broken[k]);
  }

  Check(total[1] > 100, "the run produced negative tracks to check, not just neutral ones");
  for (int k = 0; k < 3; ++k) {
    if (broken[k] != 0) {
      std::printf("  FAIL: %lld of %lld %s tracks are recorded as disconnected pieces.\n"
                  "        A segment must end where the next one starts. The worst has %lld\n"
                  "        loose ends. This is what recording a step before multiple\n"
                  "        scattering displaces the track looks like - docs/RISK.md V13.\n",
                  broken[k], total[k], name[k], worst_gap_track);
      ++g_fails;
    }
  }

  vis::free_trajectory(traj);
  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
