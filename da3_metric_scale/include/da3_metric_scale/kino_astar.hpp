// Kinodynamic (hybrid-state) A* for a quadrotor, following Zhou et al.,
// "Robust and Efficient Quadrotor Trajectory Generation for Fast Autonomous Flight" (RA-L 2019), Sec. III.
//
// State  : x = [p, v]   (double integrator, n = 2)
// Input  : u = a        (constant over each primitive, each axis in [-max_acc, max_acc])
// Edge   : motion primitive of duration tau, (2r+1)^3 discretised inputs
// Cost   : J = integral(|u|^2) dt + rho * T
// Heur.  : closed-form optimal unconstrained trajectory to the goal (Pontryagin), quartic in T
// Prune  : one node per voxel (lowest f)
// Finish : goal reached, analytic expansion succeeds, or the planning horizon is hit (receding horizon)
//
// Dependencies: Eigen3 only. Obstacle info comes through the `is_free` callback, so any map
// (nvblox ESDF, occupancy grid, ...) can be plugged in.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cmath>
#include <functional>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

namespace kino {

struct Params {
  double max_vel = 3.0;      // v_max, per axis [m/s]
  double max_acc = 2.0;      // u_max, per axis [m/s^2]
  int r = 2;                 // input discretisation -> (2r+1)^3 primitives per node
  double tau = 0.5;          // primitive duration [s]
  double rho = 1.0;          // weight on time in J. Rest-to-rest over distance d has T* = (36 d^2 / rho)^(1/4);
                             // pick rho so the analytic shot's peak vel/acc (~1.5 d/T, ~6 d/T^2) fit max_vel/max_acc,
                             // otherwise the analytic expansion is rejected and the search floods the map
  double resolution = 0.1;   // voxel size used for pruning / closed set [m]
  double horizon = 7.0;      // stop once a node is this far from the start [m]
  double goal_tol = 0.2;     // position tolerance for "reached goal" [m]
  double check_dt = 0.05;    // sampling step for collision / limit checks [s]
  double sample_dt = 0.1;    // sampling step of the returned trajectory [s]
  double lambda_heu = 1.0;   // heuristic inflation (>1 = faster, less optimal)
  double tie_breaker = 1.0 + 1.0 / 10000.0;
  int max_nodes = 100000;    // safety cap on node expansions
};

struct State {
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();  // input applied from this sample onward
  double t = 0.0;
};

enum class Status { REACH_GOAL, REACH_HORIZON, NO_PATH };

struct Result {
  Status status = Status::NO_PATH;
  std::vector<State> traj;  // sampled every sample_dt
  double cost = 0.0;        // g of the last search node (excludes analytic tail)
  int expanded = 0;
};

class KinodynamicAStar {
 public:
  using FreeFn = std::function<bool(const Eigen::Vector3d&)>;  // true if the point is collision-free

  KinodynamicAStar(const Params& params, FreeFn is_free) : P_(params), is_free_(std::move(is_free)) {}

  Result search(const Eigen::Vector3d& start_pos, const Eigen::Vector3d& start_vel,
                const Eigen::Vector3d& goal_pos, const Eigen::Vector3d& goal_vel = Eigen::Vector3d::Zero()) {
    Result res;
    pool_.clear();
    nodes_.clear();
    start_pos_ = start_pos;
    goal_pos_ = goal_pos;
    goal_vel_ = goal_vel;

    std::priority_queue<QEntry, std::vector<QEntry>, std::greater<QEntry>> open;

    Node* s = newNode();
    s->pos = start_pos;
    s->vel = start_vel;
    s->idx = toIdx(start_pos);
    s->g = 0.0;
    double T;
    s->f = P_.lambda_heu * P_.tie_breaker * heuristic(s->pos, s->vel, goal_pos_, goal_vel_, T);
    nodes_[s->idx] = s;
    open.push({s->f, s});

    // Pre-build the discretised input set.
    std::vector<Eigen::Vector3d> inputs = makeInputs();

    while (!open.empty() && res.expanded < P_.max_nodes) {
      QEntry e = open.top();
      open.pop();
      Node* cur = e.node;
      if (cur->closed || e.f > cur->f + 1e-9) continue;  // stale queue entry
      cur->closed = true;
      ++res.expanded;

      const double dist_goal = (cur->pos - goal_pos_).norm();

      // 1) reached the goal region
      if (dist_goal < P_.goal_tol) {
        res.traj = retrieve(cur, nullptr);
        res.status = Status::REACH_GOAL;
        res.cost = cur->g;
        return res;
      }

      // 2) analytic expansion straight to the goal state
      if (dist_goal <= P_.horizon) {
        std::vector<State> tail;
        if (analyticExpand(*cur, tail)) {
          res.traj = retrieve(cur, &tail);
          res.status = Status::REACH_GOAL;
          res.cost = cur->g;
          return res;
        }
      }

      // 3) receding horizon: goal is far, stop at the edge of the known region
      if ((cur->pos - start_pos_).norm() > P_.horizon) {
        res.traj = retrieve(cur, nullptr);
        res.status = Status::REACH_HORIZON;
        res.cost = cur->g;
        return res;
      }

      expand(cur, inputs, open);
    }
    return res;  // NO_PATH
  }

 private:
  struct Node {
    Eigen::Vector3d pos, vel;
    Eigen::Vector3i idx;
    Eigen::Vector3d u = Eigen::Vector3d::Zero();  // input of the primitive that led here
    double dur = 0.0;
    double g = 0.0, f = 0.0;
    Node* parent = nullptr;
    bool closed = false;
  };

  struct QEntry {
    double f;
    Node* node;
    bool operator>(const QEntry& o) const { return f > o.f; }
  };

  struct IdxHash {
    size_t operator()(const Eigen::Vector3i& v) const {
      return (static_cast<size_t>(v.x()) * 73856093u) ^ (static_cast<size_t>(v.y()) * 19349663u) ^
             (static_cast<size_t>(v.z()) * 83492791u);
    }
  };

  using NodeMap = std::unordered_map<Eigen::Vector3i, Node*, IdxHash>;

  Node* newNode() {
    pool_.emplace_back(new Node());
    return pool_.back().get();
  }

  Eigen::Vector3i toIdx(const Eigen::Vector3d& p) const {
    return Eigen::Vector3i(static_cast<int>(std::floor(p.x() / P_.resolution)),
                           static_cast<int>(std::floor(p.y() / P_.resolution)),
                           static_cast<int>(std::floor(p.z() / P_.resolution)));
  }

  std::vector<Eigen::Vector3d> makeInputs() const {
    std::vector<Eigen::Vector3d> in;
    const int n = 2 * P_.r + 1;
    in.reserve(n * n * n);
    auto val = [&](int k) { return -P_.max_acc + k * P_.max_acc / P_.r; };  // k = 0..2r
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k) in.emplace_back(val(i), val(j), val(k));
    return in;
  }

  // Constant-acceleration propagation (closed form of e^{At}x0 + int e^{A(t-s)}B u ds for a double integrator).
  static void propagate(const Eigen::Vector3d& p, const Eigen::Vector3d& v, const Eigen::Vector3d& u, double t,
                        Eigen::Vector3d& p_out, Eigen::Vector3d& v_out) {
    p_out = p + v * t + 0.5 * u * t * t;
    v_out = v + u * t;
  }

  // CheckFeasible(): collision-free along the primitive and velocity limit respected at its end.
  bool primitiveFeasible(const Eigen::Vector3d& p, const Eigen::Vector3d& v, const Eigen::Vector3d& u,
                         double dur) const {
    Eigen::Vector3d pe, ve;
    propagate(p, v, u, dur, pe, ve);
    if ((ve.array().abs() > P_.max_vel).any()) return false;  // velocity is linear in t, end check suffices
    for (double t = P_.check_dt; t < dur; t += P_.check_dt) {
      Eigen::Vector3d pt, vt;
      propagate(p, v, u, t, pt, vt);
      if (!is_free_(pt)) return false;
    }
    return is_free_(pe);
  }

  // Optimal cost of the unconstrained problem p0,v0 -> p1,v1 with free end time T:
  //   J(T) = rho*T + 12|dp|^2/T^3 - 12 (v0+v1).dp/T^2 + 4(|v0|^2 + v0.v1 + |v1|^2)/T
  // dJ/dT = 0  <=>  rho T^4 - 4(...) T^2 + 24 (v0+v1).dp T - 36|dp|^2 = 0   (quartic; T^3 term is 0)
  double heuristic(const Eigen::Vector3d& p0, const Eigen::Vector3d& v0, const Eigen::Vector3d& p1,
                   const Eigen::Vector3d& v1, double& T_opt) const {
    const Eigen::Vector3d dp = p1 - p0;
    const double c4 = P_.rho;
    const double c2 = -4.0 * (v0.squaredNorm() + v0.dot(v1) + v1.squaredNorm());
    const double c1 = 24.0 * (v0 + v1).dot(dp);
    const double c0 = -36.0 * dp.squaredNorm();

    auto J = [&](double T) {
      return P_.rho * T + 12.0 * dp.squaredNorm() / (T * T * T) - 12.0 * (v0 + v1).dot(dp) / (T * T) +
             4.0 * (v0.squaredNorm() + v0.dot(v1) + v1.squaredNorm()) / T;
    };

    // Positive real roots of q(T) = c4 T^4 + c2 T^2 + c1 T + c0.
    // q'(T) = 4 c4 T^3 + 2 c2 T + c1 is a depressed cubic, so its real roots (the critical points of q)
    // have a closed form. Between consecutive critical points q is monotonic, so each sign change
    // contains exactly one root, which is bracketed and bisected. Much faster than an eigen-solve.
    auto q = [&](double T) { return (((c4 * T) * T + c2) * T + c1) * T + c0; };

    double crit[3];
    int nc = 0;
    {
      const double pp = 2.0 * c2 / (4.0 * c4);  // T^3 + pp T + qq = 0
      const double qq = c1 / (4.0 * c4);
      const double disc = -(4.0 * pp * pp * pp + 27.0 * qq * qq);
      if (disc > 0.0) {  // three real roots (trigonometric form)
        const double m = 2.0 * std::sqrt(-pp / 3.0);
        const double arg = std::max(-1.0, std::min(1.0, 3.0 * qq / (pp * m)));
        const double th = std::acos(arg) / 3.0;
        for (int k = 0; k < 3; ++k) crit[nc++] = m * std::cos(th - 2.0 * M_PI * k / 3.0);
      } else {  // one real root (Cardano)
        const double s = std::sqrt(qq * qq / 4.0 + pp * pp * pp / 27.0);
        crit[nc++] = std::cbrt(-qq / 2.0 + s) + std::cbrt(-qq / 2.0 - s);
      }
    }
    std::sort(crit, crit + nc);

    const double T_hi = 1.0 + std::max({std::abs(c2), std::abs(c1), std::abs(c0)}) / c4;  // Cauchy bound
    double knots[5];
    int nk = 0;
    knots[nk++] = 1e-3;
    for (int i = 0; i < nc; ++i)
      if (crit[i] > knots[0] && crit[i] < T_hi) knots[nk++] = crit[i];
    knots[nk++] = T_hi;

    double best_T = -1.0, best_J = std::numeric_limits<double>::infinity();
    for (int i = 0; i + 1 < nk; ++i) {
      double lo = knots[i], hi = knots[i + 1];
      double qlo = q(lo), qhi = q(hi);
      if (qlo * qhi > 0.0) continue;
      for (int it = 0; it < 40; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double qm = q(mid);
        if (qlo * qm <= 0.0) {
          hi = mid;
        } else {
          lo = mid;
          qlo = qm;
        }
      }
      const double T = 0.5 * (lo + hi);
      const double j = J(T);
      if (j < best_J) {
        best_J = j;
        best_T = T;
      }
    }
    if (best_T < 0.0) {  // degenerate (e.g. already at the goal): fall back
      best_T = std::max(dp.norm() / P_.max_vel, 1e-2);
      best_J = J(best_T);
    }
    T_opt = best_T;
    return best_J;
  }

  void expand(Node* cur, const std::vector<Eigen::Vector3d>& inputs,
              std::priority_queue<QEntry, std::vector<QEntry>, std::greater<QEntry>>& open) {
    struct Cand {
      Eigen::Vector3d pos, vel, u;
      Eigen::Vector3i idx;
      double g, f;
    };
    std::vector<Cand> cands;
    cands.reserve(inputs.size());
    std::unordered_map<Eigen::Vector3i, int, IdxHash> best_in_voxel;  // Prune(): one primitive per voxel

    for (const auto& u : inputs) {
      Cand c;
      propagate(cur->pos, cur->vel, u, P_.tau, c.pos, c.vel);
      c.u = u;
      c.idx = toIdx(c.pos);
      if (c.idx == cur->idx) continue;  // did not leave the voxel, adds nothing
      c.g = cur->g + (u.squaredNorm() + P_.rho) * P_.tau;
      double T;
      c.f = c.g + P_.lambda_heu * P_.tie_breaker * heuristic(c.pos, c.vel, goal_pos_, goal_vel_, T);

      auto it = best_in_voxel.find(c.idx);
      if (it == best_in_voxel.end()) {
        best_in_voxel[c.idx] = static_cast<int>(cands.size());
        cands.push_back(c);
      } else if (c.f < cands[it->second].f) {
        cands[it->second] = c;
      }
    }

    for (const auto& c : cands) {
      if (!primitiveFeasible(cur->pos, cur->vel, c.u, P_.tau)) continue;

      auto it = nodes_.find(c.idx);
      if (it != nodes_.end()) {
        Node* ex = it->second;
        if (ex->closed) continue;
        if (c.g >= ex->g) continue;  // existing open node is at least as good
        ex->pos = c.pos;
        ex->vel = c.vel;
        ex->u = c.u;
        ex->dur = P_.tau;
        ex->g = c.g;
        ex->f = c.f;
        ex->parent = cur;
        open.push({ex->f, ex});
      } else {
        Node* n = newNode();
        n->pos = c.pos;
        n->vel = c.vel;
        n->idx = c.idx;
        n->u = c.u;
        n->dur = P_.tau;
        n->g = c.g;
        n->f = c.f;
        n->parent = cur;
        nodes_[n->idx] = n;
        open.push({n->f, n});
      }
    }
  }

  // AnalyticExpand(): closed-form optimal trajectory from the node to the goal state,
  // accepted if it is collision-free and respects velocity / acceleration limits.
  bool analyticExpand(const Node& n, std::vector<State>& tail) const {
    double T;
    heuristic(n.pos, n.vel, goal_pos_, goal_vel_, T);
    if (T <= 1e-3 || T > 30.0) return false;

    const Eigen::Vector3d dp = goal_pos_ - n.pos - n.vel * T;
    const Eigen::Vector3d dv = goal_vel_ - n.vel;
    // [alpha; beta] = 1/T^3 * [-12, 6T; 6T, -2T^2] * [dp; dv]   (per axis)
    const Eigen::Vector3d alpha = (-12.0 * dp + 6.0 * T * dv) / (T * T * T);
    const Eigen::Vector3d beta = (6.0 * T * dp - 2.0 * T * T * dv) / (T * T * T);

    std::vector<State> out;
    const int steps = std::max(1, static_cast<int>(std::ceil(T / P_.check_dt)));
    for (int i = 0; i <= steps; ++i) {
      const double t = std::min(T, i * T / steps);
      State s;
      s.pos = n.pos + n.vel * t + 0.5 * beta * t * t + alpha * t * t * t / 6.0;
      s.vel = n.vel + beta * t + 0.5 * alpha * t * t;
      s.acc = beta + alpha * t;
      s.t = t;
      if ((s.vel.array().abs() > P_.max_vel + 1e-6).any()) return false;
      if ((s.acc.array().abs() > P_.max_acc + 1e-6).any()) return false;
      if (!is_free_(s.pos)) return false;
      out.push_back(s);
    }
    tail = std::move(out);
    return true;
  }

  // Walk back through parents, resample every primitive, and append the analytic tail if any.
  std::vector<State> retrieve(Node* last, const std::vector<State>* tail) const {
    std::vector<Node*> chain;
    for (Node* n = last; n != nullptr; n = n->parent) chain.push_back(n);
    std::reverse(chain.begin(), chain.end());

    std::vector<State> traj;
    double t_acc = 0.0;
    for (size_t k = 1; k < chain.size(); ++k) {
      const Node* par = chain[k - 1];
      const Node* ch = chain[k];
      const int steps = std::max(1, static_cast<int>(std::ceil(ch->dur / P_.sample_dt)));
      for (int i = 0; i < steps; ++i) {
        const double t = i * ch->dur / steps;
        State s;
        propagate(par->pos, par->vel, ch->u, t, s.pos, s.vel);
        s.acc = ch->u;
        s.t = t_acc + t;
        traj.push_back(s);
      }
      t_acc += ch->dur;
    }

    if (tail) {
      const double dt_tail = P_.sample_dt;
      double next = 0.0;
      for (const auto& s : *tail) {
        if (s.t + 1e-9 < next && &s != &tail->back()) continue;
        State o = s;
        o.t += t_acc;
        traj.push_back(o);
        next = s.t + dt_tail;
      }
    } else {
      State s;
      s.pos = last->pos;
      s.vel = last->vel;
      s.t = t_acc;
      traj.push_back(s);
    }
    return traj;
  }

  Params P_;
  FreeFn is_free_;
  Eigen::Vector3d start_pos_, goal_pos_, goal_vel_;
  std::vector<std::unique_ptr<Node>> pool_;
  NodeMap nodes_;
};

}  // namespace kino
