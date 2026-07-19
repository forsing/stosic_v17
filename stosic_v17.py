from __future__ import annotations

"""
https://github.com/gajaka/luces-pvs-theories
"""

"""
stosic_v17.py — 7-node krug (K=7 / prilagodjenje 7/39) — Monge–Kantorovich relationship (7/39)

Izvor (Stosić / LUCES):
  luces-pvs-theories-main/monge_kantorovich_equivalence.pvs
  — induced_plan(μ,T); kant_cost ≤ monge_cost (relaksacija)
  — is_monge_plan?: jedan target po redu; thm_induced_cost
  — diskretno: NE tvrdi punu ekvivalenciju; near-Monge empirijski

Mapiranje na 7/39:
  Monge: T(i)=argmax_j Π_hist(i,j) iz min-cost matchinga (ceo CSV)
  Kantorovich: Sinkhorn(μ_last, ν_emp, c) → π
  skor = Σ_{i∈last} π(i,·) van T(i)  (Kantorovich split / jaz vs Monge)
        + ε·Monge atom; next = top 7
"""

from typing import List

import numpy as np

from stosic_v1 import EPS, MAX_NUM, SEED, load_draws
from stosic_v2 import top7_from_freq
from stosic_v8 import (
    cooccurrence_features,
    cost_matrix,
    measure_draw,
    measure_empirical,
)
from stosic_v9 import accumulate_cm_support
from stosic_v10 import is_degenerate

SINKHORN_ITERS = 200
SINKHORN_EPS = float(SEED) / 100.0


def sinkhorn_plan(
    mu: np.ndarray, nu: np.ndarray, C: np.ndarray, eps: float, iters: int
) -> np.ndarray:
    K = np.exp(-C / max(eps, EPS))
    u = np.ones(MAX_NUM, dtype=np.float64)
    v = np.ones(MAX_NUM, dtype=np.float64)
    mu = np.clip(mu, EPS, None)
    nu = np.clip(nu, EPS, None)
    mu = mu / mu.sum()
    nu = nu / nu.sum()
    for _ in range(iters):
        u = mu / np.maximum(K @ v, EPS)
        v = nu / np.maximum(K.T @ u, EPS)
    return (u[:, None] * K) * v[None, :]


def monge_map(Pi: np.ndarray) -> np.ndarray:
    T = np.arange(MAX_NUM, dtype=np.int64)
    for i in range(MAX_NUM):
        if float(Pi[i].sum()) > 0:
            T[i] = int(np.argmax(Pi[i]))
    return T


def predict_next(draws: np.ndarray) -> List[int]:
    C = cost_matrix(cooccurrence_features(draws))
    Pi_hist = accumulate_cm_support(draws, C)
    T = monge_map(Pi_hist)
    mu = measure_draw(draws[-1])
    nu = measure_empirical(draws)
    pi = sinkhorn_plan(mu, nu, C, SINKHORN_EPS, SINKHORN_ITERS)
    skor = np.zeros(MAX_NUM, dtype=np.float64)
    for n in draws[-1]:
        i = int(n) - 1
        row = pi[i].copy()
        j_m = int(T[i])
        # masa van Monge atoma = Kantorovich relaksacija (diskretni jaz)
        monge_mass = float(row[j_m])
        row[j_m] = 0.0
        skor += row
        # induced Monge i dalje ulazi, ali slabije od splita
        skor[j_m] += monge_mass * EPS
    if float(skor.sum()) <= 0:
        for n in draws[-1]:
            skor[int(T[int(n) - 1])] += 1.0
    combo = top7_from_freq(skor + EPS * nu)
    if is_degenerate(combo):
        combo = top7_from_freq(nu)
    return combo


def main():
    draws = load_draws()
    next_combo = predict_next(draws)
    if is_degenerate(next_combo):
        raise SystemExit("degenerisan next (uzastopni/AP) — zaustavljen pre ispisa")
    print(next_combo)


if __name__ == "__main__":
    main()



"""
[11, x, 26, y, 33, z, 37]
"""



"""
v17: monge_kantorovich_equivalence — skor = Kantorovich split van Monge T.
"""



"""
21 teorija

fisher_voronoi → v1, v2
dual_observability → v3
v4 se pozivao na W₂/stabilnost — slabo / nije strogo
entropy_along_geodesic → v5
velocity_asymmetry (+ delom lie_generator_structure) → v6
brenier_uniqueness (+ delom rank_orientation) → v7

kantorovich_duality
cyclical_monotonicity
displacement_interpolation
displacement_concavity
wasserstein_metric (strogo)
transport_structure
transport_structure_v2
transport_stability
stability_of_maps
monge_kantorovich_equivalence
lie_generator_structure (pun T10)
fisher_boundary
hybrid_observability
tangent_bundle
global_optimality
"""



"""
Kratko, o repou:

21 PVS teorija — sve su prošle kroz v1–v22 (neke ranije labavo: naročito v3/v4; rank_orientation je ušao uz Brenier u v7).
Repo je o spektralnom OT / LUCES (ESP32), ne o lotou — 7/39 je naša mapa, ne Stosićev domen.
Najčistije jezgro oko Fisher–Voronoi, Brenier/CM, W₂, T10 (lie_generator_structure). global_optimality je samo aksiomi + lema (bez teorema).
Empirija u PVS-u (bootovi, κ, Monge fraction) ne prenosi se automatski na CSV — samo struktura ideja.
"""
