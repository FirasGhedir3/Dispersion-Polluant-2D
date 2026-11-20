"""Comparaison LU creuse / Jacobi / Gauss-Seidel / SOR sur le système implicite du schéma.

À chaque pas de temps il faut résoudre A u_new = B u_old, avec A creuse (5 diagonales).
On regarde : rayon spectral, dominance diagonale, nombre d'itérations, temps, mémoire.

Lancer depuis la racine du projet :  python experiences/02_solveurs_lineaires.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from dispersion.modele import Probleme
from dispersion.outils import sauver_chiffres, sauver_figure
from dispersion.solveur import SolveurDispersion
from dispersion.solveurs_lineaires import (SOR, GaussSeidel, Jacobi, SolveurLU, dominance_diagonale,
                                           omega_young, pente_convergence, rayon_spectral, rayon_spectral_dense)

pb = Probleme()
theta = 0.5
chiffres = {}

# ---------------------------------------------------------------- 1. effet de dt sur la convergence
# Le terme I/dt est sur la diagonale de A : plus dt est grand, moins A est dominante,
# et plus Jacobi/Gauss-Seidel ont du mal.
print("== influence de dt (grille 31x31, démarrage à zéro, tolérance 1e-8)")
nx = 31
dx = pb.Lx / (nx - 1)
S = 2 / dx**2
lignes = []
courbes = {}
for dt in [0.05, 0.25, 1.0, 4.0, 16.0]:
    s = SolveurDispersion(pb, nx, nx, dt=dt, theta=theta)
    A = s.A
    b = s.B @ s.etat_initial()
    rho_j = rayon_spectral(A, "jacobi")
    rho_gs = rayon_spectral(A, "gs")
    # omega optimal de Young : 2 / (1 + sqrt(1 - rho_J^2)). Il est valable parce que la matrice est
    # "consistently ordered" (stencil à 5 points, numérotation naturelle), donc rho_GS = rho_J^2.
    w_opt = omega_young(rho_gs)
    rho_sor = rayon_spectral(A, "sor", w_opt)
    r_j = Jacobi(A).resoudre(b, tol=1e-8, max_iter=50000)
    r_gs = GaussSeidel(A).resoudre(b, tol=1e-8, max_iter=50000)
    r_sor = SOR(A, w_opt).resoudre(b, tol=1e-8, max_iter=50000)
    # valeur théorique pour Péclet <= 2 : somme hors diagonale / diagonale = 1 - (1/dt)/diag
    ratio_theo = 2 * theta * pb.kappa * S / (1 / dt + 2 * theta * pb.kappa * S)
    lig = dict(dt=dt, rho_jacobi_carre=rho_j**2, dominance=dominance_diagonale(A), dominance_theorique=ratio_theo,
               rho_jacobi=rho_j, rho_gs=rho_gs, omega_opt=w_opt, rho_sor=float(rho_sor),
               it_jacobi=r_j.nb_iter, it_gs=r_gs.nb_iter, it_sor=r_sor.nb_iter,
               pente_jacobi=pente_convergence(r_j.residus), log10_rho_jacobi=float(np.log10(rho_j)),
               pente_gs=pente_convergence(r_gs.residus), log10_rho_gs=float(np.log10(rho_gs)))
    lignes.append(lig)
    courbes[dt] = (r_j.residus, r_gs.residus, r_sor.residus)
    print(f"  dt={dt:6.2f} dominance={lig['dominance']:.4f} (théo {ratio_theo:.4f})  "
          f"rho J/GS/SOR = {rho_j:.4f}/{rho_gs:.4f}/{lig['rho_sor']:.4f} (w={w_opt:.2f})  "
          f"itérations J/GS/SOR = {r_j.nb_iter}/{r_gs.nb_iter}/{r_sor.nb_iter}  (rho_J^2 = {rho_j**2:.4f})")
    print(f"           pente mesurée J={lig['pente_jacobi']:.4f} (log10 rho = {lig['log10_rho_jacobi']:.4f})  "
          f"GS={lig['pente_gs']:.4f} (log10 rho = {lig['log10_rho_gs']:.4f})")
chiffres["influence_dt"] = lignes

# petit contrôle : le calcul dense des valeurs propres est faux pour Jacobi dès qu'il y a de l'advection
s_ctrl = SolveurDispersion(pb, nx, nx, dt=16.0)
chiffres["controle_rho_jacobi_dt16"] = dict(
    puissances=rayon_spectral(s_ctrl.A, "jacobi"), dense=rayon_spectral_dense(s_ctrl.A, "jacobi"),
    racine_rho_gs=rayon_spectral(s_ctrl.A, "gs") ** 0.5)
print("  contrôle (dt=16) :", chiffres["controle_rho_jacobi_dt16"])

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
for ax, dt in zip(axes, (0.25, 16.0)):
    rj, rg, rs = courbes[dt]
    ax.semilogy(rj, label="Jacobi")
    ax.semilogy(rg, label="Gauss-Seidel")
    ax.semilogy(rs, label="SOR (omega optimal)")
    ax.set(title=f"dt = {dt}", xlabel="itération")
    ax.legend()
axes[0].set_ylabel("résidu relatif")
sauver_figure(fig, "02_convergence_residus.png")

fig, ax = plt.subplots(figsize=(5.4, 3.6))
dts = [l["dt"] for l in lignes]
ax.semilogx(dts, [l["rho_jacobi"] for l in lignes], "o-", label="Jacobi")
ax.semilogx(dts, [l["rho_gs"] for l in lignes], "o-", label="Gauss-Seidel")
ax.semilogx(dts, [l["rho_sor"] for l in lignes], "o-", label="SOR optimal")
ax.axhline(1, color="k", lw=0.8, ls=":")
ax.set(xlabel="dt", ylabel="rayon spectral", title="Rayon spectral de la matrice d'itération")
ax.legend()
sauver_figure(fig, "02_rayon_spectral.png")

# ---------------------------------------------------------------- 2. simulation complète : temps et itérations
# Ici dt = h, démarrage "à chaud" (on repart de la solution du pas précédent).
print("== simulation complète jusqu'à T = 20, dt = h, tolérance 1e-10")
tailles = [41, 81, 121, 161]
methodes = ["lu", "jacobi", "gauss-seidel", "sor"]
temps = {m: [] for m in methodes}
iters = {m: [] for m in methodes}
omegas_sim = []
for nx in tailles:
    h = pb.Lx / (nx - 1)
    A_ref = SolveurDispersion(pb, nx, nx, dt=h).A
    w = omega_young(rayon_spectral(A_ref, "gs", nb_iter=1500))
    omegas_sim.append(w)
    for m in methodes:
        t0 = time.perf_counter()
        s = SolveurDispersion(pb, nx, nx, dt=h, methode=m, omega=w)
        s.integrer(20.0)
        dt_calc = time.perf_counter() - t0
        nsteps = int(round(20.0 / h))
        temps[m].append(dt_calc)
        iters[m].append(s.nb_iter_total / nsteps)
    print(f"  nx={nx:4d} (n={nx*nx:6d})  " + "  ".join(
        f"{m}: {temps[m][-1]:.2f}s ({iters[m][-1]:.0f} it/pas)" for m in methodes) + f"  (omega={w:.2f})")
chiffres["simulation"] = {"tailles": tailles, "omega_sor": [round(w, 3) for w in omegas_sim], "temps_s": temps, "iterations_par_pas": iters}

fig, ax = plt.subplots(figsize=(5.6, 3.8))
for m in methodes:
    ax.loglog([n * n for n in tailles], temps[m], "o-", label=m)
ax.set(xlabel="nombre d'inconnues n", ylabel="temps total (s)", title="Simulation jusqu'à T = 20")
ax.legend()
sauver_figure(fig, "02_temps.png")

# ---------------------------------------------------------------- 3. mémoire
print("== mémoire")
lignes_mem = []
for nx in [41, 81, 161]:
    s = SolveurDispersion(pb, nx, nx, dt=1.0)
    n = nx * nx
    csr = s.A.data.nbytes + s.A.indices.nbytes + s.A.indptr.nbytes
    dense = n * n * 8
    lignes_mem.append(dict(n=n, nnz=int(s.A.nnz), csr_Mo=csr / 1e6, dense_Mo=dense / 1e6))
    print(f"  n={n:6d}  nnz={s.A.nnz:7d}  CSR={csr/1e6:8.2f} Mo   dense={dense/1e6:10.1f} Mo")
chiffres["memoire"] = lignes_mem

# comparaison avec le solveur dense de numpy pour un pas de temps (la solution naïve : matrice dense + solveur générique)
print("== un pas de temps : creux contre dense")
for nx in (41, 81):
    s = SolveurDispersion(pb, nx, nx, dt=1.0)
    b = s.B @ s.etat_initial()
    t0 = time.perf_counter(); SolveurLU(s.A).resoudre(b); t_creux = time.perf_counter() - t0
    Ad = s.A.toarray()
    t0 = time.perf_counter(); np.linalg.solve(Ad, b); t_dense = time.perf_counter() - t0
    print(f"  n={nx*nx:5d}  LU creuse (factorisation + solve) {t_creux*1e3:7.1f} ms   numpy dense {t_dense*1e3:8.1f} ms")
    chiffres.setdefault("creux_vs_dense", []).append(dict(n=nx * nx, creux_ms=t_creux * 1e3, dense_ms=t_dense * 1e3))

sauver_chiffres("solveurs_lineaires", chiffres)
