import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dispersion.modele import Probleme, erreur_L2_relative
from dispersion.solveur import SolveurDispersion
from dispersion.solveurs_lineaires import (SOR, GaussSeidel, Jacobi, descente, dominance_diagonale,
                                           gauss_seidel_boucle_csr, lu_maison, rayon_spectral, remontee)

PB = Probleme()
T = 60.0


def erreur(nx, **kw):
    h = PB.Lx / (nx - 1)
    s = SolveurDispersion(PB, nx, nx, dt=h, **kw)
    u, _ = s.integrer(T)
    return erreur_L2_relative(u, PB.exacte(s.X, s.Y, T)), s, u


# ------------------------------------------------------------------ le solveur

def test_ordre_deux():
    e1, _, _ = erreur(31)
    e2, _, _ = erreur(61)
    assert np.log2(e1 / e2) == pytest.approx(2.0, abs=0.15)


def test_masse_conservee():
    _, s, u = erreur(61)
    assert s.masse(u) / PB.masse_exacte() == pytest.approx(1.0, abs=1e-3)


def test_les_solveurs_lineaires_donnent_la_meme_solution():
    e_lu, _, u_lu = erreur(41, methode="lu")
    for methode in ("jacobi", "gauss-seidel"):
        _, _, u = erreur(41, methode=methode)
        assert np.allclose(u, u_lu, atol=1e-7)


def test_noeuds_dirichlet_nuls():
    _, _, u = erreur(41)
    assert np.all(u[0, :] == 0) and np.all(u[:, 0] == 0)


def test_centre_oscille_a_grand_peclet_mais_pas_upwind():
    h = PB.Lx / 60
    p = Probleme(kappa=PB.vx * h / 16)          # Péclet de maille = 16
    for adv, doit_etre_negatif in (("centre", True), ("upwind", False)):
        s = SolveurDispersion(p, 61, 61, dt=h, advection=adv)
        u, _ = s.integrer(T)
        assert (u.min() < -1e-3) == doit_etre_negatif


def test_vitesse_nulle_le_pic_ne_bouge_pas():
    p = Probleme(vx=0.0, vy=0.0)
    s = SolveurDispersion(p, 41, 41, dt=1.0)
    u, _ = s.integrer(10.0)
    j, i = np.unravel_index(np.argmax(u), u.shape)
    assert abs(s.x[i] - p.x0) <= s.dx and abs(s.y[j] - p.y0) <= s.dy


# ------------------------------------------------------------------ algèbre linéaire

def test_lu_maison():
    A = np.array([[-2., 1, -1, 1], [2, 0, 4, -3], [-4, -1, -12, 9], [-2, 1, 1, -4]])
    b = np.array([1.5, 4, -14, -6.5])
    L, U = lu_maison(A)
    assert np.allclose(L @ U, A)
    assert np.allclose(remontee(U, descente(L, b)), np.linalg.solve(A, b))


def test_gauss_seidel_boucle_csr_comme_la_version_rapide():
    s = SolveurDispersion(PB, 21, 21, dt=1.0)
    b = np.random.default_rng(0).random(s.A.shape[0])
    x0 = np.zeros_like(b)
    lent = gauss_seidel_boucle_csr(s.A, b, x0, nb_balayages=3)
    gs, rapide = GaussSeidel(s.A), x0.copy()
    for _ in range(3):
        rapide = gs._lu.solve(b + gs.N @ rapide)
    assert np.allclose(lent, rapide)


def test_jacobi_converge_si_diagonale_dominante():
    s = SolveurDispersion(PB, 21, 21, dt=1.0)
    assert dominance_diagonale(s.A) < 1
    b = np.random.default_rng(1).random(s.A.shape[0])
    res = Jacobi(s.A).resoudre(b, tol=1e-10)
    assert res.converge
    assert np.allclose(s.A @ res.x, b, atol=1e-8)


def test_rho_gauss_seidel_egal_rho_jacobi_au_carre():
    s = SolveurDispersion(PB, 21, 21, dt=4.0)
    rj, rg = rayon_spectral(s.A, "jacobi"), rayon_spectral(s.A, "gs")
    assert rg == pytest.approx(rj ** 2, rel=1e-4)


def test_sor_omega_un_est_gauss_seidel():
    s = SolveurDispersion(PB, 21, 21, dt=2.0)
    b = np.random.default_rng(2).random(s.A.shape[0])
    assert np.allclose(SOR(s.A, 1.0).resoudre(b, tol=1e-12).x, GaussSeidel(s.A).resoudre(b, tol=1e-12).x)


# ------------------------------------------------------------------ PINN (juste pour vérifier que ça tourne)

def test_pinn_tourne_et_la_perte_baisse():
    pytest.importorskip("torch")
    from dispersion.pinn import entrainer, fabriquer_donnees
    pb = Probleme(Lx=1.0, Ly=1.0, kappa=0.01, vx=0.5, vy=0.5, x0=0.5, y0=0.5, sigma=0.1)
    s = SolveurDispersion(pb, 21, 21, dt=0.25 / 20)
    _, imgs = s.integrer(0.25, nb_images=20)
    ts, us = np.array([t for t, _ in imgs]), np.array([u for _, u in imgs])
    donnees = fabriquer_donnees(((ts, us), (s.x, s.y)), 200, 0.0)
    _, hist, _ = entrainer(pb, 0.25, donnees, couches=3, largeur=32, iterations=200,
                           nb_colloc=200, nb_ic=100, nb_bords=100, verbeux=False)
    assert hist.perte[-1] < hist.perte[0]
