"""Solveurs de systèmes linéaires A x = b pour les matrices creuses du schéma implicite.

Ce qu'il y a ici :
  - lu_maison / descente / remontee : la LU du TD, sur matrice dense (pour mémoire et pour tester)
  - SolveurLU        : LU creuse (SuperLU) factorisée une seule fois
  - Jacobi, SOR      : méthodes itératives (SOR avec omega = 1 donne Gauss-Seidel)
  - gauss_seidel_boucle_csr : un balayage de Gauss-Seidel écrit à la main avec indptr/indices/data
  - rayon_spectral   : rayon spectral de la matrice d'itération (méthode des puissances), pour comparer à ce qu'on observe

Convention : A = D + Li + Ls  (diagonale, strictement inférieure, strictement supérieure).
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------- LU du cours (dense)

def lu_maison(A):
    """Factorisation LU sans pivot, A = L U. Plante si un pivot est nul."""
    U = np.array(A, dtype=float)
    n = U.shape[0]
    L = np.eye(n)
    for k in range(n - 1):
        for j in range(k + 1, n):
            alpha = U[j, k] / U[k, k]
            U[j, k:] -= alpha * U[k, k:]
            L[j, k] = alpha
    return L, U


def descente(L, b):
    """Résout L y = b, L triangulaire inférieure."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def remontee(U, y):
    """Résout U x = y, U triangulaire supérieure."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


# ---------------------------------------------------------------- LU creuse

class SolveurLU:
    """On factorise une fois, puis chaque pas de temps ne coûte qu'une descente-remontée."""

    def __init__(self, A):
        self._lu = spla.splu(sp.csc_matrix(A))

    def resoudre(self, b, x0=None, **_):
        return Resultat(x=self._lu.solve(b), nb_iter=0)


# ---------------------------------------------------------------- méthodes itératives

@dataclass
class Resultat:
    x: np.ndarray
    nb_iter: int
    residus: list = field(default_factory=list)   # ||b - A x|| / ||b|| à chaque itération
    erreurs: list = field(default_factory=list)   # ||x - x_exact|| si on donne x_exact
    converge: bool = True


def _boucle_iterative(A, b, x0, tol, max_iter, x_exact, une_iteration):
    x = np.zeros_like(b) if x0 is None else x0.copy()
    nb = np.linalg.norm(b)
    nb = nb if nb > 0 else 1.0
    res = [np.linalg.norm(b - A @ x) / nb]
    err = [np.linalg.norm(x - x_exact)] if x_exact is not None else []
    k = 0
    while res[-1] > tol and k < max_iter:
        x = une_iteration(x)
        k += 1
        res.append(np.linalg.norm(b - A @ x) / nb)
        if x_exact is not None:
            err.append(np.linalg.norm(x - x_exact))
        if not np.isfinite(res[-1]):
            break
    return Resultat(x=x, nb_iter=k, residus=res, erreurs=err, converge=res[-1] <= tol)


class Jacobi:
    def __init__(self, A):
        self.A = sp.csr_matrix(A)
        self.d = self.A.diagonal()
        self.R = (self.A - sp.diags(self.d)).tocsr()   # tout sauf la diagonale

    def resoudre(self, b, x0=None, tol=1e-8, max_iter=20000, x_exact=None):
        def it(x):
            return (b - self.R @ x) / self.d
        return _boucle_iterative(self.A, b, x0, tol, max_iter, x_exact, it)


class SOR:
    """SOR : (D/w + Li) x_new = b - (Ls + (1 - 1/w) D) x_old.  w = 1 -> Gauss-Seidel.

    M = D/w + Li est triangulaire, donc sa "factorisation" ne crée aucun remplissage :
    on la fait une fois avec splu et chaque balayage est une descente rapide.
    """

    def __init__(self, A, omega=1.0):
        self.A = sp.csr_matrix(A)
        self.omega = omega
        D = sp.diags(self.A.diagonal())
        Li = sp.tril(self.A, k=-1)
        Ls = sp.triu(self.A, k=1)
        self.M = (D / omega + Li).tocsc()
        self.N = -(Ls + (1 - 1 / omega) * D).tocsr()
        self._lu = spla.splu(self.M, permc_spec="NATURAL")

    def resoudre(self, b, x0=None, tol=1e-8, max_iter=20000, x_exact=None):
        def it(x):
            return self._lu.solve(b + self.N @ x)
        return _boucle_iterative(self.A, b, x0, tol, max_iter, x_exact, it)


def GaussSeidel(A):
    return SOR(A, omega=1.0)


def gauss_seidel_boucle_csr(A, b, x0, nb_balayages=1):
    """Gauss-Seidel "à la main" sur le format CSR (lent, c'est juste pour vérifier).

    Pour la ligne i, les coefficients non nuls sont data[indptr[i]:indptr[i+1]]
    et leurs colonnes indices[indptr[i]:indptr[i+1]].
    """
    A = sp.csr_matrix(A)
    x = x0.copy()
    for _ in range(nb_balayages):
        for i in range(A.shape[0]):
            debut, fin = A.indptr[i], A.indptr[i + 1]
            somme, diag = 0.0, 0.0
            for p in range(debut, fin):
                j = A.indices[p]
                if j == i:
                    diag = A.data[p]
                else:
                    somme += A.data[p] * x[j]   # x[j] est déjà à jour si j < i
            x[i] = (b[i] - somme) / diag
    return x


# ---------------------------------------------------------------- analyse

def matrice_iteration(A, methode, omega=1.0):
    """Matrice d'itération T (dense) telle que x_new = T x_old + c."""
    A = sp.csr_matrix(A)
    D = sp.diags(A.diagonal())
    if methode == "jacobi":
        return (-(sp.diags(1 / A.diagonal())) @ (A - D)).toarray()
    Li, Ls = sp.tril(A, k=-1), sp.triu(A, k=1)
    M = (D / omega + Li).toarray()
    N = (-(Ls + (1 - 1 / omega) * D)).toarray()
    return np.linalg.solve(M, N)


def rayon_spectral_dense(A, methode, omega=1.0):
    """Rayon spectral par diagonalisation dense. Bien pour une petite matrice symétrique,
    mais pas fiable quand l'advection rend T très non normale (valeurs propres mal conditionnées)."""
    T = matrice_iteration(A, methode, omega)
    return float(np.max(np.abs(np.linalg.eigvals(T))))


def rayon_spectral(A, methode, omega=1.0, nb_iter=4000, graine=0):
    """Rayon spectral de la matrice d'itération par la méthode des puissances.

    On applique T à un vecteur aléatoire en renormalisant, et on prend la moyenne géométrique
    du facteur de croissance sur la deuxième moitié des itérations (ça marche aussi quand les
    valeurs propres dominantes sont par paires +mu/-mu, ce qui est le cas ici).
    """
    A = sp.csr_matrix(A)
    if methode == "jacobi":
        d = A.diagonal()
        R = (A - sp.diags(d)).tocsr()

        def appliquer(x):
            return -(R @ x) / d
    else:
        sor = SOR(A, omega)

        def appliquer(x):
            return sor._lu.solve(sor.N @ x)

    x = np.random.default_rng(graine).standard_normal(A.shape[0])
    x /= np.linalg.norm(x)
    logs = []
    for _ in range(nb_iter):
        y = appliquer(x)
        ny = np.linalg.norm(y)
        if ny == 0:
            return 0.0
        logs.append(np.log(ny))
        x = y / ny
    return float(np.exp(np.mean(logs[nb_iter // 2:])))


def dominance_diagonale(A):
    """Plus grand rapport sum_{j!=i}|a_ij| / |a_ii|. Strictement < 1 : Jacobi converge à coup sûr."""
    A = sp.csr_matrix(A)
    d = np.abs(A.diagonal())
    hors_diag = np.asarray(abs(A).sum(axis=1)).ravel() - d
    return float(np.max(hors_diag / d))


def pente_convergence(residus, debut=0.2):
    """Pente de log10(résidu) en fonction de l'itération, ajustée sur la partie "régime établi"."""
    r = np.log10(np.asarray(residus))
    k = np.arange(len(r))
    i0 = int(debut * len(r))
    return float(np.polyfit(k[i0:], r[i0:], 1)[0])


def omega_young(rho_gs):
    """omega optimal de SOR quand rho_GS = rho_J^2 (matrice "consistently ordered") : 2 / (1 + sqrt(1 - rho_J^2))."""
    return 2.0 / (1.0 + np.sqrt(1.0 - rho_gs))
