"""Solveur aux différences finies pour l'advection-diffusion 2D.

Schéma en temps : theta-schéma (theta = 0.5 -> Crank-Nicolson, 1 -> Euler implicite).
Schéma en espace : différences centrées pour la diffusion ; pour l'advection au choix
centré (ordre 2) ou upwind/décentré amont (ordre 1).

Conditions aux limites :
  - Dirichlet u = 0 à l'ouest (x = 0) et au sud (y = 0)
  - Neumann du/dn = 0 à l'est et au nord, pour laisser sortir le polluant

Les inconnues sont rangées dans un vecteur, noeud (i, j) -> k = j*nx + i.
"""
import numpy as np
import scipy.sparse as sp

from .solveurs_lineaires import SOR, Jacobi, SolveurLU


def _operateurs_1d(n, h):
    """Dérivées 1D sur n points espacés de h, avec les conditions de bord.

    La ligne 0 (Dirichlet) sera écrasée plus tard, donc ce qu'on y met n'a pas d'importance.
    La dernière ligne est le bord Neumann : on utilise un point fantôme symétrique
    (u_n = u_{n-2}) pour la dérivée seconde, et on met les dérivées premières à 0.
    """
    d2 = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="lil") / h**2
    centre = sp.diags([-1.0, 1.0], [-1, 1], shape=(n, n), format="lil") / (2 * h)
    amont = sp.diags([-1.0, 1.0], [-1, 0], shape=(n, n), format="lil") / h   # (u_i - u_{i-1})/h
    aval = sp.diags([-1.0, 1.0], [0, 1], shape=(n, n), format="lil") / h     # (u_{i+1} - u_i)/h

    d2[n - 1, :] = 0
    d2[n - 1, n - 2] = 2 / h**2
    d2[n - 1, n - 1] = -2 / h**2
    for op in (centre, amont, aval):
        op[n - 1, :] = 0
    return d2.tocsr(), centre.tocsr(), amont.tocsr(), aval.tocsr()


class SolveurDispersion:
    def __init__(self, pb, nx, ny, dt, theta=0.5, advection="centre",
                 methode="lu", omega=1.0, tol=1e-10, vx=None, vy=None):
        """vx, vy : champs de vitesse optionnels, tableaux (ny, nx) ou fonctions (X, Y) -> tableau.
        Sinon on prend la vitesse constante de pb."""
        self.pb, self.nx, self.ny, self.dt = pb, nx, ny, dt
        self.dx, self.dy = pb.Lx / (nx - 1), pb.Ly / (ny - 1)
        self.x = np.linspace(0, pb.Lx, nx)
        self.y = np.linspace(0, pb.Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)          # formes (ny, nx)
        self.tol = tol
        self.nb_iter_total = 0

        def champ(v, defaut):
            if v is None:
                return np.full((ny, nx), defaut)
            return v(self.X, self.Y) if callable(v) else v

        vx, vy = champ(vx, pb.vx), champ(vy, pb.vy)

        L = self._operateur_spatial(vx.ravel(), vy.ravel(), advection)
        n = nx * ny
        I = sp.identity(n, format="csr")

        # noeuds de Dirichlet : première colonne (i=0) et première ligne (j=0)
        dirichlet = np.zeros((ny, nx), dtype=bool)
        dirichlet[:, 0] = True
        dirichlet[0, :] = True
        P = sp.diags((~dirichlet).ravel().astype(float))      # 1 sur les noeuds libres

        A = I / dt - theta * L
        B = I / dt + (1 - theta) * L
        # sur un noeud de Dirichlet : A -> identité, B -> 0, donc u_new = 0
        self.A = (P @ A + (I - P)).tocsr()
        self.B = (P @ B).tocsr()
        self.solveur = self._choisir_solveur(methode, omega)

    def _operateur_spatial(self, vx, vy, advection):
        nx, ny = self.nx, self.ny
        kappa = self.pb.kappa
        d2x, cx, mx, px = _operateurs_1d(nx, self.dx)
        d2y, cy, my, py = _operateurs_1d(ny, self.dy)
        Ix, Iy = sp.identity(nx), sp.identity(ny)

        # une dérivée en x agit sur l'indice rapide i, une dérivée en y sur l'indice lent j
        Dxx, Dyy = sp.kron(Iy, d2x), sp.kron(d2y, Ix)
        if advection == "centre":
            Dx, Dy = sp.kron(Iy, cx), sp.kron(cy, Ix)
        elif advection == "upwind":
            # on prend la différence du côté d'où vient le polluant, selon le signe de v
            Dx = sp.diags((vx > 0).astype(float)) @ sp.kron(Iy, mx) + sp.diags((vx <= 0).astype(float)) @ sp.kron(Iy, px)
            Dy = sp.diags((vy > 0).astype(float)) @ sp.kron(my, Ix) + sp.diags((vy <= 0).astype(float)) @ sp.kron(py, Ix)
        else:
            raise ValueError(f"advection inconnue : {advection}")
        return kappa * (Dxx + Dyy) - sp.diags(vx) @ Dx - sp.diags(vy) @ Dy

    def _choisir_solveur(self, methode, omega):
        if methode == "lu":
            return SolveurLU(self.A)
        if methode == "jacobi":
            return Jacobi(self.A)
        if methode in ("gauss-seidel", "sor"):
            return SOR(self.A, omega if methode == "sor" else 1.0)
        raise ValueError(f"méthode inconnue : {methode}")

    # ------------------------------------------------------------------

    def etat_initial(self):
        return self.pb.initiale(self.X, self.Y).ravel()

    def pas(self, u):
        """Un pas de temps : A u_new = B u_old."""
        res = self.solveur.resoudre(self.B @ u, x0=u, tol=self.tol)
        self.nb_iter_total += res.nb_iter
        return res.x

    def integrer(self, T, nb_images=0, u0=None):
        """Avance jusqu'à T. Renvoie u final (ny, nx) et, si demandé, des images régulières."""
        nt = int(round(T / self.dt))
        u = self.etat_initial() if u0 is None else u0.ravel().copy()
        pas_image = max(1, nt // nb_images) if nb_images else None
        images = [(0.0, u.reshape(self.ny, self.nx).copy())] if nb_images else []
        for n in range(1, nt + 1):
            u = self.pas(u)
            if pas_image and n % pas_image == 0:
                images.append((n * self.dt, u.reshape(self.ny, self.nx).copy()))
        return u.reshape(self.ny, self.nx), images

    def masse(self, u):
        return float(u.sum() * self.dx * self.dy)
