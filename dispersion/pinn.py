"""PINN pour l'équation d'advection-diffusion 2D : u(t, x, y) approché par un petit réseau.

La perte a quatre morceaux (comme dans le papier de Battina et al.) :
  - données  : écart avec des mesures (ici simulées, avec du bruit)
  - physique : résidu de l'EDP aux points de collocation, dérivées par différentiation automatique
  - initiale : écart avec la condition initiale
  - bords    : u = 0 sur les quatre côtés du carré

Pour le problème inverse, D et la vitesse deviennent des paramètres appris en même temps que le réseau.
"""
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)


class Reseau(nn.Module):
    def __init__(self, couches=9, largeur=128, T=1.0):
        super().__init__()
        self.T = T
        blocs = [nn.Linear(3, largeur), nn.Tanh()]
        for _ in range(couches - 1):
            blocs += [nn.Linear(largeur, largeur), nn.Tanh()]
        blocs.append(nn.Linear(largeur, 1))
        self.net = nn.Sequential(*blocs)

    def forward(self, t, x, y):
        # on remet t entre 0 et 1 pour que les trois entrées aient la même échelle
        return self.net(torch.stack([t / self.T, x, y], dim=-1)).squeeze(-1)


@dataclass
class Poids:
    donnees: float = 1.0
    physique: float = 1.0
    initiale: float = 1.0
    bords: float = 1.0


@dataclass
class Historique:
    iteration: list = field(default_factory=list)
    perte: list = field(default_factory=list)
    D: list = field(default_factory=list)
    vx: list = field(default_factory=list)
    vy: list = field(default_factory=list)
    duree_s: float = 0.0


def _derivees(u, t, x, y):
    """du/dt, du/dx, du/dy, d2u/dx2, d2u/dy2 par différentiation automatique."""
    un = torch.ones_like(u)
    u_t, u_x, u_y = torch.autograd.grad(u, (t, x, y), un, create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return u_t, u_x, u_y, u_xx, u_yy


def entrainer(pb, T, donnees, *, couches=9, largeur=128, iterations=6000, lr=6e-3,
              nb_colloc=2000, nb_ic=500, nb_bords=500, poids=Poids(),
              D=None, vitesse=None, apprendre=None, graine=0, verbeux=True):
    """Entraîne un PINN et renvoie (reseau, historique, parametres appris).

    pb        : Probleme (carré unité) ; sert pour la condition initiale et, par défaut, pour D et v
    donnees   : tableau (N, 4) de colonnes t, x, y, u_mesuré
    vitesse   : fonction (x, y) -> (vx, vy) sur tenseurs torch, pour une vitesse non uniforme
    apprendre : dict de valeurs de départ, par exemple {"D": 0.05, "vx": 0.2, "vy": 0.8},
                pour apprendre ces paramètres (problème inverse)
    """
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    reseau = Reseau(couches, largeur, T)
    donnees = torch.tensor(donnees, dtype=torch.float32)
    L = pb.Lx

    params_phys = {}
    if apprendre:
        if "D" in apprendre:
            params_phys["logD"] = nn.Parameter(torch.tensor(float(np.log(apprendre["D"]))))
        if "vx" in apprendre:
            params_phys["vx"] = nn.Parameter(torch.tensor(float(apprendre["vx"])))
        if "vy" in apprendre:
            params_phys["vy"] = nn.Parameter(torch.tensor(float(apprendre["vy"])))

    def valeurs_physiques(x, y):
        Dv = torch.exp(params_phys["logD"]) if "logD" in params_phys else (pb.kappa if D is None else D)
        if vitesse is not None:
            vxv, vyv = vitesse(x, y)
        else:
            vxv = params_phys["vx"] if "vx" in params_phys else pb.vx
            vyv = params_phys["vy"] if "vy" in params_phys else pb.vy
        return Dv, vxv, vyv

    opt = torch.optim.Adam(list(reseau.parameters()) + list(params_phys.values()), lr=lr)
    # le learning rate descend en douceur jusqu'à 1 % de sa valeur de départ
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.01 ** (1.0 / iterations))

    hist = Historique()
    t0 = time.time()
    for it in range(1, iterations + 1):
        opt.zero_grad()

        # --- données
        pred = reseau(donnees[:, 0], donnees[:, 1], donnees[:, 2])
        perte_d = torch.mean((pred - donnees[:, 3]) ** 2)

        # --- physique (points tirés au hasard à chaque itération)
        pts = torch.tensor(rng.random((nb_colloc, 3)), dtype=torch.float32)
        t = (pts[:, 0] * T).requires_grad_(True)
        x = (pts[:, 1] * L).requires_grad_(True)
        y = (pts[:, 2] * L).requires_grad_(True)
        u = reseau(t, x, y)
        u_t, u_x, u_y, u_xx, u_yy = _derivees(u, t, x, y)
        Dv, vxv, vyv = valeurs_physiques(x, y)
        residu = u_t + vxv * u_x + vyv * u_y - Dv * (u_xx + u_yy)
        perte_p = torch.mean(residu ** 2)

        # --- condition initiale
        xy = torch.tensor(rng.random((nb_ic, 2)) * L, dtype=torch.float32)
        u0 = reseau(torch.zeros(nb_ic), xy[:, 0], xy[:, 1])
        cible = torch.tensor(pb.initiale(xy[:, 0].numpy(), xy[:, 1].numpy()), dtype=torch.float32)
        perte_i = torch.mean((u0 - cible) ** 2)

        # --- bords : u = 0
        tb = torch.tensor(rng.random(nb_bords) * T, dtype=torch.float32)
        s = torch.tensor(rng.random(nb_bords) * L, dtype=torch.float32)
        cote = rng.integers(0, 4, nb_bords)
        xb = torch.where(torch.tensor(cote == 0), torch.zeros_like(s),
                         torch.where(torch.tensor(cote == 1), torch.full_like(s, L), s))
        yb = torch.where(torch.tensor(cote == 2), torch.zeros_like(s),
                         torch.where(torch.tensor(cote == 3), torch.full_like(s, L), s))
        perte_b = torch.mean(reseau(tb, xb, yb) ** 2)

        perte = (poids.donnees * perte_d + poids.physique * perte_p
                 + poids.initiale * perte_i + poids.bords * perte_b)
        perte.backward()
        opt.step()
        sched.step()

        if it % 100 == 0 or it == 1:
            hist.iteration.append(it)
            hist.perte.append(float(perte.detach()))
            if params_phys:
                hist.D.append(float(torch.exp(params_phys["logD"])) if "logD" in params_phys else np.nan)
                hist.vx.append(float(params_phys["vx"]) if "vx" in params_phys else np.nan)
                hist.vy.append(float(params_phys["vy"]) if "vy" in params_phys else np.nan)
        if verbeux and it % 1000 == 0:
            msg = f"    it {it:5d}  perte {float(perte):.3e}  (données {float(perte_d):.1e}, physique {float(perte_p):.1e})"
            if params_phys:
                msg += "  " + "  ".join(f"{k}={float(v):.4f}" if k != "logD" else f"D={float(torch.exp(v)):.4f}"
                                        for k, v in params_phys.items())
            print(msg, flush=True)
    hist.duree_s = time.time() - t0

    appris = {}
    if "logD" in params_phys:
        appris["D"] = float(torch.exp(params_phys["logD"]))
    for k in ("vx", "vy"):
        if k in params_phys:
            appris[k] = float(params_phys[k])
    return reseau, hist, appris


@torch.no_grad()
def predire(reseau, t, X, Y):
    """Évalue le réseau sur une grille (X, Y) numpy à l'instant t."""
    tt = torch.full((X.size,), float(t))
    u = reseau(tt, torch.tensor(X.ravel(), dtype=torch.float32), torch.tensor(Y.ravel(), dtype=torch.float32))
    return u.numpy().reshape(X.shape)


def temps_inference_ms(reseau, X, Y, t, repetitions=50):
    """Temps moyen pour prédire un champ complet (toute la grille) à un instant donné."""
    predire(reseau, t, X, Y)  # échauffement
    t0 = time.perf_counter()
    for _ in range(repetitions):
        predire(reseau, t, X, Y)
    return (time.perf_counter() - t0) / repetitions * 1e3


def fabriquer_donnees(images, nb_points, bruit, graine=0):
    """Tire nb_points mesures au hasard dans (t, x, y), puis ajoute du bruit gaussien.

    images : liste de (t, u) avec u de forme (ny, nx), plus un tuple (x, y) pour la grille.
    bruit  : écart-type du bruit, en fraction du maximum initial (1 pour notre gaussienne).
    """
    rng = np.random.default_rng(graine)
    (ts, us), (x, y) = images
    nt, ny, nx = us.shape
    idx = rng.choice(nt * ny * nx, size=nb_points, replace=False)
    n, j, i = np.unravel_index(idx, (nt, ny, nx))
    vrai = us[n, j, i]
    mesure = vrai + bruit * us.max() * rng.standard_normal(nb_points)
    return np.column_stack([ts[n], x[i], y[j], mesure])
