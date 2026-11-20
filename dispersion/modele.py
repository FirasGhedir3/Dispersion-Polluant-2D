"""Le problème physique : paramètres, condition initiale et solution exacte.

On résout  du/dt + v.grad(u) = kappa * laplacien(u)  sur un rectangle,
avec une tache gaussienne comme condition initiale.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Probleme:
    # Domaine 100x100, kappa et v de l'ordre de grandeur d'un rejet côtier.
    # sigma = 5 : la tache doit couvrir plusieurs mailles (dx ~ 1.7 sur la grille 61x61),
    # sinon elle n'est pas résolue. Le centre (20, 20) laisse de la marge avant les bords.
    Lx: float = 100.0
    Ly: float = 100.0
    kappa: float = 0.9
    vx: float = 0.5
    vy: float = 0.5
    x0: float = 20.0
    y0: float = 20.0
    sigma: float = 5.0

    def exacte(self, X, Y, t):
        """Solution exacte, valable tant que la tache ne touche pas le bord.

        Une gaussienne exp(-r²/sigma²) reste une gaussienne : elle est translatée
        à la vitesse v et son carré de largeur grandit de 4*kappa*t.
        """
        s2 = self.sigma**2 + 4 * self.kappa * t
        r2 = (X - self.x0 - self.vx * t) ** 2 + (Y - self.y0 - self.vy * t) ** 2
        return self.sigma**2 / s2 * np.exp(-r2 / s2)

    def initiale(self, X, Y):
        return self.exacte(X, Y, 0.0)

    def masse_exacte(self):
        # intégrale de exp(-r²/sigma²) sur le plan, conservée au cours du temps
        return np.pi * self.sigma**2


def erreur_L2_relative(u, u_ref):
    """Erreur relative en norme L2 discrète (la même que dans le papier PINN)."""
    return np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref)
