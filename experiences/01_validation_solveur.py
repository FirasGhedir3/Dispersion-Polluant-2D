"""Validation du solveur : solution exacte, ordre de convergence, masse, nombre de Péclet.

Lancer depuis la racine du projet :  python experiences/01_validation_solveur.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from dispersion.modele import Probleme, erreur_L2_relative
from dispersion.outils import MEDIA, sauver_chiffres, sauver_figure
from dispersion.solveur import SolveurDispersion

pb = Probleme()
T = 60.0
chiffres = {}

# ---------------------------------------------------------------- 1. convergence en maillage
# On prend dt = h pour que l'erreur en espace et l'erreur en temps diminuent ensemble.
print("== convergence en maillage")
tailles = [21, 31, 41, 61, 81, 121, 161]
hs, erreurs = [], []
for nx in tailles:
    h = pb.Lx / (nx - 1)
    s = SolveurDispersion(pb, nx, nx, dt=h)
    u, _ = s.integrer(T)
    e = erreur_L2_relative(u, pb.exacte(s.X, s.Y, T))
    hs.append(h)
    erreurs.append(e)
    print(f"  nx={nx:4d}  h={h:.3f}  erreur L2 relative = {e:.2e}")
ordre_espace = np.polyfit(np.log(hs), np.log(erreurs), 1)[0]
print(f"  ordre observé : {ordre_espace:.2f}")

# ordre en temps : on fixe le maillage et on compare à la même grille avec un dt très petit,
# comme ça l'erreur d'espace s'annule et il ne reste que l'erreur en temps
print("== convergence en temps (maillage fixe nx=101)")
s_ref = SolveurDispersion(pb, 101, 101, dt=T / 960)
u_ref, _ = s_ref.integrer(T)
dts = [4.0, 2.0, 1.0, 0.5, 0.25]
err_t = []
for dt in dts:
    s = SolveurDispersion(pb, 101, 101, dt=dt)
    u, _ = s.integrer(T)
    err_t.append(erreur_L2_relative(u, u_ref))
    print(f"  dt={dt:5.2f}  écart à la référence = {err_t[-1]:.2e}")
ordre_temps = np.polyfit(np.log(dts), np.log(err_t), 1)[0]
print(f"  ordre en temps : {ordre_temps:.2f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
a1.loglog(hs, erreurs, "o-", label="solveur")
a1.loglog(hs, erreurs[0] * (np.array(hs) / hs[0]) ** 2, "k--", lw=1, label="pente 2")
a1.set(xlabel="pas d'espace h  (dt = h)", ylabel="erreur L2 relative", title="Convergence en espace")
a1.legend()
a2.loglog(dts, err_t, "o-", label="solveur")
a2.loglog(dts, err_t[0] * (np.array(dts) / dts[0]) ** 2, "k--", lw=1, label="pente 2")
a2.set(xlabel="pas de temps dt", ylabel="écart à la solution à dt très petit", title="Convergence en temps (h fixé)")
a2.legend()
sauver_figure(fig, "01_convergence.png")
chiffres["ordre_espace"] = round(ordre_espace, 2)
chiffres["ordre_temps"] = round(ordre_temps, 2)
chiffres["erreurs_espace"] = {str(n): float(f"{e:.3e}") for n, e in zip(tailles, erreurs)}

# ---------------------------------------------------------------- 2. conservation de la masse
print("== masse")
s = SolveurDispersion(pb, 121, 121, dt=0.5)
_, images = s.integrer(T, nb_images=60)
masse = [s.masse(u) / pb.masse_exacte() for _, u in images]
temps = [t for t, _ in images]
print(f"  masse / masse exacte : min {min(masse):.5f}  max {max(masse):.5f}")
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(temps, masse)
ax.set(xlabel="t", ylabel="masse / masse exacte", title="Conservation de la masse", ylim=(0.99, 1.01))
sauver_figure(fig, "01_masse.png")
chiffres["masse_min"] = round(min(masse), 5)
chiffres["masse_max"] = round(max(masse), 5)

# ---------------------------------------------------------------- 3. Péclet : centré vs upwind
# Péclet de maille = |v| h / kappa. Au-dessus de 2, le schéma centré se met à osciller.
print("== Péclet")
nx = 61
h = pb.Lx / (nx - 1)
pecs = [0.5, 1, 2, 4, 8, 16]
res = {"centre": {"err": [], "min": []}, "upwind": {"err": [], "min": []}}
profils = {}
for pe in pecs:
    p = Probleme(kappa=pb.vx * h / pe)
    for adv in ("centre", "upwind"):
        s = SolveurDispersion(p, nx, nx, dt=h, advection=adv)
        u, _ = s.integrer(T)
        ex = p.exacte(s.X, s.Y, T)
        res[adv]["err"].append(erreur_L2_relative(u, ex))
        res[adv]["min"].append(float(u.min() / ex.max()))
        if pe == 8:
            profils[adv] = np.diag(u)
            profils["exacte"] = np.diag(ex)
            profils["s"] = np.sqrt(2) * s.x
    print(f"  Pe={pe:5.2f}  centré: err={res['centre']['err'][-1]:.3f} min={res['centre']['min'][-1]:+.3f}"
          f"   upwind: err={res['upwind']['err'][-1]:.3f} min={res['upwind']['min'][-1]:+.3f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
for adv, c in (("centre", "C0"), ("upwind", "C1")):
    a1.loglog(pecs, res[adv]["err"], "o-", color=c, label=adv)
    a2.semilogx(pecs, res[adv]["min"], "o-", color=c, label=adv)
for a in (a1, a2):
    a.axvline(2, color="gray", ls=":")
    a.set_xlabel("Péclet de maille  |v| h / kappa")
    a.legend()
a1.set(ylabel="erreur L2 relative", title="Erreur selon le Péclet")
a2.set(ylabel="min(u) / max(u exact)", title="Valeurs négatives (oscillations)")
a2.axhline(0, color="k", lw=0.8)
sauver_figure(fig, "01_peclet.png")

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(profils["s"], profils["exacte"], "k-", lw=2, label="exacte")
ax.plot(profils["s"], profils["centre"], "C0.-", label="centré")
ax.plot(profils["s"], profils["upwind"], "C1.-", label="upwind")
ax.set(xlabel="abscisse le long de la diagonale", ylabel="u", title="Profil à t = 60 pour Péclet = 8")
ax.legend()
sauver_figure(fig, "01_profil_peclet.png")
chiffres["peclet"] = {
    "pe": pecs,
    "erreur_centre": [round(e, 4) for e in res["centre"]["err"]],
    "erreur_upwind": [round(e, 4) for e in res["upwind"]["err"]],
    "min_centre": [round(m, 4) for m in res["centre"]["min"]],
    "min_upwind": [round(m, 4) for m in res["upwind"]["min"]],
}

# ---------------------------------------------------------------- 4. animation
print("== animation")
s = SolveurDispersion(pb, 101, 101, dt=0.5)
_, images = s.integrer(T, nb_images=40)
fig, ax = plt.subplots(figsize=(4.6, 4))
vmax = images[0][1].max()
im = ax.imshow(images[0][1], origin="lower", extent=[0, pb.Lx, 0, pb.Ly], vmin=0, vmax=vmax, cmap="viridis")
titre = ax.set_title("t = 0")
ax.set(xlabel="x", ylabel="y")
fig.colorbar(im, ax=ax, label="concentration")
fig.tight_layout()


def maj(k):
    t, u = images[k]
    im.set_data(u)
    titre.set_text(f"t = {t:.0f}")
    return im, titre


anim = animation.FuncAnimation(fig, maj, frames=len(images), interval=100)
anim.save(MEDIA / "dispersion_simulation.gif", writer="pillow", fps=10)
plt.close(fig)
print("gif :", MEDIA / "dispersion_simulation.gif")

sauver_chiffres("validation", chiffres)
