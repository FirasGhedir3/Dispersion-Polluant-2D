"""PINN : cas du papier, sensibilité, problème inverse, vitesse non uniforme.

Chaque expérience enregistre son résultat dans resultats/pinn/ dès qu'elle est finie, et
une expérience déjà faite n'est pas refaite (pratique, certaines sont longues).

    python experiences/03_pinn.py papier        # réseau 9x128, 6000 itérations (long : ~25 min sur CPU)
    python experiences/03_pinn.py etudes        # sensibilité + inverse + cisaillement (réseaux plus petits)
    python experiences/03_pinn.py figures       # fabrique les figures à partir des résultats enregistrés
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

from dispersion.modele import Probleme, erreur_L2_relative
from dispersion.outils import RESULTATS, sauver_chiffres, sauver_figure
from dispersion.pinn import Poids, entrainer, fabriquer_donnees, predire, temps_inference_ms
from dispersion.solveur import SolveurDispersion

DOSSIER = RESULTATS / "pinn"
DOSSIER.mkdir(exist_ok=True)
T = 0.25
NB_IMAGES_EVAL = 11        # on évalue l'erreur sur 11 instants régulièrement espacés


# ------------------------------------------------------------------ le cas du papier

def monde_papier():
    """Carré unité, v = (0.5, 0.5), D = 0.01, T = 0.25, grille 51x51 et 100 pas de temps.

    Le papier dit juste "gaussienne centrée dans le domaine" : le centre (0.5, 0.5) et la
    largeur sigma = 0.1 sont donc mon choix.
    """
    pb = Probleme(Lx=1.0, Ly=1.0, kappa=0.01, vx=0.5, vy=0.5, x0=0.5, y0=0.5, sigma=0.1)
    s = SolveurDispersion(pb, 51, 51, dt=T / 100)
    t0 = time.perf_counter()
    _, imgs = s.integrer(T, nb_images=100)
    duree_fd = time.perf_counter() - t0
    ts = np.array([t for t, _ in imgs])
    us = np.array([u for _, u in imgs])
    return pb, s, ts, us, duree_fd


def erreurs_sur_instants(reseau, s, pb, ts, us):
    """Erreur L2 relative aux 11 instants, contre la solution exacte et contre le solveur FD."""
    idx = np.linspace(0, len(ts) - 1, NB_IMAGES_EVAL).round().astype(int)
    e_ex, e_fd, preds = [], [], []
    for k in idx:
        p = predire(reseau, ts[k], s.X, s.Y)
        preds.append(p)
        e_ex.append(erreur_L2_relative(p, pb.exacte(s.X, s.Y, ts[k])))
        e_fd.append(erreur_L2_relative(p, us[k]))
    preds = np.array(preds)
    ex = np.array([pb.exacte(s.X, s.Y, ts[k]) for k in idx])
    globale_ex = float(np.linalg.norm(preds - ex) / np.linalg.norm(ex))
    globale_fd = float(np.linalg.norm(preds - us[idx]) / np.linalg.norm(us[idx]))
    return dict(instants=ts[idx].tolist(), err_exacte=e_ex, err_fd=e_fd,
                globale_exacte=globale_ex, globale_fd=globale_fd)


def sauver(nom, dico):
    (DOSSIER / f"{nom}.json").write_text(json.dumps(dico, indent=1))


def existe(nom):
    return (DOSSIER / f"{nom}.json").exists()


def charger(nom):
    return json.loads((DOSSIER / f"{nom}.json").read_text())


# ------------------------------------------------------------------ expérience 1 : réseau du papier

def run_papier():
    nom = "papier"
    if existe(nom):
        print("déjà fait :", nom)
        return
    pb, s, ts, us, duree_fd = monde_papier()
    erreur_fd_exacte = erreur_L2_relative(us[-1], pb.exacte(s.X, s.Y, T))
    donnees = fabriquer_donnees(((ts, us), (s.x, s.y)), nb_points=2000, bruit=0.01)
    print("réseau du papier : 9 couches x 128, Adam lr=6e-3, 6000 itérations", flush=True)
    reseau, hist, _ = entrainer(pb, T, donnees, couches=9, largeur=128, iterations=6000, lr=6e-3,
                                nb_colloc=1000, nb_ic=300, nb_bords=300)
    ev = erreurs_sur_instants(reseau, s, pb, ts, us)
    sauver(nom, dict(
        **ev, erreur_fd_vs_exacte_T=erreur_fd_exacte, duree_entrainement_s=hist.duree_s,
        duree_fd_s=duree_fd, inference_ms=temps_inference_ms(reseau, s.X, s.Y, T),
        nb_parametres=sum(p.numel() for p in reseau.parameters()),
        perte=hist.perte, iteration=hist.iteration,
        champ_pinn_T=predire(reseau, T, s.X, s.Y).tolist(), champ_fd_T=us[-1].tolist(),
        champ_exact_T=pb.exacte(s.X, s.Y, T).tolist()))
    print(f"  erreur globale vs FD {ev['globale_fd']:.4f}, vs exacte {ev['globale_exacte']:.4f}")


# ------------------------------------------------------------------ expérience 2 : sensibilité

PETIT = dict(couches=5, largeur=64, iterations=3000, lr=6e-3, nb_colloc=1000, nb_ic=300, nb_bords=300)


def une_run_petite(nom, nb_points, bruit, poids=Poids(), graine=0):
    if existe(nom):
        print("déjà fait :", nom)
        return
    pb, s, ts, us, _ = monde_papier()
    donnees = fabriquer_donnees(((ts, us), (s.x, s.y)), nb_points, bruit, graine=graine)
    reseau, hist, _ = entrainer(pb, T, donnees, poids=poids, graine=graine, verbeux=False, **PETIT)
    ev = erreurs_sur_instants(reseau, s, pb, ts, us)
    sauver(nom, dict(nb_points=nb_points, bruit=bruit, poids_physique=poids.physique,
                     globale_exacte=ev["globale_exacte"], globale_fd=ev["globale_fd"],
                     duree_s=hist.duree_s))
    print(f"  {nom:32s} erreur vs exacte {ev['globale_exacte']:.4f}", flush=True)


def run_sensibilite():
    print("== sensibilité (réseau 5x64, 3000 itérations, graine 0)")
    une_run_petite("sens_base", 2000, 0.01)
    for b in (0.0, 0.05, 0.10):
        une_run_petite(f"sens_bruit_{b}", 2000, b)
    for n in (100, 500):
        une_run_petite(f"sens_N_{n}", n, 0.01)
    for n in (100, 500, 2000):
        une_run_petite(f"sens_N_{n}_sans_physique", n, 0.01, Poids(physique=0.0))
    for w in (0.1, 10.0):
        une_run_petite(f"sens_wphys_{w}", 2000, 0.01, Poids(physique=w))


# ------------------------------------------------------------------ expérience 3 : problème inverse

def run_inverse():
    print("== problème inverse : retrouver D et v (départ D=0.05, v=(0.2, 0.8) ; vrai D=0.01, v=(0.5, 0.5))")
    pb, s, ts, us, _ = monde_papier()
    cas = [("inv_bruit_0.0", 2000, 0.0), ("inv_bruit_0.01", 2000, 0.01),
           ("inv_bruit_0.05", 2000, 0.05), ("inv_bruit_0.1", 2000, 0.10), ("inv_N200_bruit_0.05", 200, 0.05)]
    for nom, n, b in cas:
        if existe(nom):
            print("déjà fait :", nom)
            continue
        donnees = fabriquer_donnees(((ts, us), (s.x, s.y)), n, b)
        reseau, hist, appris = entrainer(pb, T, donnees, apprendre={"D": 0.05, "vx": 0.2, "vy": 0.8},
                                         verbeux=False, **{**PETIT, "iterations": 4000})
        sauver(nom, dict(nb_points=n, bruit=b, appris=appris, vrai=dict(D=0.01, vx=0.5, vy=0.5),
                         historique=dict(iteration=hist.iteration, D=hist.D, vx=hist.vx, vy=hist.vy),
                         duree_s=hist.duree_s))
        print(f"  {nom:24s} D={appris['D']:.4f} vx={appris['vx']:.4f} vy={appris['vy']:.4f}", flush=True)


def run_inverse_variantes():
    """D sort sous-estimé dans le cas de base : on regarde si c'est un problème de réglage."""
    print("== problème inverse, variantes d'entraînement (bruit 1 %, 2000 mesures)")
    pb, s, ts, us, _ = monde_papier()
    donnees = fabriquer_donnees(((ts, us), (s.x, s.y)), 2000, 0.01)
    variantes = [("inv_var_10000_iterations", dict(iterations=10000)),
                 ("inv_var_lr_2e-3", dict(iterations=8000, lr=2e-3))]
    for nom, opts in variantes:
        if existe(nom):
            print("déjà fait :", nom)
            continue
        reseau, hist, appris = entrainer(pb, T, donnees, apprendre={"D": 0.05, "vx": 0.2, "vy": 0.8},
                                         verbeux=False, **{**PETIT, **opts})
        sauver(nom, dict(nb_points=2000, bruit=0.01, appris=appris, vrai=dict(D=0.01, vx=0.5, vy=0.5),
                         historique=dict(iteration=hist.iteration, D=hist.D, vx=hist.vx, vy=hist.vy),
                         duree_s=hist.duree_s, options=opts))
        print(f"  {nom:28s} D={appris['D']:.4f} vx={appris['vx']:.4f} vy={appris['vy']:.4f}", flush=True)


# ------------------------------------------------------------------ expérience 4 : vitesse non uniforme

def champ_cisaillement_np(X, Y):
    return 0.5 * np.ones_like(X), 0.4 * np.sin(2 * np.pi * X)


def champ_cisaillement_torch(x, y):
    return 0.5 * torch.ones_like(x), 0.4 * torch.sin(2 * torch.pi * x)


def run_cisaillement():
    nom = "cisaillement"
    if existe(nom):
        print("déjà fait :", nom)
        return
    print("== vitesse non uniforme : v = (0.5, 0.4 sin(2 pi x))")
    pb = Probleme(Lx=1.0, Ly=1.0, kappa=0.01, vx=0.5, vy=0.0, x0=0.3, y0=0.4, sigma=0.1)

    def fd(n, nt):
        s = SolveurDispersion(pb, n, n, dt=T / nt, vx=lambda X, Y: champ_cisaillement_np(X, Y)[0],
                              vy=lambda X, Y: champ_cisaillement_np(X, Y)[1])
        _, imgs = s.integrer(T, nb_images=nt)
        return s, np.array([t for t, _ in imgs]), np.array([u for _, u in imgs])

    s_ref, ts_ref, us_ref = fd(201, 400)          # référence fine
    s, ts, us = fd(51, 100)                       # celle qui fournit les "mesures"
    # est-ce que la grille 51 est assez fine ? on la compare à la référence sur les points communs
    ecart_fd = erreur_L2_relative(us[-1], us_ref[-1][::4, ::4])
    donnees = fabriquer_donnees(((ts, us), (s.x, s.y)), 2000, 0.01)
    reseau, hist, _ = entrainer(pb, T, donnees, couches=6, largeur=80, iterations=5000, lr=6e-3,
                                nb_colloc=1000, nb_ic=300, nb_bords=300, vitesse=champ_cisaillement_torch,
                                verbeux=False)
    idx = np.linspace(0, len(ts) - 1, NB_IMAGES_EVAL).round().astype(int)
    preds = np.array([predire(reseau, ts[k], s.X, s.Y) for k in idx])
    ref = np.array([us_ref[4 * k][::4, ::4] for k in idx])    # 400 pas = 4 fois plus que 100
    globale = float(np.linalg.norm(preds - ref) / np.linalg.norm(ref))
    sauver(nom, dict(ecart_fd51_vs_fd201=ecart_fd, globale_vs_reference=globale, duree_s=hist.duree_s,
                     champ_pinn_T=preds[-1].tolist(), champ_ref_T=ref[-1].tolist()))
    print(f"  FD 51 vs FD 201 : {ecart_fd:.4f}   PINN vs référence : {globale:.4f}", flush=True)


# ------------------------------------------------------------------ figures

def figures():
    chiffres = {}
    # --- réseau du papier
    if existe("papier"):
        r = charger("papier")
        pinn, fd = np.array(r["champ_pinn_T"]), np.array(r["champ_fd_T"])
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
        vmax = fd.max()
        for ax, champ, titre in zip(axes[:2], (fd, pinn), ("solveur FD (référence)", "PINN 9x128")):
            im = ax.imshow(champ, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=vmax)
            ax.set_title(f"{titre}, t = T")
            ax.grid(False)
        im2 = axes[2].imshow(np.abs(pinn - fd), origin="lower", extent=[0, 1, 0, 1], cmap="magma")
        axes[2].set_title("|PINN - FD|")
        axes[2].grid(False)
        fig.colorbar(im2, ax=axes[2], fraction=0.046)
        axes[3].semilogy(r["iteration"], r["perte"])
        axes[3].set(xlabel="itération", ylabel="perte totale", title="Apprentissage")
        sauver_figure(fig, "03_pinn_papier.png")

        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(r["instants"], r["err_fd"], "o-", label="contre le solveur FD")
        ax.plot(r["instants"], r["err_exacte"], "s--", label="contre la solution exacte")
        ax.set(xlabel="t", ylabel="erreur L2 relative", title="Erreur du PINN au cours du temps")
        ax.legend()
        sauver_figure(fig, "03_pinn_erreur_temps.png")
        chiffres["papier"] = {k: r[k] for k in ("globale_fd", "globale_exacte", "erreur_fd_vs_exacte_T",
                                                "duree_entrainement_s", "duree_fd_s", "inference_ms", "nb_parametres")}

    # --- sensibilité
    noms = [p.stem for p in DOSSIER.glob("sens_*.json")]
    if noms:
        res = {n: charger(n) for n in noms}
        g = lambda n: res[n]["globale_exacte"] if n in res else np.nan
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        bruits = [0.0, 0.01, 0.05, 0.10]
        axes[0].plot([100 * b for b in bruits],
                     [g("sens_base") if b == 0.01 else g(f"sens_bruit_{b}") for b in bruits], "o-")
        axes[0].set(xlabel="bruit (% du maximum)", ylabel="erreur L2 relative", title="Effet du bruit (2000 mesures)")
        Ns = [100, 500, 2000]
        avec = [g("sens_base") if n == 2000 else g(f"sens_N_{n}") for n in Ns]
        sans = [g(f"sens_N_{n}_sans_physique") for n in Ns]
        axes[1].semilogx(Ns, avec, "o-", label="avec la physique")
        axes[1].semilogx(Ns, sans, "s--", label="données seules (poids physique = 0)")
        axes[1].set(xlabel="nombre de mesures", title="Effet du nombre de mesures (bruit 1 %)")
        axes[1].legend()
        ws = [0.1, 1.0, 10.0]
        axes[2].semilogx(ws, [g("sens_wphys_0.1"), g("sens_base"), g("sens_wphys_10.0")], "o-")
        axes[2].set(xlabel="poids du terme physique", title="Effet de la pondération (2000 mesures)")
        sauver_figure(fig, "03_sensibilite.png")
        chiffres["sensibilite"] = {n: round(v["globale_exacte"], 4) for n, v in sorted(res.items())}

    # --- inverse
    noms = sorted(p.stem for p in DOSSIER.glob("inv_*.json"))
    if noms:
        res = {n: charger(n) for n in noms}
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        courbes = [(n, lbl) for n, lbl in (("inv_bruit_0.01", "4000 itérations"),
                                           ("inv_var_10000_iterations", "10000 itérations")) if n in res]
        for ax, cle, vrai in zip(axes, ("D", "vx", "vy"), (0.01, 0.5, 0.5)):
            for n, lbl in courbes:
                h = res[n]["historique"]
                ax.plot(h["iteration"], h[cle], label=lbl)
            ax.axhline(vrai, color="k", ls="--", lw=1, label="vraie valeur")
            ax.set(xlabel="itération", title=f"{cle} appris (bruit 1 %)")
            ax.legend()
        sauver_figure(fig, "04_inverse.png")
        chiffres["inverse"] = {n: dict(appris={k: round(v, 4) for k, v in r["appris"].items()},
                                       nb_points=r["nb_points"], bruit=r["bruit"]) for n, r in res.items()}

    # --- cisaillement
    if existe("cisaillement"):
        r = charger("cisaillement")
        pinn, ref = np.array(r["champ_pinn_T"]), np.array(r["champ_ref_T"])
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
        X, Y = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
        vx, vy = champ_cisaillement_np(X, Y)
        axes[0].quiver(X, Y, vx, vy)
        axes[0].set_title("champ de vitesse")
        for ax, champ, titre in zip(axes[1:3], (ref, pinn), ("référence FD fine", "PINN")):
            ax.imshow(champ, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=ref.max())
            ax.set_title(f"{titre}, t = T")
            ax.grid(False)
        im = axes[3].imshow(np.abs(pinn - ref), origin="lower", extent=[0, 1, 0, 1], cmap="magma")
        axes[3].set_title("|PINN - référence|")
        axes[3].grid(False)
        fig.colorbar(im, ax=axes[3], fraction=0.046)
        sauver_figure(fig, "05_cisaillement.png")
        chiffres["cisaillement"] = {k: r[k] for k in ("ecart_fd51_vs_fd201", "globale_vs_reference", "duree_s")}

    sauver_chiffres("pinn", chiffres)
    print(json.dumps(chiffres, indent=1))


if __name__ == "__main__":
    commande = sys.argv[1] if len(sys.argv) > 1 else "etudes"
    if commande == "papier":
        run_papier()
    elif commande == "etudes":
        run_cisaillement()
        run_sensibilite()
        run_inverse()
    elif commande == "inverse_variantes":
        run_inverse_variantes()
    elif commande == "figures":
        figures()
    else:
        sys.exit(f"commande inconnue : {commande}")
