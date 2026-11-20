"""Petits utilitaires partagés par les scripts d'expériences."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # pas de fenêtre : on enregistre les figures dans des fichiers
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parent.parent
RESULTATS = RACINE / "resultats"
MEDIA = RACINE / "media"
RESULTATS.mkdir(exist_ok=True)
MEDIA.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def sauver_chiffres(cle, valeurs):
    """Range les chiffres d'une expérience dans resultats/chiffres.json (pour le README)."""
    chemin = RESULTATS / "chiffres.json"
    donnees = json.loads(chemin.read_text()) if chemin.exists() else {}
    donnees[cle] = valeurs
    chemin.write_text(json.dumps(donnees, indent=2, ensure_ascii=False))


def sauver_figure(fig, nom):
    fig.tight_layout()
    fig.savefig(RESULTATS / nom)
    plt.close(fig)
    print("figure :", RESULTATS / nom)
