# Étude de Dispersion de Polluant

Ce projet simule la dispersion d'un polluant dans un domaine 2D en utilisant des méthodes numériques pour résoudre l'équation de convection-diffusion.

## Description

Le code implémente une simulation numérique de la dispersion d'un polluant initialement concentré en un point, se propageant sous l'effet de la convection et de la diffusion. La simulation utilise:

- **Méthodes directes**: Factorisation LU pour résoudre les systèmes linéaires
- **Méthodes itératives**: Jacobi et Gauss-Seidel pour comparaison de convergence
- **Résolution PDE**: Méthode des différences finies implicite pour l'équation de convection-diffusion
- **Visualisation**: Animation 3D en temps réel de l'évolution de la concentration

## Structure du Projet

```
Dispersion-Polluant/
├── main.py               # Code complet (calculs + visualisation)
├── .gitignore            # Fichiers à ignorer par Git
├── requirements.txt      # Dépendances Python
├── README.md             # Ce fichier
└── media/
    └── dispersion_simulation.gif  # Animation de démonstration
```

## Installation

1. Cloner le repository:
```bash
git clone <url-du-repo>
cd Dispersion-Polluant
```

2. Créer un environnement virtuel:
```bash
python -m venv .venv
# Sur Windows:
.venv\Scripts\activate
# Sur Linux/Mac:
source .venv/bin/activate
```

3. Installer les dépendances:
```bash
pip install -r requirements.txt
```

## Utilisation

Lancer la simulation:
```bash
python main.py
```

La simulation va:
1. Effectuer les tests des méthodes LU, Jacobi et Gauss-Seidel
2. Afficher une visualisation 3D en temps réel de la dispersion
3. Sauvegarder une animation GIF dans `media/dispersion_simulation.gif`

## Paramètres de Simulation

- **Domaine**: 100x100 unités
- **Grille**: 30x30 points
- **Temps**: 100 unités (200 pas de temps)
- **Vitesse**: [0.5, 0.5] unités/temps
- **Coefficient de diffusion**: κ = 0.9
- **Condition initiale**: Gaussienne centrée en (10,10) avec σ = 0.5

## Résultats

La simulation montre:
- La propagation convective du polluant
- L'effet diffusif qui lisse la concentration
- La convergence des méthodes itératives
- Une animation visuelle de l'évolution temporelle

## Dépendances

- numpy: Calculs numériques
- scipy: Algèbre linéaire (LU, lstsq)
- matplotlib: Visualisation 3D et animation
- pillow: Sauvegarde des GIFs

## Auteur

[Firas] - Étude numérique de dispersion de polluant
