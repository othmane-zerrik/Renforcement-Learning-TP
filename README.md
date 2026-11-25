# TP 4 : Apprentissage par renforcement (RL) avec CartPole et PPO


## 1. Objectif

Comprendre les bases du RL et entraîner un agent **PPO** (Stable-Baselines3) sur l'environnement **CartPole-v1** (Gymnasium), puis analyser les résultats.

## 2. Principe du RL

Un **agent** interagit avec un **environnement** : à chaque pas il observe un état, choisit une action et reçoit une récompense. Son but est de maximiser la récompense cumulée. Le cadre formel est le **MDP** (S, A, R, P, γ).

| Élément | Dans CartPole |
|---|---|
| États S | position et vitesse du chariot, angle et vitesse angulaire de la tige |
| Actions A | pousser à gauche (0) ou à droite (1) |
| Récompense R | +1 à chaque pas où la tige reste droite |
| Transition P | dynamique physique du système |
| γ | facteur d'actualisation (0,99 par défaut dans SB3) |

L'épisode se termine si la tige tombe (terminated) ou après 500 pas (truncated). La **récompense maximale est donc 500**.

## 3. Algorithme utilisé : PPO

PPO (Proximal Policy Optimization) est une méthode **Actor-Critic** :
- **Actor** : le réseau qui choisit les actions (la politique).
- **Critic** : le réseau qui estime la valeur d'un état.
- Le paramètre `clip_range = 0,2` limite l'ampleur de chaque mise à jour, ce qui stabilise l'apprentissage.

## 4. Installation

```powershell
python -m venv rl_tp
rl_tp\Scripts\activate
pip install gymnasium "stable-baselines3[extra]" matplotlib seaborn pandas
```

Environnement utilisé : Python 3.12, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0 (CPU).

## 5. Fichiers

| Fichier | Rôle |
|---|---|
| `exemple1.py` | entraîne PPO (10 000 pas) puis teste 1 épisode avec affichage |
| `exemple2.py` | entraîne PPO puis teste 5 épisodes et trace la courbe |
| `courbe_recompense.png` | courbe générée par l'exemple 2 |

Lancement : `python exemple1.py` puis `python exemple2.py`.

## 6. Résultats

### Entraînement (mesuré)

| Pas d'entraînement | Exemple 1 `ep_rew_mean` | Exemple 2 `ep_rew_mean` |
|---|---|---|
| 2 048 | 22 | 23,2 |
| 4 096 | 27,4 | 26 |
| 6 144 | 37,2 | 34 |
| 8 192 | 47,3 | 44,6 |
| 10 240 | 60,9 | 58,2 |

### Test de l'agent entraîné

- Exemple 1, récompense de l'épisode : **... (à compléter avec la valeur affichée par `exemple1.py`)**
- Exemple 2, récompenses des 5 épisodes : **248, 500, 500, 500, 500** (moyenne = 449,6)

![Courbe des récompenses](courbe_recompense.png)

## 7. Explication des résultats

**Exemple 1.** On crée CartPole, on entraîne PPO pendant 10 000 pas, puis on teste un épisode. Comme la récompense est +1 par pas, `ep_rew_mean` est égale à `ep_len_mean` : c'est la durée moyenne pendant laquelle la tige reste droite. Elle passe d'environ 22 à 61 : l'agent a presque triplé sa durée d'équilibre, ce qui prouve qu'il apprend. Elle reste loin de 500, car 10 000 pas sont insuffisants pour converger.

**Exemple 2.** Même entraînement, mais le test porte sur 5 épisodes et on trace la récompense par épisode. La courbe ne montre pas l'apprentissage, mais la **régularité de la politique finale** : des valeurs proches indiquent une politique stable. Avec `deterministic=True`, l'agent applique toujours l'action la plus probable, donc seules les conditions initiales aléatoires font varier les résultats.

**Lecture des résultats réels de l'exemple 2.** L'agent obtient 248 au premier épisode, puis 500 aux quatre suivants, c'est-à-dire le maximum de CartPole-v1 (l'épisode s'arrête par limite de temps, pas par échec). La moyenne est de 449,6. Ce résultat peut surprendre : pendant l'entraînement, `ep_rew_mean` n'était que d'environ 58. L'écart s'explique ainsi :
- pendant l'entraînement, les actions sont **tirées au hasard** selon la politique (exploration), ce qui provoque des erreurs et des chutes précoces ;
- pendant le test, `deterministic=True` fait choisir **toujours l'action la plus probable** (exploitation), ce qui est bien plus performant ;
- `ep_rew_mean` est une moyenne glissante sur les derniers épisodes de la phase de collecte, donc elle sous-estime la qualité de la politique finale.

Le premier épisode à 248 montre que la politique n'est pas encore parfaite : sur certaines conditions initiales, l'agent perd l'équilibre avant 500 pas. Un entraînement plus long ou plusieurs graines permettraient de confirmer la stabilité.

**Indicateurs d'entraînement.**
- `entropy_loss` passe de -0,686 à -0,607 : la politique devient moins aléatoire (moins d'exploration, plus d'exploitation).
- `explained_variance` monte de ~0 à ~0,3 : le critique prédit de mieux en mieux les retours.
- Les deux exécutions ont des courbes très proches : la tendance est reproductible, seule la graine aléatoire change les valeurs exactes.

**Exploration / exploitation.** Pendant l'entraînement, l'agent tire ses actions au hasard selon sa politique (exploration). Pendant le test, il choisit toujours la meilleure action connue (exploitation).

## 8. Pistes d'amélioration

- Augmenter `total_timesteps` (50 000 à 100 000) : l'agent approche généralement 500.
- Régler les hyperparamètres : `learning_rate`, `gamma`, `n_steps`, `batch_size`.
- Répéter avec plusieurs graines et moyenner les résultats.
- Utiliser `evaluate_policy` (20 épisodes) pour une évaluation fiable.
- Comparer avec d'autres algorithmes : DQN, A2C.
- Visualiser l'apprentissage avec TensorBoard ou `Monitor` + courbe de `ep_rew_mean`.

## 9. Conclusion

L'agent PPO apprend bien à équilibrer la tige : la récompense moyenne augmente de façon régulière au fil des itérations. Avec un entraînement aussi court, la performance n'est pas encore maximale, mais la tendance confirme le bon fonctionnement de l'algorithme.

## Références

- Gymnasium : https://gymnasium.farama.org
- Stable-Baselines3 : https://stable-baselines3.readthedocs.io
- OpenAI Spinning Up : https://spinningup.openai.com
