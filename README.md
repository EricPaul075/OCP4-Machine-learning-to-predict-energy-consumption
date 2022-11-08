# OCP4-Machine-learning-to-predict-energy-consumption
Machine learning regression model to predict energy consumption and GHG emission

Le dataset est assemblé à partir de données relevées sur deux années et les variables cibles sélectionnées.
Un traitement approfondi est mis en place pour traiter les nombreuses valeurs manquantes et les valeurs erronées.
Le jeu de données est filtré pour se concentrer sur les features pertinentes pour la modélisation et éviter les risques de data leakage. Il est également simplifié s'agissant de certaines variables catégorielles qui contiennent un trop grand nombre de catégories.
L'analyse univariée met en évidence la nécessité de transformer certaines données, dont les variables cibles, afin de disposer d'une répartition favorisant la performance du machine learning.
L'analyse multivariée met en évidence des corrélations permettant de filtrer davantage les features du jeu de données.

Le machine learning s'appuie sur la librairie scikit-learn et sur XGBoost, en comparant différents types de modèles entre eux et par rapport à la baseline du dummyregressor: modèles linéaires (ridge, lasso, elasticnet), modèles non linéaires (Kernel SVM, kernel ridge et réseau de neurones), et modèles ensemblistes (random forest, adaboost et xgboost).
La comparaison s'effectue selon un unique processus pour tous les modèles et comprenant:
- Le découpage du jeu en une partie pour l'apprentissage et l'autre pour le test ;
- La fixation d'un random state pour assurer la reproductibilité ;
- la définition de la métrique principale et des métriques secondaires ;
- la mesure du temps d'apprentissage de chaque modèle ;
- l'optimisation des hyperparamètres de chaque modèle, par une recherche sur grille avec validation croisée et la visualisation graphique de l'impact des paramètres sur le score ;
- l'examen des courbes d'apprentissage, pour identifier un possible sur-apprentissage ou fuite de données, en plus de la capacité d'apprentissage de chaque modèle.

Un bilan permet de sélectionner le meilleur modèle et examiner les possibilités de simplification.
L'analyse de l'importance des features permet par ailleurs de conclure sur l'intérêt de disposer de l’ENERGY STAR Score pour la prédiction des émissions de GES.
