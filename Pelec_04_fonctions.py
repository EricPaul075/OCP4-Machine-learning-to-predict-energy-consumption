#***********************************************************************************************************************
# Liste des fonctions de ce module :
#
# Fonctions d'export vers MS-Excel:
# - illegal_char_remover : retire certains caractères provoquant des erreurs d'écriture vers Excel
# - writeDF2XLfile : écrit un dataframe dans un fichier excel
#
# Fonctions pour l'analyse exploratoire
# - polyreg : renvoie la régression linéaire
# - pair_plot : trace le nuage x,y avec possibilité de tracé de nuage partiel
# - eta_squared : calcule le rapport de corrélation entre une variable catégorielle et une variable numérique
# - df_style_fct, df_style : fonction de style pour l'affichage des dataframes de corrélation
# - sort_columns, sort_index, sort_labels: fonctions utilisées par cor_table
# - cor_table: calcule et affiche la table des corrélations de Pearson d'une liste de variables numériques 2 à 2
# - eta_table: calcule et affiche la table des rapport de corrélations entre variables catégorielles et numériques
#
# Fonctions de transformation inverse des cibles de la modélisation
# t_nrj, t_ghg
#
# Fonctions d'initialisation de l'étude de modélisation
# - init_target_list : initialise la liste des variables cibles de la modélisation
# - init_scoring_list : initialise la liste des scorings qui seront utilisés par les fonctions de modélisation
# - init_criterion_list : initialise la liste des critères de comparaison des modèles
# - model_name : retourne le nom du modèle sous forme de chaine de caractères, utilisée par append_model
# - append_model : ajoute un nouveau modèle à la liste de comparaison
# - init_result_table : initialise la table des résultats
# - df_styler : fonction de style pour l'affichage de la table des résultats
#
# Fonctions utilisées pour la modélisation
# - search_best_model : entrainement pour la recherche sur grille et affichage du résultat
# - model_eval : évaluation de modèle sur l'échantillon de test et enregistrement du résultat
# - eval_results : affichage du résultat comparé des modèles et enregistrement dans un fichier csv
# - plot_model_comparison : affichage graphique de la comparaison des modèles (bargraph horizontal)
# - learning_graph : affiche la learning curve pour 3 scorings (r2, MSE, MAE)
# - plot_estimator_coef : bargraph horizontal des coefficients du modèle de régression linéaire
# - plot_feature_importance : bargraph horizontal de l'importance des features du modèle ensembliste
# - coef_vs_alpha : graphe des coefficient de la régression linéaire en fonction de alpha
# - show_gram : affiche graphiquement des 100x100 premières lignes et colonnes de la matrice de Gram
# - get_param : sous-fonction de plt_grid pour ordonner les paramètres de grille
# - plt_grid : affichage des graphes de score en fonction des paramètres de grille (max 3)
#
#***********************************************************************************************************************

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from numpy.polynomial import polynomial as P
import scipy.stats as st
from scipy.stats import f
import seaborn as sns
import re
import sys

# Display options
from IPython.display import (display, display_html, display_png, display_svg)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 199)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# Colorama
from colorama import Fore, Back, Style
# Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Style: DIM, NORMAL, BRIGHT, RESET_ALL

# Répertoires de sauvegarde
dir_data = ".\Pelec_05_data"


# Remove ILLEGAL CHARACTER to write DataFrame in a MS-Excel file
ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]|[\x00-\x1f\x7f-\x9f]|[\uffff]')
def illegal_char_remover(df):
    if isinstance(df, str):
        return ILLEGAL_CHARACTERS_RE.sub("", df)
    else:
        return df

# Ecriture dans un fichier MS-Excel
def writeDF2XLfile(data, fileName):
    data4xl = data.applymap(illegal_char_remover)
    data4xl.to_excel(fileName + ".xlsx", encoding='utf-8-sig')


# Régression avec np.polynomial.polynomial.polyfit entre les colonnes xlabel et ylabel du DataFrame data
# Si l'argument 'full' n'est pas spécifié, renvoie les coefficients du polynôme: poly[i] coefficient du terme en X^i
# Sinon retourne 'reg' avec:
# - poly=reg[0] coefficients du polynôme: poly[i] coefficient du terme en X^i
# - list=reg[1] permet de calculer r2 = 1 - list[0][0] / (np.var(yData) * len(yData))
def polyreg(data, xlabel, ylabel, full=False, deg=1):
    xData = data[xlabel].copy(deep=True)
    yData = data[ylabel].copy(deep=True)
    if not full: return P.polyfit(xData, yData, deg=deg)
    else: return P.polyfit(xData, yData, deg=deg, full=True)



# Tracé de graphe par paire de variables numériques avec :
# - à gauche le nuage de l'ensemble des points
# - à droite, si un ou plusieurs arguments sont spécifiés, le nuage réduit avec la courbe de tendance
def pair_plot(data, pair, exclude_x=None, exclude_y=None, xmin=None, xmax=None, ymin=None, ymax=None):
    # Initialisation des paramètres avec les kwargs
    if exclude_x is not None: bx=True
    else: bx=False
    if exclude_y is not None: by=True
    else: by=False
    if xmin is not None: bxmin=True
    else: bxmin=False
    if xmax is not None: bxmax=True
    else: bxmax=False
    if ymin is not None: bymin=True
    else: bymin=False
    if ymax is not None: bymax=True
    else: bymax=False

    df = data[pair].copy(deep=True)
    coef_p = st.pearsonr(df[pair[0]], df[pair[1]])[0]
    # Pour éventuellement examiner la corrélation en excluant les valeurs nulles
    if not bx and not by and not bxmin and not bxmax and not bymin and not bymax:
        print("Nuage complet, Pearson=" + f"{coef_p:.2f}")
        sns.jointplot(data=df, x=pair[0], y=pair[1], kind="reg", marginal_kws=dict(bins=20, fill=True))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        suptitle = "Pairplot " + pair[0] + " et " + pair[1]
        fig.suptitle(suptitle)
        sns.scatterplot(ax=axes[0], data=df, x=pair[0], y=pair[1])
        title = "Nuage complet, Pearson=" + f"{coef_p:.2f}"
        axes[0].set_title(title)
        if bx:
            df = df[df[pair[0]] != exclude_x]
        if by:
            df = df[df[pair[1]] != exclude_y]
        if bxmin:
            df = df[df[pair[0]] >= xmin]
        if bxmax:
            df = df[df[pair[0]] <= xmax]
        if bymin:
            df = df[df[pair[1]] >= ymin]
        if bymax:
            df = df[df[pair[1]] <= ymax]
        coef_p_ve = st.pearsonr(df[pair[0]], df[pair[1]])[0]
        g = sns.scatterplot(ax=axes[1], data=df, x=pair[0], y=pair[1])
        poly = polyreg(df, pair[0], pair[1])
        g.axline(xy1=(0, poly[0]), slope=poly[1], color='k', dashes=(5, 2))
        title = "Nuage partiel, Pearson=" + f"{coef_p_ve:.2f}"
        axes[1].set_title(title)
    plt.tight_layout()
    plt.show()
    # plt.savefig("Figure - pairplot recherche correlation.png", dpi=150)


# Calcul du rapport de corrélation entre une variable catégorielle x et une variable quantitative y
def eta_squared(x, y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT


# Fonctions de style d'affichage de dataframe avec display
# Code type: df.style.pipe(df_style).applymap(df_style_fct)
# Seuil de mise en évidence de valeurs, à ajuster dans le Notebook
thresshold = 0.5
# Ajoute une bordure rouge épaisse aux valeurs supérieures au seuil
def df_style_fct(val):
    if (val > thresshold) and (val!=1):
        border = '3px solid red'
    else:
        border = '1px solid black'
    return 'border: %s' % border
# Règle le style d'affichage du dataframe
# - Nombres float avec 2 chiffres après la virgule
# - Gradient de couleurs avec la colormap "BuGn" entre les valeurs vmin et vmax
def df_style(styler):
    styler.format("{:.2f}")
    styler.background_gradient(axis=None, vmin=0, vmax=1, cmap="seismic")
    return styler

def sort_columns(data):
    return data[sorted(data.columns.tolist())]

def sort_index(data):
    return data.reindex(sorted(data.index.tolist()))

def sort_labels(data):
    return sort_index(sort_columns(data))

def cor_table(data, num_features, thresshold=0.5, XLexport=None):
    # Etablissement de la table des corrélations entre variables numériques
    df_cor = np.array([st.pearsonr(data[feature], data[other_feature])[0] for feature in num_features for other_feature in num_features])
    df_cor = pd.DataFrame(np.reshape(df_cor, (len(num_features), len(num_features))), index=num_features, columns=num_features)

    # Export optionnel de la table dans un fichier excel
    if XLexport is not None:
        writeDF2XLfile(df_cor, XLexport)

    # Filtrage avec la valeur de seuil
    df_hit = df_cor.applymap(lambda x: x >= thresshold)
    print(Fore.LIGHTGREEN_EX + "* Recherche de corrélations potentielle entre les caractéristiques numériques",
          ", coefficient de Pearson >", thresshold, ":" + Style.RESET_ALL)
    list_cor = []
    list_pair_cor = []
    for feature in num_features:
        for other_feature in df_hit.index[df_hit[feature]==True].tolist():
            if other_feature != feature:
                if other_feature not in list_cor:
                    list_cor.append(other_feature)
                if [other_feature, feature] not in list_pair_cor:
                    list_pair_cor.append([feature, other_feature])
        df_cor.at[feature, feature] = 1
    df_cor = df_cor.loc[list_cor, list_cor]
    df_cor = sort_labels(df_cor)
    # Affichage du résultat
    print(Fore.LIGHTBLUE_EX + "  -> Liste des", len(list_pair_cor), "paires de caractéristiques à examiner :" + Style.RESET_ALL)
    display(list_pair_cor)
    # Affichage des correlations en surbrillance
    print(Fore.LIGHTBLUE_EX + "  -> Table des corrélations :" + Style.RESET_ALL)
    display(df_cor.style.pipe(df_style).applymap(df_style_fct))


# Etablissement d'une table de calcul des η² par paires de variables catégorielles,
# dont la liste est dans cat_features, et numériques, dont la liste est dans num_features
# La valeur de seuil (thresshold) filtre les valeurs supérieures
# Possibilité d'export de la table vers MS-Excel en spécifiant le nom de fichier
def eta_table(data, cat_features, num_features, thresshold=0.5, XLexport=None):
    # Matrice des corrélations
    df_eta = np.array(
        [eta_squared(data[feature], data[other_feature]) for feature in cat_features for other_feature in
         num_features])
    df_eta = pd.DataFrame(np.reshape(df_eta, (len(cat_features), len(num_features))), index=cat_features,
                          columns=num_features)

    # Export vers MS-Excel avec le fichier spécifié dans 'save'
    if XLexport is not None:
        writeDF2XLfile(df_eta, XLexport)

    # Test de la matrice par rapport à un seuil
    df_hit = df_eta.applymap(lambda x: x >= thresshold)
    print(
        Fore.LIGHTGREEN_EX + "* Recherche de corrélations potentielle entre caractéristiques catégorielles et numériques",
        ", coefficient η² >", thresshold, ":" + Style.RESET_ALL)

    # Liste des paires avec η² > thresshold et liste des exclusions par overfitting (trop de catégories)
    list_pair_eta = []
    cat_exclusions = []
    thr_cat_excl = int(data.shape[0] / 100)
    for num_feature in num_features:
        for cat_feature in df_hit.index[df_hit[num_feature] == True].tolist():
            if len(data[cat_feature].unique()) > thr_cat_excl:
                if cat_feature not in cat_exclusions:
                    cat_exclusions.append(cat_feature)
            elif [cat_feature, num_feature] not in list_pair_eta:
                list_pair_eta.append([cat_feature, num_feature])
    list_pair_eta.sort()

    # Affichage des exclusions par overfitting
    if len(cat_exclusions) > 0:
        print(Fore.LIGHTBLUE_EX + "  -> Caractéristiques exclues de l'analyse car comportant trop (>",
              thr_cat_excl, ", soit >{:.1f}%) de catégories (situation d'overfitting):".format(
                round(100.0 * thr_cat_excl / data.shape[0], 1)) +
              Style.RESET_ALL)
        display(cat_exclusions)
        df_eta = sort_labels(df_eta)
        display(df_eta.loc[cat_exclusions].style.pipe(df_style).applymap(df_style_fct))

    # Affichage des paires de caractéristiques au-dessus du seuil de corrélation
    print(Fore.LIGHTBLUE_EX + "  -> Liste des", len(list_pair_eta), "paires de caractéristiques à examiner :" +
          Style.RESET_ALL)
    display(list_pair_eta)
    cat_items = list(set([item[0] for item in list_pair_eta]))
    num_items = list(set([item[1] for item in list_pair_eta]))
    df_eta = df_eta.loc[cat_items, num_items]
    df_eta = sort_labels(df_eta)
    print(Fore.LIGHTBLUE_EX + "\n  -> Table des corrélations :" + Style.RESET_ALL)
    display(df_eta.style.pipe(df_style).applymap(df_style_fct))

#***********************************************************************************************************************
# Fonctions de transformation inverse des cibles
# → les variables de facteur d'échelle initialisées pour chaque fonction sont à fixer dans le Notebook1
#***********************************************************************************************************************

# Transformation inverse 'SiteEnergyUse(kBtu)'
min_nrj = 11441
max_nrj = 873923712
mean_nrj = 8047522
def t_nrj(df):
    df = np.log(min_nrj) + (np.log(max_nrj) - np.log(min_nrj)) * df
    df = np.exp(df)
    return df

# Transformation inverse 'TotalGHGEmissions'
min_ghg = 0
max_ghg = 16871
mean_ghg = 178
eps = 0.175
def t_ghg(df):
    df = df * (np.log(max_ghg+eps) - np.log(eps)) + np.log(eps)
    df = eps + np.exp(df)
    return df
#***********************************************************************************************************************
#
#Fonctions d'initialisation pour le machone learning et de formatage du tableau de résultats
#
# Initialise la liste des cibles
targets = []
def init_target_list(target_list):
    for target in target_list:
        targets.append(target)
    return targets

# Initialise la liste des scoring d'analyse des modèles
scorings = []
def init_scoring_list(scoring_list):
    for scoring in scoring_list:
        scorings.append(scoring)
    return scorings

# Initialise la liste des critères de comparaison des modèles
criterions = []
def init_criterion_list(criterion_list):
    for criterion in criterion_list:
        criterions.append(criterion)
    return criterions

# Retourne le nom du modèle (sans les '()')
def model_name(fname):
    return str(fname).split('(')[0]

# Initialisation du tableau des résultats pour chaque cible de modélisation
model_names = []
def append_model(model, suffix=''):
    name = model_name(model) + str(suffix)
    if name not in model_names:
        model_names.append(name)
    return clone(model), name

result_tables = {}
def init_result_table():
    columns = criterions.copy()
    columns.append('model')
    for target in targets:
        result_tables[target] = pd.DataFrame([], columns=columns)

# Style d'affichage des tables de résultat
formatter = {'r2': '{:.3f}',
             'neg_mean_squared_error': '{:.6f}',
             'neg_mean_absolute_error': '{:.6f}',
             'neg_median_absolute_error': '{:.6f}',
             'fit_time': '{:.3f}s'
             }
def df_styler(styler):
    styler.format(formatter, na_rep='-')
    styler.set_table_styles([
        {'selector': 'th.col_heading', 'props': 'text-align: center; border: 1px solid black; background-color: #4b8bbe'},
        {'selector': 'th.row_heading', 'props': 'text-align: right; border: 1px solid black; background-color: #4b8bbe'},
        {'selector': 'td', 'props': 'text-align: right; font-weight: bold; border: 1px solid black'}
    ], overwrite=False)
    #styler.highlight_min(color='red', axis=0)
    styler.highlight_max(color='steelblue', axis=0)
    return styler

########################################################################################################################
#Fonctions de machine learning

# Fonction d'évaluation de modèle
from sklearn import metrics
import timeit
from sklearn.base import clone
def model_eval(model, model_name, target, X_train, y_train, X_test, y_test):
    # Vérification qu'il s'agit d'un modèle de la table de résultat
    if model_name not in model_names:
        print("Le nom de modèle ne figure pas dans la liste\n")
        return None

    # Calcul des résultats
    columns = criterions.copy()
    columns.append('model')
    df = pd.DataFrame([], index=[model_name], columns=columns)
    mdl = clone(model)
    start_time = timeit.default_timer()
    mdl.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    y_pred = model.predict(X_test)
    if 'r2' in criterions:
        df.at[model_name, 'r2'] = metrics.r2_score(y_test, y_pred)
    if 'neg_mean_squared_error' in criterions:
        df.at[model_name, 'neg_mean_squared_error'] = -metrics.mean_squared_error(y_test, y_pred)
    if 'neg_mean_absolute_error' in criterions:
        df.at[model_name, 'neg_mean_absolute_error'] = -metrics.mean_absolute_error(y_test, y_pred)
    if 'neg_median_absolute_error' in criterions:
        df.at[model_name, 'neg_median_absolute_error'] =-metrics.median_absolute_error(y_test, y_pred)
    if 'fit_time' in criterions:
        df.at[model_name, 'fit_time'] = -elapsed
    df.at[model_name, 'model'] = mdl

    # Ajout du résultat dans la table
    if model_name in result_tables[target].index:
        result_tables[target].loc[model_name] = df.loc[model_name]
    else:
        result_tables[target] = pd.concat([result_tables[target], df], axis=0)

    # Enregistrement et affichage des résultats
    print(Fore.GREEN + f"  →  Evaluation sur l'échantillon de test :" + Style.RESET_ALL)
    display(result_tables[target][criterions].style.pipe(df_styler))

    return result_tables[target]


# Affichage du tableau de résultat
def eval_results(target):
    display(result_tables[target][criterions].style.pipe(df_styler))
    result_tables[target].to_csv(dir_data + "\\" + target + ".csv", encoding='utf-8-sig', index=True, sep=';')

# Tracé de l'histogramme de comparaison des modèles
def plot_model_comparison(target, criterions=criterions, subplots=True):
    df = result_tables[target][criterions].copy()
    if subplots==False:
        for col in df.columns:
            df[col] = 100 * (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        df.plot.barh(figsize=(15, int(len(df)/2)+1), legend=True, grid=True)
        plt.title(f"Comparaison des modèles pour {target} (base 0-100 pour min-max)", fontsize=14)
    else:
        df.plot.barh(subplots=True, figsize=(15, int(len(criterions)/2)+1), legend=False, grid=True)
    plt.tight_layout()
    plt.show()


# Fonction d'entrainement de modèle et affichage des résultats
# Retourne la grille de modèles entrainés
from datetime import timedelta
def search_best_model(grid, target, X_train, y_train):
    start_time = timeit.default_timer()
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    elapsed = timeit.default_timer() - start_time
    # Affichage des résultats
    print(Fore.GREEN + "► Modélisation de" + Style.RESET_ALL, target)
    print(Fore.GREEN + f"  → Meilleur modèle =" + Style.RESET_ALL, f"{model}")
    print(Fore.GREEN + f"  → Meilleurs paramètres =" + Style.RESET_ALL, f"{grid.best_params_}")
    print(Fore.GREEN + f"  → Meilleur score d'entrainement  =" + Style.RESET_ALL,
          f"{grid.best_score_:.4f}")
    print(Fore.GREEN + f"  → Temps de recherche =" + Style.RESET_ALL, f"{timedelta(seconds=elapsed)}")
    return grid


# Graphe de la learning curve pour une régression
from sklearn.model_selection import learning_curve
from scipy.special import ndtri, ndtr
from sklearn.base import clone
def learning_graph(model, X, y, scoring='neg_mean_squared_error', cv=4, n_pts=20, err=2*(1-ndtr(1)),
           n_jobs=-1, random_state=None):

    print(Fore.GREEN + f"  → Courbe d'apprentissage :" + Style.RESET_ALL)
    scorings = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

    n_pts = max(3, int(n_pts))
    train_sizes = []
    q = 0.01**(1/n_pts)
    for i in range(n_pts-1):
        train_sizes.append(1-q**(i+1))
    train_sizes.append(1)

    fig = plt.figure(figsize=(5*len(scorings), 4))
    for idx in range(len(scorings)):
        ax = fig.add_subplot(1, len(scorings), idx+1)
        N, train_score, val_score, fit_times, score_times = learning_curve(clone(model), X, y, scoring=scorings[idx],
                                                                           cv=cv,train_sizes=train_sizes, n_jobs=n_jobs,
                                                                           random_state=random_state, return_times=True)

        # Paramètres pour les courbes de train_score
        ts_mean = np.mean(train_score, axis=1)
        ts_std = np.std(train_score, axis=1)
        if idx == 0:
            ts_min = ts_mean - ndtri(1 - err / 2) * ts_std
            ts_max = ts_mean + ndtri(1 - err / 2) * ts_std
            ts_max[ts_max > 1] = 1
        else:
            ts_min = ts_mean - ndtri(1 - err / 2) * ts_std
            ts_max = ts_mean + ndtri(1 - err / 2) * ts_std
            ts_max[ts_max > 0] = 0

        # Paramètres pour les courbes de val_score
        vs_mean = np.mean(val_score, axis=1)
        vs_std = np.std(val_score, axis=1)
        if idx==0:
            vs_min = vs_mean - ndtri(1 - err / 2) * vs_std
            vs_max = vs_mean + ndtri(1 - err / 2) * vs_std
            vs_max[vs_max>1] = 1
        else:
            vs_min = vs_mean - ndtri(1 - err / 2) * vs_std
            vs_max = vs_mean + ndtri(1 - err / 2) * vs_std
            vs_max[vs_max>0] = 0

        # Recouvrement entre les faisceaux de courbes de training et validation
        rec = (ts_min[-1]<=vs_mean[-1]) or (vs_max[-1]>=ts_mean[-1])

        # Tracé des courbes de score
        ax.set_title(f"Scoring='{scorings[idx]}'\n(vs={vs_mean[-1]:.4f}, ts={ts_mean[-1]:.4f}, rec={rec})")
        ax.plot(N, ts_mean, color='coral', label='Train score')
        ax.fill_between(N, ts_max, ts_min, color='coral', alpha=0.2)
        ax.plot(N, vs_mean, color='steelblue', label='Validation score')
        ax.fill_between(N, vs_max, vs_min, color='steelblue', alpha=0.2)

        # Fixation des limites de tracé
        ax.set_xlim(left=N[0])
        if np.amax(val_score)>0:
            ax.set_ylim(bottom=max(0, np.amin(vs_mean)), top=np.amax(ts_max))
        else:
            ax.set_ylim(bottom=np.amin(vs_mean), top=min(0, np.amax(ts_max)))

        if scorings[idx] == scoring:
            # Tangente en fin de courbe de val_score
            n = max(int(n_pts/3), 3)
            vs_poly = P.polyfit(N[-n:], vs_mean[-n:], deg=1)
            vs_lc = max(vs_poly[1], 0)
            ts_poly = P.polyfit(N[-n:], ts_mean[-n:], deg=1)
            ts_lc = min(ts_poly[1], 0)
            dN = 0 if vs_lc==ts_lc else (ts_poly[0] - min(ts_std[-1], vs_std[-1]) - vs_poly[0]) / (vs_lc - ts_lc)
            ax.axline(xy1=(0, vs_poly[0]), slope=vs_poly[1], color='steelblue', ls='--', lw=1)
            ax.axline(xy1=(0, ts_poly[0]), slope=ts_poly[1], color='coral', ls='--', lw=1)

        plt.grid(visible=True)
        ax.legend()
    plt.suptitle("Courbes d'apprentissages selon différentes méthodes de scoring", fontsize=16)
    plt.tight_layout()
    plt.show()
    #return (vs_mean, ts_mean-vs_mean, ts_std, rec, dN)
    return dN

def plot_estimator_coef(model, labels_X):
    height = int(len(labels_X) / 4)
    pd.DataFrame(model.coef_, index=labels_X).plot.barh(figsize=(10, height))
    plt.gca().invert_yaxis()
    legend = plt.legend()
    legend.remove()
    plt.grid(visible=True)
    plt.title("Coefficients du meilleur modèle de régression", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, labels_X):
    height = int(len(labels_X) / 4)
    pd.DataFrame(model.feature_importances_, index=labels_X).plot.barh(figsize=(10, height))
    plt.gca().invert_yaxis()
    legend = plt.legend()
    legend.remove()
    plt.grid(visible=True)
    plt.title("Importance des features", fontsize=16)
    plt.tight_layout()
    plt.show()


# Graphe des coefficients de la régression en fonction de alpha (régularisation)
def coef_vs_alpha(grid, X_train, y_train, X_test, y_test):
    coefs = []
    mse = []

    model_name = str(grid.best_estimator_).split('(')[0]
    model = clone(grid.best_estimator_)

    n_alphas = 100
    best_alpha = grid.best_params_['alpha']
    alpha_min = np.log(best_alpha) - 1
    alpha_max = np.log(best_alpha) + 4
    alphas = np.logspace(alpha_min, alpha_max, n_alphas)

    for a in alphas:
        model.set_params(alpha=a)
        model.fit(X_train, y_train)
        coefs.append(model.coef_)
        mse.append(metrics.mean_squared_error(y_test, model.predict(X_test)))

    print(Fore.GREEN + f"  → Coefficients en fonction de la régularisation alpha :" + Style.RESET_ALL)

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(alphas, coefs)
    ax1.plot([best_alpha, best_alpha], [np.amin(coefs), np.amax(coefs)], ls='--', lw=1)
    ax1.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('coefficients')
    plt.title('Coefficients en fonction de la régularisation')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(alphas, mse)
    ax2.plot([best_alpha, best_alpha], [np.amin(mse), np.amax(mse)], ls='--', lw=1)
    ax2.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('MSE')
    plt.grid(visible=True)
    plt.title('MSE en fonction de la régularisation')
    plt.axis('tight')
    plt.suptitle(f"Regression {model_name} avec les meilleurs paramètres calculés", fontsize=16)
    plt.tight_layout()
    plt.show()

def show_gram(grid, X_train):
    kmatrix = metrics.pairwise.pairwise_kernels(X_train, metric=grid.best_params_['kernel'])
    print(Fore.GREEN + f"  → Matrice de Gram, noyau de dimensions" + Style.RESET_ALL, f"{kmatrix.shape}")
    kmatrix100 = kmatrix[:100, :100]
    plt.pcolor(kmatrix100, cmap=mpl.cm.PuRd)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.gca().xaxis.tick_top()
    plt.tight_layout()
    plt.show()

# Fonction de préparation des paramètres de grille pour utilisation avec plt_grid
# Au moins un des paramètres de param_grid doit être un np.ndarray pour l'axe des abscisses (x)
# Le paramètre sort=True permet de minimiser le nombre de graphiques
# Le paramètre x peut être spécifié: s'il n'est pas de type np.ndarray il y a tentative
# d'en trouver un autre
from operator import itemgetter
def get_param(param_grid, sort=True, x=None):
    elt = []
    for k, v in param_grid.items():
        elt.append([k, len(v), v])
    if sort: elt = sorted(elt, key=itemgetter(1))
    if x in param_grid.keys():
        for index in range(len(elt)):
            if elt[index][0] == x: break
        elt.append(elt.pop(index))
    if isinstance((elt[-1][2]), np.ndarray):
        return elt
    else:
        for index in range(len(elt)-1):
            if isinstance((elt[index][2]), np.ndarray):
                elt.append(elt.pop(index))
                return elt
        print("Error: x must be 'ndarray' type")
        return None

# Graphes des scores en fonction des paramètres de la grille
# Paramètres:
# - grid: la grille entrainée avec GridSearchCV (retour de la fonction eval_model())
# - param_grid: les paramètres d'entrainement de la grille ; compris entre 1 et 3 paramètres
# - sort=True pour minimiser le nombre de graphiques
# - x: permet de spécifier le paramètre pour l'axe des abscisses
# - scale: spécifie l'échelle de l'axe des abcises
def plt_grid(grid, param_grid, return_train_score=False, sort=True, x=None, scale='log'):
    df = pd.DataFrame(grid.cv_results_).copy()
    p = get_param(param_grid, sort=sort, x=x)

    print(Fore.GREEN + "  → Scores en fonction des paramètres de la grille :" + Style.RESET_ALL)

    if len(param_grid)==1:
        best_x = grid.best_params_[p[0][0]]
        plt.figure(figsize=(7, 6))
        if return_train_score:
            label_trs = "tr_sc"
            x = df[f"param_{p[0][0]}"].values
            y = df['mean_train_score'].values
            plt.plot(x, y, label=label_trs, ls='dotted', lw=1)
        label_tes = "te_sc"
        x = df[f"param_{p[0][0]}"].values
        y = df['mean_test_score'].values
        plt.plot(x, y, label=label_tes, ls='solid', lw=1)
        plt.plot([best_x, best_x], plt.gca().get_ylim(), color='k', ls='-', lw=2)
        plt.title(f"Paramètre {p[0][0]} (best={best_x:.2e}, score={df['mean_test_score'].max():.2e})", fontsize=12)
        plt.xscale(scale)
        plt.xlabel(p[0][0])
        plt.ylabel('Scores')
        plt.legend()
        plt.grid(visible=True)
        plt.tight_layout()
        plt.show()

    elif len(param_grid)==2:
        best_x = grid.best_params_[p[1][0]]
        plt.figure(figsize=(7, 6))
        for pl in range(p[0][1]):
            if return_train_score:
                label_trs = "tr_sc_" + f"{p[0][0]}={p[0][2][pl]}"
                x = df.loc[df[f"param_{p[0][0]}"]==p[0][2][pl], f"param_{p[1][0]}"].values
                y = df.loc[df[f"param_{p[0][0]}"]==p[0][2][pl], 'mean_train_score'].values
                plt.plot(x, y, label=label_trs, ls='dotted', lw=1)
            label_tes = "te_sc_" + f"{p[0][0]}={p[0][2][pl]}"
            x = df.loc[df[f"param_{p[0][0]}"]==p[0][2][pl], f"param_{p[1][0]}"].values
            y = df.loc[df[f"param_{p[0][0]}"]==p[0][2][pl], 'mean_test_score'].values
            plt.plot(x, y, label=label_tes, ls='solid', lw=1)
        plt.plot([best_x, best_x], plt.gca().get_ylim(), color='k', ls='-', lw=2)
        plt.title(f"Paramètres de la grille (best: {p[1][0]}={best_x:.2e},"
                  f"score={df['mean_test_score'].max():.2e})", fontsize=12)
        plt.xscale(scale)
        plt.xlabel(p[1][0])
        plt.ylabel('Scores')
        plt.legend()
        plt.grid(visible=True)
        plt.tight_layout()
        plt.show()

    elif len(param_grid)==3:
        best_x = grid.best_params_[p[2][0]]
        if p[0][1] >= 3: n_width = 3
        else: n_width = p[0][1] % 3
        n_height = 1 + int((p[0][1]-1) / 3)
        fig = plt.figure(figsize=(7*n_width, 6*n_height))
        for gph, pval in zip(range(p[0][1]), p[0][2]):
            df_gph = df.loc[df[f"param_{p[0][0]}"]==p[0][2][gph]].copy()
            max_x = df_gph.loc[df_gph['mean_test_score'].idxmax(), f"param_{p[2][0]}"]
            ax = fig.add_subplot(n_height, n_width, gph+1)
            for pl in range(p[1][1]):
                if return_train_score:
                    label_trs = "tr_sc_" + f"{p[1][0]}={p[1][2][pl]}"
                    x = df_gph.loc[df_gph[f"param_{p[1][0]}"]==p[1][2][pl], f"param_{p[2][0]}"].values
                    y = df_gph.loc[df_gph[f"param_{p[1][0]}"]==p[1][2][pl], 'mean_train_score'].values
                    ax.plot(x, y, label=label_trs, ls='dotted', lw=1)
                label_tes = "te_sc_" + f"{p[1][0]}={p[1][2][pl]}"
                #dxy = df_gph.loc[df_gph[f"param_{p[1][0]}"] == p[1][2][pl], [f"param_{p[2][0]}", 'mean_test_score']]
                x = df_gph.loc[df_gph[f"param_{p[1][0]}"]==p[1][2][pl], f"param_{p[2][0]}"].values
                y = df_gph.loc[df_gph[f"param_{p[1][0]}"]==p[1][2][pl], 'mean_test_score'].values
                ax.plot(x, y, label=label_tes, ls='solid', lw=1)
            ax.plot([max_x, max_x], ax.get_ylim(), color='k', ls='--', lw=1)
            ax.plot([best_x, best_x], ax.get_ylim(), color='k', ls='-', lw=2)
            ax.set_title(f"{p[0][0]}={p[0][2][gph]} (best: {p[2][0]}={max_x:.2e},"
                         f"score={df_gph['mean_test_score'].max():.2e})", fontsize=12)
            ax.set_xscale(scale)
            plt.xlabel(p[2][0])
            plt.ylabel('Scores')
            plt.legend()
            plt.grid(visible=True)
        plt.suptitle(f"Scores (selon 'scoring') en fonctions des paramètres de la grille", fontsize=16)
        plt.tight_layout()
        plt.show()

# Filtre les features dont l'importance est inférieure ou égale à une seuil
def feature_filter(model, thres=0.1):
    mask = model.feature_importances_>thres
    return(mask)