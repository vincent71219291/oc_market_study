import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestCentroid


def preprocessing(df, ind_names=None, names_in_index=False):
    if df.isna().sum().sum():
        raise ValueError("Le dataframe contient des valeurs manquantes.")

    if ind_names is not None and names_in_index:
        raise ValueError(
            "Les paramètres `ind_names` et `names_in_index` sont mutuellement "
            "exclusifs."
        )

    # on sélectionne les colonnes avec des valeurs numériques
    df_numerical = df.select_dtypes(include='number')
    X = df_numerical.values

    # on stocke les noms des variables
    features = list(df_numerical.columns)

    # on stocke les noms des individus
    if ind_names is not None:
        names_arr = df[ind_names].to_numpy()
    elif names_in_index:
        names_arr = df.index.to_numpy()

    # on affiche les listes des colonnes utilisées et ignorées
    cols_not_selected = list(df.select_dtypes(exclude='number'))
    print(f"Colonnes utilisées :\n{features}\n")
    print(f"Colonnes ignorées car non numériques :\n{cols_not_selected}")

    # on centre et on réduit les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(
        data=X_scaled,
        index=names_arr,
        columns=features
    )

    result = {
        'X_scaled': X_scaled,
        'X_scaled_df': X_scaled_df,
        'scaler': scaler,
        'features': features,
        'ind_names': names_arr
    }

    return result


# la fonction plot_dendrogram provient du site de scikit-learn
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_davies_bouldin_scores(
        X_scaled,
        method=None,
        n_cluster_max=20,
        ax=None,
        **kwargs
):
    methods = ['ahc', 'kmeans']
    if method not in methods:
        raise ValueError(
            "La méthode de clustering doit figurer dans la liste suivante :\n"
            f"{methods}"
        ) 
    
    db_scores = []

    n_list = range(2, n_cluster_max + 1)
    
    for n in n_list:
        if method == 'ahc':
            model = AgglomerativeClustering(n_clusters=n, linkage='ward')
        else:
            model = KMeans(n_clusters=n, random_state=0)
        
        y = model.fit_predict(X_scaled)
        db_score = davies_bouldin_score(X_scaled, y)
        db_scores.append(db_score)
    
    if ax is None:
        ax = plt.gca()

    sns.lineplot(x=n_list, y=db_scores, ax=ax, **kwargs)
    ax.set_title("Indice de Davies-Bouldin")
    ax.set_xlabel("Nombre de clusters")
    ax.set_xlim(2, n_cluster_max)
    ax.set_xticks(n_list)


def plot_silhouette_scores(
    X_scaled,
    method=None,
    n_cluster_max=20,
    ax=None,
    **kwargs
):
    methods = ['ahc', 'kmeans']
    if method not in methods:
        raise ValueError(
            "La méthode de clustering doit figurer dans la liste suivante :\n"
            f"{methods}"
        ) 
    
    sil_scores = []

    n_list = range(2, n_cluster_max + 1)
    
    for n in n_list:
        if method == 'ahc':
            model = AgglomerativeClustering(n_clusters=n, linkage='ward')
        else:
            model = KMeans(n_clusters=n, random_state=0)
        
        y = model.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, y, metric='euclidean')
        sil_scores.append(sil_score)
    
    if ax is None:
        ax = plt.gca()

    sns.lineplot(x=n_list, y=sil_scores, ax=ax, **kwargs)
    ax.set_title("Coefficient de silhouette")
    ax.set_xlabel("Nombre de clusters")
    ax.set_xlim(2, n_cluster_max)
    ax.set_xticks(n_list)


def plot_elbow(X, k_max=20, ax=None, **kwargs):
    inertia = []
    
    k_list = range(2, k_max + 1)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for k in k_list:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        
    if ax is None:
        ax = plt.gca()
    
    sns.lineplot(x=k_list, y=inertia, ax=ax, **kwargs)

    ax.set_xlabel("Nombre de classes (clusters)")
    ax.set_ylabel("Inertie")
    ax.set_title("Méthode du coude")
    ax.set_xlim(2, k_max)
    ax.set_xticks(k_list)


def get_ahc_clusters(X_scaled, n_clusters):
    ahc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    clusters = ahc.fit_predict(X_scaled)
    return clusters


def get_kmeans_clusters(X, k):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    clusters = kmeans.labels_
    return clusters


def get_centroids(X_scaled, clusters, features):
    clf = NearestCentroid()
    clf.fit(X_scaled, clusters)
    centroids_df = pd.DataFrame(
        data=clf.centroids_,
        index=clf.classes_,
        columns=features
    ).rename_axis(index='cluster', columns='feature')
    return centroids_df


def plot_centroids(centroids_df, palette=None, ax=None, **kwargs):    
    if palette is None:
        cluster_max = max(centroids_df.index)
        palette = sns.color_palette(n_colors=(cluster_max + 1))
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    centroids_melted = centroids_df.melt(ignore_index=False).reset_index()

    sns.scatterplot(
        data=centroids_melted,
        x='value',
        y='feature',
        hue='cluster',
        palette=palette,
        s=100,
        ax=ax,
        **kwargs
    )
        
    ax.axvline(x=0, color='grey', linewidth=1)
    xmax = max(ax.get_xlim(), key=abs)
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(title='Classe', loc='center left', bbox_to_anchor=(1, 0.5))


def draw_scree_plot(pca):
    n = pca.n_components_
    x_axis = range(1, n + 1)

    var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)

    plt.bar(x_axis, var_ratio)
    plt.plot(x_axis, cum_var_ratio, color='red', marker='o')

    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Ratio d'inertie")
    plt.title("Eboulis des valeurs propres")

    for x, y in zip(x_axis, cum_var_ratio):
        plt.annotate(text=round(y, 2),
                     xy=(x, y), xycoords='data',
                     xytext=(0, 10), textcoords='offset points',
                     ha='center')

    plt.ylim(0, 1.15)
    plt.draw()


def pca_modelisation(X_scaled, n_components=6, ind_names=None, scree_plot=False):
    
    # on instancie et on entraîne le modèle
    pca = PCA(n_components=n_components)
    X_projected = pca.fit_transform(X_scaled)
    
    # on représente l'éboulis des valeurs propres
    if scree_plot:
        draw_scree_plot(pca)
    
    cols = ['F' + str(i + 1) for i in range(n_components)]
    X_projected_df = pd.DataFrame(data=X_projected, index=ind_names, columns=cols)
    
    result = {
        'X_projected': X_projected,
        'X_projected_df': X_projected_df,
        'pca': pca
    }
    
    return result


def text_alignment(x, y):   
    if x <= 0:
        ha='right'
    else:
        ha='left'
    if y <= 0:
        va='top'
    else:
        va='bottom'
    return ha, va


@plt.rc_context({'axes.labelsize': 14})
def draw_corr_circle(pca, components=(0, 1), features=None, ax=None):
    x_comp, y_comp = components

    for i in range(pca.components_.shape[1]):
        x = pca.components_[x_comp, i]
        y = pca.components_[y_comp, i]
        
        # texte
        ha, va = text_alignment(x, y)
        ax.annotate(
            features[i],
            xy=(x, y),
            ha=ha,
            va=va,
            bbox=dict(facecolor='lightgrey', alpha=0.5)
        )
        
        # flèches
        ax.annotate(
            '',
            xy=(0, 0),
            xytext=(x, y),
            arrowprops=dict(
                color=sns.color_palette()[0],
                lw=2,
                arrowstyle='<-'  # '<|-'
            )
        )
    
    # on représente un cercle de rayon 1
    angles = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(angles), np.sin(angles))
    ax.axis('equal')
    
    # on représente les axes
    ax.axhline(y=0, color='grey', linestyle='dashed', linewidth=1)
    ax.axvline(x=0, color='grey', linestyle='dashed', linewidth=1)
    
    # nom des axes
    x_comp_var_pct = pca.explained_variance_ratio_[x_comp] * 100
    y_comp_var_pct = pca.explained_variance_ratio_[y_comp] * 100
    ax.set_xlabel(f'F{x_comp + 1} ({x_comp_var_pct:.1f} %)')
    ax.set_ylabel(f'F{y_comp + 1} ({y_comp_var_pct:.1f} %)')
    
    # titre                            
    ax.set_title(f'Cercle des corrélations (F{x_comp + 1} et F{y_comp + 1})')


def repr_ind_projected(
    X_projected,
    pca,
    components=(0, 1),
    clusters=None,
    centroids_df=None,
    palette=None,
    ax=None
):
    X_projected_df = pd.DataFrame(X_projected)
    
    if clusters is not None:
        X_projected_df['cluster'] = clusters
        hue = 'cluster'
        if palette is None:
            palette = sns.color_palette(n_colors=max(clusters) + 1)
    else:
        hue = None
        
    # si on représente les centroïdes, on réduit l'opacité des autres points
    if centroids_df is not None:
        alpha = 0.7
    else:
        alpha = None

    x_comp, y_comp = components

    # représentation des projections des individus
    sns.scatterplot(
        data=X_projected_df,
        x=x_comp,
        y=y_comp,
        hue=hue,
        palette=palette,
        alpha=alpha,
        ax=ax
    )

    # représentation des centroïdes
    ## on projette les centroïdes sur les axes principaux d'inertie
    if centroids_df is not None:
        centroids_projected = pca.transform(centroids_df.values)
        centroids_projected_df = pd.DataFrame(
            data=centroids_projected,
            index=centroids_df.index
        )
        sns.scatterplot(
            data=centroids_projected_df,
            x=x_comp,
            y=y_comp,
            marker='s',
            color='black',
            ax=ax
        )

        ## on annote les centroïdes
        for centroid, coords in centroids_projected_df.iterrows():
            xy_annot = coords[[x_comp, y_comp]]
            ax.annotate(
                text=centroid,
                xy=xy_annot,
                xytext=(0, 8),
                textcoords='offset points',
                ha='center'
            )
    
    # représentation des axes
    ax.axhline(y=0, color='grey', linestyle='dashed', linewidth=1)
    ax.axvline(x=0, color='grey', linestyle='dashed', linewidth=1)
    
    # nom des axes
    x_comp_var_pct = pca.explained_variance_ratio_[x_comp] * 100
    y_comp_var_pct = pca.explained_variance_ratio_[y_comp] * 100
    ax.set_xlabel(f'F{x_comp + 1} ({x_comp_var_pct:.1f} %)')
    ax.set_ylabel(f'F{y_comp + 1} ({y_comp_var_pct:.1f} %)')
    
    # titre du graphique                          
    ax.set_title(f'Projection des individus sur les axes F{x_comp + 1} et F{y_comp + 1}')
    
    # limites des axes
    xmax = max(ax.get_xlim(), key=abs)
    ymax = max(ax.get_ylim(), key=abs)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)


def clustering_and_pca(X_scaled, n_clusters, method='ahc', features=None):
    methods = ['ahc', 'kmeans']
    if method not in methods:
        raise ValueError(
            "La méthode de clustering doit figurer dans la liste suivante :\n"
            f"{methods}"
        )
    method_name = 'CAH' if method == 'ahc' else 'KMeans'
    title_color = (0.122, 0.467, 0.706)
    palette = sns.color_palette(n_colors=n_clusters)

    fig_clustering, axes_clustering = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 6))

    fig_pca, axes_pca = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(16, 12)
    )

    # clustering
    if method == 'ahc':
        clusters = get_ahc_clusters(X_scaled, n_clusters)
    else:
        clusters = get_kmeans_clusters(X_scaled, n_clusters)
        
    ## nombre d'observations par cluster
    n_obs = pd.DataFrame(clusters, columns=['cluster']) \
        .groupby('cluster').agg(count=('cluster', 'count'))
    display(n_obs)
        
    ## on récupère les centroïdes des classes
    centroids_df = get_centroids(X_scaled, clusters, features=features)

    ## on représente les centroïdes
    plot_centroids(
        centroids_df,
        palette=palette,
        alpha=0.7,
        ax=axes_clustering[0]
    )

    sns.heatmap(
        centroids_df.T,
        vmin=-2,
        vmax=2,
        center=0,
        cmap='coolwarm',
        square=True,
        ax=axes_clustering[1]
    )

    # ACP
    ## modélisation de l'ACP
    pca_res = pca_modelisation(X_scaled, n_components=4)
    X_projected = pca_res['X_projected']
    pca = pca_res['pca']
    
    ## représentation des cercles de corrélation
    ### F1 et F2
    draw_corr_circle(pca, components=(0, 1), features=features, ax=axes_pca[0, 0])
    ### F3 et F4
    draw_corr_circle(pca, components=(2, 3), features=features, ax=axes_pca[1, 0])
    
    ## représentation des projections des individus
    ### sur le plan formé par F1 et F2
    repr_ind_projected(
        X_projected,
        pca,
        components=(0, 1),
        clusters=clusters,
        centroids_df=centroids_df,
        #palette=palette,
        ax=axes_pca[0, 1]
    )
    ### sur le plan formé par F3 et F4
    repr_ind_projected(
        X_projected,
        pca,
        components=(2, 3),
        clusters=clusters,
        centroids_df=centroids_df,
        palette=palette,
        ax=axes_pca[1, 1]
    )

    # mise en forme des graphiques
    ## on supprime le ylabel de la heatmap
    axes_clustering[1].set_ylabel('')
    
    for ax in axes_pca[:, 1]:
        ax.legend(
            title=f"Classe ({method_name})",
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )
    
    fig_clustering.suptitle(
        f"Représentation des centroïdes des classes ({method_name})",
        color=title_color
    )
    fig_pca.suptitle(
        "\nACP - Représentation des projections des variables et des individus\n",
        color=title_color
    )
    for fig in (fig_clustering, fig_pca):
        fig.tight_layout()
    plt.show()


def get_cluster_sets(data, clustering=None, ind_names=None, names_in_index=False):
    if ind_names is not None and names_in_index:
        raise ValueError(
            "Les paramètres `ind_names` et `names_in_index` sont mutuellement "
            "exclusifs."
        )
    if names_in_index:
        ind_names = data.index.name if data.index.name is not None else 'index'
        data = data.reset_index()
    return data.groupby(clustering)[ind_names].agg(set)


def compare_clusterings(
    data,
    clusterings,
    prefixes=None,
    ind_names=None,
    names_in_index=False,
    figsize=(16, 8)
):
    if prefixes is None:
        prefixes = ('', '')
    
    kwargs = dict(
        data=data,
        ind_names=ind_names,
        names_in_index=names_in_index
    )
    
    # on recupère les sets contenant les observations pour chaque cluster de
    # chacun des deux clusterings
    cluster_sets1 = get_cluster_sets(clustering=clusterings[0], **kwargs)
    cluster_sets2 = get_cluster_sets(clustering=clusterings[1], **kwargs)
    
    # on initialise les matrices des comparaisons et des annotations
    comp_matrix = np.zeros((cluster_sets1.size, cluster_sets2.size))
    annot = np.empty_like(comp_matrix, dtype='object')
    
    # on compare les clusters deux à deux
    for i, set1 in enumerate(cluster_sets1):
        for j, set2 in enumerate(cluster_sets2):
            count_intersection = len(set1.intersection(set2))
            count_union = len(set1.union(set2))
            comp_matrix[i, j] = count_intersection / count_union
            annot[i, j] = f"{count_intersection}/{count_union}"
    
    # on représente la matrice des comparaisons sous forme de heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        comp_matrix,
        vmin=0,
        vmax=1,
        cmap='Blues',
        annot=annot,
        fmt='',
        square=True
    )
    ax.set_xlabel(clusterings[1])
    ax.set_ylabel(clusterings[0])
    
    # on ajoute le nombre d'observations dans les xtickslabels des clusters
    xticklabels = [
        f"{prefixes[1]}{cluster}\n({len(cluster_set)})"
        for cluster, cluster_set in cluster_sets2.iteritems()
    ]
    yticklabels = [
        f"{prefixes[0]}{cluster}\n({len(cluster_set)})"
        for cluster, cluster_set in cluster_sets1.iteritems()
    ]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels, rotation=0)
    plt.show()