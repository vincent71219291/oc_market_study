# Projet : Produisez une étude de marché
Projet du parcours Data analyst d'OpenClassrooms

## Scénario du projet
<img src="imgs/poule.png" width="500"/>

Dans le scénario du projet, nous sommes data analyst pour une entreprise agroalimentaire qui ambitionne de se développer à l'étranger. Nous avons été chargé d'identifier des pays cibles pour l'exportation de volailles.

## Commentaires

### Données
Les données brutes ont été téléchargées à partir des bases de données publiques suivantes :
* [Organisation des Nations unies pour l'alimentation et l'agriculture (FAO)](https://www.fao.org/faostat/fr/#data)
* [World Bank](https://databank.worldbank.org/home)
* [Google](https://developers.google.com/public-data/docs/canonical/countries_csv)

Elles sont disponibles dans le dossier [`data`](https://github.com/vincent71219291/oc_market_study/tree/main/data) sous forme de fichiers CSV.


### Organisation des fichiers du projet
Les analyses ont été effectuées avec Python en utilisant Jupyter Notebook. Pour plus de lisibilité, le projet a été divisé en deux notebooks qui contiennent :
* [le nettoyage](https://github.com/vincent71219291/oc_market_study/blob/main/oc_market_study_notebook_01.ipynb) (assez fastidieux) et l'imputation des données ;
* [l'analyse des données](https://github.com/vincent71219291/oc_market_study/blob/main/oc_market_study_notebook_02.ipynb), qui inclue notamment un clustering et une ACP.

Le fichier [`P9_functions.py`](https://github.com/vincent71219291/oc_market_study/blob/main/scripts/P9_functions.py) (dans le dossier `scripts`) contient des fonctions personnalisées dont la programmation représente une grosse partie du travail effectué pour ce projet.

<b>Note :</b> Certains graphiques interactifs (cartes du monde) du notebook 2 ne s'affichent pas correctement sur GitHub, mais il est possible de les visualiser à l'aide de nbviewer [à cette adresse](https://nbviewer.org/github/vincent71219291/oc_market_study/blob/main/oc_market_study_notebook_02.ipynb).

## Compétences évaluées :
* Nettoyer des données
* Réaliser un clustering
* Réaliser une ACP