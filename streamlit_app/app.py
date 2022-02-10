"""
@name : app.py

@date : 26/12/21

@description : Code permettant de visualiser l'application Streamlit

@authors: 
    Caolan GUEGUEN
    Louis LEVEAUX
    Maxime CERISIER
"""

import streamlit as st
import app_session as session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from PIL import Image

df = pd.read_csv("CSV/df_demo_streamlit.csv", index_col = 0)

def main():
    state = session.get_state()
    
    pages = {
        "Le projet SpotiPy": page_presentation,
        "Datasets": page_datasets,
        "DataViz'" : page_dataviz,
        "Clustering": page_clustering,
        "Classification": page_classification,
        "Recommandation": page_recommandation,
        "Conclusion": page_conclusion,
        "A propos de l'équipe": page_equipe,
    }
    
    st.sidebar.title("SpotiPy")
    st.sidebar.subheader("Menu") 
    selection = st.sidebar.radio("", tuple(pages.keys()))
    pages[selection](state)
    
def page_presentation(state):
    image = Image.open('Images/projet_spotipy.png')
    st.image(image, use_column_width=True)
    st.title('')
    st.write("A partir d'un jeu de données contenant différents noms de morceaux \
             de musique et leurs caractéristiques techniques, le projet Spotipy a \
             deux objectifs :")
    st.write("- Prédire la popularité d'une musique")
    st.write("- Développer un algorithme de recommandation musicale")
    st.info("Au début du projet nous avions 3 datasets et l'objectif était de faire \
            de l'analyse de sentiment à partir de tweets et des caractéristiques des \
            morceaux de musique. Cependant après quelques temps nous nous sommes \
            rendu compte que les datasets étaient inexploitables, nous avons donc \
            choisis un autre jeu de données avec de nouveaux objectifs \
            (ceux cités ci-dessus).")

def page_datasets(state):
    st.title('DATASETS')
    st.markdown('Pour réaliser notre projet nous avons utilisé deux datasets :'
        '\n\n # 1er Dataset'
        "\n\n Le premier Dataset est constitué de différentes colonnes :"
        "\n\n   - Le nom de la musique, de son album et de son artiste"
        "\n\n   - Les identifiants spotify leur correspondant"
        "\n\n   - Le numéro de la chanson et du disque"
        "\n\n   - Les différentes caractéristiques  de la chanson : 'explicit', 'danceability', 'energy','key',"
        " 'loudness','mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms','time_signature'"
        "\n\n   - L'année et la date de parution de la chanson"
        "\n\nNous avons ajouté à ce jeu de données deux colonnes supplémentaire que nous avons obtenu grâce à l'api spotify :"
        "\n\n   - La première colonne est la popularité comprise entre 0 et 100 qui représente la 'popularité actuelle'"
        " de la musique, plus la musique est écoutée récemment plus la note est élevée"
        "\n\n   - La seconde colonne est le genre musical où l'on trouve si l'api Spotify avait l'information du genre musical de l'album de la chanson")
    df_musique=pd.read_csv('CSV/extrait_musique.csv',index_col=0)
    with st.expander("Extrait"):
        st.dataframe(df_musique)
    st.markdown('\n\n # 2ème Dataset'
        "\n\n Le deuxième Dataset à était obtenue par extraction d'information contenu dans des dictionnaires au format JSON."
        "\n\n Il contient deux informations, un numéro donné à chaque playlist de manière arbitraire "
        "et les différents identifiant spotify de chanson associée à chaque playlist"
        )
    
            
    with st.expander("Exemple d'une entrée du dictionnaire JSON"):
        st.code('''
            {
                "name": "musical",
                "collaborative": "false",
                "pid": 5,
                "modified_at": 1493424000,
                "num_albums": 7,
                "num_tracks": 12,
                "num_followers": 1,
                "num_edits": 2,
                "duration_ms": 2657366,
                "num_artists": 6,
                "tracks": [
                    {
                        "pos": 0,
                        "artist_name": "Degiheugi",
                        "track_uri": "spotify:track:7vqa3sDmtEaVJ2gcvxtRID",
                        "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
                        "track_name": "Finalement",
                        "album_uri": "spotify:album:2KrRMJ9z7Xjoz1Az4O6UML",
                        "duration_ms": 166264,
                        "album_name": "Dancing Chords and Fireflies"
                    },
                    {
                        "pos": 1,
                        "artist_name": "Degiheugi",
                        "track_uri": "spotify:track:23EOmJivOZ88WJPUbIPjh6",
                        "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
                        "track_name": "Betty",
                        "album_uri": "spotify:album:3lUSlvjUoHNA8IkNTqURqd",
                        "duration_ms": 235534,
                        "album_name": "Endless Smile"
                    },
                    {
                        "pos": 2,
                        "artist_name": "Degiheugi",
                        "track_uri": "spotify:track:1vaffTCJxkyqeJY7zF9a55",
                        "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
                        "track_name": "Some Beat in My Head",
                        "album_uri": "spotify:album:2KrRMJ9z7Xjoz1Az4O6UML",
                        "duration_ms": 268050,
                        "album_name": "Dancing Chords and Fireflies"
                    },
                    // 8 tracks omitted
                    {
                        "pos": 11,
                        "artist_name": "Mo' Horizons",
                        "track_uri": "spotify:track:7iwx00eBzeSSSy6xfESyWN",
                        "artist_uri": "spotify:artist:3tuX54dqgS8LsGUvNzgrpP",
                        "track_name": "Fever 99\u00b0",
                        "album_uri": "spotify:album:2Fg1t2tyOSGWkVYHlFfXVf",
                        "duration_ms": 364320,
                        "album_name": "Come Touch The Sun"
                    }
                ],

            }
            
            
            
            ''')
    df_playlist=pd.read_csv('CSV/extrait_playlist.csv',index_col=0)
    with st.expander("Extrait"):
        st.dataframe(df_playlist)  

def page_dataviz(state):
    st.title('VISUALISATION DES DONNEES')
    st.header('Distribution des variables')
    
    L_select = [0 for i in range(18)]
    
    L_name = ['energy','acousticness','loudness',
                'instrumentalness','liveness','speechiness',
                'danceability','valence','tempo','mode','key',
                'explicit','duration_ms','time_signature','year',
                'track_number','disc_number','popularity']
    
    intervalle = np.arange(0,1.1,0.01)
    
    L_intervalle = [intervalle, intervalle, np.linspace(-60,8,100), intervalle,
                    intervalle, intervalle, intervalle, intervalle, ]
    
    with st.expander("Choisissez les variables à visualiser :"):
        box_all      = st.checkbox('Tout sélectionner')        
        L_select[0]  = st.checkbox(L_name[0])
        L_select[1]  = st.checkbox(L_name[1])
        L_select[2]  = st.checkbox(L_name[2])
        L_select[3]  = st.checkbox(L_name[3])
        L_select[4]  = st.checkbox(L_name[4])
        L_select[5]  = st.checkbox(L_name[5])
        L_select[6]  = st.checkbox(L_name[6])
        L_select[7]  = st.checkbox(L_name[7])
        L_select[8]  = st.checkbox(L_name[8])
        L_select[9]  = st.checkbox(L_name[9])
        L_select[10] = st.checkbox(L_name[10])
        L_select[11] = st.checkbox(L_name[11])
        L_select[12] = st.checkbox(L_name[12])
        L_select[13] = st.checkbox(L_name[13])
        L_select[14] = st.checkbox(L_name[14])
        L_select[15] = st.checkbox(L_name[15])
        L_select[16] = st.checkbox(L_name[16])
        L_select[17] = st.checkbox(L_name[17])

        bt_display = st.button('Afficher')

    if bt_display:
        k=0
        
        fig = plt.figure(figsize=(17,30))

        if L_select[0] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_1 = sns.histplot(df['energy'], bins=intervalle, color='orange')
        if L_select[1] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_2 = sns.histplot(df['acousticness'], bins=intervalle, color='orange')
        if L_select[2] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_3 = sns.histplot(df['loudness'], bins=np.linspace(-60,8,100), color='orange')
        if L_select[3] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_4 = sns.histplot(df['instrumentalness'], bins=intervalle, color='orange')
        if L_select[4] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_5 = sns.histplot(df['liveness'], bins=intervalle, color='orange')
        if L_select[5] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_6 = sns.histplot(df['speechiness'], bins=intervalle, color='orange')
        if L_select[6] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_7 = sns.histplot(df['danceability'], bins=intervalle, color='orange')
        if L_select[7] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_8 = sns.histplot(df['valence'], bins=intervalle, color='orange')
        if L_select[8] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_9 = sns.histplot(df['tempo'], bins=np.linspace(50,250,100), color='orange')
        if L_select[9] or box_all:
            k +=1
            plt.subplot(6,3,k)
            plt.pie(df['mode'].value_counts(normalize=True), labels=['1','0'], autopct='%.0f%%')
            plt.xlabel('mode')
        if L_select[10] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_11 = sns.countplot(df['key'])
        if L_select[11] or box_all:
            k +=1
            plt.subplot(6,3,k)
            plt.pie(df['explicit'].value_counts(normalize=True), labels=['1','0'], autopct='%.0f%%')
            plt.xlabel('explicit')
        if L_select[12] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_13 = sns.histplot(df['duration_ms'], bins=np.linspace(1e3,2e6,100), color='orange')
        if L_select[13] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_14 = sns.countplot(df['time_signature'])
        if L_select[14] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_15 = sns.countplot(df['year'])
            plt.xticks([])
        if L_select[15] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_16 = sns.countplot(df['track_number'])
            plt.xticks([0,10,20,30,40,50])
        if L_select[16] or box_all:
            k +=1
            plt.subplot(6,3,k)
            seuil = 1
            df_nb_disc_desequilibre = pd.Series(np.where(df['disc_number']<=seuil,1,0))
            plt.pie(df_nb_disc_desequilibre.value_counts(normalize=True), labels=['= 1','> 1'], autopct='%.0f%%')
            plt.xlabel('disc_number')
        if L_select[17] or box_all:
            k +=1
            plt.subplot(6,3,k)
            graph_18 = sns.histplot(df['popularity'], bins=np.linspace(0,100,20), color='orange')
            
        st.pyplot(fig)
    st.info("Pour la démo streamlit nous avons réduit le jeu de donnée pour minimiser le temps de calcul. "
            "Cela explique notamment pourquoi les visualisations ne sont pas strictement les même que dans le rapport.")
    
    st.header('Etude des corrélations')
    
    # mat = df.corr()
    # fig, ax = plt.subplots(figsize=(12,12))
    # sns.heatmap(mat, annot=True, ax=ax, cmap='flare')
    # st.pyplot(fig)
    image = Image.open('Images/matrice_correlation.png')
    st.image(image, use_column_width=True)
    
    
    st.markdown('La grande majorité des données ne sont pas corrélées, en revanche on constate tout de même que : \
              \n\n - Les colonnes “energy” et “loudness” ont tendance à être proportionnelles \
              \n\n - Les colonnes “energy” et “accousticness” ont tendance à être inversement proportionnelles')
    with st.expander("Confirmation des tendances observées ..."):
        image = Image.open('Images/confirmations_correlations.png')
        st.image(image, use_column_width=True)
        

def page_clustering(state):
    st.title('CLUSTERING')
    st.markdown('### Preprocessing - Choix des variables pour le clustering')
    st.write('Toutes les variables de notre dataset ne sont pas forcément \
             utiles ou utilisables pour effectuer un clustering. \
             Voici la liste des variables conservées :')
    st.write('- instrumentalness')     
    st.write('- liveness')   
    st.write('- speechiness')   
    st.write('- danceability')   
    st.write('- valence')   
    st.write('- loudness')   
    st.write('- tempo')   
    st.write('- acousticness')   
    st.write('- energy')
    st.write('- mode')   
    st.write('- key')   
    st.write('- duration_ms')   
    
    st.markdown('### Preprocessing - Choix du Scaler sur les données')
    st.write('Pour faire notre clustering il est important de normaliser nos \
             données, nous allons utiliser le scaler MinMax.')
    
    st.markdown('### Clustering - Méthode du coude')
    st.write('Après avoir appliqué le Scaler MinMax nous effectuons \
             la méthode du coude pour déterminer le nombre de clusters \
             que nous choisirons pour notre algorithme de clustering.')
    st.image('Images/Méthode_du_Coude_Clustering.png',caption='Méthode du Coude')
    st.write('Nous apercevons plusieurs coudes qui se démarquent, \
             particulièrement pour 3 et 5 clusters.')
    st.write('Nous choisirons 5 clusters.')
    
    st.markdown("### Clustering - Application de l'algorithme du KMeans")
    st.write("Après avoir lancé notre algorithme du KMeans avec 5 clusters en entrée \
             voici la répartition du nombre de morceaux de musique par cluster :")
    st.image("Images/Nombre_par_label_Clustering.png",caption='Nombre de Morceaux par labels')
    
    st.markdown('### Evaluations du modèle - PCA')
    st.write("Maintenant il est nécessaire d'observer à quoi ressemble notre clustering. \
             Or notre clustering s'effectue à l'aide de 12 variables, ainsi il est \
             nécessaire d'appliquer une réduction de dimension afin d'observer \
             notre clustering sur un graphique.")
    PCA=st.selectbox(label='Quel module de PCA voulez-vous montrer ?',
                 options=['Module sklearn','Module Prince'])
    if PCA == 'Module sklearn':
        st.write("Dans un premier temps nous avons opté pour une PCA issue de sklearn \
                 en réduisant l'espace à un espace de 3 dimensions.")
        st.write("Il est maintenant nécessaire de vérifier que notre \
                 réduction de dimensions est performante et qu'il n'y a pas \
                 trop d'informations perdues.")
        st.write("Nous traçons donc les cercles de corrélations de nos \
                 3 composantes de PCA avec les composantes de bases :")
        st.image("Images/Cercle_Cor_PCA_1_2_sklearn_Clustering.png",caption='Cercle de corrélation PCA 1 & 2 module sklearn')
        st.image("Images/Cercle_Cor_PCA_1_3_sklearn_Clustering.png",caption='Cercle de corrélation PCA 1 & 3 module sklearn')
        st.table({"Composantes":['PCA 1','PCA 2','PCA 3','Total'],"Ratio":[28.56,23.83,12.79,28.56+23.83+12.79]})
        st.write("Les ratios de variance de nos variables et les cercles de corrélation \
                 montrent que notre PCA explique de façon peu convaincante \
                 notre jeu de données. Par exemple on remarque que PCA 2 n'est \
                 composée que de la variable mode. Finalement les variables \
                 réellement conservées par la réduction de dimension sont :")
        st.write("- acousticness")
        st.write("- instrumentalness")
        st.write("- energy")
        st.write("- mode")
        st.write("Nous remarquons en plus que la somme de nos ratios de variance \
                 n'est que de 65 %, pour une bonne PCA on espère plutôt être aux \
                 alentours de 90 %.")
        st.write("Voici les représentations graphiques de la PCA du module sklearn :")
        st.image("Images/PCA_3D_sklearn_Clustering.png",caption='PCA 3 Dimensions module sklearn')
        st.image("Images/PCA_2D_1_2_sklearn_Clustering.png",caption="PCA 2 Dimensions (PCA 1 & 2) module sklearn")
        st.image("Images/PCA_2D_2_3_sklearn_Clustering.png",caption="PCA 2 Dimensions (PCA 2 & 3) module sklearn")        
        st.write("Les représentations graphiques sont cependant intéressantes, \
                 elles nous permettent de voir que nos clusters sont bien délimités \
                 dans notre espace réduit.")
    else:
        st.write("N'étant pas satisfait de la PCA du module sklearn nous avons essayé \
                 un autre module de réduction de dimensions, Prince.")
        st.info("Nous reduisons toujours à un espace 3 Dimensions")
        st.image("Images/Cercle_Cor_PCA_1_2_prince_Clustering.png",caption='Cercle de corrélation PCA 1 & 2 module Prince')
        st.image("Images/Cercle_Cor_PCA_1_3_prince_Clustering.png",caption='Cercle de corrélation PCA 1 & 3 module Prince')
        st.table({"Composantes":['PCA 1','PCA 2','PCA 3','Total'],"Ratio":[26.45,10.89,9.72,26.45+10.89+9.72]})
        st.write("Cette fois-ci les résulats de notre PCA sont encore moins bons sur les \
                 ratios de variance malgré des cercles de corrélations plus lisibles que \
                 sur le module sklearn. Contrairement au module de sklearn, \
                 les variables issues de notre réduction de dimensions de Prince \
                 sont composées de la quasi intégralité de nos variables de départ, \
                 ce qui est un bon point.")
        st.write("Cependant nous remarquons aussi que la somme des ratios de variance \
                 ne dépasse pas 47 %, ce qui est trop faible.")
        st.write("Voici les représentations graphiques de la PCA du module Prince :")
        st.image("Images/PCA_3D_prince_Clustering.png",caption='PCA 3 Dimensions module Prince')
        st.image("Images/PCA_2D_1_2_prince_Clustering.png",caption="PCA 2 Dimensions (PCA 1 & 2) module Prince")
        st.image("Images/PCA_2D_2_3_prince_Clustering.png",caption="PCA 2 Dimensions (PCA 2 & 3) module Prince")
        st.write("Lors de la PCA de sklearn nous avons réussi à observer des graphiques \
                 intéressants (surement parce que cette PCA ne conservait réellement que \
                 4 des variables de départ). Les graphiques de la PCA Prince sont quant à \
                 eux beaucoup moins interprétables. En y ajoutant les ratios de variances \
                 très faibles nous ne pouvons pas tirer de conclusions exploitables \
                 sur ce modèle. Il est donc plus intéressant de garder en tête les résultats \
                 de la PCA issue de sklearn.")
                 
    st.markdown('### Résultats du modèle')
    st.write("Pour conclure sur notre clustering nous allons comparer les différents \
             morceaux/artistes entre eux et voir dans quel cluster ils ont été placés.")
    st.write("La première chose qu'on a fait a été de déterminer le nombre de clusters \
             distincts par artiste. L'idée étant qu'un artiste fait souvent des morceaux \
             d'un même style ainsi ces morceaux devraient être dans le même cluster.")
    st.write("On a donc fait notre regroupement et tracé un histogramme du nombre \
             d'artistes en fonction du nombre de clusters distincts :")
    st.image("Images/Analyse_graph_artiste_Clustering.png",caption="Nombre d'artistes par nombre de clusters distincts")
    st.table({"Nombre de clusters distincts":[1,2,3,4,5],"Pourcentage d'artistes":[59.9,19.6,11.4,6.2,2.9]})
    st.write("Ce résultat est assez bon, on remarque que presque 80 % des artistes sont \
             affectés à moins de 2 clusters différents sur l'ensemble de leurs morceaux.")
    st.write("Cependant en cherchant dans le dataset , on a pu remarquer quelques \
             incohérences dans notre clustering. En effet certais morceaux se retrouvent \
             dans le même cluster alors qu'ils n'ont pas grand-chose à voir. Par exemple :")
    st.write("Rage Against the Machine - Killing in the Name On")
    st.audio("Audios/Rage Against The Machine Killing In the Name.mp3")
    st.write("Daryl Hall John Oates - Do it For Love :")
    st.audio("Audios/Daryl Hall John Oates Do it For Love.mp3")
    st.write("Le rapprochement entre ces deux musiques est du à plusieurs choses :")
    st.write("- Le nombre de cluster est trop faible")
    st.write("- Notre modèle de clustering est trop simple pour faire de la recommandation \
             sur un jeu de données aussi vaste")
    st.write("Pour conclure, le clustering ne parait pas être une bonne solution pour \
             effectuer une recommandation satisfaisante. Nous utiliserons donc une autre méthode \
             plus tard dans la partie recommandation")

def page_classification(state):
    st.title('CLASSIFICATION')
    st.header('Type de classification')
    choix = ('A partir de la popularité', 'A partir des vues de YouTube')
    option = st.selectbox('',choix)
    
    st.subheader('Déséquilibre de classe')
    
    
    if option == choix[0]:
        col1, col2 = st.columns([1.5, 1,])
        with col1:
            st.markdown("Nos données sont très concentrées autour de 0. Cette répartition n’a \
                        rien d’étonnant car on comprend assez simplement qu'à l'échelle de la musique, \
                        seule une poignée de morceaux sont populaires. En revanche, dans le cadre de notre étude, cette \
                        disparité est un problème car elle engendre un fort déséquilibre de classe. Pour pallier celui-ci, \
                        nous avons scindé nos données en deux groupes distincts :")

        with col2:
            image = Image.open('Images/repartition_popularite.png')
            st.image(image, use_column_width=True)
        
        st.markdown("- Les musiques considérées populaires (note de popularité >= 40) \
                    \n\n - Les musiques considérées impopulaires (note de popularité < 40)")
        
        st.info("Le seuil de 40 a été choisi car il est assez discriminant pour qu’on ne récupère que les musiques réellement \
                 populaires, tout en en gardant un nombre suffisant pour entraîner nos modèles de machine learning.")
        st.success("Un **Random Under Sampler** a ensuite été utilisé pour équilibrer ces classes.")
        
    else :
        st.markdown("Après avoir tenté d’établir un modèle de régression (qui n'a pas donné de résultats convainquants), \
                        nous nous sommes tournés vers des modèles de classification avec les classes suivantes :")
        
        col1, col2 = st.columns([1.5, 1,])    
            
        with col1:
            st.markdown("\n\n - Classe 0 : vues < 1M \
                        \n\n - Classe 1 : vues ∈ [ 1M, 10M [ \
                        \n\n - Classe 2 : vues ∈ [ 10M, 100M ] \
                        \n\n - Classe 3 : vues > 100M ")

        with col2:
            image = Image.open('Images/repartition_vues_YT.png')
            st.image(image, use_column_width=True)
        
        st.info("Pour cette partie, nous n'avons étudié que les 12 000 musiques les plus populaires \
                car le web scraping aurait été trop long si nous avions du considérer l'entièreté du jeu de données.")
        st.success("Un **Random Over Sampler** a ensuite été utilisé pour équilibrer ces classes.")

    st.subheader('Modèles')
    
    st.markdown('Nous avons entraîné 5 modèles différents (avec une grille de recherche) dans le but d’avoir \
                 la meilleure prédiction possible : \
                 \n\n - Régression logistique \
                 \n\n - SVM \
                 \n\n - K plus proches voisins \
                 \n\n - Forêt aléatoire \
                 \n\n - XGBoost \n\n'
                 'Nous avons abouti aux résultats suivants :')
    
    if option == choix[0]:
        image = Image.open('Images/comparaison_accuracy_popu.png')
        st.image(image, use_column_width=True)
        st.markdown('On peut cependant contraster ces résultats en observant le score f1 pour chacune des \
                     classes :')
        image = Image.open('Images/comparaison_f1_score_popu.png')
        st.image(image, use_column_width=True)
        st.warning('On remarque effectivement que les prédictions sur les musiques populaires sont nettement \
                    moins fiables que celles pour les musiques impopulaires. Cette différence est due au fort \
                    déséquilibre entre nos deux classes au sein de l’ensemble de test. L’algorithme est donc bien \
                    plus performant sur la classe majoritaire que sur la classe minoritaire.')
    else:
        image = Image.open('Images/comparaison_accuracy_vues.png')
        st.image(image, use_column_width=True)
        image = Image.open('Images/comparaison_f1_score_vues.png')
        st.markdown('On peut également regarder le score f1 pour chacune des classes :')
        st.image(image, use_column_width=True)
        st.warning('Les résultats sont bien plus regroupés que dans les modèles utilisant la popularité \
                   comme variable cible, en revanche la valeur des scores reste assez faible.') 
        
    st.header('Interprétation')
    st.markdown("Il n’est pas étonnant que nous n’arrivions pas à prédire parfaitement l'appréciation d'une \
                musique étant donné que nous nous basons **exclusivement sur ses caractéristiques audios**. \
                Effectivement, on peut facilement intuiter que cela dépend de plusieurs autres facteurs \
                notamment l'exposition médiatique, la popularité de l'artiste, etc...")
                
    st.success("De ce fait, nos résultats ne sont clairement pas ridicules. Il serait utopique de \
              s’attendre à des résultats de l’ordre de 90% au vue des données passées en entrée")
        
    
def page_recommandation(state):
    st.title('RECOMMANDATION')
    st.markdown("# Les différentes méthodes de recommandation"
        "\n\n Pour réaliser notre système de recommandation, nous pouvions utiliser deux méthodes :"
        "\n\n - La recommandation basée sur le contenu utilisant les informations et caractéristiques d'une chanson pour en recommander une autre"
        "\n\n - Le filtrage collaboratif utilisant les informations des autres utilisateurs pour effectuer une recommandation")
    img1 = Image.open("Images/Content-based-filtering-vs-Collaborative-filtering-Source.jpg") 
    st.image(img1)

    st.markdown(" # La recommandation basée sur le contenu")
    st.markdown("Le premier système de recommandation que nous avons réalisé est un système de recommandation basé sur le contenu. " 
        "Dans notre cas nous avons utilisé le genre musical de la chanson et sa popularité"
        "\n\nCe système de recommandation détermine pour chaque musique mis en entrée les musiques les plus proches du point de vue du genre musical et nous avons "
        "recommandé parmi ces dernières celles ayant la plus grande popularité."
        )
    with st.expander("Exemple"):
        im_exemple1=Image.open("Images/recommandation_genre.jpg")
        st.image(im_exemple1)


    st.markdown("# Le filtrage collaboratif")
    st.markdown("Pour mettre en œuvre un système de filtrage collaboratif, nous avons eu besoin d'utilisé des données utilisateur, "
        "pour ce faire nous avons utilisé un jeu de données contenant les playlist d'un grand nombre d'utilisateur."
        "\n\n Nous pouvons ensuite effectuer deux types de recommandations différentes :"
        )
    img2 = Image.open("Images/filtrage_collaborative.jpg") 
    st.image(img2)
    st.markdown(
        "> - La recommandation “item item” qui prendra un produit particulier, "
        "trouvera les utilisateurs qui ont aimé ce produit et trouvera d'autres éléments que ces utilisateurs ont également aimé. "
        "Dans notre cas, nous recommandons des musiques en fonction d’une autre musique."
        "\n\n > - La recommandation “user user” prend un utilisateur, recherche des utilisateurs similaires à ce dernier sur la base d'évaluations "
        "ou de comportements similaires et recommande des produits que ces utilisateurs similaires ont aimés. "
        "Dans notre cas, nous recommandons des musiques en fonction d’une liste de musique correspondant à un utilisateur."
        "\n\n ## Recommandation “Item Item”"
        "\n\n Ce système de recommandation détermine pour chaque musique mis en entrée les playlists contenant cette musique, "
        "il recommande ensuite les musiques qui apparaissent le plus souvent dans ces playlists."
        
            )
    with st.expander("Exemple “item item”"):
        im_exemple2=Image.open("Images/recommandation_musique.jpg")
        st.image(im_exemple2)

    st.markdown("## Recommandation “User User”"
        "\n\n Ce système de recommandation détermine prend en entrée une liste de musique, il recherche ensuite pour chaque musique les playlists contenant cette dernière, "
        "on obtient une liste de playlist (une playlist peut apparaitre plusieurs fois dans cette liste et donc être pris en compte plusieurs fois si elle contient plusieurs musiques de la liste de musique) "
        "ensuite le système détermine les musiques les plus présentes dans la liste de playlist et les recommande."
    )

    with st.expander("Exemple “user user”"):
        im_exemple3=Image.open("Images/recommandation_playlist.jpg")
        st.image(im_exemple3)
    
def page_conclusion(state):
    st.title('CONCLUSION')
    st.write("Lors de la Data Visualisation nous avons remarqué que nos variables étaien très peu corrélées \
             entre elles ou avec la variable de sortie (popularité). C'est dans ce contexte \
             que nous avons fait de la classification, du clustering et enfin de la recommandation.")
    st.write("Pour réaliser l'objectif de prédiction de popularité nous avons opté pour \
             de la classification. Dans deux contextes différents (note de popularité \
             et nombre de vue) nous obtenons des résultats corrects sans être extraordinaires, \
             on peut donc dire que notre objectif et partiellement atteint. \
             Ces résultats corrects sont explicables, car il aurait été étonnant \
             de prédire précisément des comportements de consommateurs avec uniquement les caractéristiques \
             techniques d'un morceau de musique.")
    st.write("Pour le second objectif (recommandation de musique) nous avons choisi \
             deux méthodes différentes. \
             \nLa première méthode utilisée a été le clustering. Nous avons \
             utilisé l'algorithme du KMeans et deux PCA différentes, mais les résultats \
             de ces dernières étaient peu convaincants. On peut l'expliquer à cause \
             de nos variables très peu corrélées entre elles et de notre nombre de clusters \
             surement trop faible par rapport aux nombres de morceaux différents du jeu \
             de données. Nous avons donc choisis de partir sur la seconde méthode. \
             \nLa deuxième idée a été de faire une recommandation à l'aide du genre \
             des musiques extraits de l'API Spotify, et de playlists obtenues à l'aide d'un \
             autre jeu de données. Cette fois ci nos algorithmes effectuent des recommandations \
             cohérentes en fonction des morceaux qu'on leur indique en entrée. Le deuxième objectif \
             est donc atteint.")
    
def page_equipe(state):
    st.title("A PROPOS DE L'EQUIPE")
    st.info("Le projet à été mené par trois datascientists :"
            "\n\n - Louis Leveaux ([GitHub](https://github.com/LouisLvx) / [LinkedIn](https://www.linkedin.com/in/louis-leveaux-311865182/))"
            "\n\n - Caolan Gueguen ([GitHub](https://github.com/CaolanGu) / [LinkedIn](https://www.linkedin.com/in/caolan-gueguen-906218182/))"
            "\n\n - Maxime Cerisier ([GitHub](https://github.com/MSV94) / [LinkedIn](https://www.linkedin.com/in/maxime-cerisier/))")
    st.info("Cette application web fait partie de la formation DataScientest, Cursus DS - Bootcamp octobre 2021." 
            "\n\n Lien du site web de DataScientest : https://datascientest.com/en/home-page")

########################################################   
# EXECUTION DU CODE
########################################################

if __name__ == '__main__':
    main()
    
