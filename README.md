# SpotiPy

## Presentation

This repository contains the code for our project **SpotiPy**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to :
- 1/ Predict whether a music will be more or less popular.
- 2/ Develop a music recommandation algorithm.

At the beginning of this project, we had 3 datasets from [Kaggle/#nowplaying-rs](https://www.kaggle.com/chelseapower/nowplayingrs?select=sentiment_values.csv) : 
- **context_content_features.csv** containing all context and content features for each listening event.
- **sentiment_values.csv containing** the hashtag itself and the sentiment values gathered via four different sentiment dictionaries (AFINN, Opinion Lexicon, Sentistrength Lexicon and vader). We changed it a little into: sentiment_values_v2.csv, because some columns had no name.
- **user_track_hashtag_timestamp.csv** containing context features from Twitter (2014).

However those datasets were not exploitable due to :
- the track IDs were not usable (they didn't correspond to any Spotify music).
- the link between sentimental values and hashtags were not reliable.

To bypass those issues we found 2 other datasets which allows us to come up with workable outcome :
- **track_features.csv** from [Kaggle/Spotify 1.2M+ Songs](https://www.kaggle.com/rodolfofigueroa/spotify-12m-songs) containing all context and content features for each listening event (and usable with Spotify's API)
- **spotify_million_playlist_dataset** from [Kaggle/spotify-millions-playlist](https://www.kaggle.com/adityak80/spotify-millions-playlist) containing music IDs and their playlists associated in JSON format.

This project was developed by the following team :

- Louis Leveaux ([GitHub](https://github.com/LouisLvx) / [LinkedIn](https://www.linkedin.com/in/louis-leveaux-311865182/))
- Caolan Gueguen ([GitHub](https://github.com/CaolanGu) / [LinkedIn](https://www.linkedin.com/in/caolan-gueguen-906218182/)
- Maxime Cerisier ([GitHub](https://github.com/MSV94) / [LinkedIn](https://www.linkedin.com/in/maxime-cerisier/))

## Installation

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

```
pip install imblearn
pip install xgboost
pip install prince
pip install spotipy --upgrade
```

## Streamlit App

To run the app :

```shell
cd streamlit_app
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
