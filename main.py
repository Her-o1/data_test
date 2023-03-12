# 1. Library imports
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI
import gunicorn
import uvicorn
# import sqlalchemy
# from sqlalchemy import create_engine
# import pymysql

# 2. Create the app object
app = FastAPI()
# engine = create_engine("mysql+pymysql://sql9605015:a6DAZVzvqL@sql9.freesqldatabase.com:3306/sql9605015")
## Load the dataset
# df = pd.read_sql_query("SELECT * FROM spoti", engine)
df = pd.read_csv("Spotify.csv")
# Define features of songs
genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
               'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

#. Load the model

model = NearestNeighbors()


# Define predict function
#  defining a function called predict which will take the input and internally uses PyCaret’s predict_model function to generate predictions and return the value as a dictionary
@app.get('/predict')
def predict(genre, test_feat):
    genre = genre.lower()
    test_feat = np.array(list(map(float, test_feat.strip('[]').split(','))))
    genre_data = df[(df["genres"] == genre) ]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:20]
    model.fit(genre_data[audio_feats].to_numpy())
    n_neighbors = model.kneighbors([test_feat], n_neighbors=len(
        genre_data), return_distance=False)[0]

    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return {'uris': uris }
