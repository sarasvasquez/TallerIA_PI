import os
import json
import numpy as np
from openai import OpenAI

# Cargar el archivo .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), 'api_keys.env'))

# Obtener la ruta absoluta del archivo JSON
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(BASE_DIR, 'movie_descriptions_embeddings.json')

# Ahora abre el archivo JSON usando la ruta absoluta
with open(json_file_path, 'r') as file:
    file_content = file.read()
    movies = json.loads(file_content)

# Funciones auxiliares para embeddings y similitud coseno
def get_embedding(text, client, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Función de recomendación (el resto de tu código sigue igual)
def return_recommendation(req):
    client = OpenAI(api_key=os.getenv('openai_api_key'))

    emb = get_embedding(req, client)
    sim = []
    for i in range(len(movies)):
        sim.append(cosine_similarity(emb, movies[i]['embedding']))
    sim = np.array(sim)
    idx = np.argmax(sim)
    return movies[idx]['title']
#print(return_recommendation("pelicula de personas felices"))

