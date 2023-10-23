import streamlit as st
import openai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Configura tu clave de API de OpenAI
openai.api_key = 'sk-openai-key'  # Reemplaza con tu clave de API

# Define el nombre del bot
nombre_bot = "MiBot"

# Define la función para obtener el embedding
def get_embedding(text, engine="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=[text],
        model=engine
    )
    embedding_result = response['data'][0]['embedding']
    return embedding_result

# Define la función para cargar datos y calcular embeddings
def embed_text(path="chatbot_qa.csv"):
    conocimiento_df = pd.read_csv(path)
    conocimiento_df['Embedding'] = conocimiento_df['texto'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    conocimiento_df.to_csv('embeddings.csv', index=False)
    return conocimiento_df

# Carga los datos y calcula los embeddings al inicio
texto_emb = embed_text("chatbot_qa.csv")

# Define la función para buscar y devolver el texto más parecido
def buscar(busqueda, datos):
    busqueda_embed = get_embedding(busqueda, engine="text-embedding-ada-002")
    
    # Convierte la columna "Similitud" a un tipo de dato numérico
    datos["Similitud"] = datos['Embedding'].apply(lambda x: cosine_similarity([x], [busqueda_embed])[0]).astype(float)
    
    # Encuentra el índice del resultado más parecido
    indice_resultado_parecido = datos["Similitud"].idxmax()
    
    # Obtiene el texto del resultado más parecido
    texto_resultado_parecido = datos.loc[indice_resultado_parecido, "texto"]
    
    return texto_resultado_parecido

# Interfaz de Streamlit
st.title("Hifest-AI")

busqueda = st.text_input("How can I help you? ")
if busqueda:
    if st.session_state.get("ultima_busqueda") != busqueda:
        resultado_parecido = buscar(busqueda, texto_emb)
        st.session_state.ultima_busqueda = busqueda

        # Agrega estilo al mensaje de respuesta y muestra el nombre del bot
        st.markdown('<div style="background-color:#f2f2f2; padding:10px; border-radius:10px;">'
                    f'<p style="color:#333; font-size:16px;">Hifest-AI says : {resultado_parecido}</p>'
                    '</div>', unsafe_allow_html=True)
