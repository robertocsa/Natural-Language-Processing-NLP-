# Carrega modelo GloVe
import gensim.downloader
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# Carregando o modelo pré-treinado
model = gensim.downloader.load("glove-wiki-gigaword-50")

# Lista de palavras para plotar
words = ["tower", "skyscraper", "roof", "built", "constructed"]

# Obtendo os vetores de embedding
vectors = [model[word] for word in words]

# Redução para 3 dimensões com PCA
pca = PCA(n_components=3)
vectors_3d = pca.fit_transform(vectors)

# Separar as coordenadas X, Y, Z
x, y, z = vectors_3d[:,0], vectors_3d[:,1], vectors_3d[:,2]

# Criar o gráfico 3D interativo
fig = go.Figure()

# Adicionar cada ponto com anotação
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers+text',
    marker=dict(size=6, color='blue'),
    text=words,
    textposition="top center"
))

# Personalizar layout
fig.update_layout(
    title="Embeddings GloVe - Representação 3D",
    scene=dict(
        xaxis_title='PCA 1',
        yaxis_title='PCA 2',
        zaxis_title='PCA 3'
    )
)

# Mostrar o gráfico
fig.show()
