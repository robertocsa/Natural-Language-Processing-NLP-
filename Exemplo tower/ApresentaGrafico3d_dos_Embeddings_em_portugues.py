# Criado no ChatGPT em 25/05/2025
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# Carregando o modelo FastText em português
print("Carregando modelo FastText em português...")
# Fonte para download: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz
model = KeyedVectors.load_word2vec_format("cc.pt.300.vec.gz", binary=False)
print("Modelo carregado.")

# Palavras em português para visualização
words = ["torre", "arranha-céu", "telhado", "construído", "edificado"]

# Verificar se todas as palavras estão no vocabulário
missing = [word for word in words if word not in model]
if missing:
    print("Palavras ausentes do modelo:", missing)
    exit()

# Obter os vetores
vectors = [model[word] for word in words]

# Redução de dimensionalidade para 3D
pca = PCA(n_components=3)
vectors_3d = pca.fit_transform(vectors)

# Separar componentes X, Y, Z
x, y, z = vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2]

# Criar visualização 3D interativa
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers+text',
    text=words,
    textposition='top center',
    marker=dict(size=8, color='royalblue')
))

fig.update_layout(
    title='Embeddings de Palavras em Português (FastText)',
    scene=dict(
        xaxis_title='Componente 1 (PCA)',
        yaxis_title='Componente 2 (PCA)',
        zaxis_title='Componente 3 (PCA)',
        xaxis=dict(showspikes=False),
        yaxis=dict(showspikes=False),
        zaxis=dict(showspikes=False),
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()
