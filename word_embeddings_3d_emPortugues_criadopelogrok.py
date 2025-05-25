# Criado com o Grok em 25/05/2025
import numpy as np
import plotly.graph_objects as go
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from gensim.downloader import load

# Lista de palavras em português
palavras = ['prédio', 'edifício', 'construção', 'casa', 'apartamento']

# Carregar modelo fasttext pré-treinado
print("Carregando modelo fasttext...")
model = load('fasttext-wiki-news-subwords-300')  # Modelo pré-treinado

# Obter embeddings das palavras
embeddings = np.array([model[word] for word in palavras if word in model])

# Verificar se todas as palavras foram encontradas
if len(embeddings) != len(palavras):
    print("Aviso: Algumas palavras não foram encontradas no modelo.")

# Reduzir dimensionalidade para 3D usando PCA
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

# Criar gráfico 3D interativo com Plotly
fig = go.Figure()

# Adicionar pontos para as palavras
fig.add_trace(go.Scatter3d(
    x=embeddings_3d[:, 0],
    y=embeddings_3d[:, 1],
    z=embeddings_3d[:, 2],
    mode='markers+text',
    text=palavras[:len(embeddings_3d)],  # Rótulos das palavras
    marker=dict(size=8, color='blue'),
    textposition='top center'
))

# Configurar layout do gráfico
fig.update_layout(
    title="Visualização 3D de Embeddings de Palavras",
    scene=dict(
        xaxis_title="Componente 1",
        yaxis_title="Componente 2",
        zaxis_title="Componente 3",
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True),
        zaxis=dict(showgrid=True, zeroline=True)
    ),
    showlegend=False
)

# Exibir gráfico interativo
fig.show()

# Opcional: Salvar o gráfico como HTML para visualização offline
fig.write_html("embeddings_3d.html")
print("Gráfico salvo como 'embeddings_3d.html'")