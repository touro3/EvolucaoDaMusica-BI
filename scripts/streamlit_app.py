import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

# Configurar Spark
spark = SparkSession.builder.appName('MusicDataAnalysis').getOrCreate()

# Carregar dados usando PySpark
data_df = spark.read.csv('/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/data.csv', header=True, inferSchema=True).toPandas()
musicdata_df = spark.read.csv('/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/musicdata.csv', header=True, inferSchema=True).toPandas()
charts_df = spark.read.csv('/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/charts.csv', header=True, inferSchema=True).toPandas()
new_df = spark.read.csv('/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/Best selling music artists.csv', header=True, inferSchema=True).toPandas()
new_artists_df = spark.read.csv('/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/Artists.csv', header=True, inferSchema=True).toPandas()

st.title('Análise da Evolução da Música')

# Visualização da popularidade das músicas ao longo dos anos
st.subheader('Popularidade das Músicas ao Longo dos Anos')
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_df, x='year', y='popularity')
plt.title('Popularidade das Músicas ao Longo dos Anos')
plt.xlabel('Ano')
plt.ylabel('Popularidade Média')
plt.grid(True)
st.pyplot(plt)

# Visualização da evolução das vendas de música por formato
st.subheader('Evolução das Vendas de Música por Formato')
plt.figure(figsize=(14, 7))
sns.lineplot(data=musicdata_df, x='year', y='value_actual', hue='format', marker='o')
plt.title('Evolução das Vendas de Música por Formato')
plt.xlabel('Ano')
plt.ylabel('Vendas (Unidades)')
plt.legend(title='Formato', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
st.pyplot(plt)

# Visualização da evolução dos rankings do Billboard Hot 100
st.subheader('Evolução dos Rankings do Billboard Hot 100')
plt.figure(figsize=(12, 6))
sns.lineplot(data=charts_df, x='date', y='rank', ci=None)
plt.gca().invert_yaxis()  # Inverte o eixo Y para que o ranking 1 esteja no topo
plt.title('Evolução dos Rankings do Billboard Hot 100')
plt.xlabel('Data')
plt.ylabel('Rank')
plt.grid(True)
st.pyplot(plt)

# Função para integrar gênero e criar gráfico para diferentes eras
def plot_top_artists_by_sales_with_genre(dataframe, genre_dataframe, start_year, end_year, era_name):
    era_df = dataframe[(dataframe['Release year of first charted record'] >= start_year) & (dataframe['Release year of first charted record'] <= end_year)]
    artists_sales = era_df.groupby('Artist name')['Total certified units'].sum().reset_index()
    top_artists_sales = artists_sales.sort_values(by='Total certified units', ascending=False).head(10)
    
    # Integrar com dados de gênero
    top_artists_sales = top_artists_sales.merge(genre_dataframe[['Name', 'Genres']], left_on='Artist name', right_on='Name', how='left')
    
    # Preencher valores ausentes de gênero com "Indefinido"
    top_artists_sales['Genres'].fillna('Indefinido', inplace=True)

    # Visualizar os resultados
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_artists_sales, x='Total certified units', y='Artist name', hue='Genres', dodge=False, palette='viridis')
    plt.title(f'Top 10 Artistas com Mais Vendas Certificadas Durante a Era {era_name} ({start_year}-{end_year})')
    plt.xlabel('Vendas Certificadas (em milhões)')
    plt.ylabel('Artista')
    plt.legend(title='Gênero', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)

# Analisar e visualizar diferentes eras com gêneros
st.subheader('Top Artistas por Vendas Certificadas por Era')
plot_top_artists_by_sales_with_genre(new_df, new_artists_df, 1950, 1980, 'dos Vinis')
plot_top_artists_by_sales_with_genre(new_df, new_artists_df, 1970, 1990, 'dos Cassetes')
plot_top_artists_by_sales_with_genre(new_df, new_artists_df, 1980, 2000, 'dos CDs')
plot_top_artists_by_sales_with_genre(new_df, new_artists_df, 2000, 2020, 'Digital')

# Analisar popularidade digital dos artistas
st.subheader('Top 10 Artistas Mais Populares Durante a Era Digital')
# Garantir que 'Popularity' é numérico
new_artists_df['Popularity'] = pd.to_numeric(new_artists_df['Popularity'], errors='coerce')

artists_digital_popularity = new_artists_df.groupby(['Name', 'Genres'])['Popularity'].mean().reset_index()
top_artists_digital_popularity = artists_digital_popularity.sort_values(by='Popularity', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_artists_digital_popularity, x='Popularity', y='Name', hue='Genres', dodge=False, palette='viridis')
plt.title('Top 10 Artistas Mais Populares Durante a Era Digital (atualizado no fim de 2023)')
plt.xlabel('Popularidade')
plt.ylabel('Artista')
plt.legend(title='Gênero', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(plt)
