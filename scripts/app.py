import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st 
import re
import numpy as np
from pyspark.sql import SparkSession

# Inicializar SparkSession
spark = SparkSession.builder.appName('music_analysis').getOrCreate()

# Carregar os dados processados com PySpark
data_df = spark.read.parquet('/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/processed_data_df.parquet').toPandas()
musicdata_df = spark.read.parquet('/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/processed_musicdata_df.parquet').toPandas()
charts_df = spark.read.parquet('/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/processed_charts_df.parquet').toPandas()
new_df = spark.read.parquet('/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/processed_new_df.parquet').toPandas()
new_artists_df = spark.read.parquet('/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/processed_new_artists_df.parquet').toPandas()
final_df = spark.read.parquet('/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/processed_final_df.parquet').toPandas()

# Configurar os dados
data_df['year'] = data_df['year'].astype(int)
data_df['popularity'] = data_df['popularity'].astype(float)

# Função para visualizar a popularidade das músicas ao longo dos anos
def plot_popularity_over_years(data_df):
    plt.figure(figsize=(12,6))
    sns.lineplot(data=data_df, x='year', y='popularity')
    plt.title('Popularidade das Músicas ao Longo dos Anos')
    plt.xlabel('Ano')
    plt.ylabel('Popularidade Média')
    plt.grid(True)
    st.pyplot(plt)

# Função para visualizar a evolução das vendas de música por formato
def plot_sales_by_format(musicdata_df):
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=musicdata_df, x='year', y='value_actual', hue='format', marker='o')
    plt.title('Evolução das Vendas de Música por Formato')
    plt.xlabel('Ano')
    plt.ylabel('Vendas (Unidades)')
    plt.legend(title='Formato', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    st.pyplot(plt)

# Função para visualizar a evolução dos rankings do Billboard Hot 100
def plot_billboard_rankings(charts_df):
    plt.figure(figsize=(12,6))
    sns.lineplot(data=charts_df, x='date', y='rank', ci=None)
    plt.gca().invert_yaxis()  # Inverte o eixo Y para que o ranking 1 esteja no topo
    plt.title('Evolução dos Rankings do Billboard Hot 100')
    plt.xlabel('Data')
    plt.ylabel('Rank')
    plt.grid(True)
    st.pyplot(plt)

# Função para extrair e somar as vendas certificadas
def extract_sales(units_string):
    units = re.findall(r'(\d+(?:\.\d+)?)(?: million)?', units_string.replace(',', ''))
    return sum(float(num) for num in units)

# Aplicando a função na coluna 'Total certified units'
new_df['Total Sales (millions)'] = new_df['Total certified units'].apply(extract_sales)

# Definição das eras
eras = {
    'Vinyl Era': (1950, 1980),
    'Cassette Era': (1970, 1990),
    'CD Era': (1980, 2000),
    'Digital Era': (2000, 2020)
}

# Função para filtrar artistas ativos durante uma dada era
def filter_artists_by_era(df, start_year, end_year):
    def active_during(artist_years):
        years_active = re.findall(r'\d{4}', artist_years)
        if not years_active:
            return False
        artist_start, artist_end = int(years_active[0]), int(years_active[-1])
        return (artist_start <= end_year) and (artist_end >= start_year)
    
    return df[df['Active years'].apply(active_during)].nlargest(10, 'Total Sales (millions)')

# Filtrando artistas para cada era
top_artists_per_era = {era: filter_artists_by_era(new_df, start, end) for era, (start, end) in eras.items()}

# Função para plotar os gráficos para cada era com os gêneros musicais e cores customizadas
def plot_top_artists_with_genres_custom_colors(era_name, df):
    # Cores em degradê de azul escuro
    dark_colors = plt.cm.Blues(np.linspace(0.5, 1, len(df)))  # Invertendo o degradê para escuro no topo
    fig, ax = plt.subplots(figsize=(12, 8))
    df = df.sort_values('Total Sales (millions)', ascending=True)
    bars = ax.barh(df['Artist name'], df['Total Sales (millions)'], color=dark_colors)
    ax.set_title(f'Top 10 Selling Artists in the {era_name} (Millions of Units Sold)', fontsize=14)
    ax.set_xlabel('Total Sales (Millions)', fontsize=12)
    ax.set_ylabel('Artist Name', fontsize=12)

    # Adicionando os gêneros musicais dentro das barras
    for bar, genre in zip(bars, df['Genre']):
        text = genre.strip('[]').replace("'", "")
        ax.text(bar.get_width() - 5, bar.get_y() + bar.get_height()/2, f"{text}",
                va='center', ha='right', color='white', fontsize=10)

    plt.tight_layout()
    return fig

# Processar dados dos artistas mais populares atualmente
new_artists_df['Popularity'] = pd.to_numeric(new_artists_df['Popularity'], errors='coerce')
artists_df = new_artists_df.dropna(subset=['Popularity'])
top_10_current = artists_df.nlargest(10, 'Popularity')

# Função para plotar os gráficos com degradê de cores escuras
def plot_top_artists_by_popularity_very_dark_colors(df, title):
    sorted_df = df.sort_values('Popularity', ascending=True)
    dark_colors = plt.cm.Blues(np.linspace(0.5, 1, len(sorted_df)))
    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.barh(sorted_df['Name'], sorted_df['Popularity'], color=dark_colors)
    ax.set_xlabel('Popularity')
    ax.set_title(title)
    ax.set_yticklabels(sorted_df['Name'], fontdict={'horizontalalignment': 'right'})

    for bar, genre in zip(bars, sorted_df['Genres']):
        text = genre.strip('[]').replace("'", "")
        ax.text(bar.get_width() - 5, bar.get_y() + bar.get_height()/2, f"{text}",
                va='center', ha='right', color='white', fontsize=10)

    plt.tight_layout()
    return fig

# Processar dados dos artistas com mais discos vendidos (vinil)
final_df['price'] = pd.to_numeric(final_df['price'], errors='coerce')
final_df['stock'] = pd.to_numeric(final_df['stock'], errors='coerce')
final_df['sales'] = final_df['price'] * final_df['stock']
final_df = final_df[final_df['artist'] != 'Various Artists']
artists_vinyl_sales = final_df.groupby('artist')['sales'].sum().reset_index()
top_artists_vinyl_sales = artists_vinyl_sales.sort_values(by='sales', ascending=False).head(10)

# Função para plotar os gráficos dos artistas com mais vendas estimadas na era do vinil
def plot_top_artists_vinyl_sales(df, title):
    plt.figure(figsize=(12,6))
    sns.barplot(data=df, x='sales', y='artist', palette='viridis')
    plt.title(title)
    plt.xlabel('Vendas Estimadas (em dólares)')
    plt.ylabel('Artista')
    plt.grid(True)
    st.pyplot(plt)

# Título do Aplicativo
st.title('Análise da Evolução da Música')
st.write('Este aplicativo analisa a evolução da música ao longo dos anos, incluindo a popularidade das músicas, as vendas de música por formato, os rankings do Billboard Hot 100, os artistas mais ouvidos por formato de mídia e suas eras musicais, os artistas mais populares atualmente e os artistas com mais discos vendidos na era do vinil.')

# Sidebar para navegação
st.sidebar.title('Menu')

# Adicionar opção para visualizar a popularidade ao longo dos anos
if st.sidebar.checkbox('Mostrar Popularidade das Músicas ao Longo dos Anos'):
    st.header('Popularidade das Músicas ao Longo dos Anos')
    plot_popularity_over_years(data_df)

# Adicionar opção para visualizar a evolução das vendas de música por formato
if st.sidebar.checkbox('Mostrar Evolução das Vendas de Música por Formato'):
    st.header('Evolução das Vendas de Música por Formato')
    plot_sales_by_format(musicdata_df)

# Adicionar opção para visualizar a evolução dos rankings do Billboard Hot 100
if st.sidebar.checkbox('Mostrar Evolução dos Rankings do Billboard Hot 100'):
    st.header('Evolução dos Rankings do Billboard Hot 100')
    plot_billboard_rankings(charts_df)

show_era_graph = st.sidebar.checkbox('Mostrar Gráfico das Eras Musicais')
if show_era_graph:
    era_option = st.sidebar.selectbox('Selecionar Era Musical', list(eras.keys()))
    if era_option:
        st.header(f'Top 10 Artistas Mais Ouvidos na {era_option}')
        fig = plot_top_artists_with_genres_custom_colors(era_option, top_artists_per_era[era_option])
        st.pyplot(fig)

# Adicionar opção para visualizar os artistas mais populares atualmente
if st.sidebar.checkbox('Mostrar Artistas Mais Populares Atualmente'):
    st.header('Top 10 Artistas Mais Populares da Atualidade')
    fig_current_popular_very_dark_colors = plot_top_artists_by_popularity_very_dark_colors(
        top_10_current, 'Top 10 Most Popular Artists of the Present (Very Dark Colors)')
    st.pyplot(fig_current_popular_very_dark_colors)

# Adicionar opção para visualizar os artistas com mais discos vendidos (vinil)
if st.sidebar.checkbox('Mostrar Artistas com Mais Discos Vendidos (Vinil)'):
    st.header('Top 10 Artistas com Mais Vendas Estimadas na Era do Vinil')
    plot_top_artists_vinyl_sales(top_artists_vinyl_sales, 'Top 10 Artistas com Mais Vendas Estimadas na Era do Vinil')

# Exibir os dataframes carregados
if st.sidebar.checkbox('Mostrar Dataframes Carregados'):
    st.subheader('Data DF')
    st.write(data_df.head())

    st.subheader('Music Data DF')
    st.write(musicdata_df.head())

    st.subheader('Charts DF')
    st.write(charts_df.head())

    st.subheader('New DF')
    st.write(new_df.head())

    st.subheader('New Artists DF')
    st.write(new_artists_df.head())

    st.subheader('Final DF')
    st.write(final_df.head())
