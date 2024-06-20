from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year
import pandas as pd

# Inicializar a sessão Spark
spark = SparkSession.builder.appName("MusicDataAnalysis").getOrCreate()

# Função para carregar dados
def load_data(file_path):
    return spark.read.csv(file_path, header=True, inferSchema=True)

# Função para limpar e tratar os dados
def clean_data(df, date_col=None):
    df = df.dropna()  # Remover valores nulos
    if date_col:
        df = df.withColumn(date_col, to_date(df[date_col], 'yyyy-MM-dd'))
        df = df.filter(year(col(date_col)).between(1900, pd.Timestamp.now().year))  # Filtro de ano
    df = df.dropDuplicates()  # Remover duplicatas
    return df

# Carregar datasets
data_paths = {
    "data_df": '/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/data.csv',
    "musicdata_df": '/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/musicdata.csv',
    "charts_df": '/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/charts.csv',
    "new_df": '/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/Best selling music artists.csv',
    "new_artists_df": '/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/Artists.csv',
    "final_df": '/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/final.csv'
}

datasets = {name: load_data(path) for name, path in data_paths.items()}

# Tratar cada dataset
datasets = {name: clean_data(df, date_col='release_date' if 'release_date' in df.columns else None) for name, df in datasets.items()}

# Tratar outliers no dataset 'data_df'
datasets['data_df'] = datasets['data_df'].filter(col('popularity') <= 100)

# Salvar os dados tratados
for name, df in datasets.items():
    df.write.parquet(f'/home/gabipereira/tremBI/EvolucaoDaMusica-BI/data/processed_{name}.parquet', mode='overwrite')

# Encerrar sessão Spark
spark.stop()
