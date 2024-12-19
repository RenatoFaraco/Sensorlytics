
#!pip install process_spectra

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq

from scipy.signal import savgol_filter
from process_spectra.funcs import get_approximate_valley

def ler_multiplos_txt(diretorio,lambda_min=1400,lambda_max=1650):
    """
    Lê múltiplos arquivos `.txt` de um diretório, filtra os dados por comprimento de onda 
    e retorna uma lista de DataFrames.

    Parameters
    ----------
    diretorio : str
        Caminho para o diretório contendo os arquivos `.txt`.
    lambda_min : float, optional
        Valor mínimo do comprimento de onda para filtragem (default: 1400).
    lambda_max : float, optional
        Valor máximo do comprimento de onda para filtragem (default: 1650).

    Returns
    -------
    list of pd.DataFrame
        Lista de DataFrames contendo os dados filtrados de cada arquivo.
    """

    dfs = [] 
    
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith(".txt"):
            caminho_arquivo = os.path.join(diretorio, arquivo)
            df = pd.read_csv(caminho_arquivo, sep=';') 
            df = df.rename(columns={df.columns[0]: 'Wavelength', df.columns[1]: 'Level'})
            df = df[(df.Wavelength > lambda_min) & (df.Wavelength < lambda_max)]
            dfs.append(df)
    
    return dfs

def plotar_dfs(dataframes):
    """
    Plota os dados de uma lista de DataFrames contendo colunas `Wavelength` e `Level`.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        Lista de DataFrames a serem plotados. Cada DataFrame deve conter as colunas
        'Wavelength' e 'Level'.

    Returns
    -------
    None
        Exibe o gráfico gerado.
    """
    
    sns.set(style='ticks', palette='jet')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, df in enumerate(dataframes):
        ax.plot(df['Wavelength'], df['Level'])
    
    ax.set_xlabel('Wavelength', fontsize=12)
    ax.set_ylabel('Level', fontsize=12)
    ax.set_title('Plot de múltiplos DataFrames', fontsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.tick_params(axis='both', which='both', labelsize=10)
    plt.tight_layout()
    plt.show()
    
def get_mean(level):
    """
    Calcula a média suavizada de uma série de níveis utilizando o filtro de Savitzky-Golay.

    Parameters
    ----------
    level : array-like
        Série de valores de nível para cálculo da média suavizada.

    Returns
    -------
    numpy.ndarray
        Série suavizada dos níveis.
    """

    wind = len(level)/1.1
    wind = int(np.floor(wind))
    wind = wind + wind%2 + 1
    return savgol_filter(level, wind, 2)

def get_fringe(level):
    """
    Calcula as franjas de visibilidade subtraindo a média suavizada de uma série de níveis.

    Parameters
    ----------
    level : array-like
        Série de valores de nível.

    Returns
    -------
    numpy.ndarray
        Série de franjas de visibilidade suavizadas.
    """
    fringe = level-get_mean(level)
    fringe = savgol_filter(fringe, 11, 2)
    return fringe

def varre_dados_plot(dados_amostra,color='r'):
    """
    Calcula as franjas de visibilidade para cada DataFrame na lista e plota os resultados.

    Parameters
    ----------
    dados_amostra : list of pd.DataFrame
        Lista de DataFrames contendo as colunas `Wavelength` e `Level`.
    color : str, optional
        Cor da linha do gráfico (default: 'r').

    Returns
    -------
    None
        Exibe os gráficos das franjas de visibilidade.
    """
    Franjas_media = []
    for i, x in enumerate(dados_amostra):
        wl = dados_amostra[i].Wavelength
        level = dados_amostra[i].Level
        x_m = get_fringe(level)
        plt.plot(wl, x_m, color=color)
        Franjas_media.append(x_m)
    Franjas_media = np.array(Franjas_media)
        
def varre_dados(dados_amostra):
    """
    Calcula as franjas de visibilidade para cada DataFrame na lista, sem gerar gráficos.

    Parameters
    ----------
    dados_amostra : list of pd.DataFrame
        Lista de DataFrames contendo as colunas `Wavelength` e `Level`.

    Returns
    -------
    numpy.ndarray
        Array contendo as franjas de visibilidade para cada amostra.
    """
    Franjas_media = []
    for i, x in enumerate(dados_amostra):
        wl = dados_amostra[i].Wavelength
        level = dados_amostra[i].Level
        x_m = get_fringe(level)
        Franjas_media.append(x_m)
    Franjas_media = np.array(Franjas_media)
    return(Franjas_media)

def save_list_to_json(data, filename):
    """
    Salva dados processados em um arquivo JSON.

    Parameters
    ----------
    data : list, dict, or pd.DataFrame
        Dados a serem salvos no arquivo JSON. Se for um DataFrame, ele será convertido
        para um formato adequado antes de ser salvo.
    filename : str
        Nome do arquivo JSON (com extensão .json) onde os dados serão salvos.

    Returns
    -------
    None
        Salva o arquivo JSON no local especificado e exibe uma mensagem de confirmação.
    """
    if hasattr(data, "to_dict"):
        data_dict = {"data": data.to_dict(orient="records")}
    else:
        data_dict = {"data": data}

    with open(filename, "w") as f:
        json.dump(data_dict, f)
    print(f"Arquivo JSON '{filename}' salvo com sucesso.")
    
def load_json_file(file_path):
    """
    Carrega dados de um arquivo JSON e os retorna como um dicionário.

    Parameters
    ----------
    file_path : str
        Caminho completo para o arquivo JSON.

    Returns
    -------
    dict
        Dados carregados do arquivo JSON em formato de dicionário.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def dict_to_vector(dictionary):
    """
    Converte um dicionário em um vetor numpy.

    Parameters
    ----------
    dictionary : dict
        Dicionário cujos valores serão convertidos para um vetor.

    Returns
    -------
    numpy.ndarray
        Vetor contendo os valores do dicionário, reduzido a uma dimensão.
    """
    vector = np.array(list(dictionary.values()))
    vector = np.squeeze(vector)
    return vector

def salvar_espectros(lista_dataframes, nome_arquivo, pasta_destino):
    """
    Salva uma lista de DataFrames processados em um arquivo JSON.

    Parameters
    ----------
    lista_dataframes : list of pd.DataFrame
        Lista de DataFrames a serem salvos.
    nome_arquivo : str
        Nome do arquivo JSON (com extensão .json).
    pasta_destino : str
        Caminho da pasta onde o arquivo será salvo.

    Returns
    -------
    None
        Salva o arquivo JSON no local especificado e exibe uma mensagem de confirmação.
    """
    os.makedirs(pasta_destino, exist_ok=True)
    caminho_arquivo = os.path.join(pasta_destino, nome_arquivo)
    lista_dicionarios = [df.to_dict(orient='records') for df in lista_dataframes]

    with open(caminho_arquivo, 'w') as outfile:
        json.dump(lista_dicionarios, outfile, indent=4)
    
    print(f"Arquivo JSON salvo com sucesso em: {caminho_arquivo}")
    
def list_dict_to_list_df(list_dict):
    """
    Converte uma lista de dicionários em uma lista de DataFrames.

    Essa função é utilizada para transformar dados provenientes de arquivos JSON
    contendo múltiplos espectros, como os obtidos de um OSA Thorlabs, em DataFrames
    no formato comum.

    Parameters
    ----------
    list_dict : list of dict
        Lista de dicionários representando os dados dos espectros.

    Returns
    -------
    list of pd.DataFrame
        Lista de DataFrames convertidos a partir dos dicionários.

    Example
    -------
    >>> data = [{"Wavelength": [1, 2], "Level": [0.5, 0.7]}, {"Wavelength": [3, 4], "Level": [0.6, 0.8]}]
    >>> dfs = list_dict_to_list_df(data)
    >>> print(dfs[0])
       Wavelength  Level
    0           1    0.5
    1           2    0.7
    """
    lista_dataframes = []
    for dicionario in list_dict:
        df = pd.DataFrame.from_dict(dicionario)
        lista_dataframes.append(df)
    return(lista_dataframes)

def cria_franjas(specs_experimento):
    """
    Gera as franjas para todos os espectros de um experimento.

    Aplica a função `varre_dados` para uma lista contendo listas de DataFrames
    representando os espectros de cada amostra.

    Parameters
    ----------
    specs_experimento : list of list of pd.DataFrame
        Lista contendo as listas de DataFrames de espectros para cada amostra.

    Returns
    -------
    tuple of (numpy.ndarray, list of int)
        - Um array contendo todas as franjas processadas.
        - Uma lista com os comprimentos das franjas para cada conjunto de espectros.

    Example
    -------
    >>> specs_dia_2 = [spec_1_dia_2, spec_2_dia_2]
    >>> franjas, len_franjas = cria_franjas(specs_dia_2)
    """
    franjas = []  
    len_franjas = []  
    for spec in specs_experimento:
        franjas_spec = np.array(varre_dados(spec))
        len_spec = (len(franjas_spec))
        len_franjas.append(len_spec)
        franjas.extend(franjas_spec)
    return np.array(franjas), len_franjas

def calcular_wl_res_list(dados_amostra):
    """
    Calcula a lista de comprimentos de onda ressonantes para os espectros fornecidos.

    Utiliza a função `get_approximate_valley` para determinar os valores ressonantes
    de comprimento de onda para cada DataFrame em `dados_amostra`.

    Parameters
    ----------
    dados_amostra : list of pd.DataFrame
        Lista de DataFrames contendo os espectros com colunas `Wavelength` e `Level`.

    Returns
    -------
    list of float
        Lista de valores de comprimento de onda ressonantes calculados.

    Example
    -------
    >>> dados_amostra = [df1, df2]
    >>> wl_res_list = calcular_wl_res_list(dados_amostra)
    """
    stdout_temp = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    wl_res_list = []
    for df in dados_amostra:
        spec = np.array([df['Wavelength'].values, df['Level'].values])
        spec, info = get_approximate_valley(spec.T, {}, prominence=0.5)
        i = info['best_index']
        wl_res = info[f'resonant_wl_{i}']
        wl_res_list.append(wl_res)
    sys.stdout = stdout_temp

    return wl_res_list


def find_pdf(data):
    """
    Calcula a PDF (Probabilidade de Densidade) para os valores únicos em um conjunto de dados.

    Parameters
    ----------
    data : numpy.ndarray or list
        Dados para os quais a PDF será calculada.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        - Os valores únicos presentes nos dados.
        - A PDF correspondente a cada valor único.

    Example
    -------
    >>> data = [1, 2, 2, 3, 3, 3]
    >>> unique_values, pdf = find_pdf(data)
    >>> print(unique_values, pdf)
    [1 2 3] [0.16666667 0.33333333 0.5]
    """
    unique_values, counts = np.unique(data, return_counts=True)
    pdf = counts / np.sum(counts)
    return unique_values, pdf


def plot_histogram(data, ax=0):
    """
    Plota um histograma para os dados fornecidos.

    Parameters
    ----------
    data : array-like
        Dados para os quais o histograma será gerado.
    ax : int, optional
        Índice do eixo (não utilizado no código atual, mas pode ser estendido para múltiplos eixos).

    Returns
    -------
    None
        Exibe o histograma gerado.

    Example
    -------
    >>> data = [1, 2, 2, 3, 3, 3, 4]
    >>> plot_histogram(data)
    """
    plt.hist(data, bins='auto', alpha=0.7, rwidth=0.85, color='r')
    plt.xlabel('Valores')
    plt.ylabel('Frequência')
    plt.title('Histograma')
    plt.grid(True)
    plt.show()

def save_to_parquet(data, dia, sample, rodada, pasta_destino):
    """
    Salva os dados fornecidos em formato Parquet em um diretório especificado.

    Parameters
    ----------
    data : list or pd.DataFrame
        Dados a serem salvos. Podem ser uma lista ou um DataFrame.
    dia : str
        Identificador do dia.
    sample : str
        Identificador da amostra.
    rodada : str
        Identificador da rodada.
    pasta_destino : str
        Caminho para o diretório onde o arquivo será salvo.

    Returns
    -------
    None
        Salva os dados em um arquivo Parquet no diretório especificado.

    Example
    -------
    >>> data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> save_to_parquet(data, '2024-12-19', 'sample1', 'rodada1', './dados')
    """
    os.makedirs(pasta_destino, exist_ok=True)
    nome_arquivo = f"dados_{dia}_{sample}_{rodada}.parquet"
    caminho_arquivo = os.path.join(pasta_destino, nome_arquivo)
    
    if isinstance(data, list):
        data = pd.DataFrame(data)
    
    try:
        data.to_parquet(caminho_arquivo, engine='pyarrow')
        print(f"Arquivo Parquet salvo com sucesso em: {caminho_arquivo}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo {caminho_arquivo}: {e}")

def save_to_json(data, nome_arquivo, pasta_destino):
    """
    Salva os dados fornecidos em formato JSON em um diretório especificado.

    Parameters
    ----------
    data : list, dict, or pd.DataFrame
        Dados a serem salvos. Podem ser uma lista, um dicionário ou um DataFrame.
    nome_arquivo : str
        Nome do arquivo JSON (com extensão `.json`).
    pasta_destino : str
        Caminho para o diretório onde o arquivo será salvo.

    Returns
    -------
    None
        Salva os dados em um arquivo JSON no diretório especificado.

    Example
    -------
    >>> data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> save_to_json(data, 'dados.json', './dados')
    """
    os.makedirs(pasta_destino, exist_ok=True)
    caminho_arquivo = os.path.join(pasta_destino, nome_arquivo)
 
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="records")
    
    try:
        with open(caminho_arquivo, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Arquivo JSON salvo com sucesso em: {caminho_arquivo}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo {caminho_arquivo}: {e}")