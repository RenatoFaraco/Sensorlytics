from Sensorlytics import ler_multiplos_txt, varre_dados_plot

# Diretório onde estão os arquivos .txt
diretorio = "path/to/your/directory"

# Leia os arquivos .txt do diretório para criar os DataFrames
dataframes = ler_multiplos_txt(diretorio)

# Plote as franjas de visibilidade para os dados
varre_dados_plot(dataframes, color='b')