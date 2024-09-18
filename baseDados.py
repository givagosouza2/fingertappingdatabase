import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
import matplotlib.patches as patches


def fit_ellipse(x, y):
    # Fit an ellipse to the data
    cov_matrix = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Get the angle and semi-axes lengths
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    semi_major_axis = 2.0 * np.sqrt(eigenvalues[0])*1.96
    semi_minor_axis = 2.0 * np.sqrt(eigenvalues[1])*1.96

    # Get the center of the ellipse
    center = np.mean(np.column_stack((x, y)), axis=0)

    # Create the Ellipse instance with the fitted parameters
    ellipse = Ellipse(xy=center, width=semi_major_axis,
                      height=semi_minor_axis, angle=angle)

    # Generate points on the fitted ellipse for plotting
    fit_x, fit_y = ellipse.get_verts().T
    return fit_x, fit_y, semi_major_axis, semi_minor_axis, angle


st.set_page_config(layout="wide")
st.title('Brazilian Database for Smartphone-Based Finger Tapping Test')
st.write('The Brazilian Database for the Smartphone-Based Finger Tapping Test is a collaborative research effort involving investigators from the Federal University of Pará, Federal University of West Pará, Federal Institute of São Paulo, and Federal University of Maranhão. The primary objective of this database is to serve as a healthy reference for the Brazilian population. None participant has declared the identity, but other information such as sex and age are available. Currently, the database exclusively includes data from right-handers. However, future releases are anticipated to incorporate information from left-handers as well. The Finger Tapping Test performances are recorded at the center of the smartphone screen. For any inquiries or requests related to the database, please feel free to contact us via email at givagosouza@ufpa.br.')
st.write('Instruction of use. This web application has two tools to explore the database: (i) one to evaluate each individual dataset and (ii) another to evaluate the whole database or in filtered conditions of sample features.')
tab1, tab2, tab3 = st.tabs(
    ["Individual data", "Database statistics", "Releases"])
with tab1:
    st.title('Individual data')
    caminho_base_de_dados = 'database.csv'
    base_de_dados = pd.read_csv(caminho_base_de_dados)

    # Criar um dicionário para armazenar listas de dados para cada paciente
    participantes = {}

    # Iterar sobre cada linha do DataFrame
    for indice, linha in base_de_dados.iterrows():
        # Obter o código do paciente
        codigo_participante = linha['Code']
        if codigo_participante not in participantes:
            # Se não estiver, criar uma entrada para o paciente no dicionário
            participantes[codigo_participante] = {
                'codigo': [],
                'codigo_tentativa': [],
                'tempo': [],
                'x_coord': [],
                'y_coord': [],
                'sexo': [],
                'mao': [],
                'smartphone': [],
                'x_resolucao': [],
                'y_resolucao': [],
                'ano_nascimento': [],
                'ano_teste': [],
                'idade_teste': []
            }

        # Adicionar os valores das colunas para o paciente atual

        participantes[codigo_participante]['codigo'].append(
            linha['Code'])
        participantes[codigo_participante]['tempo'].append(
            linha['time_of_tap'])
        participantes[codigo_participante]['x_coord'].append(
            linha['x_coord'])
        participantes[codigo_participante]['y_coord'].append(
            linha['y_coord'])
        participantes[codigo_participante]['sexo'].append(linha['Sex'])
        participantes[codigo_participante]['mao'].append(linha['Hand'])
        participantes[codigo_participante]['smartphone'].append(
            linha['Smartphone'])
        participantes[codigo_participante]['x_resolucao'].append(
            linha['x_resolution'])
        participantes[codigo_participante]['y_resolucao'].append(
            linha['y_resolution'])
        participantes[codigo_participante]['ano_nascimento'].append(
            linha['Year of birth'])
        participantes[codigo_participante]['ano_teste'].append(
            linha['Year of test'])
        participantes[codigo_participante]['idade_teste'].append(
            linha['Age during the test'])

    codigo_participante_unico = sorted(list(set(base_de_dados['Code'])))

    codigo_participante_especifico = st.selectbox(
        'Select the participant', codigo_participante_unico)

    # Verifique se o código do paciente está presente no dicionário
    if codigo_participante_especifico in participantes:
        dados_do_participante = participantes[codigo_participante_especifico]
        mao_testada = sorted(list(set(dados_do_participante['mao'])))
        mao_participante_especifico = st.radio(
            'Select the hand', mao_testada)

        valores_extraidos = []
        # Usando um loop for para percorrer as listas
        for i, mao in enumerate(dados_do_participante['mao']):
            if mao == mao_participante_especifico:
                valores_extraidos.append(i)

                # Use os índices para obter os valores correspondentes
        t = [dados_do_participante['tempo'][i] for i in valores_extraidos]
        x_dados = [dados_do_participante['x_coord'][i]
                   for i in valores_extraidos]
        y_dados = [dados_do_participante['y_coord'][i]
                   for i in valores_extraidos]
        codigo_participante = dados_do_participante['codigo'][valores_extraidos[0]]
        sexo_participante = dados_do_participante['sexo'][valores_extraidos[0]]
        mao_de_teste = dados_do_participante['mao'][valores_extraidos[0]]
        smartphone_de_teste = dados_do_participante['smartphone'][valores_extraidos[0]]
        x_resolucao_de_teste = dados_do_participante['x_resolucao'][valores_extraidos[0]]
        y_resolucao_de_teste = dados_do_participante['y_resolucao'][valores_extraidos[0]]
        ano_nascimento_participante = dados_do_participante['ano_nascimento'][valores_extraidos[0]]
        ano_teste_participante = dados_do_participante['ano_teste'][valores_extraidos[0]]
        idade_teste_participante = dados_do_participante['idade_teste'][valores_extraidos[0]]

        intervals = np.diff(t)
        x_resolution = x_resolucao_de_teste
        y_resolution = y_resolucao_de_teste
        x = [0, 0, x_resolution, x_resolution, 0]
        y = [0, y_resolution, y_resolution, 0, 0]

        x_squared = [a**2 for a in x_dados]
        y_squared = [b**2 for b in y_dados]
        total_deviation = np.sum(np.sqrt(x_squared+y_squared))

        x_fit, y_fit, ellipse_semimajor_axis, ellipse_semiminor_axis, angle = fit_ellipse(
            x_dados, y_dados)
        ellipse_area = np.pi * ellipse_semiminor_axis * ellipse_semimajor_axis

        col1, col2, col3 = st.columns([0.5, 1.25, 1.2])
        with col1:
            st.subheader('Description of the file')
            code_text = ['Code: ' + str(codigo_participante)]
            st.write(code_text[0])
            sex_text = ['Sex: ' + str(sexo_participante)]
            st.write(sex_text[0])
            hand_text = ['Hand: ' + str(mao_de_teste)]
            st.write(hand_text[0])
            birthyear_text = ['Year of birth: ' +
                              str(ano_nascimento_participante)]
            st.write(birthyear_text[0])
            testyear_test = ['Year of test: ' + str(ano_teste_participante)]
            st.write(testyear_test[0])
            ages_text = ['Age during the test: ' +
                         str(idade_teste_participante)]
            st.write(ages_text[0])
            smartphone_text = ['Smartphone: ' + str(smartphone_de_teste)]
            st.write(smartphone_text[0])
            resolution = [str(x_resolucao_de_teste) +
                          ' x ' + str(y_resolucao_de_teste)]
            resolution_text = ['Smartphone resolution: ' + str(resolution)]
            st.write(resolution_text[0])

        with col2:
            fig, ax = plt.subplots()
            ax.plot(t[0:-1], intervals, 'o-',
                    color=[0, 0.8, 1], markeredgecolor=[0, 0, 1])
            ax.set_ylim(np.min(intervals)*0.5, np.max(intervals)*1.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Interval inter tap (ms)')
            ax.set_title('Interval inter tap as a function of the time')
            st.pyplot(fig)
        with col3:
            fig, ax = plt.subplots()
            ax.plot(x, y, 'k')
            ax.plot(x_dados, y_dados, '+k')
            plt.plot(x_fit, y_fit, 'blue')
            ax.set_xlim(-y_resolution/2, y_resolution)
            ax.set_ylim(0, y_resolution)
            ax.set_title('Taps distribution in the screen')
            ax.axis('off')
            st.pyplot(fig)
        st.subheader('Summary of the performance')
        number_of_taps_text = ['Number of taps: ' + str(len(intervals))]
        st.write(number_of_taps_text[0])
        mean_interval_text = [
            'Mean interval (ms): ' + str(round(np.mean(intervals), 4))]
        st.write(mean_interval_text[0])
        std_interval_text = [
            'Standard deviation of intervals (ms): ' + str(round(np.std(intervals), 4))]
        st.write(std_interval_text[0])
        maximum_interval_text = [
            'Maximum interval (ms): ' + str(round(np.max(intervals), 4))]
        st.write(maximum_interval_text[0])
        minimum_interval_text = [
            'Minimum interval (ms): ' + str(round(np.min(intervals), 4))]
        st.write(minimum_interval_text[0])
        ellipse_area_text = [
            'Ellipse area (px^2): ' + str(round(ellipse_area, 4))]
        st.write(ellipse_area_text[0])
        total_deviation_text = [
            'Total deviation (px): ' + str(round(total_deviation, 4))]
        st.write(total_deviation_text[0])
with tab2:
    todos_participantes = {}
    for indice, linha in base_de_dados.iterrows():
        # Obter o código do paciente
        codigo_tentativa_participante = linha['Code_trial']

        if codigo_tentativa_participante not in todos_participantes:
            # Se não estiver, criar uma entrada para o paciente no dicionário
            todos_participantes[codigo_tentativa_participante] = {
                'codigo': [],
                'codigo_tentativa': [],
                'tempo': [],
                'x_coord': [],
                'y_coord': [],
                'sexo': [],
                'mao': [],
                'smartphone': [],
                'x_resolucao': [],
                'y_resolucao': [],
                'ano_nascimento': [],
                'ano_teste': [],
                'idade_teste': []
            }

        # Adicionar os valores das colunas para o paciente atual

        todos_participantes[codigo_tentativa_participante]['codigo'].append(
            linha['Code'])
        todos_participantes[codigo_tentativa_participante]['codigo_tentativa'].append(
            linha['Code_trial'])
        todos_participantes[codigo_tentativa_participante]['tempo'].append(
            linha['time_of_tap'])
        todos_participantes[codigo_tentativa_participante]['x_coord'].append(
            linha['x_coord'])
        todos_participantes[codigo_tentativa_participante]['y_coord'].append(
            linha['y_coord'])
        todos_participantes[codigo_tentativa_participante]['sexo'].append(
            linha['Sex'])
        todos_participantes[codigo_tentativa_participante]['mao'].append(
            linha['Hand'])
        todos_participantes[codigo_tentativa_participante]['smartphone'].append(
            linha['Smartphone'])
        todos_participantes[codigo_tentativa_participante]['x_resolucao'].append(
            linha['x_resolution'])
        todos_participantes[codigo_tentativa_participante]['y_resolucao'].append(
            linha['y_resolution'])
        todos_participantes[codigo_tentativa_participante]['ano_nascimento'].append(
            linha['Year of birth'])
        todos_participantes[codigo_tentativa_participante]['ano_teste'].append(
            linha['Year of test'])
        todos_participantes[codigo_tentativa_participante]['idade_teste'].append(
            linha['Age during the test'])

    st.title('Database Descriptive Statistics')
    smartphones_disponiveis = []
    for participante in todos_participantes:
        smartphones_disponiveis.append(
            todos_participantes[participante]['smartphone'])
    smartphones_unicos = []
    for i in smartphones_disponiveis:
        smartphones_unicos.extend(sorted(list(set(i))))
    smartphones_unicos_ordenados = sorted(list(set(smartphones_unicos)))

    with st.form("Filtering datasets"):
        condicao_1 = st.selectbox(
            'Select hand condition', ['Both hands', 'Dominant', 'Non-dominant'])
        condicao_2 = st.selectbox(
            'Select sex condition', ['Both sexes', 'Male', 'Female'])
        condicao_3 = st.selectbox(
            'Select age condition', ['All ages', 'Young adult (18-30 yo)', 'Young adult (31-40 yo)','Young adult (41-50 yo)','Young adult (51-60 yo)','Old adult (> 60 yo)'])
        smartphones_unicos_ordenados.insert(0, 'All devices')
        condicao_4 = st.selectbox(
            'Select smartphone device', smartphones_unicos_ordenados)
        filtro = st.form_submit_button("Show statistics")

    if filtro:
        features_num_taps = []
        features_mean = []
        features_std = []
        features_max = []
        features_min = []
        features_area = []
        features_deviation = []
        condicao_1_selected = []
        condicao_2_selected = []
        condicao_3_selected = []
        condicao_4_selected = []

        indice_1 = []
        a = 0
        for participante in todos_participantes:
            if condicao_1 == 'Both hands':
                indice_1.append(a)
            else:
                if todos_participantes[participante]['mao'][0] == condicao_1:
                    indice_1.append(a)
            a = a + 1

        indice_2 = []
        a = 0
        for participante in todos_participantes:
            if condicao_2 == 'Both sexes':
                indice_2.append(a)
            else:
                if todos_participantes[participante]['sexo'][0] == condicao_2:
                    indice_2.append(a)
            a = a + 1

        indice_3 = []
        a = 0
        for participantes in todos_participantes:

            if condicao_3 == "All ages":
                indice_3.append(a)
            else:
                if condicao_3 == 'Young adult (18-30 yo)':
                    if todos_participantes[participantes]['idade_teste'][0] < 31:
                        indice_3.append(a)
                elif condicao_3 == 'Young adult (31-40 yo)':
                    if todos_participantes[participantes]['idade_teste'][0] > 30 and todos_participantes[participantes]['idade_teste'][0] < 41:
                        indice_3.append(a)
                elif condicao_3 == 'Young adult (41-50 yo)':
                    if todos_participantes[participantes]['idade_teste'][0] > 40 and todos_participantes[participantes]['idade_teste'][0] < 51:
                        indice_3.append(a)       
                elif condicao_3 == 'Young adult (51-60 yo)':
                    if todos_participantes[participantes]['idade_teste'][0] > 50 and todos_participantes[participantes]['idade_teste'][0] < 61:
                        indice_3.append(a)        
                elif condicao_3 == 'Old adult (> 60 yo)':
                    if todos_participantes[participantes]['idade_teste'][0] > 60:
                        indice_3.append(a)
                else:
                    print('0')
            a = a + 1

        indice_4 = []
        a = 0
        for participante in todos_participantes:
            if condicao_4 == "All devices":
                indice_4.append(a)
            else:
                if todos_participantes[participante]['smartphone'][0] == condicao_4:
                    indice_4.append(a)
            a = a + 1

        conjunto1 = set(indice_1)
        conjunto2 = set(indice_2)
        conjunto3 = set(indice_3)
        conjunto4 = set(indice_4)

        # Encontrar elementos comuns usando a interseção de conjuntos
        elementos_comuns = conjunto1 & conjunto2 & conjunto3 & conjunto4
        print(elementos_comuns)

        a = 0
        for participante in todos_participantes:
            if a in elementos_comuns:
                t = todos_participantes[participante]['tempo']
                intervalos_base_de_dados = np.diff(t)
                features_num_taps.append(len(t))
                features_mean.append(np.mean(intervalos_base_de_dados))
                features_std.append(np.std(intervalos_base_de_dados))
                features_max.append(np.max(intervalos_base_de_dados))
                features_min.append(np.min(intervalos_base_de_dados))
                x_coordenadas = todos_participantes[participante]['x_coord']
                y_coordenadas = todos_participantes[participante]['y_coord']
                x_fit, y_fit, ellipse_semimajor_axis, ellipse_semiminor_axis, angle = fit_ellipse(
                    x_coordenadas, y_coordenadas)
                ellipse_area = np.pi * ellipse_semiminor_axis * ellipse_semimajor_axis
                features_area.append(ellipse_area)
                x_squared = [a**2 for a in x_coordenadas]
                y_squared = [b**2 for b in y_coordenadas]
                total_deviation = np.sum(np.sqrt(x_squared+y_squared))
                
                features_deviation.append(total_deviation)
                
            a = a + 1

        st.write('Number of datasets: ' + str(len(elementos_comuns)))
        media = np.mean(features_num_taps)
        desvio_padrao = np.std(features_num_taps)
        limite_inferior = media - 3 * desvio_padrao
        limite_superior = media + 3 * desvio_padrao
        dados_filtrados_1 = [
            x for x in features_num_taps if limite_inferior <= x <= limite_superior]
        num_outliers_1 = len(features_num_taps) - len(dados_filtrados_1)
        st.write('Mean number of touches (ms): ' + str(round(np.mean(dados_filtrados_1), 4)
                                                       ) + ' ± ' + str(round(np.std(dados_filtrados_1), 4)))

        media = np.mean(features_mean)
        desvio_padrao = np.std(features_mean)
        limite_inferior = media - 3 * desvio_padrao
        limite_superior = media + 3 * desvio_padrao
        dados_filtrados_2 = [
            x for x in features_mean if limite_inferior <= x <= limite_superior]
        num_outliers_2 = len(features_mean) - len(dados_filtrados_2)
        st.write('Mean interval (ms): ' + str(round(np.mean(dados_filtrados_2), 4)
                                              ) + ' ± ' + str(round(np.std(dados_filtrados_2), 4)))

        media = np.mean(features_std)
        desvio_padrao = np.std(features_std)
        limite_inferior = media - 3 * desvio_padrao
        limite_superior = media + 3 * desvio_padrao
        dados_filtrados_3 = [
            x for x in features_std if limite_inferior <= x <= limite_superior]
        num_outliers_3 = len(features_std) - len(dados_filtrados_3)
        st.write('Mean standard deviation (ms): ' + str(round(np.mean(dados_filtrados_3), 4)
                                                        ) + ' ± ' + str(round(np.std(dados_filtrados_3), 4)))

        media = np.mean(features_max)
        desvio_padrao = np.std(features_max)
        limite_inferior = media - 3 * desvio_padrao
        limite_superior = media + 3 * desvio_padrao
        dados_filtrados_4 = [
            x for x in features_max if limite_inferior <= x <= limite_superior]
        num_outliers_4 = len(features_max) - len(dados_filtrados_4)
        st.write('Maximum intervals (ms): ' + str(round(np.mean(dados_filtrados_4), 4)
                                                  ) + ' ± ' + str(round(np.std(dados_filtrados_4), 4)))

        media = np.mean(features_min)
        desvio_padrao = np.std(features_min)
        limite_inferior = media - 3 * desvio_padrao
        limite_superior = media + 3 * desvio_padrao
        dados_filtrados_5 = [
            x for x in features_min if limite_inferior <= x <= limite_superior]
        num_outliers_5 = len(features_min) - len(dados_filtrados_5)
        st.write('Minimum intervals (ms): ' + str(round(np.mean(dados_filtrados_5), 4)
                                                  ) + ' ± ' + str(round(np.std(dados_filtrados_5), 4)))

        media = np.mean(features_area)
        desvio_padrao = np.std(features_area)
        limite_inferior = media - 3 * desvio_padrao
        limite_superior = media + 3 * desvio_padrao
        dados_filtrados_6 = [
            x for x in features_area if limite_inferior <= x <= limite_superior]
        num_outliers_6 = len(features_area) - len(dados_filtrados_6)
        st.write('Area (px^2): ' + str(round(np.mean(dados_filtrados_6), 4)
                                       ) + ' ± ' + str(round(np.std(dados_filtrados_6), 4)))

        media = np.mean(features_deviation)
        desvio_padrao = np.std(features_deviation)
        limite_inferior = media - 3 * desvio_padrao
        limite_superior = media + 3 * desvio_padrao
        dados_filtrados_7 = [
            x for x in features_deviation if limite_inferior <= x <= limite_superior]
        num_outliers_7 = len(features_deviation) - len(dados_filtrados_7)
        st.write('Total deviation (px): ' + str(round(np.mean(dados_filtrados_7), 4)
                                       ) + ' ± ' + str(round(np.std(dados_filtrados_7), 4)))

        # Convertendo a lista em um DataFrame
  
        df = pd.DataFrame({
            "Number of taps": features_num_taps,
            "Mean interval (ms)": features_mean,
            "Standard deviation (ms)": features_std,
            "Maximum interval (ms)": features_max,
            "Minimum interval": features_min,
            "Area": features_area,
            "Deviation": features_deviation,
        })

        # Convertendo o DataFrame para CSV
        csv = df.to_csv(index=False)

        # Botão de download para a lista de dados numéricos
        st.download_button(
            label="Download filtered data",
            data=csv,
            file_name='filtered database.csv',
            mime='text/csv')
        col5, col6, col7 = st.columns([1, 1, 1])
        with col5:
            fig, ax = plt.subplots()
            ax.hist(dados_filtrados_1, color='skyblue', edgecolor='blue')
            ax.set_xlabel('Number of taps')
            ax.set_ylabel('Number of participants')
            ax.set_title('Number of taps')
            st.pyplot(fig)
            st.write(f'Number of outliers = {num_outliers_1}')

            fig, ax = plt.subplots()
            ax.hist(dados_filtrados_2, color='skyblue', edgecolor='blue')
            ax.set_xlabel('Interval inter tap (ms)')
            ax.set_ylabel('Number of participants')
            ax.set_title('Mean interval inter tap')
            st.pyplot(fig)
            st.write(f'Number of outliers = {num_outliers_2}')

        with col6:
            fig, ax = plt.subplots()
            ax.hist(dados_filtrados_3, color='skyblue', edgecolor='blue')
            ax.set_xlabel('Maximum interval inter tap (ms)')
            ax.set_ylabel('Number of participants')
            ax.set_title('Maximum interval inter tap')
            st.pyplot(fig)
            st.write(f'Number of outliers = {num_outliers_3}')

            fig, ax = plt.subplots()
            ax.hist(dados_filtrados_4, color='skyblue', edgecolor='blue')
            ax.set_xlabel('Minimum interval inter tap (ms)')
            ax.set_ylabel('Number of participants')
            ax.set_title('Minimum interval inter tap')
            st.pyplot(fig)
            st.write(f'Number of outliers = {num_outliers_4}')

        with col7:
            fig, ax = plt.subplots()
            ax.hist(dados_filtrados_5, color='skyblue', edgecolor='blue')
            ax.set_xlabel('Standard deviation of intervals inter tap (ms)')
            ax.set_ylabel('Number of participants')
            ax.set_title('Standard deviation of intervals inter tap')
            st.pyplot(fig)
            st.write(f'Number of outliers = {num_outliers_5}')

            fig, ax = plt.subplots()
            ax.hist(dados_filtrados_6, color='skyblue', edgecolor='blue')
            ax.set_xlabel('Ellipse area (px^2)')
            ax.set_ylabel('Number of participants')
            ax.set_title('Ellipse area')
            st.pyplot(fig)
            st.write(f'Number of outliers = {num_outliers_6}')
