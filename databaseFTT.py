import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
import matplotlib.patches as patches


# Fit an ellipse to cover 95% of the points

def fit_ellipse_v2(x, y):
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
st.write('The Brazilian Database for the Smartphone-Based Finger Tapping Test is a collaborative research effort involving investigators from the Federal University of Pará, Federal University of West Pará, and Federal University of Maranhão. The primary objective of this database is to serve as a healthy reference for the Brazilian population. None participant has declared the identity, but other information such as sex and age are available. Currently, the database exclusively includes data from right-handers. However, future releases are anticipated to incorporate information from left-handers as well. The Finger Tapping Test performances are recorded at the center of the smartphone screen. For any inquiries or requests related to the database, please feel free to contact us via email at givagosouza@ufpa.br. Last release at 7th January, 2024.')
st.write('Instruction of use. This web application has two tools to explore the database: (i) one to evaluate each individual dataset and (ii) another to evaluate the whole database or in filtered conditions of sample features.')

st.title('Individual data')
col1, col2, col3, col4 = st.columns([0.5, 0.5, 0.7, 1])

caminho_time_csv = 'time_of_tap.csv'
time = pd.read_csv(caminho_time_csv)

caminho_x_coordinates_csv = 'x_coordinates_of_tap.csv'
x_coordinates = pd.read_csv(caminho_x_coordinates_csv)

caminho_y_coordinates_csv = 'y_coordinates_of_tap.csv'
y_coordinates = pd.read_csv(caminho_y_coordinates_csv)

caminho_info_csv = 'info_participants.csv'
info = pd.read_csv(caminho_info_csv)
single_codes = sorted(list(set(info['Code'])))
single_smartphones = sorted(list(set(info['Smartphone'])))

with col1:
    st.subheader('Select an individual file')
    participant = st.selectbox('Select the participant', single_codes)
    position = []
    for i, code in enumerate(info['Code']):
        if code == participant:
            position.append(i)
    chosen_hand = st.radio('Select the hand', info['Hand'][position])
    for i, hand in enumerate(info['Hand'][position]):
        if hand == chosen_hand:
            choice = position[i]
            break
with col2:
    st.subheader('Description of the file')
    st.write(f'Code: {info['Code'][choice]}')
    st.write(f'Sex: {info['Sex'][choice]}')
    st.write(f'Hand: {info['Hand'][choice]}')
    st.write(f'Year of birth: {info['Year of birth'][choice]}')
    st.write(f'Year of test: {info['Year of test'][choice]}')
    ages = info['Year of test'][choice]-info['Year of birth'][choice]
    st.write(f'Age during the test: {ages}')
    st.write(f'Smartphone: {info['Smartphone'][choice]}')
    resolution = [str(info['x_resolution'][choice]) +
                  ' x ' + str(info['y_resolution'][choice])]
    st.write(f'Smartphone resolution: {resolution}')

    with col3:
        intervals = np.diff(time.iloc[:, choice].dropna())
        time_trial = time.iloc[1:-1, choice].dropna()
        x_resolution = info['x_resolution'][choice]
        y_resolution = info['y_resolution'][choice]
        x = [0, 0, x_resolution, x_resolution, 0]
        y = [0, y_resolution, y_resolution, 0, 0]
        x_data = x_coordinates.iloc[1:-1, choice].dropna()
        y_data = y_coordinates.iloc[1:-1, choice].dropna()

        x_fit, y_fit, ellipse_semimajor_axis, ellipse_semiminor_axis, angle = fit_ellipse_v2(
            x_data, y_data)
        ellipse_area = np.pi * ellipse_semiminor_axis * ellipse_semimajor_axis

        st.subheader('Summary of the performance')
        st.write(f'Number of taps: {len(intervals)}')
        st.write(f'Mean interval (ms): {round(np.mean(intervals), 4)}')
        st.write(f'Standard deviation of intervals (ms): {
            round(np.std(intervals), 4)}')
        st.write(f'Maximum interval (ms): {
            round(np.max(intervals), 4)}')
        st.write(f'Minimum interval (ms): {
            round(np.min(intervals), 4)}')
        st.write(f'Amplitude (ms): {
            round(np.max(intervals) - np.min(intervals), 4)}')
        st.write(f'Ellipse area (px^2): {
            round(ellipse_area, 4)}')

    with col4:
        fig, ax = plt.subplots()
        ax.plot(time_trial/1000, intervals, 'o-',
                color=[0, 0.8, 1], markeredgecolor=[0, 0, 1])
        ax.set_ylim(np.min(intervals)*0.5, np.max(intervals)*1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Interval inter tap (ms)')
        ax.set_title('Interval inter tap as a function of the time')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(x, y, 'k')
        ax.plot(x_data, y_data, '+k')
        plt.plot(x_fit, y_fit, 'blue')
        ax.set_xlim(-y_resolution/2, y_resolution)
        ax.set_ylim(0, y_resolution)
        ax.set_title('Taps distribution in the screen')
        ax.axis('off')
        st.pyplot(fig)
st.title('Database Descriptive Statistics')
if st.checkbox('Proceed database statisticcs') == True:
    with st.form("Filtering datasets"):
        condition_1 = st.selectbox(
            'Select hand condition', ['Both hands', 'Dominant', 'Non-dominant'])
        condition_2 = st.selectbox(
            'Select sex condition', ['Both sexes', 'Male', 'Female'])
        condition_3 = st.selectbox(
            'Select age condition', ['All ages', 'Young adult (18-60 yo)', 'Old adult (> 60 yo)'])
        single_smartphones.insert(0, 'All devices')
        condition_4 = st.selectbox(
            'Select smartphone device', single_smartphones)

        submitted = st.form_submit_button("Show statistics")
    if submitted:
        features_num_taps = []
        features_mean = []
        features_std = []
        features_max = []
        features_min = []
        features_area = []
        condition_1_selected = []
        condition_2_selected = []
        condition_3_selected = []
        condition_4_selected = []
        for i, hand in enumerate(info['Hand']):
            if condition_1 == 'Both hands':
                condition_1_selected.append(i)
            else:
                if hand == condition_1:
                    condition_1_selected.append(i)

        for i, sex in enumerate(info['Sex']):
            if condition_2 == 'Both sexes':
                condition_2_selected.append(i)
            else:
                if sex == condition_2:
                    condition_2_selected.append(i)

        for i, age in enumerate(info['Age during the test']):
            if condition_3 == 'All ages':
                condition_3_selected.append(i)
            else:
                if condition_3 == 'Young adult (18-60 yo)':
                    if age < 60:
                        condition_3_selected.append(i)
                elif condition_3 == 'Old adult (> 60 yo)':
                    if age >= 60:
                        condition_3_selected.append(i)

        for i, smartphone in enumerate(info['Smartphone']):
            if condition_4 == 'All devices':
                condition_4_selected.append(i)
            else:
                if smartphone == condition_4:
                    condition_4_selected.append(i)
        # Converter as listas em conjuntos
        conjunto1 = set(condition_1_selected)
        conjunto2 = set(condition_2_selected)
        conjunto3 = set(condition_3_selected)
        conjunto4 = set(condition_4_selected)

        # Encontrar elementos comuns usando a interseção de conjuntos
        elementos_comuns = conjunto1 & conjunto2 & conjunto3 & conjunto4
        for i in elementos_comuns:
            intervals = np.diff(time.iloc[:, i].dropna())
            features_num_taps.append(len(intervals)+1)
            features_mean.append(np.mean(intervals))
            features_std.append(np.std(intervals))
            features_max.append(np.max(intervals))
            features_min.append(np.min(intervals))
            x_data = x_coordinates.iloc[1:-1, i].dropna()
            y_data = y_coordinates.iloc[1:-1, i].dropna()
            x_fit, y_fit, ellipse_semimajor_axis, ellipse_semiminor_axis, angle = fit_ellipse_v2(
                x_data, y_data)
            ellipse_area = np.pi * ellipse_semiminor_axis * ellipse_semimajor_axis
            features_area.append(ellipse_area)

        st.write('Number of datasets: ' + str(len(features_area)))

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
