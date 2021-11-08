# APP SENTINEL HUB CALCULO DE INDICES
#------------------------------------
# Librerias
import streamlit as st 
import rasterio     # import the main rasterio function
from rasterio.plot import show, show_hist # some specific rasterio functions we'll need
import matplotlib   # matplotlib is the primary python plotting and viz library
import matplotlib.pyplot as plt
from PIL import Image
# this bit of magic allows matplotlib to plot inline ina  jupyter notebook
import folium       # folium is an interactive mapping library
import numpy as np
import pandas as pd 
import plotly.express as px
import math
import os
import datetime
#Librerias Sentinel
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, DownloadRequest
from utils import plot_image
from sentinelhub import SHConfig
#librerias creadas
import concesiones 
import evaluacion_suelo
import calidad_agua

#Deteccion de cambios
import imageio #Lectura y escritura de imagenes
from sklearn.decomposition import PCA #Principal Component Analysis Reduccion de dimensionalidad lineal 
import sklearn    
from sklearn.cluster import KMeans
from collections import Counter
#import cv2
# Inicio de sesion SentinelHub

st.set_option('deprecation.showPyplotGlobalUse', False)
CLIENT_ID ='3fc9aefe-7d67-400b-935b-96c7d6406921'
CLIENT_SECRET = 'Rga7fW/I].<xXH:Y<^zD8Nyypvwp&LxHO<B1*rR7'
config = SHConfig()
if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id= CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET    
if config.sh_client_id == '' or  config.sh_client_secret == '':
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

#Inicio de la Aplicación
#-----------------------
st.title('Sistema de Monitoreo ANZU')

video_file = open('monitoring1.mp4', 'rb')
video_bytes = video_file.read()
st.sidebar.video(video_bytes ,format="video/mp4", start_time=0)
#st.sidebar.write('Sentinel')

st.sidebar.title('Menu')
option = st.sidebar.radio('Seleccione una opcion', ['💧 Concesiones', '🛰 Visualización satelital', '⛰ Evaluación de suelo', '🧪 Evaluación de la calidad el agua', '📊 Análisis temporal puntos de agua','🔎 Detección de cambios'])

if option == '💧 Concesiones':

	st.info('Concesiones Mineras')

	columna1, columna2 = st.columns(2)
	with columna1:
		select_concesion = st.selectbox('🌐 Seleccione',['⚒ Concesiones','Zona 1 red','Zona 2 yellow','Zona 3 cian','Anzu Norte', 'Berta 1', 'Confluencia', 'Cristobal','Genial', 'Vista Anzu'])

	with columna2:
		if select_concesion == '⚒ Concesiones':
			figura_conciones = concesiones.imagen_concesiones()
			st.image(figura_conciones)

		else:
			year_zona = st.slider('Seleccione el año',2017,2021,2017,step=1)
			zona = concesiones.seleccion_zona(select_concesion, year_zona)
			st.image(zona)

if option == '🛰 Visualización satelital':

	st.warning('Coordenadas WGS84 para Sentinel: http://bboxfinder.com/#0.000000,0.000000,0.000000,0.000000')


	st.warning('Ingrese coordenadas WGS84de para el area a monitorear')


	col1, col2,col3 = st.columns(3)

	with col1:

		latitud_inferior = st.number_input('latitud esquina inferior izquierda',min_value=None, max_value=None, value=-77.920532,  step=0.01 )
		longitud_inferior = st.number_input('longitud esquina inferior izquierda',min_value=None, max_value=None, value=-1.254012,  step=0.01 )
		latitud_superior = st.number_input('latitud esquina superior derecha',min_value=None, max_value=None, value=-77.752132,  step=0.01 )
		longitud_superior = st.number_input('longitud esquina superior derecha',min_value=None, max_value=None, value=-1.031028,  step=0.01 )
		resolucion = st.number_input('Ingrese la resolucion',min_value=0, max_value=None, value=10,  step=10 )
		
	with col2:

			
		with st.expander('Ingreso de fechas'):
			fecha_inicio = st.text_input('Ingrese fecha inicio (yyyy-mm-dd)',value ='2020-01-01')
			fecha_fin = st.text_input('Ingrese fecha fin (yyyy-mm-dd)',value ='2020-12-30')
			

		betsiboka_coords_wgs84 = [latitud_inferior, longitud_inferior, latitud_superior,longitud_superior]
		resolution = int(resolucion)
		betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
		betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)

		print(f'Image shape at {resolution} m resolution: {betsiboka_size} pixels')

		evalscript_true_color = """
		    //VERSION=3

		    function setup() {
		        return {
		            input: [{
		                bands: ["B02", "B03", "B04"]
		            }],
		            output: {
		                bands: 3
		            }
		        };
		    }

		    function evaluatePixel(sample) {
		        return [sample.B04, sample.B03, sample.B02];
		    }
		"""

		request_true_color = SentinelHubRequest(
		    evalscript=evalscript_true_color,
		    input_data=[
		        SentinelHubRequest.input_data(
		            data_collection=DataCollection.SENTINEL2_L1C,
		            time_interval=(fecha_inicio, fecha_fin),
		            mosaicking_order='leastCC'
		        )
		    ],
		    responses=[
		        SentinelHubRequest.output_response('default', MimeType.PNG)
		    ],
		    bbox=betsiboka_bbox,
		    size=betsiboka_size,
		    config=config
		)
		true_color_imgs = request_true_color.get_data()
		print(f'Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}.')
		print(f'Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}')
		image = true_color_imgs[0]
		print(f'Image type: {image.dtype}')

		# plot function
		# factor 1/255 to scale between 0-1
		# factor 3.5 to increase brightness
		a = plot_image(image, factor=3.5/255, clip_range=(0,1))
		st.write('Imagen en color verdadero')
		st.pyplot(a)


	with col3:

		with st.expander('Selección de bandas'):



			select_banda = st.selectbox('Seleccione la banda de interes', ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'])

		st.write('Valores de reflectancia')

		if select_banda == 'B01':

			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B01"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }

			    function evaluatePixel(sample) {
			        return [sample.B01
			                ];
			    }
			"""
		if select_banda == 'B02':
				
			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B02"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }

			    function evaluatePixel(sample) {
			        return [sample.B02
			                ];
			    }
			"""
		if select_banda == 'B03':
			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B03"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }

			    function evaluatePixel(sample) {
			        return [sample.B03
			                ];
			    }
			"""
		if select_banda == 'B04':
			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B04"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }

			    function evaluatePixel(sample) {
			        return [sample.B04


			                ];
			    }
			"""
		if select_banda == 'B05':
			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B05"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }
				    function evaluatePixel(sample) {
			        return [sample.B05
				                ];
			    }
			"""	
		if select_banda == 'B06':
			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B06"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }

			    function evaluatePixel(sample) {
			        return [sample.B06
			                ];
			    }
			"""
		if select_banda == 'B07':
			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B07"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }

			    function evaluatePixel(sample) {
			        return [sample.B07
			                ];
			    }
			"""
		if select_banda == 'B08':
			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B08"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }

			    function evaluatePixel(sample) {
			        return [sample.B08
			                ];
			    }
			"""
		if select_banda == 'B8A':
			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B8A"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }

			    function evaluatePixel(sample) {
			        return [sample.B8A
			                ];
			    }
			"""
		if select_banda == 'B09':
			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B09"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }

			    function evaluatePixel(sample) {
			        return [sample.B09
			                ];
			    }
			"""
		if select_banda == 'B10':
			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B10"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }

			    function evaluatePixel(sample) {
			        return [sample.B10
			                ];
			    }
			"""
		if select_banda == 'B11':
			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B11"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }

			    function evaluatePixel(sample) {
			        return [sample.B11
			                ];
			    }
			"""
		if select_banda == 'B12':
			evalscript_all_bands = """
			    //VERSION=3
			    function setup() {
			        return {
			            input: [{
			                bands: ["B12"],
			                units: "REFLECTANCE"
			            }],
			            output: {
			                bands: 1,
			                sampleType: "FLOAT32"
			            }
			        };
			    }

			    function evaluatePixel(sample) {
			        return [sample.B12
			                ];
			    }
			"""


		request_all_bands = SentinelHubRequest(
		    evalscript=evalscript_all_bands,
		    input_data=[
		        SentinelHubRequest.input_data(
		            data_collection=DataCollection.SENTINEL2_L1C,
		            time_interval=(fecha_inicio, fecha_fin),
		            mosaicking_order='leastCC'
		    )],
		    responses=[
		        SentinelHubRequest.output_response('default', MimeType.TIFF)
		    ],
		    bbox=betsiboka_bbox,
		    size=betsiboka_size,
		    config=config
		)

		all_bands_response = request_all_bands.get_data()


		b = plot_image(all_bands_response[0])

		st.pyplot(b)

if option == '⛰ Evaluación de suelo':

	zona_opcion = st.selectbox('🌐 Seleccione una zona',['Zona 1 red','Zona 2 yellow','Zona 3 cian','Anzu Norte', 'Berta 1', 'Confluencia', 'Cristobal','Genial', 'Vista Anzu'])
	year_option = st.slider('Elija un año', 2017,2021,2017)

	s2_bands = evaluacion_suelo.bandas(zona_opcion, year_option)

	arrs = []
	for band in s2_bands:
	    with rasterio.open(band) as f:
	        arrs.append(f.read(1))

	sentinel_img = np.array(arrs, dtype=arrs[0].dtype)

	clipped_img = sentinel_img

	np.seterr(divide='ignore', invalid='ignore')

	band02 = clipped_img[0] 
	band03 = clipped_img[1] 
	band04 = clipped_img[2] 
	band08 = clipped_img[3] 
	band11 = clipped_img[4] 
	band12 = clipped_img[5]

	with st.expander('Análisis NDVI y BSI'):
		col5, col6 =st.columns(2)

		with col5:
			ndvi_index = (band08.astype(float)-band04.astype(float) )/ (band08.astype(float)+band04.astype(float))
			fig_index, ax = plt.subplots()
			ax.imshow(ndvi_index, cmap="RdYlGn")

			st.pyplot(fig_index)

			st.write('\nMax NDVI: {m}'.format(m=ndvi_index.max()))
			st.write('Mean NDVI: {m}'.format(m=ndvi_index.mean()))
			st.write('Median NDVI: {m}'.format(m=np.median(ndvi_index)))
			st.write('Min NDVI: {m}'.format(m=ndvi_index.min()))

		with col6:

			bsi_index = ((band12.astype(float)+band04.astype(float))-(band08.astype(float)+band02.astype(float)))/((band12.astype(float)+band04.astype(float))+(band08.astype(float)+band02.astype(float)))
			fig_index2, ax = plt.subplots()
			ax.imshow(bsi_index, cmap= "YlOrRd")
			st.pyplot(fig_index2)

			st.write('\nMax BSI: {m}'.format(m=bsi_index.max()))
			st.write('Mean BSI: {m}'.format(m=bsi_index.mean()))
			st.write('Median BSI: {m}'.format(m=np.median(bsi_index)))
			st.write('Min BSI: {m}'.format(m=bsi_index.min()))

	with st.expander('Ubicación NDVI y BSI'):

		option_index_suelo = st.radio('Seleccione indice', ['NDVI','BSI'])

		if option_index_suelo =='NDVI':

			fig_ndvi = px.imshow(ndvi_index, title='NDVI')
			st.write(fig_ndvi)

		if option_index_suelo =='BSI':
			fig_bsi = px.imshow(bsi_index,title='BSI')
			st.write(fig_bsi)

if option =='🧪 Evaluación de la calidad el agua':

	with st.expander('Análisis anual de Índices'):

		zona_opcion_agua = st.selectbox('🌐 Seleccione una zona',['Zona 1 red','Zona 2 yellow','Zona 3 cian','Anzu Norte', 'Berta 1', 'Confluencia', 'Cristobal','Genial', 'Vista Anzu'])
		year_option_agua = st.slider('Elija un año', 2017,2021,2017)
		s2_bands = evaluacion_suelo.bandas(zona_opcion_agua, year_option_agua)
			
		st.header('Ínidices de calidad de agua')

		arrs = []
		for band in s2_bands:
		    with rasterio.open(band) as f:
		        arrs.append(f.read(1))
		sentinel_img = np.array(arrs, dtype=arrs[0].dtype)
		clipped_img = sentinel_img[:, 0:1080:, 0:1080]
		np.seterr(divide='ignore', invalid='ignore')
		band02 = clipped_img[0] 
		band03 = clipped_img[1] 
		band04 = clipped_img[2] 
		band08 = clipped_img[3] 
		band11 = clipped_img[4] 
		band12 = clipped_img[5]

		NH3_N_map = math.e**(-8.8129 - 1.7044*np.log(band02.astype(float))  + 1.7620*np.log(band03.astype(float)) -1.8647*np.log(band04.astype(float))- 1.4377*np.log(band08.astype(float)))
		COD_map = 2.76 -(17.27*band02.astype(float))+(72.15*band03.astype(float))-(12.11*band04.astype(float))
		BOD_map = 1.79 -(0.789*band02.astype(float))+(52.36*band03.astype(float))-(3.28*band04.astype(float))
		TOC_map = 6.41 -(85.29*band02.astype(float))+(2.05*band03.astype(float))-(29.96*band04.astype(float))

		cola, colb ,colc, cold=st.columns(4)
				
		with cola:
				
			fig_index_agua1, ax = plt.subplots()

			ax.imshow(NH3_N_map, cmap="RdYlGn" )
			ax.set_title('NH3')
			st.pyplot(fig_index_agua1)

			st.write('\nMax NH3: {m}'.format(m=round(NH3_N_map.max(),2)))
			st.write('Mean NH3: {m}'.format(m=round(NH3_N_map.mean(),2)))
			st.write('Median NH3: {m}'.format(m=round(np.median(NH3_N_map),2)))
			st.write('Min NH3: {m}'.format(m=round(NH3_N_map.min(),2)))
		with colb:
			fig_index_agua2, ax = plt.subplots()
			ax.imshow(COD_map, cmap="RdYlGn")
			ax.set_title('COD')
			st.pyplot(fig_index_agua2)
			st.write('\nMax COD: {m}'.format(m=round(COD_map.max(),2)))
			st.write('Mean COD: {m}'.format(m=round(COD_map.mean(),2)))
			st.write('Median COD: {m}'.format(m=round(np.median(COD_map),2)))
			st.write('Min COD: {m}'.format(m=round(COD_map.min(),2)))
		with colc:
			fig_index_agua3, ax = plt.subplots()
			ax.imshow(BOD_map, cmap="RdYlGn")
			ax.set_title('BOD')
			st.pyplot(fig_index_agua3)
			st.write('\nMax BOD: {m}'.format(m=round(BOD_map.max(),2)))
			st.write('Mean BOD: {m}'.format(m=round(BOD_map.mean(),2)))
			st.write('Median BOD: {m}'.format(m=round(np.median(BOD_map),2)))
			st.write('Min BOD: {m}'.format(m=round(BOD_map.min(),2)))
		with cold:
			fig_index_agua4, ax = plt.subplots()
			ax.imshow(TOC_map, cmap="RdYlGn")
			ax.set_title('TOC')
			st.pyplot(fig_index_agua4)
			st.write('\nMax TOC: {m}'.format(m=round(TOC_map.max(),2)))
			st.write('Mean TOC: {m}'.format(m=round(TOC_map.mean(),2)))
			st.write('Median TOC: {m}'.format(m=round(np.median(TOC_map),2)))
			st.write('Min TOC: {m}'.format(m=round(TOC_map.min(),2)))


	with st.expander('Análisis por ubicaciÓn'):

		indice_plot =st.selectbox('Seleccione el indice a analizar',['NH3','COD','BOD','TOC'])

		if indice_plot == 'NH3':

			fig = px.imshow(NH3_N_map, title='NH3')
			st.write(fig)
		elif indice_plot == 'COD':

			fig = px.imshow(COD_map, title='COD')
			st.write(fig)
		elif indice_plot == 'BOD':

			fig = px.imshow(BOD_map, title='BOD')
			st.write(fig)
		elif indice_plot == 'TOC':

			fig = px.imshow(TOC_map, title='TOC')
			st.write(fig)

if option == '📊 Análisis temporal puntos de agua':
	st.info('Analisis en diferentes periodos de tiempos')

	columna1, columna2 = st.columns(2)
	with columna1:
		select_zona_agua = st.selectbox('🌐 Seleccione',['Zona 1 red','Zona 2 yellow','Zona 3 cian','Anzu Norte', 'Berta 1', 'Confluencia', 'Cristobal','Genial', 'Vista Anzu'])

		year_zona = st.slider('Seleccione el año',2017,2021,2017,step=1)


	with columna2:
		zona = calidad_agua.seleccion_zona(select_zona_agua, year_zona)
		st.pyplot(zona)

	with st.expander('VisualizaciÓn map'):

		s2_bands = evaluacion_suelo.bandas(select_zona_agua, year_zona)

		arrs = []
		for band in s2_bands:
		    with rasterio.open(band) as f:
		        arrs.append(f.read(1))
		sentinel_img = np.array(arrs, dtype=arrs[0].dtype)
		clipped_img = sentinel_img[:, 0:1080:, 0:1080]
		np.seterr(divide='ignore', invalid='ignore')
		band02 = clipped_img[0] 
		band03 = clipped_img[1] 
		band04 = clipped_img[2] 
		band08 = clipped_img[3] 
		band11 = clipped_img[4] 
		band12 = clipped_img[5]


		NH3_N_map = math.e**(-8.8129 - 1.7044*np.log(band02.astype(float))  + 1.7620*np.log(band03.astype(float)) -1.8647*np.log(band04.astype(float))- 1.4377*np.log(band08.astype(float)))
		COD_map = 2.76 -(17.27*band02.astype(float))+(72.15*band03.astype(float))-(12.11*band04.astype(float))
		BOD_map = 1.79 -(0.789*band02.astype(float))+(52.36*band03.astype(float))-(3.28*band04.astype(float))
		TOC_map = 6.41 -(85.29*band02.astype(float))+(2.05*band03.astype(float))-(29.96*band04.astype(float))

		cola, colb ,colc, cold=st.columns(4)
				
		with cola:
				
			fig_index_agua1, ax = plt.subplots()

			ax.imshow(NH3_N_map, cmap="RdYlGn" )
			ax.set_title('NH3')
			st.pyplot(fig_index_agua1)
		with colb:
			fig_index_agua2, ax = plt.subplots()
			ax.imshow(COD_map, cmap="RdYlGn")
			ax.set_title('COD')
			st.pyplot(fig_index_agua2)
		with colc:
			fig_index_agua3, ax = plt.subplots()
			ax.imshow(BOD_map, cmap="RdYlGn")
			ax.set_title('BOD')
			st.pyplot(fig_index_agua3)
		with cold:
			fig_index_agua4, ax = plt.subplots()
			ax.imshow(TOC_map, cmap="RdYlGn")
			ax.set_title('TOC')
			st.pyplot(fig_index_agua4)

		s2_bands2017 = evaluacion_suelo.bandas(select_zona_agua, 2017)

		arrs = []
		for band in s2_bands2017:
		    with rasterio.open(band) as f:
		        arrs.append(f.read(1))
		sentinel_img = np.array(arrs, dtype=arrs[0].dtype)
		clipped_img = sentinel_img[:, 0:1080:, 0:1080]
		np.seterr(divide='ignore', invalid='ignore')
		band02 = clipped_img[0] 
		band03 = clipped_img[1] 
		band04 = clipped_img[2] 
		band08 = clipped_img[3] 
		band11 = clipped_img[4] 
		band12 = clipped_img[5]

		df_b02 = pd.DataFrame(band02)  #azul   
		df_b03 = pd.DataFrame(band03) # verde
		df_b04 = pd.DataFrame(band04) # roja
		df_b08 = pd.DataFrame(band08) # NIR
		df_b11 = pd.DataFrame(band11)
		df_b12 = pd.DataFrame(band12) 



		dato_b02 = df_b02[210][206]
		dato_b03 = df_b03[210][206]
		dato_b04 = df_b04[210][206]
		dato_b08 = df_b08[210][206]
		dato_b11 = df_b11[210][206]


		NH3_N = math.e**(-8.8129 - 1.7044*math.log(dato_b02)  + 1.7620*math.log(dato_b03) -1.8647*math.log(dato_b04)- 1.4377*math.log(dato_b08))
		BOD = 1.79 -(0.789*dato_b02)+(52.36*dato_b03)-(3.28*dato_b04)
		COD = 2.76 -(17.27*dato_b02)+(72.15*dato_b03)-(12.11*dato_b04)
		TOC = 6.41 -(85.29*dato_b02)+(2.05*dato_b03)-(29.96*dato_b04)

		year_2017 = [2017,2017,2017,2017]
		l_indice = ['NH3','BOD','COD','TOC']
		data_2017 = [NH3_N,BOD,COD,TOC]

		df2017={'Año':year_2017,
				'Indice':l_indice,
				'Valor':data_2017}

		df_2017 = pd.DataFrame(df2017)

		#st.write(df_2017)

		s2_bands2018 = evaluacion_suelo.bandas(select_zona_agua, 2018)

		arrs = []
		for band in s2_bands2018:
		    with rasterio.open(band) as f:
		        arrs.append(f.read(1))
		sentinel_img = np.array(arrs, dtype=arrs[0].dtype)
		clipped_img = sentinel_img[:, 0:1080:, 0:1080]
		np.seterr(divide='ignore', invalid='ignore')
		band02 = clipped_img[0] 
		band03 = clipped_img[1] 
		band04 = clipped_img[2] 
		band08 = clipped_img[3] 
		band11 = clipped_img[4] 
		band12 = clipped_img[5]

		df_b02 = pd.DataFrame(band02)  #azul   
		df_b03 = pd.DataFrame(band03) # verde
		df_b04 = pd.DataFrame(band04) # roja
		df_b08 = pd.DataFrame(band08) # NIR
		df_b11 = pd.DataFrame(band11)
		df_b12 = pd.DataFrame(band12) 



		dato_b02 = df_b02[210][206]
		dato_b03 = df_b03[210][206]
		dato_b04 = df_b04[210][206]
		dato_b08 = df_b08[210][206]
		dato_b11 = df_b11[210][206]


		NH3_N = math.e**(-8.8129 - 1.7044*math.log(dato_b02)  + 1.7620*math.log(dato_b03) -1.8647*math.log(dato_b04)- 1.4377*math.log(dato_b08))
		BOD = 1.79 -(0.789*dato_b02)+(52.36*dato_b03)-(3.28*dato_b04)
		COD = 2.76 -(17.27*dato_b02)+(72.15*dato_b03)-(12.11*dato_b04)
		TOC = 6.41 -(85.29*dato_b02)+(2.05*dato_b03)-(29.96*dato_b04)

		year_2018 = [2018,2018,2018,2018]
		l_indice = ['NH3','BOD','COD','TOC']
		data_2018 = [NH3_N,BOD,COD,TOC]

		df2018={'Año':year_2018,
				'Indice':l_indice,
				'Valor':data_2018}

		df_2018 = pd.DataFrame(df2018)

		#st.write(df_2018)

		s2_bands2019 = evaluacion_suelo.bandas(select_zona_agua, 2019)

		arrs = []
		for band in s2_bands2019:
		    with rasterio.open(band) as f:
		        arrs.append(f.read(1))
		sentinel_img = np.array(arrs, dtype=arrs[0].dtype)
		clipped_img = sentinel_img[:, 0:1080:, 0:1080]
		np.seterr(divide='ignore', invalid='ignore')
		band02 = clipped_img[0] 
		band03 = clipped_img[1] 
		band04 = clipped_img[2] 
		band08 = clipped_img[3] 
		band11 = clipped_img[4] 
		band12 = clipped_img[5]

		df_b02 = pd.DataFrame(band02)  #azul   
		df_b03 = pd.DataFrame(band03) # verde
		df_b04 = pd.DataFrame(band04) # roja
		df_b08 = pd.DataFrame(band08) # NIR
		df_b11 = pd.DataFrame(band11)
		df_b12 = pd.DataFrame(band12) 



		dato_b02 = df_b02[210][206]
		dato_b03 = df_b03[210][206]
		dato_b04 = df_b04[210][206]
		dato_b08 = df_b08[210][206]
		dato_b11 = df_b11[210][206]


		NH3_N = math.e**(-8.8129 - 1.7044*math.log(dato_b02)  + 1.7620*math.log(dato_b03) -1.8647*math.log(dato_b04)- 1.4377*math.log(dato_b08))
		BOD = 1.79 -(0.789*dato_b02)+(52.36*dato_b03)-(3.28*dato_b04)
		COD = 2.76 -(17.27*dato_b02)+(72.15*dato_b03)-(12.11*dato_b04)
		TOC = 6.41 -(85.29*dato_b02)+(2.05*dato_b03)-(29.96*dato_b04)

		year_2019 = [2019,2019,2019,2019]
		l_indice = ['NH3','BOD','COD','TOC']
		data_2019 = [NH3_N,BOD,COD,TOC]

		df2019={'Año':year_2019,
				'Indice':l_indice,
				'Valor':data_2019}

		df_2019 = pd.DataFrame(df2019)

		#st.write(df_2019)

		s2_bands2020 = evaluacion_suelo.bandas(select_zona_agua, 2020)

		arrs = []
		for band in s2_bands2020:
		    with rasterio.open(band) as f:
		        arrs.append(f.read(1))
		sentinel_img = np.array(arrs, dtype=arrs[0].dtype)
		clipped_img = sentinel_img[:, 0:1080:, 0:1080]
		np.seterr(divide='ignore', invalid='ignore')
		band02 = clipped_img[0] 
		band03 = clipped_img[1] 
		band04 = clipped_img[2] 
		band08 = clipped_img[3] 
		band11 = clipped_img[4] 
		band12 = clipped_img[5]

		df_b02 = pd.DataFrame(band02)  #azul   
		df_b03 = pd.DataFrame(band03) # verde
		df_b04 = pd.DataFrame(band04) # roja
		df_b08 = pd.DataFrame(band08) # NIR
		df_b11 = pd.DataFrame(band11)
		df_b12 = pd.DataFrame(band12) 



		dato_b02 = df_b02[210][206]
		dato_b03 = df_b03[210][206]
		dato_b04 = df_b04[210][206]
		dato_b08 = df_b08[210][206]
		dato_b11 = df_b11[210][206]


		NH3_N = math.e**(-8.8129 - 1.7044*math.log(dato_b02)  + 1.7620*math.log(dato_b03) -1.8647*math.log(dato_b04)- 1.4377*math.log(dato_b08))
		BOD = 1.79 -(0.789*dato_b02)+(52.36*dato_b03)-(3.28*dato_b04)
		COD = 2.76 -(17.27*dato_b02)+(72.15*dato_b03)-(12.11*dato_b04)
		TOC = 6.41 -(85.29*dato_b02)+(2.05*dato_b03)-(29.96*dato_b04)

		year_2020 = [2020,2020,2020,2020]
		l_indice = ['NH3','BOD','COD','TOC']
		data_2020 = [NH3_N,BOD,COD,TOC]

		df2020={'Año':year_2020,
				'Indice':l_indice,
				'Valor':data_2020}

		df_2020 = pd.DataFrame(df2020)

		#st.write(df_2020)

		s2_bands2021 = evaluacion_suelo.bandas(select_zona_agua, 2021)

		arrs = []
		for band in s2_bands2021:
		    with rasterio.open(band) as f:
		        arrs.append(f.read(1))
		sentinel_img = np.array(arrs, dtype=arrs[0].dtype)
		clipped_img = sentinel_img[:, 0:1080:, 0:1080]
		np.seterr(divide='ignore', invalid='ignore')
		band02 = clipped_img[0] 
		band03 = clipped_img[1] 
		band04 = clipped_img[2] 
		band08 = clipped_img[3] 
		band11 = clipped_img[4] 
		band12 = clipped_img[5]

		df_b02 = pd.DataFrame(band02)  #azul   
		df_b03 = pd.DataFrame(band03) # verde
		df_b04 = pd.DataFrame(band04) # roja
		df_b08 = pd.DataFrame(band08) # NIR
		df_b11 = pd.DataFrame(band11)
		df_b12 = pd.DataFrame(band12) 



		dato_b02 = df_b02[210][206]
		dato_b03 = df_b03[210][206]
		dato_b04 = df_b04[210][206]
		dato_b08 = df_b08[210][206]
		dato_b11 = df_b11[210][206]


		NH3_N = math.e**(-8.8129 - 1.7044*math.log(dato_b02)  + 1.7620*math.log(dato_b03) -1.8647*math.log(dato_b04)- 1.4377*math.log(dato_b08))
		BOD = 1.79 -(0.789*dato_b02)+(52.36*dato_b03)-(3.28*dato_b04)
		COD = 2.76 -(17.27*dato_b02)+(72.15*dato_b03)-(12.11*dato_b04)
		TOC = 6.41 -(85.29*dato_b02)+(2.05*dato_b03)-(29.96*dato_b04)

		year_2021 = [2021,2021,2021,2021]
		l_indice = ['NH3','BOD','COD','TOC']
		data_2021 = [NH3_N,BOD,COD,TOC]

		df2021={'Año':year_2021,
				'Indice':l_indice,
				'Valor':data_2021}

		df_2021 = pd.DataFrame(df2021)

		#st.write(df_2021)

		df_resul_index = pd.concat([df_2017,df_2018,df_2019,df_2020,df_2021])

		#st.write(df_resul_index)

		fig_final= px.line(df_resul_index, x = 'Año', y='Valor', color='Indice')
		st.write(fig_final)

if option == '🔎 Detección de cambios':
	st.info('Deteccion de cambios en la Zona 1 2017 y Zona 1 2020')
	video_file = open('deteccion_cambios.mp4', 'rb')
	video_bytes = video_file.read()
	st.video(video_bytes ,format="video/mp4", start_time=0)
	
