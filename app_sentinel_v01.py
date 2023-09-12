# APP SENTINEL HUB CALCULO DE INDICES
# Elaborado por Ing. Carlos Carrillo Villavicencio Master Degree TIC
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
CLIENT_ID ='0d7e8598-eb42-4313-923c-3707afa2efd4'
CLIENT_SECRET = 'hLc7Lg;&0%&mW:2?U&>j0N>jzIpCQAEK)2JkiAlZ'
config = SHConfig()
if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id= CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET    
if config.sh_client_id == '' or  config.sh_client_secret == '':
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

#Inicio de la Aplicación
#-----------------------
st.title('Sistema para monitoreo de bioindicadores ANZU')

video_file = open('monitoring1.mp4', 'rb')
video_bytes = video_file.read()
st.sidebar.video(video_bytes ,format="video/mp4", start_time=0)
#st.sidebar.write('Sentinel')

st.sidebar.title('Menu')
option = st.sidebar.radio('Seleccione una opcion', ['💧 MAPE', '🛰 Visualización satelital', '⛰ Evaluación de cobertura vegetal', '🧪 Evaluación de la calidad el agua'])


qr = Image.open('qrcode_app-sentinel-v1.herokuapp.com.png')
st.sidebar.image(qr,caption='Quick Response', width=150)


if option == '💧 MAPE':

	st.info('MAPE - Amazonia ecuatoriana')

	columna1, columna2 = st.columns(2)
	with columna1:
		select_concesion = st.selectbox('🌐 Seleccione',['⚒ Concesiones','Zona de monitoreo 1','Zona de monitoreo 2','Zona de monitoreo 3','Anzu Norte', 'Berta 1', 'Confluencia', 'Cristobal','Genial', 'Vista Anzu'])

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


	st.warning('Ingrese coordenadas WGS84 para el area a monitorear')


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

if option == '⛰ Evaluación de cobertura vegetal':
	
	try:
		zona_opcion = st.selectbox('🌐 Seleccione una zona',['Zona de monitoreo 1','Zona de monitoreo 2','Zona de monitoreo 3','Anzu Norte', 'Berta 1', 'Confluencia', 'Cristobal','Genial', 'Vista Anzu'])
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
		band09 = clipped_img[6]
		band05 = clipped_img[7]
		band06 = clipped_img[8]

		with st.expander('Indices de vegetacion'):
			
			col1,col2 = st.columns(2)
			with col1:
			
				ndvi_index = (band08.astype(float)-band04.astype(float) )/(band08.astype(float)+band04.astype(float))

				fig_ndvi = px.imshow(ndvi_index,title='Índice de vegetación de diferencia normalizada', labels=dict(x="pixel x", y="pixel y", color="NDVI"))
				fig_ndvi.update_layout(margin=dict( l=0, r=10, b=10, t=30,pad=1),coloraxis_colorbar=dict( title="NDVI", len=0.8 , thickness=5))
				st.plotly_chart(fig_ndvi,use_container_width=True)

				st.write('\nMax NDVI: {m}'.format(m=ndvi_index.max()))
				st.write('Mean NDVI: {m}'.format(m=ndvi_index.mean()))
				st.write('Median NDVI: {m}'.format(m=np.median(ndvi_index)))
				st.write('Min NDVI: {m}'.format(m=ndvi_index.min()))
				st.success('Es un indicador numérico que utiliza las bandas espectrales roja y cercana al infrarrojo. El NDVI está altamente asociado con el contenido de vegetación. ')

			
			with col2:
				
				bsi_index = ((band12.astype(float)+band04.astype(float))-(band08.astype(float)+band02.astype(float)))/((band12.astype(float)+band04.astype(float))+(band08.astype(float)+band02.astype(float)))

				fig_bsi = px.imshow(bsi_index,title='Índice de suelo desnudo', labels=dict(x="pixel x", y="pixel y", color="BSI"))
				fig_bsi.update_layout(margin=dict( l=0, r=10, b=10, t=30,pad=1),coloraxis_colorbar=dict( title="BSI", len=0.8 , thickness=5))
				st.plotly_chart(fig_bsi,use_container_width=True)

				st.write('\nMax BSI: {m}'.format(m=bsi_index.max()))
				st.write('Mean BSI: {m}'.format(m=bsi_index.mean()))
				st.write('Median BSI: {m}'.format(m=np.median(bsi_index)))
				st.write('Min BSI: {m}'.format(m=bsi_index.min()))
				st.success('Es un indicador numérico que combina bandas espectrales azules, rojas, infrarrojas cercanas e infrarrojas de onda corta para capturar las variaciones del suelo. Estas bandas espectrales se utilizan de manera normalizada.')


			
			gndvi_index = (band08.astype(float)-band03.astype(float) )/ (band08.astype(float)+band03.astype(float))
			ndmi_index = (band08.astype(float)-band11.astype(float))/(band08.astype(float)+band11.astype(float))
			gci_index = (band09.astype(float)/band03.astype(float))-1
			ndwi_index = (band03.astype(float)-band08.astype(float))/(band03.astype(float)+band08.astype(float))

			cola,colb = st.columns(2)
			
			with cola:
				
				fig_gndvi_index = px.imshow(gndvi_index,title='Índice de Vegetación de la Diferencia Normalizada Verde', labels=dict(x="pixel x", y="pixel y", color="GNDVI"))
				fig_gndvi_index.update_layout(margin=dict( l=0, r=10, b=10, t=30,pad=1),coloraxis_colorbar=dict( title="GNDVI", len=0.8 , thickness=5))
				st.plotly_chart(fig_gndvi_index,use_container_width=True)
				st.success('Es una versión modificada del NDVI para que sea más sensible a la variación del contenido de clorofila en el cultivo.')
				
				fig_gci_index = px.imshow(gci_index,title='Índice de Clorofila', labels=dict(x="pixel x", y="pixel y", color="GCI"))
				fig_gci_index.update_layout(margin=dict( l=0, r=10, b=10, t=30,pad=1),coloraxis_colorbar=dict( title="GCI", len=0.8 , thickness=5))
				st.plotly_chart(fig_gci_index,use_container_width=True)
				st.success('Se utiliza para estimar el contenido de clorofila en las hojas de diversas especies de plantas. El contenido de clorofila refleja el estado fisiológico de la vegetación; disminuye en las plantas estresadas y, por lo tanto, puede utilizarse como medida de la salud de las plantas (EOS, 2019).')
				
			with colb:
				
				fig_ndmi_index = px.imshow(ndmi_index,title='Índice de Diferencia Normalizada de Humedad' , labels=dict(x="pixel x", y="pixel y", color="NDMI"))
				fig_ndmi_index.update_layout(margin=dict( l=0, r=10, b=10, t=30,pad=1),coloraxis_colorbar=dict( title="NDMI", len=0.8 , thickness=5))
				st.plotly_chart(fig_ndmi_index,use_container_width=True)
				st.success('Se utiliza para determinar el contenido de agua de la vegetación.')
							
				fig_ndwi_index = px.imshow(ndwi_index,title='Índice Diferencial de Agua Normalizado', labels=dict(x="pixel x", y="pixel y", color="NDWI"))
				fig_ndwi_index.update_layout(margin=dict( l=0, r=10, b=10, t=30,pad=1),coloraxis_colorbar=dict( title="NDWI", len=0.8 , thickness=5))
				st.plotly_chart(fig_ndwi_index,use_container_width=True)
				st.success('Se utiliza para el análisis de masas de agua. El índice utiliza bandas verdes y casi infrarrojas de imágenes de teledetección. El NDWI puede mejorar la información sobre el agua de manera eficiente en la mayoría de los casos. Es sensible a la acumulación de tierra y resulta en la sobreestimación de los cuerpos de agua.')
				
	
	except:
		st.sidebar.error("Zona de muestreo no explorada")
		

		
	with st.expander('Índices complementarios'):
		indice_ndvi = (band08.astype(float)-band04.astype(float) )/ (band08.astype(float)+band04.astype(float))
		indice_ndwi =  (band08.astype(float)-(band05.astype(float)*1))
		indice_tgi = band03.astype(float)-0.39*band04.astype(float)-0.61*band02.astype(float)
		indice_evi2 = (2.5*(band08.astype(float)-band04.astype(float))/ (band08.astype(float)+6*(band04.astype(float))+2.4*(band02.astype(float))+1))
		indice_dbsi= band11.astype(float)-(band03.astype(float)/band11.astype(float))+(band03.astype(float)-((band08.astype(float)-band04.astype(float))/ (band08.astype(float)+band04.astype(float))))
		#NDBI  = ((band11.astype(float)-band08.astype(float))/(band11.astype(float)+band08.astype(float)))
		#NDVI  = ((band08.astype(float)-band04.astype(float))/(band08.astype(float)+band04.astype(float)))
		#MNDWI = ((band03.astype(float)-band11.astype(float))/(band03.astype(float)+band11.astype(float)))		
		indice_ibi = ((band11.astype(float)-band08.astype(float))/(band11.astype(float)+band08.astype(float))) - (((band08.astype(float)-band04.astype(float))/(band08.astype(float)+band04.astype(float))) + ((band03.astype(float)-band11.astype(float))/(band03.astype(float)+band11.astype(float)))/2)/((band11.astype(float)-band08.astype(float))/(band11.astype(float)+band08.astype(float))) + (((band08.astype(float)-band04.astype(float))/(band08.astype(float)+band04.astype(float))) + ((band03.astype(float)-band11.astype(float))/(band03.astype(float)+band11.astype(float)))/2)
		indice_gci = (band09.astype(float)/band03.astype(float))-1
		indice_ci = (band04.astype(float)-band03.astype(float) )/ (band04.astype(float)+band03.astype(float))
		indice_savi = ((band08.astype(float)-band05.astype(float))/(band08.astype(float)+band05.astype(float)+0.5))*(1+0.5)
		
		#def plot_indices(ingreso_indice, name):
		#	fig,ax = plt.subplots()
		#	ax.imshow(ingreso_indice, cmap="RdYlGn")
		#	ax.set_title(name,fontweight ="bold")
		#	return fig
			
			
		#cola,colb,colc = st.columns(3)
		
		#with cola:
		#	fig_indice_ndvi=plot_indices(indice_ndwi, 'Índice NDVI')
		#	st.pyplot(fig_indice_ndvi)
			
		#	fig_indice_ndwi=plot_indices(indice_ndwi, 'Índice NDWI')
		#	st.pyplot(fig_indice_ndwi)
			
		#	fig_indice_tgi=plot_indices(indice_tgi, 'Índice TGI')
		#	st.pyplot(fig_indice_tgi)
				
		#with colb:				
		#	fig_indice_evi2=plot_indices(indice_evi2, 'Índice EVI2')
		#	st.pyplot(fig_indice_evi2)
			
		#	fig_indice_dbsi=plot_indices(indice_dbsi, 'Índice DBSI')
		#	st.pyplot(fig_indice_dbsi)
			
		#	fig_indice_ibi=plot_indices(indice_ibi, 'Índice IBI')
		#	st.pyplot(fig_indice_ibi)
		#with colc:
			
		#	fig_indice_gci=plot_indices(indice_gci, 'Índice GCI')
		#	st.pyplot(fig_indice_gci)
			
		#	fig_indice_ci=plot_indices(indice_ci, 'Índice CI')
		#	st.pyplot(fig_indice_ci)
			
		#	fig_indice_savi=plot_indices(indice_savi, 'Índice SAVI')
		#	st.pyplot(fig_indice_savi)
			
			
		principales_indices2 = st.selectbox('Seleccione el indice',['NDVI','NDWI','GCI','TGI','EVI2','DBSI','IBI','CI','SAVI'])
		
		if principales_indices2 =='NDVI':
			fig = px.imshow(indice_ndvi, title='NDVI', labels=dict(x="pixel x", y="pixel y", color='NDVI'))
			st.write(fig)
			st.success(' NDVI < 0 suelo desnudo, infraestructura, cuerpos de agua 0 - 0.6 ')
		elif principales_indices2 =='NDWI':
			fig = px.imshow(indice_ndwi,title='NDWI', labels=dict(x="pixel x", y="pixel y", color='NDWI'))
			st.write(fig)
			st.success('NDWI < = 0 Cuerpos hídricos 0 < NDWI < 0.1 suelo desnudo o cubiertas infraestructura ')
		elif principales_indices2 =='GCI':
			fig = px.imshow(indice_gci,title='GCI', labels=dict(x="pixel x", y="pixel y", color='GCI'))
			st.write(fig)
		elif principales_indices2 =='TGI':
			fig = px.imshow(indice_tgi,title='TGI', labels=dict(x="pixel x", y="pixel y", color='TGI'))
			st.write(fig)
			st.success('Triangular Greeness Index Infraestructura valores negativos, agricultura, zonas afectadas valores medios en la escala produccion media clorofila asociado actividad deforestacion por la actividad minera y otras actividades.  El índice TGI permitió identificar satisfactoriamente  cambios en las coberturas asociados a procesos tales como: ganadería, agricultura, tala de  árboles y adecuaciones de la infraestructura necesaria para el desarrollo de la actividad minera llevada a cabo en estas zonas. ')
		elif principales_indices2 =='EVI2':
			fig = px.imshow(indice_evi2,title='EVI2', labels=dict(x="pixel x", y="pixel y", color='EVI2'))
			st.write(fig)
			st.success('Indice de vegetacion mejorado 2')
		elif principales_indices2 =='DBSI':
			fig = px.imshow(indice_dbsi,title='DBSI', labels=dict(x="pixel x", y="pixel y", color='DBSI'))
			st.write(fig)
			st.success('Dry Bare-Soil Index')
		elif principales_indices2 =='IBI':
			fig = px.imshow(indice_ibi,title='IBI', labels=dict(x="pixel x", y="pixel y", color='IBI'))
			st.write(fig)
			st.success('Index-based Built-up Index')
		elif principales_indices2 =='CI':
			fig = px.imshow(indice_ci,title='CI', labels=dict(x="pixel x", y="pixel y", color='CI'))
			st.write(fig)
			#st.success('Los incendios forestales son un fenómeno natural o provocado por el hombre que destruye los recursos naturales, el ganado vivo, desequilibra el medio ambiente local, libera una gran cantidad de gases de efecto invernadero, etc.')
		elif principales_indices2 =='SAVI':
			fig = px.imshow(indice_savi,title='SAVI', labels=dict(x="pixel x", y="pixel y", color='SAVI'))
			st.write(fig)
			st.success('SAVI < 0  cuerpos de agua 0 - 0.3 medio rango de menor vigor vegetal o suelos sin vegetación')
	
		
	with st.expander('Monitoreo con detección de cambios'):
		st.success('El Machine Learning o aprendizaje automático posibilita la identificación de patrones en los datos basándose en algoritmos que clasifican cada factor según su grado de influencia aprendiendo y mejorando el proceso continuamente')
		st.write('Las dos principales características que determinan la percepción de las imágenes satelitales son la resolución y la frecuencia de las imágenes utilizadas. La disponibilidad de imágenes de satélites es mayor cada día y las métricas ofertadas mejoran en consecuencia, abriendo el abanico de posibilidades para aplicaciones que aporten soluciones a actividades y empresas de todo tipo.')
		
		video_file = open('deteccion_cambios.mp4', 'rb')
		video_bytes = video_file.read()
		st.video(video_bytes ,format="video/mp4", start_time=0)

if option =='🧪 Evaluación de la calidad el agua':

	with st.expander('Análisis anual de Índices'):
		
		try:
			zona_opcion_agua = st.selectbox('🌐 Seleccione una zona',['Zona de monitoreo 1','Zona de monitoreo 2','Zona de monitoreo 3','Anzu Norte', 'Berta 1', 'Confluencia', 'Cristobal','Genial', 'Vista Anzu'])
			year_option_agua = st.slider('Elija un año', 2017,2021,2017)

			if zona_opcion_agua =='Zona de monitoreo 1':
				zona_opcion_agua='River'

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
				ax.set_title('DQO')
				st.pyplot(fig_index_agua2)
				st.write('\nMax DQO: {m}'.format(m=round(COD_map.max(),2)))
				st.write('Mean DQO: {m}'.format(m=round(COD_map.mean(),2)))
				st.write('Median DQO: {m}'.format(m=round(np.median(COD_map),2)))
				st.write('Min DQO: {m}'.format(m=round(COD_map.min(),2)))
			with colc:
				fig_index_agua3, ax = plt.subplots()
				ax.imshow(BOD_map, cmap="RdYlGn")
				ax.set_title('DBO')
				st.pyplot(fig_index_agua3)
				st.write('\nMax DBO: {m}'.format(m=round(BOD_map.max(),2)))
				st.write('Mean DBO: {m}'.format(m=round(BOD_map.mean(),2)))
				st.write('Median DBO: {m}'.format(m=round(np.median(BOD_map),2)))
				st.write('Min DBO: {m}'.format(m=round(BOD_map.min(),2)))
			with cold:
				fig_index_agua4, ax = plt.subplots()
				ax.imshow(TOC_map, cmap="RdYlGn")
				ax.set_title('COT')
				st.pyplot(fig_index_agua4)
				st.write('\nMax COT: {m}'.format(m=round(TOC_map.max(),2)))
				st.write('Mean COT: {m}'.format(m=round(TOC_map.mean(),2)))
				st.write('Median COT: {m}'.format(m=round(np.median(TOC_map),2)))
				st.write('Min COT: {m}'.format(m=round(TOC_map.min(),2)))
		except:
			st.sidebar.error("Zona de muestreo no explorada")

	with st.expander('Análisis por ubicación'):
		
		try:

			indice_plot =st.selectbox('Seleccione el indice a analizar',['NH3','DQO','DBO','COT'])

			if indice_plot == 'NH3':

				fig = px.imshow(NH3_N_map, title='NH3', labels=dict(x="pixel x", y="pixel y", color='NH3'))
				st.write(fig)
			elif indice_plot == 'DQO':

				fig = px.imshow(COD_map, title='DQO', labels=dict(x="pixel x", y="pixel y", color='DQO'))
				st.write(fig)
			elif indice_plot == 'DBO':

				fig = px.imshow(BOD_map, title='DBO', labels=dict(x="pixel x", y="pixel y", color='DBO'))
				st.write(fig)
			elif indice_plot == 'COT':

				fig = px.imshow(TOC_map, title='COT', labels=dict(x="pixel x", y="pixel y", color='COT'))
				st.write(fig)
		except:
			st.error("Zona de muestreo no explorada")

	with st.expander('Análisis en los puntos de muestreo'):
		st.info('Puntos de muestreo')

		columna1, columna2 = st.columns(2)
		with columna1:
			select_zona_agua = st.selectbox('🌐 Seleccione',['Zona de monitoreo 1','Zona de monitoreo 2','Zona de monitoreo 3','Anzu Norte', 'Berta 1', 'Confluencia', 'Cristobal','Genial', 'Vista Anzu'])

			year_zona = st.slider('Seleccione el año',2017,2021,2017,step=1)


		with columna2:
			zona = calidad_agua.seleccion_zona(select_zona_agua, year_zona)
			st.image(zona)

	with st.expander('Análisis en los puntos de muestreo'):
		
		try:
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
			l_indice = ['NH3','DBO','DQO','COT']
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
			l_indice = ['NH3','DBO','DQO','COT']
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
			l_indice = ['NH3','DBO','DQO','COT']
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
			l_indice = ['NH3','DBO','DQO','COT']
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
			l_indice = ['NH3','DBO','DQO','COT']
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
		except:
			st.error("Zona de muestreo no explorada")
