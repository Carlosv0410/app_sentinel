import streamlit as st 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

@st.cache
def  imagen_concesiones():
	im = Image.open('Zona_estudio/Zona_de_estudio3.png')
	return im

@st.cache
def  seleccion_zona(zona, year):
	if zona =='Zona de monitoreo 1':

		img1=Image.open('Galeria_concesiones/Zona1/2017.png')
		img2=Image.open('Galeria_concesiones/Zona1/2018.png')
		img3=Image.open('Galeria_concesiones/Zona1/2019.png')
		img4=Image.open('Galeria_concesiones/Zona1/2020.png')
		img5=Image.open('Galeria_concesiones/Zona1/2021.png')

		if year==2017:
			fig=img1
		if year==2018:
			fig=img2
		if year==2019:
			fig=img3
		if year==2020:
			fig=img4
		if year==2021:
			fig=img5

	elif zona =='Zona de monitoreo 2':

		img1=Image.open('Galeria_concesiones/Zona2/2017.png')
		img2=Image.open('Galeria_concesiones/Zona2/2018.png')
		img3=Image.open('Galeria_concesiones/Zona2/2019.png')
		img4=Image.open('Galeria_concesiones/Zona2/2020.png')
		img5=Image.open('Galeria_concesiones/Zona2/2021.png')

		if year==2017:
			fig=img1
		if year==2018:
			fig=img2
		if year==2019:
			fig=img3
		if year==2020:
			fig=img4
		if year==2021:
			fig=img5
	elif zona =='Zona de monitoreo 3':

		img1=Image.open('Galeria_concesiones/Zona3/2017.png')
		img2=Image.open('Galeria_concesiones/Zona3/2018.png')
		img3=Image.open('Galeria_concesiones/Zona3/2019.png')
		img4=Image.open('Galeria_concesiones/Zona3/2020.png')
		img5=Image.open('Galeria_concesiones/Zona3/2021.png')

		if year==2017:
			fig=img1
		if year==2018:
			fig=img2
		if year==2019:
			fig=img3
		if year==2020:
			fig=img4
		if year==2021:
			fig=img5

	else:

		fig = Image.open('Zona_estudio/Zona_de_estudio4.png')

	return fig
