import streamlit as st 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def  imagen_concesiones():

	im = Image.open('Zona de estudio\\Zona_de_estudio2.png')
	# Create figure and axes
	fig, ax = plt.subplots(figsize = (18,15))
	# Display the image
	ax.imshow(im)
	# Create a Rectangle patch
	rect1 = patches.Rectangle((410, 80), 50, 40, linewidth=3, edgecolor='r', facecolor='none',label='Label1')
	rect2 = patches.Rectangle((330, 280), 50, 40, linewidth=3, edgecolor='y', facecolor='none')
	rect3 = patches.Rectangle((90, 690), 50, 40, linewidth=3, edgecolor='c', facecolor='none')
	# Add the patch to the Axes
	ax.add_patch(rect1)
	ax.add_patch(rect2)
	ax.add_patch(rect3)

	return fig

def  seleccion_zona(zona, year):
	if zona =='Zona 1 red':

		img1=Image.open('Galeria concesiones\\Zona1\\2017.png')
		img2=Image.open('Galeria concesiones\\Zona1\\2018.png')
		img3=Image.open('Galeria concesiones\\Zona1\\2019.png')
		img4=Image.open('Galeria concesiones\\Zona1\\2020.png')
		img5=Image.open('Galeria concesiones\\Zona1\\2021.png')

		if year==2017:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img1)
		if year==2018:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img2)
		if year==2019:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img3)
		if year==2020:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img4)
		if year==2021:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img5)

	if zona =='Zona 2 yellow':

		img1=Image.open('Galeria concesiones\\Zona2\\2017.png')
		img2=Image.open('Galeria concesiones\\Zona2\\2018.png')
		img3=Image.open('Galeria concesiones\\Zona2\\2019.png')
		img4=Image.open('Galeria concesiones\\Zona2\\2020.png')
		img5=Image.open('Galeria concesiones\\Zona2\\2021.png')

		if year==2017:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img1)
		if year==2018:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img2)
		if year==2019:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img3)
		if year==2020:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img4)
		if year==2021:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img5)

	if zona =='Zona 3 cian':

		img1=Image.open('Galeria concesiones\\Zona3\\2017.png')
		img2=Image.open('Galeria concesiones\\Zona3\\2018.png')
		img3=Image.open('Galeria concesiones\\Zona3\\2019.png')
		img4=Image.open('Galeria concesiones\\Zona3\\2020.png')
		img5=Image.open('Galeria concesiones\\Zona3\\2021.png')

		if year==2017:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img1)
		if year==2018:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img2)
		if year==2019:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img3)
		if year==2020:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img4)
		if year==2021:
			fig, ax = plt.subplots(figsize = (18,15))
			ax.imshow(img5)
	return fig





## eveluacion de d suelo NDVI YBSI con sus nombres y siglas
#ocultar el x y y
#select box

# evaluacion de la calidad del agua

# select box de comportamiento de los indicadores en agua y suelo

# nada vertical y todos los nombres y sus siglas

