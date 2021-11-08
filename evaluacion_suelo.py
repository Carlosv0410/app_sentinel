import streamlit as st 

def bandas(zona, year):

	if zona== 'Zona 1 red':

		if year ==2017:
			s2_bands = ["Evaluacion_suelo\\Zona1\\2017\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona1\\2017\\B03.tiff",
		            	"Evaluacion_suelo\\Zona1\\2017\\B04.tiff",
		            	"Evaluacion_suelo\\Zona1\\2017\\B08.tiff",
		            	"Evaluacion_suelo\\Zona1\\2017\\B11.tiff",
		            	"Evaluacion_suelo\\Zona1\\2017\\B12.tiff"]
		if year ==2018:
			s2_bands = ["Evaluacion_suelo\\Zona1\\2018\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona1\\2018\\B03.tiff",
		            	"Evaluacion_suelo\\Zona1\\2018\\B04.tiff",
		            	"Evaluacion_suelo\\Zona1\\2018\\B08.tiff",
		            	"Evaluacion_suelo\\Zona1\\2018\\B11.tiff",
		            	"Evaluacion_suelo\\Zona1\\2018\\B12.tiff"]
		if year ==2019:
			s2_bands = ["Evaluacion_suelo\\Zona1\\2019\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona1\\2019\\B03.tiff",
		            	"Evaluacion_suelo\\Zona1\\2019\\B04.tiff",
		            	"Evaluacion_suelo\\Zona1\\2019\\B08.tiff",
		            	"Evaluacion_suelo\\Zona1\\2019\\B11.tiff",
		            	"Evaluacion_suelo\\Zona1\\2019\\B12.tiff"]
		if year ==2020:
			s2_bands = ["Evaluacion_suelo\\Zona1\\2020\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona1\\2020\\B03.tiff",
		            	"Evaluacion_suelo\\Zona1\\2020\\B04.tiff",
		            	"Evaluacion_suelo\\Zona1\\2020\\B08.tiff",
		            	"Evaluacion_suelo\\Zona1\\2020\\B11.tiff",
		            	"Evaluacion_suelo\\Zona1\\2020\\B12.tiff"]
		if year ==2021:
			s2_bands = ["Evaluacion_suelo\\Zona1\\2021\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona1\\2021\\B03.tiff",
		            	"Evaluacion_suelo\\Zona1\\2021\\B04.tiff",
		            	"Evaluacion_suelo\\Zona1\\2021\\B08.tiff",
		            	"Evaluacion_suelo\\Zona1\\2021\\B11.tiff",
		            	"Evaluacion_suelo\\Zona1\\2021\\B12.tiff"]



	if zona== 'Zona 2 yellow':

		if year ==2017:
			s2_bands = ["Evaluacion_suelo\\Zona2\\2017\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona2\\2017\\B03.tiff",
		            	"Evaluacion_suelo\\Zona2\\2017\\B04.tiff",
		            	"Evaluacion_suelo\\Zona2\\2017\\B08.tiff",
		            	"Evaluacion_suelo\\Zona2\\2017\\B11.tiff",
		            	"Evaluacion_suelo\\Zona2\\2017\\B12.tiff"]
		if year ==2018:
			s2_bands = ["Evaluacion_suelo\\Zona2\\2018\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona2\\2018\\B03.tiff",
		            	"Evaluacion_suelo\\Zona2\\2018\\B04.tiff",
		            	"Evaluacion_suelo\\Zona2\\2018\\B08.tiff",
		            	"Evaluacion_suelo\\Zona2\\2018\\B11.tiff",
		            	"Evaluacion_suelo\\Zona2\\2018\\B12.tiff"]
		if year ==2019:
			s2_bands = ["Evaluacion_suelo\\Zona2\\2019\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona2\\2019\\B03.tiff",
		            	"Evaluacion_suelo\\Zona2\\2019\\B04.tiff",
		            	"Evaluacion_suelo\\Zona2\\2019\\B08.tiff",
		            	"Evaluacion_suelo\\Zona2\\2019\\B11.tiff",
		            	"Evaluacion_suelo\\Zona2\\2019\\B12.tiff"]
		if year ==2020:
			s2_bands = ["Evaluacion_suelo\\Zona2\\2020\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona2\\2020\\B03.tiff",
		            	"Evaluacion_suelo\\Zona2\\2020\\B04.tiff",
		            	"Evaluacion_suelo\\Zona2\\2020\\B08.tiff",
		            	"Evaluacion_suelo\\Zona2\\2020\\B11.tiff",
		            	"Evaluacion_suelo\\Zona2\\2020\\B12.tiff"]
		if year ==2021:
			s2_bands = ["Evaluacion_suelo\\Zona2\\2021\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona2\\2021\\B03.tiff",
		            	"Evaluacion_suelo\\Zona2\\2021\\B04.tiff",
		            	"Evaluacion_suelo\\Zona2\\2021\\B08.tiff",
		            	"Evaluacion_suelo\\Zona2\\2021\\B11.tiff",
		            	"Evaluacion_suelo\\Zona2\\2021\\B12.tiff"]


	if zona== 'Zona 3 cian':

		if year ==2017:
			s2_bands = ["Evaluacion_suelo\\Zona3\\2017\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona3\\2017\\B03.tiff",
		            	"Evaluacion_suelo\\Zona3\\2017\\B04.tiff",
		            	"Evaluacion_suelo\\Zona3\\2017\\B08.tiff",
		            	"Evaluacion_suelo\\Zona3\\2017\\B11.tiff",
		            	"Evaluacion_suelo\\Zona3\\2017\\B12.tiff"]
		if year ==2018:
			s2_bands = ["Evaluacion_suelo\\Zona3\\2018\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona3\\2018\\B03.tiff",
		            	"Evaluacion_suelo\\Zona3\\2018\\B04.tiff",
		            	"Evaluacion_suelo\\Zona3\\2018\\B08.tiff",
		            	"Evaluacion_suelo\\Zona3\\2018\\B11.tiff",
		            	"Evaluacion_suelo\\Zona3\\2018\\B12.tiff"]
		if year ==2019:
			s2_bands = ["Evaluacion_suelo\\Zona3\\2019\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona3\\2019\\B03.tiff",
		            	"Evaluacion_suelo\\Zona3\\2019\\B04.tiff",
		            	"Evaluacion_suelo\\Zona3\\2019\\B08.tiff",
		            	"Evaluacion_suelo\\Zona3\\2019\\B11.tiff",
		            	"Evaluacion_suelo\\Zona3\\2019\\B12.tiff"]
		if year ==2020:
			s2_bands = ["Evaluacion_suelo\\Zona3\\2020\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona3\\2020\\B03.tiff",
		            	"Evaluacion_suelo\\Zona3\\2020\\B04.tiff",
		            	"Evaluacion_suelo\\Zona3\\2020\\B08.tiff",
		            	"Evaluacion_suelo\\Zona3\\2020\\B11.tiff",
		            	"Evaluacion_suelo\\Zona3\\2020\\B12.tiff"]
		if year ==2021:
			s2_bands = ["Evaluacion_suelo\\Zona3\\2021\\B02.tiff",
	        		    "Evaluacion_suelo\\Zona3\\2021\\B03.tiff",
		            	"Evaluacion_suelo\\Zona3\\2021\\B04.tiff",
		            	"Evaluacion_suelo\\Zona3\\2021\\B08.tiff",
		            	"Evaluacion_suelo\\Zona3\\2021\\B11.tiff",
		            	"Evaluacion_suelo\\Zona3\\2021\\B12.tiff"]

	return s2_bands







