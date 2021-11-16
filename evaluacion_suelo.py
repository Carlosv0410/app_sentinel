import streamlit as st 

@st.cache
def bandas(zona, year):

	if zona== 'Zona de monitoreo 1':

		if year ==2017:
			s2_bands = ["Evaluacion_suelo/Zona1/2017/B02.tiff",
	        		    "Evaluacion_suelo/Zona1/2017/B03.tiff",
		            	"Evaluacion_suelo/Zona1/2017/B04.tiff",
		            	"Evaluacion_suelo/Zona1/2017/B08.tiff",
		            	"Evaluacion_suelo/Zona1/2017/B11.tiff",
		            	"Evaluacion_suelo/Zona1/2017/B12.tiff",
				    "Evaluacion_suelo/Zona1/2017/B09.tiff",
				    "Evaluacion_suelo/Zona1/2017/B05.tiff",
				    "Evaluacion_suelo/Zona1/2017/B06.tiff"]
		if year ==2018:
			s2_bands = ["Evaluacion_suelo/Zona1/2018/B02.tiff",
	        		    "Evaluacion_suelo/Zona1/2018/B03.tiff",
		            	"Evaluacion_suelo/Zona1/2018/B04.tiff",
		            	"Evaluacion_suelo/Zona1/2018/B08.tiff",
		            	"Evaluacion_suelo/Zona1/2018/B11.tiff",
		            	"Evaluacion_suelo/Zona1/2018/B12.tiff",
				   "Evaluacion_suelo/Zona1/2018/B09.tiff",
				   "Evaluacion_suelo/Zona1/2018/B05.tiff",
				   "Evaluacion_suelo/Zona1/2018/B06.tiff"]
		if year ==2019:
			s2_bands = ["Evaluacion_suelo/Zona1/2019/B02.tiff",
	        		    "Evaluacion_suelo/Zona1/2019/B03.tiff",
		            	"Evaluacion_suelo/Zona1/2019/B04.tiff",
		            	"Evaluacion_suelo/Zona1/2019/B08.tiff",
		            	"Evaluacion_suelo/Zona1/2019/B11.tiff",
		            	"Evaluacion_suelo/Zona1/2019/B12.tiff",
				   "Evaluacion_suelo/Zona1/2019/B09.tiff",
				   "Evaluacion_suelo/Zona1/2019/B05.tiff",
				   "Evaluacion_suelo/Zona1/2019/B06.tiff"]
		if year ==2020:
			s2_bands = ["Evaluacion_suelo/Zona1/2020/B02.tiff",
	        		    "Evaluacion_suelo/Zona1/2020/B03.tiff",
		            	"Evaluacion_suelo/Zona1/2020/B04.tiff",
		            	"Evaluacion_suelo/Zona1/2020/B08.tiff",
		            	"Evaluacion_suelo/Zona1/2020/B11.tiff",
		            	"Evaluacion_suelo/Zona1/2020/B12.tiff",
				   "Evaluacion_suelo/Zona1/2020/B09.tiff",
				   "Evaluacion_suelo/Zona1/2020/B05.tiff",
				   "Evaluacion_suelo/Zona1/2020/B06.tiff"]
		if year ==2021:
			s2_bands = ["Evaluacion_suelo/Zona1/2021/B02.tiff",
	        		    "Evaluacion_suelo/Zona1/2021/B03.tiff",
		            	"Evaluacion_suelo/Zona1/2021/B04.tiff",
		            	"Evaluacion_suelo/Zona1/2021/B08.tiff",
		            	"Evaluacion_suelo/Zona1/2021/B11.tiff",
		            	"Evaluacion_suelo/Zona1/2021/B12.tiff",
				   "Evaluacion_suelo/Zona1/2021/B09.tiff",
				   "Evaluacion_suelo/Zona1/2021/B05.tiff",
				   "Evaluacion_suelo/Zona1/2021/B06.tiff"]



	if zona== 'Zona de monitoreo 2':

		if year ==2017:
			s2_bands = ["Evaluacion_suelo/Zona2/2017/B02.tiff",
	        		    "Evaluacion_suelo/Zona2/2017/B03.tiff",
		            	"Evaluacion_suelo/Zona2/2017/B04.tiff",
		            	"Evaluacion_suelo/Zona2/2017/B08.tiff",
		            	"Evaluacion_suelo/Zona2/2017/B11.tiff",
		            	"Evaluacion_suelo/Zona2/2017/B12.tiff",
				   "Evaluacion_suelo/Zona2/2017/B09.tiff",
				   "Evaluacion_suelo/Zona2/2017/B05.tiff",
				   "Evaluacion_suelo/Zona2/2017/B06.tiff"]
		if year ==2018:
			s2_bands = ["Evaluacion_suelo/Zona2/2018/B02.tiff",
	        		    "Evaluacion_suelo/Zona2/2018/B03.tiff",
		            	"Evaluacion_suelo/Zona2/2018/B04.tiff",
		            	"Evaluacion_suelo/Zona2/2018/B08.tiff",
		            	"Evaluacion_suelo/Zona2/2018/B11.tiff",
		            	"Evaluacion_suelo/Zona2/2018/B12.tiff",
				   "Evaluacion_suelo/Zona2/2018/B09.tiff",
				   "Evaluacion_suelo/Zona2/2018/B05.tiff",
				   "Evaluacion_suelo/Zona2/2018/B06.tiff"]
		if year ==2019:
			s2_bands = ["Evaluacion_suelo/Zona2/2019/B02.tiff",
	        		    "Evaluacion_suelo/Zona2/2019/B03.tiff",
		            	"Evaluacion_suelo/Zona2/2019/B04.tiff",
		            	"Evaluacion_suelo/Zona2/2019/B08.tiff",
		            	"Evaluacion_suelo/Zona2/2019/B11.tiff",
		            	"Evaluacion_suelo/Zona2/2019/B12.tiff",
				   "Evaluacion_suelo/Zona2/2019/B09.tiff",
				   "Evaluacion_suelo/Zona2/2019/B05.tiff",
				   "Evaluacion_suelo/Zona2/2019/B06.tiff"]
		if year ==2020:
			s2_bands = ["Evaluacion_suelo/Zona2/2020/B02.tiff",
	        		    "Evaluacion_suelo/Zona2/2020/B03.tiff",
		            	"Evaluacion_suelo/Zona2/2020/B04.tiff",
		            	"Evaluacion_suelo/Zona2/2020/B08.tiff",
		            	"Evaluacion_suelo/Zona2/2020/B11.tiff",
		            	"Evaluacion_suelo/Zona2/2020/B12.tiff",
				   "Evaluacion_suelo/Zona2/2020/B09.tiff",
				   "Evaluacion_suelo/Zona2/2020/B05.tiff",
				   "Evaluacion_suelo/Zona2/2020/B06.tiff"]
		if year ==2021:
			s2_bands = ["Evaluacion_suelo/Zona2/2021/B02.tiff",
	        		    "Evaluacion_suelo/Zona2/2021/B03.tiff",
		            	"Evaluacion_suelo/Zona2/2021/B04.tiff",
		            	"Evaluacion_suelo/Zona2/2021/B08.tiff",
		            	"Evaluacion_suelo/Zona2/2021/B11.tiff",
		            	"Evaluacion_suelo/Zona2/2021/B12.tiff",
				   "Evaluacion_suelo/Zona2/2021/B09.tiff",
				   "Evaluacion_suelo/Zona2/2021/B05.tiff",
				   "Evaluacion_suelo/Zona2/2021/B06.tiff"]


	if zona== 'Zona de monitoreo 3':

		if year ==2017:
			s2_bands = ["Evaluacion_suelo/Zona3/2017/B02.tiff",
	        		    "Evaluacion_suelo/Zona3/2017/B03.tiff",
		            	"Evaluacion_suelo/Zona3/2017/B04.tiff",
		            	"Evaluacion_suelo/Zona3/2017/B08.tiff",
		            	"Evaluacion_suelo/Zona3/2017/B11.tiff",
		            	"Evaluacion_suelo/Zona3/2017/B12.tiff",
				   "Evaluacion_suelo/Zona3/2017/B09.tiff",
				   "Evaluacion_suelo/Zona3/2017/B05.tiff",
				   "Evaluacion_suelo/Zona3/2017/B06.tiff"]
		if year ==2018:
			s2_bands = ["Evaluacion_suelo/Zona3/2018/B02.tiff",
	        		    "Evaluacion_suelo/Zona3/2018/B03.tiff",
		            	"Evaluacion_suelo/Zona3/2018/B04.tiff",
		            	"Evaluacion_suelo/Zona3/2018/B08.tiff",
		            	"Evaluacion_suelo/Zona3/2018/B11.tiff",
		            	"Evaluacion_suelo/Zona3/2018/B12.tiff",
				   "Evaluacion_suelo/Zona3/2018/B09.tiff",
				   "Evaluacion_suelo/Zona3/2018/B05.tiff",
				   "Evaluacion_suelo/Zona3/2018/B06.tiff"]
		if year ==2019:
			s2_bands = ["Evaluacion_suelo/Zona3/2019/B02.tiff",
	        		    "Evaluacion_suelo/Zona3/2019/B03.tiff",
		            	"Evaluacion_suelo/Zona3/2019/B04.tiff",
		            	"Evaluacion_suelo/Zona3/2019/B08.tiff",
		            	"Evaluacion_suelo/Zona3/2019/B11.tiff",
		            	"Evaluacion_suelo/Zona3/2019/B12.tiff",
				   "Evaluacion_suelo/Zona3/2019/B09.tiff",
				   "Evaluacion_suelo/Zona3/2019/B05.tiff",
				   "Evaluacion_suelo/Zona3/2019/B06.tiff"]
		if year ==2020:
			s2_bands = ["Evaluacion_suelo/Zona3/2020/B02.tiff",
	        		    "Evaluacion_suelo/Zona3/2020/B03.tiff",
		            	"Evaluacion_suelo/Zona3/2020/B04.tiff",
		            	"Evaluacion_suelo/Zona3/2020/B08.tiff",
		            	"Evaluacion_suelo/Zona3/2020/B11.tiff",
		            	"Evaluacion_suelo/Zona3/2020/B12.tiff",
				   "Evaluacion_suelo/Zona3/2020/B09.tiff",
				   "Evaluacion_suelo/Zona3/2020/B05.tiff",
				   "Evaluacion_suelo/Zona3/2020/B06.tiff"]
		if year ==2021:
			s2_bands = ["Evaluacion_suelo/Zona3/2021/B02.tiff",
	        		    "Evaluacion_suelo/Zona3/2021/B03.tiff",
		            	"Evaluacion_suelo/Zona3/2021/B04.tiff",
		            	"Evaluacion_suelo/Zona3/2021/B08.tiff",
		            	"Evaluacion_suelo/Zona3/2021/B11.tiff",
		            	"Evaluacion_suelo/Zona3/2021/B12.tiff",
				   "Evaluacion_suelo/Zona3/2021/B09.tiff",
				   "Evaluacion_suelo/Zona3/2021/B05.tiff",
				   "Evaluacion_suelo/Zona3/2021/B06.tiff"]

	if zona== 'River':

		if year ==2017:
			s2_bands = ["Evaluacion_suelo/River/bandas_2017/S2_B2_2017.tif",
	        		    "Evaluacion_suelo/River/bandas_2017/S2_B3_2017.tif",
		            	"Evaluacion_suelo/River/bandas_2017/S2_B4_2017.tif",
		            	"Evaluacion_suelo/River/bandas_2017/S2_B8_2017.tif",
		            	"Evaluacion_suelo/River/bandas_2017/S2_B11_2017.tif",
		            	"Evaluacion_suelo/River/bandas_2017/S2_B12_2017.tif"]
		if year ==2018:
			s2_bands = ["Evaluacion_suelo/River/bandas_2018/S2_B2_2018.tif",
	        		    "Evaluacion_suelo/River/bandas_2018/S2_B3_2018.tif",
		            	"Evaluacion_suelo/River/bandas_2018/S2_B4_2018.tif",
		            	"Evaluacion_suelo/River/bandas_2018/S2_B8_2018.tif",
		            	"Evaluacion_suelo/River/bandas_2018/S2_B11_2018.tif",
		            	"Evaluacion_suelo/River/bandas_2018/S2_B12_2018.tif"]
		if year ==2019:
			s2_bands = ["Evaluacion_suelo/River/bandas_2019/S2_B2_2019.tif",
	        		    "Evaluacion_suelo/River/bandas_2019/S2_B3_2019.tif",
		            	"Evaluacion_suelo/River/bandas_2019/S2_B4_2019.tif",
		            	"Evaluacion_suelo/River/bandas_2019/S2_B8_2019.tif",
		            	"Evaluacion_suelo/River/bandas_2019/S2_B11_2019.tif",
		            	"Evaluacion_suelo/River/bandas_2019/S2_B12_2019.tif"]
		if year ==2020:
			s2_bands = ["Evaluacion_suelo/River/bandas_2020/S2_B2_2020.tif",
	        		    "Evaluacion_suelo/River/bandas_2020/S2_B3_2020.tif",
		            	"Evaluacion_suelo/River/bandas_2020/S2_B4_2020.tif",
		            	"Evaluacion_suelo/River/bandas_2020/S2_B8_2020.tif",
		            	"Evaluacion_suelo/River/bandas_2020/S2_B11_2020.tif",
		            	"Evaluacion_suelo/River/bandas_2020/S2_B12_2020.tif"]
		if year ==2021:
			s2_bands = ["Evaluacion_suelo/River/bandas_2021/S2_B2_2021.tif",
	        		    "Evaluacion_suelo/River/bandas_2021/S2_B3_2021.tif",
		            	"Evaluacion_suelo/River/bandas_2021/S2_B4_2021.tif",
		            	"Evaluacion_suelo/River/bandas_2021/S2_B8_2021.tif",
		            	"Evaluacion_suelo/River/bandas_2021/S2_B11_2021.tif",
		            	"Evaluacion_suelo/River/bandas_2021/S2_B12_2021.tif"]			
	return s2_bands
