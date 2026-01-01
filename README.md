# Fysiikan loppuprojekti – GPS + kiihtyvyys (Phyphox)

Tässä projektissa analysoidaan Phyphox-sovelluksella mitattua kiihtyvyys- ja GPS-dataa ja visualisoidaan tulokset Streamlitillä.

Sovellus laskee:
- Askelmäärä suodatetusta kiihtyvyysdatasta (nollanylitykset)
- Askelmäärä Fourier-analyysilla (FFT + tehospektri)
- Kuljettu matka GPS-datasta
- Keskinopeus GPS-datasta
- Askelpituus (matka / askelmäärä)

Sovellus näyttää:
- Suodatettu kiihtyvyysdata
- Tehospektritiheys (PSD)
- Reitti kartalla

## Projektin rakenne

.
├─ Streamlit.py  
├─ Data/  
│ ├─ Accelerometer.csv  
│ └─ Location.csv  
└─ README.md  

## Riippuvuudet

Vähintään nämä:
- streamlit
- pandas
- numpy
- matplotlib
- scipy
- folium
- streamlit-folium

## Ajo paikallisesti

1. Asenna kirjastot (tarvittaessa):
   - `pip install streamlit pandas numpy matplotlib scipy folium streamlit-folium`

2. Käynnistä sovellus:
   - `streamlit run Streamlit.py`

## Ajo suoraan GitHubista (raw)

Skripti lukee datan GitHub raw -linkeistä:

- Accelerometer:
  - `https://raw.githubusercontent.com/msotkasi20/FysiikanLoppuProjekti/main/Data/Accelerometer.csv`
- Location:
  - `https://raw.githubusercontent.com/msotkasi20/FysiikanLoppuProjekti/main/Data/Location.csv`

Jos halutaan ajaa suoraan raw-skriptistä, komento on muotoa:

- `streamlit run https://raw.githubusercontent.com/msotkasi20/FysiikanLoppuProjekti/main/Streamlit.py`
