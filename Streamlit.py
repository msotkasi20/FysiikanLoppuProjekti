import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

import folium
from streamlit_folium import st_folium


# 1) Ladataan data
ACC_URL = "https://raw.githubusercontent.com/msotkasi20/FysiikanLoppuProjekti/refs/heads/main/Data/Accelerometer.csv"
GPS_URL = "https://raw.githubusercontent.com/msotkasi20/FysiikanLoppuProjekti/refs/heads/main/Data/Location.csv"

df_acc = pd.read_csv(ACC_URL)
df_gps = pd.read_csv(GPS_URL)

st.title("Kinkunsulatus 25.12.2025")


# 2) Kiihtyvyys: valitaan z-komponentti ja poistetaan painovoima
t = df_acc["Time (s)"]
data = df_acc["Acceleration z (m/s^2)"]
data = data - np.mean(data)

T_tot = t.max() - t.min()
n = len(t)
fs = n / T_tot       
nyq = fs / 2

def butter_lowpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y

order = 3
cutoff = 5.0            # 2 Hz voi leikata nopean kävelyn/hölkän -> 5 Hz toimii paremmin
data_filt = butter_lowpass_filter(data, cutoff, fs, nyq, order)

# (a) askeleet nollanylityksistä
jaksot = 0
for i in range(n - 1):
    if data_filt[i] * data_filt[i + 1] < 0:
        jaksot += 1
askeleet_a = int(np.round(jaksot / 2))


# 3) (b) Fourier-analyysi (tunnin tyyli)
signal = data
N = len(signal)
dt = np.max(t) / N      # tunnin oletus

fourier = np.fft.fft(signal, N)
psd = fourier * np.conj(fourier) / N
freq = np.fft.fftfreq(N, dt)
L = np.arange(1, int(N / 2))

# haetaan dominoiva taajuus askelalueelta
mask = (freq[L] >= 0.8) & (freq[L] <= 4.0)
Lf = L[mask] if np.any(mask) else L

f_max = freq[Lf][psd[Lf] == np.max(psd[Lf])][0]
thr = np.std(data_filt) * 0.2
active = np.abs(data_filt) > thr
T_active = active.sum() / fs
askeleet_b = int(np.round(f_max * T_active))

# 4) GPS: matka + keskinopeus
# Rajataan huono GPS pois
df_gps = df_gps[df_gps["Horizontal Accuracy (m)"] < 10].reset_index(drop=True)

lat = df_gps["Latitude (°)"].to_numpy()
lon = df_gps["Longitude (°)"].to_numpy()

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c

dist = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
total_distance_m = float(np.sum(dist))

duration_s = float(df_gps["Time (s)"].max() - df_gps["Time (s)"].min())
avg_speed_mps = total_distance_m / duration_s if duration_s > 0 else np.nan

step_length_m = total_distance_m / askeleet_a if askeleet_a > 0 else np.nan


# 5) Tulostetaan luvut
st.write("**Askelmäärä suodatetusta kiihtyvyysdatasta:**", askeleet_a)
st.write("**Askelmäärä Fourier-analyysilla:**", askeleet_b)
st.write("**Kuljettu matka (GPS):**", f"{total_distance_m/1000:.2f} km")
st.write("**Keskinopeus (GPS):**", f"{avg_speed_mps*3.6:.2f} km/h" if np.isfinite(avg_speed_mps) else "—")
st.write("**Askelpituus:**", f"{step_length_m:.2f} m" if np.isfinite(step_length_m) else "—")


# 6) Kuvaajat
st.subheader("1) Suodatettu kiihtyvyysdata (askelmäärään)")
fig1 = plt.figure(figsize=(12, 4))
plt.plot(t, data_filt, color = "red")
plt.axis([1,218,-15,25])
plt.grid()
plt.xlabel("Aika (s)")
plt.ylabel("a_z (m/s²), keskitetty")
plt.title("Suodatettu kiihtyvyys (lowpass)")
st.pyplot(fig1)

st.subheader("2) Tehospektritiheys (PSD)")
fig2 = plt.figure(figsize=(12, 4))
plt.plot(freq[L], psd[L].real, color = "red")
plt.axis([0, 5, 0, 40000])
plt.grid()
plt.xlabel("Taajuus (Hz)")
plt.ylabel("Teho")
plt.title("PSD")
st.pyplot(fig2)

st.subheader("3) Reitti kartalla")
lat_center = float(np.mean(lat))
lon_center = float(np.mean(lon))

my_map = folium.Map(location=[lat_center, lon_center], zoom_start=15)
folium.PolyLine(list(zip(lat, lon)), color="red", weight=3).add_to(my_map)
st_folium(my_map, width=900, height=650)
