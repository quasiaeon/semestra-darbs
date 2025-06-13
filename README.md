# Milky-Way Density Map · Gaia DR3 Semester Project  
*Andrejs Cvečkovskis — ac24005 · University of Latvia*

---

## 1 · Kas tas ir?

Šis repozitorijs satur pilnībā reproducējamu pipeline, kas

1. **lejupielādē** izvēlētu Gaia DR3 daļu;  
2. **attīra** paralakses, aprēķina Bayesa distance un izdzišanu;  
3. **vokseļo** Piena Ceļa disku trīs dimensijās;  
4. **propagē kļūdas** ar 1 000 Monte‑Karlo realizācijām;  
5. **zīmē** attēlus un interaktīvu HTML karti;  

Pēc noklusējuma tiek apstrādāti ≈ 20 milj. zvaigžņu (|b| < 20°, G < 18); pilna palaišana uz 16 GB portatīvā datora aizņem ~2 st.

---

## 2 · Mapes uzbūve

```
├── data/                 # nav iekļauti
│   ├── stars_raw.parquet
│   └── stars_clean.parquet
├── output/               # Vokseļkubi & figūras
│   ├── voxel_cube.npz
│   ├── error_cube.npz
│   ├── volume.png
│   └── milky_way_map.html
├── download_data.py      # 1. pakāpe
├── process_data.py       # 2. pakāpe
├── spectral_analysis.py  # 3. pakāpe (pēc izvēles)
├── create_map.py         # 4. pakāpe
├── run_pipeline.py       # 5. pakāpe

```

---

## 3 · Ātra palaišana

### 3.1 Programmatūras prasības

| Komponents | Pārbaudītā versija |
|------------|--------------------|
| Python     | 3.10 (.11) |
| NumPy      | ≥ 1.26 |
| Pandas     | ≥ 2.2 |
| AstroPy    | ≥ 6.0 |
| Astroquery | ≥ 0.4 |
| PyVO       | ≥ 1.4 |
| SciPy      | ≥ 1.11 |
| Matplotlib | 3.7.x |
| Plotly     | ≥ 5.18 |

```bash
pip install numpy==1.26.* pandas==2.2.* astropy==6.*             astroquery pyvo scipy matplotlib==3.7.* plotly requests
```
*(Papildu `tikzplotlib`, ja nepieciešams plūsmas diagrammas PDF.)*

### 3.2 Pilna cauruļvada palaišana

```bash
# 1) Lejupielāde: pilns garums, |b| < 20°, G < 18
python download_data.py --gal-l 0 360 --gal-b -20 20 --gmag-max 18

# 2) Paralakšu attīrīšana un distance
python process_data.py --ext web

# 3) (pēc izvēles) XP spektru indeksu aprēķins
python spectral_analysis.py

# 4) Vokseļa kubs un interaktīvā karte
python create_map.py

# 5) Kļūdu kubs un attēli
python run_pipeline.py
```

Rezultātus skatīt `output/milky_way_map.html` un PNG failos.

---

## 4 · Citas debess zonas

```bash
python download_data.py --gal-l 0 360 --gal-b -30 30 --gmag-max 17
```

---

## 5 · Citēšana

> A. Cvečkovskis, **“Piena Ceļa diska trīsdimensiju blīvuma kartēšana ar Gaia DR3,”** Semestra darbs, Latvijas Universitāte, 2025.

---

## 6 · Licence

Avota kods — MIT licence.  
Gaia dati © ESA, izmanto saskaņā ar Gaia datu politiku.
