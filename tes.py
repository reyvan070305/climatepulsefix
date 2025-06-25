# === ClimatePulse: Chatbot Analisis Opini Publik ===
import torch
import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from geopy.geocoders import Nominatim
from datetime import datetime
import os
import tempfile



st.set_page_config(page_title="ClimatePulse", layout="wide")


device = 0 if torch.cuda.is_available() else -1


sent_tokenizer = AutoTokenizer.from_pretrained("mdhugol/indonesia-bert-sentiment-classification")
sent_model = AutoModelForSequenceClassification.from_pretrained("mdhugol/indonesia-bert-sentiment-classification")
pipe_sent = pipeline("sentiment-analysis", model=sent_model, tokenizer=sent_tokenizer)


pipe_emo = pipeline("sentiment-analysis", model="azizp128/prediksi-emosi-indobert", device=device)


ner_tokenizer = AutoTokenizer.from_pretrained("cahya/bert-base-indonesian-NER")
ner_model = AutoModelForTokenClassification.from_pretrained("cahya/bert-base-indonesian-NER")
pipe_ner = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

label_map = {'LABEL_0': 'Positif', 'LABEL_1': 'Netral', 'LABEL_2': 'Negatif'}

page_bg = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Blue_Marble_2002.png/800px-Blue_Marble_2002.png");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    color: white;
    display: flex;
    justify-content: center;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #1f2937;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    font-family: "Segoe UI", sans-serif;
    max-width: 1200px;
    margin: auto;
}
h1, h2, h3, h4, h5 {
    font-family: 'Segoe UI', sans-serif;
    color: #10B981;
}
.stButton>button {
    background-color: #10B981;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    border: none;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #059669;
}
.stTextInput>div>div>input, .stTextArea>div>textarea {
    background-color: #1f2937;
    color: white;
    border-radius: 6px;
    border: 1px solid #374151;
}
</style>
'''

st.markdown(page_bg, unsafe_allow_html=True)


import base64

with open("logo.png", "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
    <img src='data:image/png;base64,{logo_base64}' width='60'>
    <div>
        <h2 style='margin: 0; color: #10B981;'>ClimatePulse</h2> 
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown(f"""
<div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
    <div>
        <h2 style='color: white; margin-top: 0;'>Perubahan Iklim di Media Sosial</h2> 
        <p style='color: gray;'>Telusuri opini publik, sentimen, emosi, dan entitas terkait kebijakan dan bencana iklim</p>
    </div>
</div>
""", unsafe_allow_html=True)



st.markdown("### 💬 Input Teks")
text_input = ""
submit = False

with st.form(key="input_form"):
    text_input = st.text_area("Input Teks / Tweet", placeholder="Contoh: PLTN dibangun di Papua, saya takut dan kecewa", height=120)
    submit_text = st.form_submit_button("🔍 ANALISIS TEKS")


if submit_text:
    submit = True


st.markdown("### 🧠 Contoh Kalimat untuk Dicoba")
st.markdown("""
<ul style='line-height: 1.6; font-size: 1rem;'>
    <li>PLTN dibangun di Papua, saya takut dan kecewa</li>
    <li>Banjir di Jakarta makin parah tiap tahun</li>
    <li>Perlu lebih banyak edukasi soal perubahan iklim</li>
    <li>Warga Makassar mulai sadar pentingnya energi bersih</li>
    <li>Cuaca ekstrem membuat hasil panen petani gagal</li>
    <li>Senang lihat pemerintah tanam pohon di Kalimantan</li>
</ul>
""", unsafe_allow_html=True)


if submit and text_input.strip():
    with st.spinner("Menganalisis opini publik..."):
        sent = pipe_sent(text_input)[0]
        sent_label = label_map.get(sent['label'], sent['label'])
        emo = pipe_emo(text_input)[0]['label'].capitalize()
        ner = pipe_ner(text_input)
        ents = [e['word'] for e in ner]

        lokasi_kunci = [

    "sumatera", "jawa", "kalimantan", "sulawesi", "papua", "maluku", "nusa tenggara", "kepulauan seribu",

  
    "aceh", "sumatera utara", "sumatera barat", "riau", "kepulauan riau", "jambi", "bengkulu",
    "sumatera selatan", "bangka belitung", "lampung",
    "banten", "dki jakarta", "jawa barat", "jawa tengah", "daerah istimewa yogyakarta", "jawa timur",
    "bali", "nusa tenggara barat", "nusa tenggara timur",
    "kalimantan barat", "kalimantan tengah", "kalimantan selatan", "kalimantan timur", "kalimantan utara",
    "sulawesi utara", "sulawesi tengah", "sulawesi selatan", "sulawesi tenggara", "gorontalo", "sulawesi barat",
    "maluku", "maluku utara",
    "papua", "papua barat", "papua selatan", "papua tengah", "papua pegunungan", "papua barat daya",


    "banda aceh", "medan", "padang", "pekanbaru", "tanjungpinang", "jambi", "bengkulu",
    "palembang", "pangkalpinang", "bandar lampung",
    "serang", "jakarta", "bandung", "semarang", "yogyakarta", "surabaya",
    "denpasar", "mataram", "kupang",
    "pontianak", "palangka raya", "banjarmasin", "samarinda", "tarakan",
    "manado", "palu", "makassar", "kendari", "gorontalo", "mamuju",
    "ambon", "ternate",
    "jayapura", "manokwari", "merauke", "nabire", "wamena", "fakfak", "sorong", "timika",

    "bekasi", "bogor", "depok", "tangerang", "cirebon", "tegal", "purwokerto", "solo", "magelang",
    "malang", "kediri", "sidoarjo", "pasuruan", "probolinggo", "lumajang", "blitar", "jember",
    "banyuwangi", "cilacap", "padangsidimpuan", "binjai", "sibolga", "lubuklinggau", "palopo",
    "parepare", "bitung", "tomohon", "kotamobagu", "kotabaru", "pangkalan bun", "ketapang",
    "palu", "baubau", "karangasem", "buleleng", "labuan bajo", "ende", "bima", "dompu",

    # === Lokasi Baru / Khusus / Otorita ===
    "nusantara",  # Ibu kota negara baru di Kaltim
    "penajam paser utara", "balikpapan", "samarinda", "bontang",  # Kaltim area
    "kepri", "ntb", "ntt", "kaltim", "kalteng", "kalsel", "kalbar", "kaltara",  # singkatan populer

    # === Lokasi Adat/Kultural (yang sering disebut) ===
    "minangkabau", "batak", "dayak", "asmat", "ambon", "bugis", "toraja", "sunda", "madura", "tapanuli"
]

        locs = []
        for e in ner:
            ent_text = e['word'].lower()
            if e['entity_group'] == 'LOC':
                locs.append(e['word'])
            else:
                for keyword in lokasi_kunci:
                    if keyword in ent_text:
                        locs.append(keyword.capitalize())
        locs = list(set(locs))

        geolocator = Nominatim(user_agent="climatepulse")
        geo_locs = []
        for loc in locs:
            try:
                location = geolocator.geocode(loc)
                if location:
                    geo_locs.append({
                        'lokasi': loc,
                        'lat': location.latitude,
                        'lon': location.longitude,
                        'jumlah': 1
                    })
            except:
                continue

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_log_single = pd.DataFrame([{
            "timestamp": now,
            "text": text_input,
            "sentimen": sent_label,
            "emosi": emo
        }])

        log_file = "log_tren.csv"
        if os.path.exists(log_file):
            pd.concat([pd.read_csv(log_file), df_log_single]).to_csv(log_file, index=False)
        else:
            df_log_single.to_csv(log_file, index=False)


    emoji_map = {
    "Senang": "😊",
    "Sedih": "😢",
    "Marah": "😡",
    "Takut": "😨",
    "Jijik": "🤢",
    "Cinta": "🥰"  
}

    
    st.markdown(f"""
<div style='background-color: #1f2937; padding: 1rem; border-radius: 10px;'>
    <h3 style='color: white;'>Hasil Analisis</h3>
    <p><b>Sentimen:</b> <span style='color: red;'>{sent_label}</span>  |  
       <b>Emosi:</b> <span style='color: #facc15;'>{emo} {emoji_map.get(emo, '')}</span></p>
    <p><b>📍 Lokasi:</b> {', '.join(locs) or "Tidak ditemukan"}</p>
    <p><b>🔖 Entitas:</b> {', '.join(ents) or "Tidak ditemukan"}</p>
</div>
""", unsafe_allow_html=True)

   
if os.path.exists("log_tren.csv"):
    df_log = pd.read_csv("log_tren.csv")
    lokasi_kunci = [

    "sumatera", "jawa", "kalimantan", "sulawesi", "papua", "maluku", "nusa tenggara", "kepulauan seribu",


    "aceh", "sumatera utara", "sumatera barat", "riau", "kepulauan riau", "jambi", "bengkulu",
    "sumatera selatan", "bangka belitung", "lampung",
    "banten", "dki jakarta", "jawa barat", "jawa tengah", "daerah istimewa yogyakarta", "jawa timur",
    "bali", "nusa tenggara barat", "nusa tenggara timur",
    "kalimantan barat", "kalimantan tengah", "kalimantan selatan", "kalimantan timur", "kalimantan utara",
    "sulawesi utara", "sulawesi tengah", "sulawesi selatan", "sulawesi tenggara", "gorontalo", "sulawesi barat",
    "maluku", "maluku utara",
    "papua", "papua barat", "papua selatan", "papua tengah", "papua pegunungan", "papua barat daya",

    "banda aceh", "medan", "padang", "pekanbaru", "tanjungpinang", "jambi", "bengkulu",
    "palembang", "pangkalpinang", "bandar lampung",
    "serang", "jakarta", "bandung", "semarang", "yogyakarta", "surabaya",
    "denpasar", "mataram", "kupang",
    "pontianak", "palangka raya", "banjarmasin", "samarinda", "tarakan",
    "manado", "palu", "makassar", "kendari", "gorontalo", "mamuju",
    "ambon", "ternate",
    "jayapura", "manokwari", "merauke", "nabire", "wamena", "fakfak", "sorong", "timika",


    "bekasi", "bogor", "depok", "tangerang", "cirebon", "tegal", "purwokerto", "solo", "magelang",
    "malang", "kediri", "sidoarjo", "pasuruan", "probolinggo", "lumajang", "blitar", "jember",
    "banyuwangi", "cilacap", "padangsidimpuan", "binjai", "sibolga", "lubuklinggau", "palopo",
    "parepare", "bitung", "tomohon", "kotamobagu", "kotabaru", "pangkalan bun", "ketapang",
    "palu", "baubau", "karangasem", "buleleng", "labuan bajo", "ende", "bima", "dompu",


    "nusantara", 
    "penajam paser utara", "balikpapan", "samarinda", "bontang",  # Kaltim area
    "kepri", "ntb", "ntt", "kaltim", "kalteng", "kalsel", "kalbar", "kaltara",  # singkatan populer

    
    "minangkabau", "batak", "dayak", "asmat", "ambon", "bugis", "toraja", "sunda", "madura", "tapanuli"
]
    lokasi_counter = {}
    for text in df_log['text']:
        for keyword in lokasi_kunci:
            if keyword in text.lower():
                lokasi = keyword.capitalize()
                lokasi_counter[lokasi] = lokasi_counter.get(lokasi, 0) + 1

    geo_locs = []
    geolocator = Nominatim(user_agent="climatepulse-map")
    for lokasi, jumlah in lokasi_counter.items():
        try:
            location = geolocator.geocode(lokasi)
            if location:
                geo_locs.append({
                    'lokasi': lokasi,
                    'lat': location.latitude,
                    'lon': location.longitude,
                    'jumlah': jumlah
                })
        except:
            continue

    if geo_locs:
        map_df = pd.DataFrame(geo_locs)
        st.markdown("### 🗺️ Peta Opini Publik")
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=-2.5, longitude=117.0, zoom=4, pitch=0),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position='[lon, lat]',
                    get_color='[255, 100, 100, 160]',
                    get_radius='jumlah * 10000',
                    pickable=True,
                    auto_highlight=True
                )
            ],
            tooltip={"text": "{lokasi}: {jumlah} opini"}
        ))
    else:
        st.info("❗ Tidak ada lokasi yang berhasil dipetakan dari histori log.")


    st.markdown("### 📈 Tren Waktu Sentimen")
    if os.path.exists("log_tren.csv"):
        df_log = pd.read_csv("log_tren.csv")
        df_log['timestamp'] = pd.to_datetime(df_log['timestamp'])
        df_log['tanggal'] = df_log['timestamp'].dt.date
        trend_all = df_log.groupby(['tanggal', 'sentimen']).size().reset_index(name='jumlah')
        chart = alt.Chart(trend_all).mark_line(point=True).encode(
            x='tanggal:T',
            y='jumlah:Q',
            color='sentimen:N'
        ).properties(width=600)
        st.altair_chart(chart, use_container_width=True)


st.markdown("---")
st.markdown("### 📥 Analisis CSV Massal")
uploaded_file = st.file_uploader("Upload file CSV berisi kolom 'text'", type=["csv"])

if uploaded_file is not None:
    df_csv = pd.read_csv(uploaded_file)
    st.write("Pratinjau Data:", df_csv.head())

    if "text" in df_csv.columns:
        result_data = []
        geo_locs = []
        log_rows = []
        lokasi_kunci = [

    "sumatera", "jawa", "kalimantan", "sulawesi", "papua", "maluku", "nusa tenggara", "kepulauan seribu",


    "aceh", "sumatera utara", "sumatera barat", "riau", "kepulauan riau", "jambi", "bengkulu",
    "sumatera selatan", "bangka belitung", "lampung",
    "banten", "dki jakarta", "jawa barat", "jawa tengah", "daerah istimewa yogyakarta", "jawa timur",
    "bali", "nusa tenggara barat", "nusa tenggara timur",
    "kalimantan barat", "kalimantan tengah", "kalimantan selatan", "kalimantan timur", "kalimantan utara",
    "sulawesi utara", "sulawesi tengah", "sulawesi selatan", "sulawesi tenggara", "gorontalo", "sulawesi barat",
    "maluku", "maluku utara",
    "papua", "papua barat", "papua selatan", "papua tengah", "papua pegunungan", "papua barat daya",


    "banda aceh", "medan", "padang", "pekanbaru", "tanjungpinang", "jambi", "bengkulu",
    "palembang", "pangkalpinang", "bandar lampung",
    "serang", "jakarta", "bandung", "semarang", "yogyakarta", "surabaya",
    "denpasar", "mataram", "kupang",
    "pontianak", "palangka raya", "banjarmasin", "samarinda", "tarakan",
    "manado", "palu", "makassar", "kendari", "gorontalo", "mamuju",
    "ambon", "ternate",
    "jayapura", "manokwari", "merauke", "nabire", "wamena", "fakfak", "sorong", "timika",


    "bekasi", "bogor", "depok", "tangerang", "cirebon", "tegal", "purwokerto", "solo", "magelang",
    "malang", "kediri", "sidoarjo", "pasuruan", "probolinggo", "lumajang", "blitar", "jember",
    "banyuwangi", "cilacap", "padangsidimpuan", "binjai", "sibolga", "lubuklinggau", "palopo",
    "parepare", "bitung", "tomohon", "kotamobagu", "kotabaru", "pangkalan bun", "ketapang",
    "palu", "baubau", "karangasem", "buleleng", "labuan bajo", "ende", "bima", "dompu",

    "nusantara",  
    "penajam paser utara", "balikpapan", "samarinda", "bontang",  # Kaltim area
    "kepri", "ntb", "ntt", "kaltim", "kalteng", "kalsel", "kalbar", "kaltara",  # singkatan populer


    "minangkabau", "batak", "dayak", "asmat", "ambon", "bugis", "toraja", "sunda", "madura", "tapanuli"
]

        geolocator = Nominatim(user_agent="climatepulse")

        for i, row in df_csv.iterrows():
            text = str(row["text"])
            sent = pipe_sent(text)[0]
            sent_label = label_map.get(sent['label'], sent['label'])
            emo = pipe_emo(text)[0]['label'].capitalize()
            ner = pipe_ner(text)
            ents = [e['word'] for e in ner]

            locs = []
            for e in ner:
                ent_text = e['word'].lower()
                if e['entity_group'] == 'LOC':
                    locs.append(e['word'])
                else:
                    for keyword in lokasi_kunci:
                        if keyword in ent_text:
                            locs.append(keyword.capitalize())
            locs = list(set(locs))

            for loc in locs:
                try:
                    location = geolocator.geocode(loc)
                    if location:
                        geo_locs.append({
                            'lokasi': loc,
                            'lat': location.latitude,
                            'lon': location.longitude,
                            'jumlah': 1
                        })
                except:
                    continue

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_rows.append({"timestamp": now, "text": text, "sentimen": sent_label, "emosi": emo})

            result_data.append({
                "text": text,
                "sentimen": sent_label,
                "emosi": emo,
                "entitas": ", ".join(ents)
            })

        df_result = pd.DataFrame(result_data)
        st.success("Analisis selesai!")
        st.dataframe(df_result)

        csv_download = df_result.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil CSV", csv_download, "hasil_analisis.csv", "text/csv")

        log_file = "log_tren.csv"
        df_log_append = pd.DataFrame(log_rows)
        if os.path.exists(log_file):
            pd.concat([pd.read_csv(log_file), df_log_append]).to_csv(log_file, index=False)
        else:
            df_log_append.to_csv(log_file, index=False)


        
