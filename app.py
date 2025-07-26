import streamlit as st
from supabase import create_client, Client
import pandas as pd
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


if not all([SUPABASE_URL, SUPABASE_KEY, OPENROUTER_API_KEY]):
    st.error("API anahtarları eksik. .env dosyasını kontrol edin.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

st.set_page_config(
    page_title="Satış Analiz Botu",
    page_icon="📊",
    layout="centered"
)

st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #003366, #ffffff);
            height: 100vh;
            margin: 0;
        }
        .main {
            background: transparent;
        }
        h1 {
            background: linear-gradient(90deg, rgba(2, 0, 36, 1) 0%, rgba(9, 9, 121, 1) 35%, rgba(0, 212, 255, 1) 100%);
            padding: 1rem;
            border-radius: 16px;
            color: white !important;
            text-align: center;
            box-shadow: 0px 4px 16px rgba(0,0,0,0.05);
        }
        .stButton > button {
            background-color: #003366;
            color: white;
            border-radius: 10px;
            font-weight: bold;
            padding: 0.6rem 1.5rem;
            border: none;
        }
        .stTextInput > div > input {
            border-radius: 10px;
            border: 1px solid #cbd5e0;
        }
        .input-button-container {
            display: flex;
            gap: 1rem;
            max-width: 700px;
            margin-top: 20px;
        }
        .input-button-container > div:first-child {
            flex-grow: 1;
        }
        .input-button-container > div:last-child {
            flex: 0 0 120px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Satış Analiz Botu</h1>", unsafe_allow_html=True)

if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = []
if "response_history" not in st.session_state:
    st.session_state["response_history"] = []

sidebar = st.sidebar
sidebar.header("Komut Geçmişi")
for i, p in enumerate(reversed(st.session_state["prompt_history"][-20:]), 1):
    sidebar.write(f"{i}. {p[:100]}...")

# sidebar.header("Son 5 Analiz Sonucu")
# for i, r in enumerate(reversed(st.session_state["response_history"][-5:]), 1):
#     sidebar.write(f"{i}. {r[:100]}...")

with st.container():
    st.markdown('<div class="input-button-container">', unsafe_allow_html=True)
    user_prompt = st.text_input("Analiz komutunuzu yazın", key="user_prompt", label_visibility="collapsed", placeholder="Örneğin: En çok satan ürünleri listele")
    analyze_clicked = st.button("Analiz Et")
    st.markdown("</div>", unsafe_allow_html=True)

def get_sales_data():
    """Supabase'den tüm satış verilerini çeker."""
    try:
        response = supabase.from_("sales").select("*").execute()
        
        if not response.data:
            st.warning("Veri bulunamadı")
            return None
            
        sales_data = pd.DataFrame(response.data)
        print(f"Veri çekildi: {len(sales_data)} satır") 
        return sales_data
        
    except Exception as e:
        st.error(f"Hata: {str(e)}")
        return None

def save_response_to_supabase(prompt, response):
    """Analiz sonuçlarını Supabase'e kaydeder."""
    try:
        supabase.table("responses").insert({
            "user_prompt": prompt,
            "response": response,
            "created_at": pd.Timestamp.now().isoformat()
        }).execute()
    except Exception as e:
        st.error(f"Cevap kaydetme hatası: {str(e)}")

def get_last_5_responses():
    """Supabase'den son 5 cevabı çeker."""
    try:
        response = supabase.table("responses").select("response").order("created_at", desc=True).limit(5).execute()
        return [r["response"] for r in response.data]
    except Exception as e:
        st.error(f"Son cevapları alma hatası: {str(e)}")
        return []

def analyze_with_deepseek(data: pd.DataFrame, prompt: str):
    """DeepSeek ile satış verilerini analiz eder."""
    try:
        data_str = data.to_markdown(index=False)
        last_5_responses = st.session_state.get("response_history", [])[-5:]
        response_context = "\n\n### Önceki 5 Analiz Sonucu:\n" + "\n".join([f"{i+1}. {r[:200]}..." for i, r in enumerate(last_5_responses)]) if last_5_responses else "Önceki analiz sonucu yok."
        
        full_prompt = f"""
        Sen bir profesyonel satış analiz asistanısın. Görevin, yalnızca verilen satış verisi tablosuna dayanarak kullanıcının komutunu analiz etmek ve sonuçları net, anlaşılır ve profesyonel bir Türkçe ile sunmak. Aşağıdaki kurallara sıkı sıkıya uy:

        ### Kurallar:
        1. **Veriye Bağlılık**: Analizini yalnızca verilen tablodaki verilere dayandır. Tabloda olmayan hiçbir bilgiyi varsayma veya uydurma.
        2. **Analiz Türleri**: 
           - Tabloda varsa `adet` ,`urun`, `fiyat`, `tarih`, `temsilci` gibi sütunlara göre:
             - Toplam satış tutarı, ortalama satış, en çok satan ürünler, satış yoğunluğu gibi temel metrikleri hesapla.
             - Zaman bazlı trendler (gün, hafta, ay, yıl gibi) analiz et (eğer tarih verisi varsa).
             - Ürün veya kategori bazlı karşılaştırmalar yap.
             - Satış performansını etkileyen faktörleri (ör. popüler ürünler, düşük performanslı kategoriler) belirle.
        3. **Eksik Veri**: Tabloda eksik veri varsa bunu açıkça belirt, ancak analizini engelleme; mevcut verilerle en iyi sonucu üret.
        4. **Çıktı Formatı**: 
           - Sonuçları düzenli, yapılandırılmış ve profesyonel bir şekilde sun:
             - **Başlık**: Analizin ana konusunu özetleyen bir başlık.
             - **Özet**: Kullanıcının komutuna yanıt olarak kısa bir özet (2-3 cümle).
             - **Detaylı Bulgular**: Liste veya paragraflarla, hesaplamalar ve bulguları net bir şekilde açıkla.
             - **Öneriler**: Analize dayalı 1-2 uygulanabilir iş önerisi sun.
             - **Görselleştirme Önerisi**: Veriye uygun grafik türleri öner (ör. "Bu veri için bir çubuk grafik uygun olur").
        5. **Dil ve Ton**: Profesyonel, sade ve rehber bir dil kullan. Karmaşık terimlerden kaçın, her seviyeden kullanıcının anlayabileceği şekilde yaz.
        6. **Hata Kontrolü**: Veride anormallikler (negatif fiyat, mantıksız tarihler vb.) varsa bunları belirt ve analizini buna göre uyar.
        7. **Bağlam**: Önceki analiz sonuçlarını dikkate alarak tutarlı bir analiz yap.

        ### Kullanıcının Komutu:
        {prompt}

        ### Satış Verisi (Markdown formatında):
        {data_str}

        ### Önceki Analiz Bağlamı:
        {response_context}

        ### Çıktı Yapısı:
        - **Analiz Başlığı**: [Komutun özeti]
        - **Özet**: [Komutun neyi istediğini ve ana bulguları özetle]
        - **Detaylı Bulgular**: 
          - [Madde madde veya paragraflarla analiz sonuçları]
        - **İş Önerileri**: [Analize dayalı öneriler]
        - **Görselleştirme Önerisi**: [Veriye uygun grafik türü]

        Lütfen yukarıdaki kurallara uygun olarak analizini gerçekleştir ve sonuçları belirtilen formatta sun.
        """
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[
                {"role": "system", "content": "Sen yalnızca verilen satış verisine göre analiz yapan bir asistansın."},
                {"role": "user", "content": full_prompt}
            ],
            extra_headers={
                "HTTP-Referer": "https://sales-analytics.com",
                "X-Title": "Sales Analiz Botu"
            }
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Analiz hatası: {str(e)}"

if analyze_clicked:
    if user_prompt.strip() == "":
        st.warning("Lütfen bir analiz komutu girin.")
    else:
        sales_data = get_sales_data()
        if sales_data is not None:
            with st.spinner("Analiz yapılıyor..."):
                result = analyze_with_deepseek(sales_data, user_prompt)
                st.success("Analiz tamamlandı")
                st.markdown(result)
          
                st.session_state["response_history"].append(result)
                if len(st.session_state["response_history"]) > 5:
                    st.session_state["response_history"] = st.session_state["response_history"][-5:]
                st.session_state["prompt_history"].append(user_prompt)
                save_response_to_supabase(user_prompt, result)
        else:
            st.error("Veri alınamadı.")