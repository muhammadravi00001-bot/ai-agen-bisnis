"""
AI Agent Analisis Bisnis - Versi 2
Stack gratis: Streamlit (web) + Gemini API (otak agent) + pandas (hitungan)
"""

import json
import os

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ============================================================
# PENGATURAN
# ============================================================
MODEL = "gemini-flash-latest"  # selalu mengarah ke Flash terbaru; kalau error, cek nama model di Google AI Studio
KOLOM_WAJIB = ["produk", "terjual", "harga", "modal"]
KOLOM_URUT = ["terjual", "omzet", "keuntungan", "margin_persen"]

INSTRUKSI_AGENT = """Kamu adalah asisten analisis bisnis untuk pemilik usaha kecil di Indonesia.
Jawab dalam Bahasa Indonesia yang ringkas dan mudah dipahami.
Selalu gunakan tools untuk mengambil angka. Jangan menebak atau mengarang data.
Tampilkan uang dalam format Rupiah. Beri saran praktis berdasarkan data."""

DATA_CONTOH = pd.DataFrame(
    {
        "produk": ["Kaos", "Topi", "Tas"],
        "terjual": [20, 15, 10],
        "harga": [50000, 35000, 80000],
        "modal": [30000, 18000, 50000],
    }
)


# ============================================================
# BAGIAN 1: DATA & HITUNGAN
# ============================================================
def rupiah(angka: float) -> str:
    return "Rp" + f"{angka:,.0f}".replace(",", ".")


def siapkan_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validasi CSV, hitung omzet/keuntungan/margin, gabungkan produk yang sama."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    kurang = [k for k in KOLOM_WAJIB if k not in df.columns]
    if kurang:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(kurang))

    df["produk"] = df["produk"].astype(str).str.strip()
    for kolom in ["terjual", "harga", "modal"]:
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")
    df = df.dropna(subset=KOLOM_WAJIB)

    if df.empty:
        raise ValueError("Tidak ada baris data yang valid.")

    df["omzet"] = df["terjual"] * df["harga"]
    df["keuntungan"] = df["terjual"] * (df["harga"] - df["modal"])

    hasil = df.groupby("produk", as_index=False)[
        ["terjual", "omzet", "keuntungan"]
    ].sum()
    hasil["margin_persen"] = (
        hasil["keuntungan"] / hasil["omzet"].where(hasil["omzet"] != 0) * 100
    ).round(2)
    return hasil


def ke_python(df: pd.DataFrame) -> list:
    """Ubah DataFrame jadi list biasa supaya aman dikirim ke Gemini."""
    return json.loads(df.to_json(orient="records"))


# ============================================================
# BAGIAN 2: TOOLS UNTUK AGENT
# Docstring di tiap fungsi dibaca Gemini untuk memutuskan
# kapan fungsi itu dipakai, jadi tulis dengan jelas.
# ============================================================
def buat_tools(data: pd.DataFrame) -> list:
    def ringkasan_bisnis() -> dict:
        """Ambil ringkasan bisnis: total omzet, total keuntungan, margin keseluruhan, dan jumlah produk."""
        total_omzet = float(data["omzet"].sum())
        total_untung = float(data["keuntungan"].sum())
        margin = (total_untung / total_omzet * 100) if total_omzet else 0.0
        return {
            "total_omzet": total_omzet,
            "total_keuntungan": total_untung,
            "margin_persen": round(margin, 2),
            "jumlah_produk": int(len(data)),
        }

    def peringkat_produk(urut_berdasarkan: str = "keuntungan", jumlah: int = 3) -> dict:
        """Ambil peringkat produk teratas.

        Args:
            urut_berdasarkan: salah satu dari 'terjual', 'omzet', 'keuntungan', 'margin_persen'.
            jumlah: banyaknya produk yang diambil.
        """
        if urut_berdasarkan not in KOLOM_URUT:
            return {"error": "urut_berdasarkan harus salah satu dari: " + ", ".join(KOLOM_URUT)}
        atas = data.sort_values(urut_berdasarkan, ascending=False).head(max(1, jumlah))
        return {"urut_berdasarkan": urut_berdasarkan, "produk": ke_python(atas)}

    def detail_produk(nama_produk: str) -> dict:
        """Ambil data lengkap satu produk berdasarkan namanya.

        Args:
            nama_produk: nama produk, misalnya 'Kaos'.
        """
        cocok = data[data["produk"].str.lower() == nama_produk.strip().lower()]
        if cocok.empty:
            return {"error": f"Produk '{nama_produk}' tidak ditemukan.", "daftar_produk": data["produk"].tolist()}
        return ke_python(cocok)[0]

    return [ringkasan_bisnis, peringkat_produk, detail_produk]


# ============================================================
# BAGIAN 3: AGENT (GEMINI)
# ============================================================
def tanya_agent(client, tools: list, riwayat: list, pertanyaan: str) -> str:
    isi = [
        types.Content(
            role="user" if pesan["role"] == "user" else "model",
            parts=[types.Part(text=pesan["isi"])],
        )
        for pesan in riwayat[-10:]
    ]
    isi.append(types.Content(role="user", parts=[types.Part(text=pertanyaan)]))

    respons = client.models.generate_content(
        model=MODEL,
        contents=isi,
        config=types.GenerateContentConfig(
            system_instruction=INSTRUKSI_AGENT,
            tools=tools,
            temperature=0.3,
        ),
    )
    return respons.text or "Maaf, agent tidak memberi jawaban. Coba ulangi pertanyaannya."


# ============================================================
# BAGIAN 4: TAMPILAN WEB
# ============================================================
def ambil_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input(
        "Gemini API Key", type="password"
    )


def main():
    st.set_page_config(page_title="AI Agent Bisnis", page_icon="📊", layout="wide")
    st.title("📊 AI Agent Analisis Bisnis")

    # ---- Sidebar: sumber data & API key ----
    st.sidebar.header("Pengaturan")
    api_key = ambil_api_key()
    file_csv = st.sidebar.file_uploader("Upload CSV penjualan", type="csv")
    st.sidebar.caption("Kolom wajib: produk, terjual, harga, modal")
    pakai_contoh = st.sidebar.checkbox("Pakai data contoh")

    if file_csv is not None:
        mentah = pd.read_csv(file_csv)
    elif pakai_contoh:
        mentah = DATA_CONTOH
    else:
        st.info("Upload CSV di sidebar, atau centang 'Pakai data contoh' untuk mencoba.")
        st.stop()

    try:
        data = siapkan_data(mentah)
    except ValueError as galat:
        st.error(str(galat))
        st.stop()

    # ---- Dashboard ----
    total_omzet = data["omzet"].sum()
    total_untung = data["keuntungan"].sum()
    margin = (total_untung / total_omzet * 100) if total_omzet else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Omzet", rupiah(total_omzet))
    k2.metric("Total Keuntungan", rupiah(total_untung))
    k3.metric("Margin", f"{margin:.1f}%")

    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader("Keuntungan per Produk")
        st.bar_chart(data.set_index("produk")["keuntungan"])
    with kanan:
        st.subheader("Detail Produk")
        st.dataframe(data, hide_index=True, use_container_width=True)

    # ---- Chat dengan agent ----
    st.divider()
    st.subheader("💬 Tanya Agent")

    if not api_key:
        st.warning("Masukkan Gemini API Key di sidebar untuk mengaktifkan chat.")
        st.stop()

    if "riwayat" not in st.session_state:
        st.session_state.riwayat = []

    if st.sidebar.button("Hapus riwayat chat"):
        st.session_state.riwayat = []

    for pesan in st.session_state.riwayat:
        with st.chat_message(pesan["role"]):
            st.markdown(pesan["isi"])

    pertanyaan = st.chat_input("Contoh: produk mana yang paling untung dan apa saranmu?")
    if pertanyaan:
        with st.chat_message("user"):
            st.markdown(pertanyaan)

        with st.chat_message("assistant"):
            with st.spinner("Agent sedang menganalisis..."):
                try:
                    client = genai.Client(api_key=api_key)
                    jawaban = tanya_agent(
                        client, buat_tools(data), st.session_state.riwayat, pertanyaan
                    )
                except Exception as galat:
                    jawaban = (
                        "Terjadi masalah saat menghubungi Gemini. "
                        "Cek API key, atau tunggu sebentar kalau batas gratis tercapai.\n\n"
                        f"Detail: {galat}"
                    )
            st.markdown(jawaban)

        st.session_state.riwayat.append({"role": "user", "isi": pertanyaan})
        st.session_state.riwayat.append({"role": "assistant", "isi": jawaban})


if __name__ == "__main__":
    main()
