import streamlit as st
import time
import io
from datetime import datetime

# ---------- Page config ----------
st.set_page_config(page_title="Anki CSV Builder — Quick Convert (DOS)", layout="centered")

# ---------- Retro MS-DOS CSS ----------
st.markdown("""
<style>
:root{
  --bg:#000000; --fg:#00FF66; --fg-dim:#59ffa1; --accent:#00ff99; --err:#ff3b3b;
  --box:#00AA55;
}
* { font-family: "Cascadia Mono", Consolas, "Fira Code", Menlo, Monaco, monospace !important; }
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--fg) !important; }
a { color: var(--accent) !important; text-decoration: none; }
.block-container { padding-top: 1.2rem; max-width: 860px; }
hr { border-color: var(--box); }

/* Header frame */
.header-box{
  border: 1px solid var(--box);
  padding: .75rem 1rem; margin-bottom: .75rem;
  box-shadow: 0 0 0 2px rgba(0,170,85,.15) inset;
}
.header-title{ font-weight: 700; letter-spacing: .5px; }
.nav-inline{
  margin-top:.25rem; display:flex; gap:.75rem; flex-wrap:wrap;
  color: var(--fg-dim);
}
.nav-chip{
  border:1px solid var(--box); padding:.15rem .5rem; border-radius: 2px;
  background: rgba(0,255,153,0.05);
}
.nav-chip--active{ background: rgba(0,255,153,0.15); color: var(--fg); }

/* Sticky status bar */
.sticky{
  position: sticky; top: 0; z-index: 100;
  background: #001b12; padding:.35rem .6rem; margin:-.25rem 0 .75rem;
  border: 1px dashed var(--box);
  display:flex; justify-content: space-between; gap: .75rem; align-items:center;
}
.badge{ padding:.05rem .35rem; border:1px solid var(--box); border-radius:2px; }
.badge.err{ color: var(--err); border-color: var(--err); }

/* Panels */
.panel{
  border:1px solid var(--box); padding: .9rem; margin:.75rem 0; background: rgba(0,255,153,0.03);
}
.panel-title{ color: var(--fg); font-weight:700; margin-bottom:.5rem; }
small, .help{ color: var(--fg-dim); }
.stFileUploader label, .stDownloadButton > button, .stButton > button, .stCheckbox label, .stSelectbox label, .stTextInput label{
  color: var(--fg) !important;
}

/* Buttons */
.stButton > button, .stDownloadButton > button{
  background: rgba(0,255,153,0.08); border:1px solid var(--box);
  color: var(--fg); box-shadow:none; border-radius:2px;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: rgba(0,255,153,0.18); }

/* Expanders */
.streamlit-expanderHeader{ color: var(--fg) !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
ss = st.session_state
ss.setdefault("preset", "Default")
ss.setdefault("status", "Idle")
ss.setdefault("processed", 0)
ss.setdefault("total", 0)
ss.setdefault("errors", 0)
ss.setdefault("elapsed", 0.0)
ss.setdefault("results_ready", False)
ss.setdefault("csv_bytes", b"")
ss.setdefault("zip_bytes", b"")
ss.setdefault("last_run_at", None)

# ---------- Header ----------
st.markdown("""
<div class="header-box">
  <div class="header-title">Anki CSV Builder</div>
  <div class="nav-inline">
    <span class="nav-chip nav-chip--active">Quick Convert</span>
    <span class="nav-chip">Review</span>
    <span class="nav-chip">Export</span>
    <span class="nav-chip">Settings</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Sticky status ----------
def render_status():
    status_color = "badge" if ss.status != "Error" else "badge err"
    st.markdown(
        f"""
        <div class="sticky">
          <div>
            Preset: <span class="badge">{ss.preset}</span>
          </div>
          <div>
            Status: <span class="{status_color}">{ss.status}</span>
            &nbsp;•&nbsp;Processed: <span class="badge">{ss.processed}/{ss.total}</span>
            &nbsp;•&nbsp;Errors: <span class="badge">{ss.errors}</span>
            &nbsp;•&nbsp;Time: <span class="badge">{ss.elapsed:.1f}s</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

render_status()

# ---------- Quick Convert panel ----------
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">① Загрузка файла</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Перетащите TXT/CSV/ZIP или выберите файл", type=["txt", "csv", "zip"])
colA, colB = st.columns([1,1])
with colA:
    preset = st.selectbox("Preset", ["Default", "Exam-B1-Short", "Cloze-Strict"], index=0)
with colB:
    if st.button("Очистить"):
        ss.results_ready = False
        ss.csv_bytes = b""
        ss.zip_bytes = b""
        ss.status = "Idle"
        ss.processed = ss.total = ss.errors = 0
        ss.elapsed = 0.0
        ss.last_run_at = None
        st.experimental_rerun()

# Store preset to session
ss.preset = preset

# ---------- Options (collapsed) ----------
with st.expander("▸ Опции (по желанию)"):
    c1, c2 = st.columns([1,1])
    with c1:
        st.selectbox("Язык и уровень", ["Nederlands — B1-B2"], index=0)
        st.selectbox("Формат карт", ["Cloze", "Definities", "Mix"], index=0)
    with c2:
        st.selectbox("Экспорт", ["CSV (Anki); разделитель ';'"], index=0)
        st.selectbox("Озвучка (TTS)", ["Выкл (рекомендуется базовый NL-голос)"], index=0)
    st.caption("Совет: начните с базовых настроек. Продвинутые параметры живут в Settings.")

st.markdown('</div>', unsafe_allow_html=True)  # /panel

# ---------- Generate action ----------
start = st.button("Сгенерировать колоду", use_container_width=True)

if start:
    if uploaded is None:
        st.error("Сначала выберите файл.")
        ss.status = "Error"
    else:
        ss.status = "Processing"
        ss.processed = 0
        ss.errors = 0
        # Симулируем определение количества элементов (например, строк)
        # Для демо: каждые ~20 байт файла = 1 карточка, минимум 20, максимум 400
        file_bytes = uploaded.getvalue()
        est = max(20, min(400, len(file_bytes)//20 if len(file_bytes)>0 else 120))
        ss.total = est

        # Прогресс
        progress = st.progress(0, text="Обработка…")
        t0 = time.time()
        for i in range(est):
            time.sleep(0.01)  # демо-тайм
            ss.processed = i + 1
            # демо-ошибки
            if (i+1) % 77 == 0:
                ss.errors += 1
            progress.progress(int(100 * (i+1)/est), text=f"Обработка… {i+1}/{est}")
        ss.elapsed = time.time() - t0
        ss.status = "Completed"
        ss.last_run_at = datetime.now().strftime("%H:%M:%S")

        # Заготовки для скачивания
        csv_text = "Front;Back;Tags\nvoorbeeld;example;nederlands\n"
        ss.csv_bytes = csv_text.encode("utf-8")
        ss.zip_bytes = b"PK\x05\x06"  # пустая заглушка ZIP для вида
        ss.results_ready = True
        st.experimental_rerun()

# ---------- Results / Downloads ----------
if ss.results_ready and ss.status == "Completed":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">② Готово</div>', unsafe_allow_html=True)
    st.success(f"✓ Обработано: {ss.processed}/{ss.total} • Ошибок: {ss.errors} • Время: {ss.elapsed:.1f}s • {ss.last_run_at}")
    d1, d2, d3 = st.columns([1,1,1])
    with d1:
        st.download_button("⬇️ Скачать CSV", data=ss.csv_bytes, file_name="deck.csv", mime="text/csv", use_container_width=True)
    with d2:
        st.download_button("⬇️ Скачать ZIP (аудио)", data=ss.zip_bytes, file_name="audio.zip", mime="application/zip", use_container_width=True)
    with d3:
        st.button("→ Открыть Review", use_container_width=True, help="Переход на страницу проверки (заглушка)")
    st.caption("Подсказка: для настоящих озвучек и тонкой настройки — откройте Settings.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown('<span class="help">MS-DOS skin • Quick Convert prototype • Streamlit</span>', unsafe_allow_html=True)
