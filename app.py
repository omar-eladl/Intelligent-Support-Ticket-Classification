# -*- coding: utf-8 -*-
"""
Vodafone Egypt — Intelligent Support Ticket Classification with RAG
Streamlit UI — Graduation Project
"""

import re
import warnings
import collections
import os

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from typing import List
import arabic_reshaper
from bidi.algorithm import get_display
def ar(text):
    return get_display(arabic_reshaper.reshape(str(text)))

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vodafone Egypt — Smart Support",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
    --vodafone-red:   #E60000;
    --vodafone-dark:  #1A1A1A;
    --vodafone-mid:   #2D2D2D;
    --vodafone-light: #F5F5F5;
    --vodafone-white: #FFFFFF;
    --accent-gray:    #888888;
    --border:         rgba(230,0,0,0.18);
    --card-bg:        rgba(255,255,255,0.04);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: #111111 !important;
    font-family: 'Cairo', sans-serif;
    color: #F0F0F0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1A1A1A !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: #DDDDDD !important; font-family: 'Cairo', sans-serif; }

/* Headings */
h1, h2, h3, h4 { font-family: 'Cairo', sans-serif; font-weight: 900; }

/* Input */
.stTextArea textarea, .stTextInput input {
    background: #1E1E1E !important;
    color: #F0F0F0 !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 1.05rem !important;
    direction: rtl;
    text-align: right;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--vodafone-red) !important;
    box-shadow: 0 0 0 2px rgba(230,0,0,0.20) !important;
}

/* Buttons */
.stButton > button {
    background: var(--vodafone-red) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Cairo', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: #C50000 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(230,0,0,0.35) !important;
}

/* Cards */
.answer-card {
    background: linear-gradient(135deg, #1E1E1E 60%, #2A1010);
    border: 1px solid rgba(230,0,0,0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    direction: rtl;
    text-align: right;
    font-size: 1.12rem;
    line-height: 1.9;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}

.doc-card {
    background: #1A1A1A;
    border: 1px solid #2E2E2E;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.8rem;
    direction: rtl;
    text-align: right;
    font-size: 0.92rem;
    color: #CCCCCC;
    transition: border-color 0.2s;
}
.doc-card:hover { border-color: rgba(230,0,0,0.4); }

.doc-card .doc-score {
    display: inline-block;
    background: var(--vodafone-red);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    margin-left: 8px;
    font-family: 'IBM Plex Mono', monospace;
}

.category-badge {
    display: inline-block;
    background: rgba(230,0,0,0.15);
    border: 1px solid rgba(230,0,0,0.4);
    color: #FF6666;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 3px 12px;
    border-radius: 20px;
    margin-top: 6px;
    direction: rtl;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.metric-box {
    flex: 1;
    background: #1A1A1A;
    border: 1px solid #2E2E2E;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-box .metric-val {
    font-size: 1.8rem;
    font-weight: 900;
    color: var(--vodafone-red);
    font-family: 'IBM Plex Mono', monospace;
}
.metric-box .metric-lbl {
    font-size: 0.8rem;
    color: #888;
    margin-top: 2px;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1A1A1A 0%, #2A0A0A 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    direction: rtl;
}
.hero-logo {
    font-size: 3.5rem;
    flex-shrink: 0;
}
.hero-title {
    font-size: 1.8rem;
    font-weight: 900;
    color: white;
    margin: 0;
}
.hero-sub {
    font-size: 0.95rem;
    color: #888;
    margin: 0;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #2E2E2E;
    margin: 1.5rem 0;
}

/* Tabs */
.stTabs [role="tab"] {
    font-family: 'Cairo', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [role="tab"][aria-selected="true"] {
    color: var(--vodafone-red) !important;
    border-bottom-color: var(--vodafone-red) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--vodafone-red) !important; }

/* Slider */
.stSlider [data-baseweb="slider"] { direction: ltr; }

/* Expander */
.streamlit-expanderHeader {
    font-family: 'Cairo', sans-serif !important;
    font-weight: 700 !important;
    direction: rtl;
    background: #1A1A1A !important;
    border-radius: 10px !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #1E1E1E !important;
    border-color: var(--border) !important;
    color: #F0F0F0 !important;
    font-family: 'Cairo', sans-serif !important;
}

/* Radio */
.stRadio label { font-family: 'Cairo', sans-serif !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--vodafone-red); }

/* History item */
.history-item {
    background: #1A1A1A;
    border-left: 3px solid var(--vodafone-red);
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    direction: rtl;
    cursor: pointer;
    color: #CCC;
    transition: background 0.2s;
}
.history-item:hover { background: #222; }

/* Success/info boxes */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'Cairo', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ARABIC NLP UTILS
# ─────────────────────────────────────────────
ARABIC_STOPWORDS = frozenset({
    'على','إلى','عن','مع','هذا','هذه','التي','الذي','كان','كانت',
    'ان','أن','إن',"انا",'لو','لم','لا','ما','يا','هو','هي','هم','نحن','انت','أنت',
    'بتاع','بتاعت','اللي',
    'فيه','فيها','منه','منها','عليه','عليها','يكون','تكون','اكون',
    'بيكون','هيكون','دي','ده','دول','لما',
    'فال','بال','كل','كلها','بعد','قبل','مو','غير','جدا','قوي',
    'كتير','برضو','كمان','لي','لك','له','لها','لنا','لهم','بي','بك',
    'به','بها','بنا','بهم','كانوا','ليس','ليست','عند','عندك','عنده',
    'عندها','عندنا','الى','علي','التى','الذى','هذا','هذه','هؤلاء',
})

def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = ' '.join(text.split())
    text = text.lower()
    text = re.sub('[أإآٱ]', 'ا', text)
    text = re.sub('ى', 'ي', text)
    text = re.sub('ـ', '', text)
    text = re.sub('[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[^a-zA-Z\u0600-\u06FF0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list:
    return [t for t in text.split() if len(t) > 1]

def preprocess(text: str, remove_stopwords: bool = True) -> str:
    tokens = tokenize(normalize_arabic(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ARABIC_STOPWORDS]
    return ' '.join(tokens)


# ─────────────────────────────────────────────
# RETRIEVERS
# ─────────────────────────────────────────────
class TfidfRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5), min_df=1
        )
        self.doc_vectors = None
        self.docs_df = None

    def fit(self, docs_df: pd.DataFrame, text_col: str = 'retrieval_text'):
        self.docs_df = docs_df.reset_index(drop=True).copy()
        self.doc_vectors = self.vectorizer.fit_transform(self.docs_df[text_col].tolist())

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_vectors).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = self.docs_df.iloc[top_idx].copy()
        results['score'] = sims[top_idx]
        return results[['Questions', 'Answers', 'category', 'score']]


class DenseRetriever:
    def __init__(self, model_name='intfloat/multilingual-e5-base'):
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.docs_df = None

    def fit(self, docs_df: pd.DataFrame, text_col: str = 'retrieval_text'):
        self.docs_df = docs_df.reset_index(drop=True).copy()
        texts = ["passage: " + t for t in self.docs_df[text_col].fillna('').tolist()]
        self.doc_embeddings = self.model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
        )

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        q_emb = self.model.encode(
            ["query: " + query], convert_to_tensor=True, normalize_embeddings=True
        )
        scores = (q_emb @ self.doc_embeddings.T).cpu().numpy()[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = self.docs_df.iloc[top_idx].copy()
        results['score'] = scores[top_idx]
        return results[['Questions', 'Answers', 'category', 'score']]


class HybridRetriever:
    def __init__(self, sparse_retriever, dense_retriever, alpha=0.4):
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.alpha = alpha
        self.docs_df = None

    def fit(self, docs_df: pd.DataFrame, text_col: str = 'retrieval_text'):
        self.docs_df = docs_df.reset_index(drop=True).copy()
        self.sparse.fit(self.docs_df, text_col=text_col)
        self.dense.fit(self.docs_df, text_col=text_col)

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        all_k = len(self.docs_df)
        sparse_results = self.sparse.search(query, top_k=all_k).rename(columns={"score": "sparse_score"})
        dense_results = self.dense.search(query, top_k=all_k).rename(columns={"score": "dense_score"})

        merged = self.docs_df.copy()
        merged = merged.merge(
            sparse_results[["Questions", "Answers", "category", "sparse_score"]],
            on=["Questions", "Answers", "category"], how="left"
        )
        merged = merged.merge(
            dense_results[["Questions", "Answers", "category", "dense_score"]],
            on=["Questions", "Answers", "category"], how="left"
        )
        merged["sparse_score"] = merged["sparse_score"].fillna(0.0)
        merged["dense_score"] = merged["dense_score"].fillna(0.0)

        scaler = MinMaxScaler()
        merged[["sparse_score", "dense_score"]] = scaler.fit_transform(
            merged[["sparse_score", "dense_score"]]
        )
        merged["score"] = (1 - self.alpha) * merged["sparse_score"] + self.alpha * merged["dense_score"]
        merged = merged.sort_values("score", ascending=False).head(top_k).copy()
        return merged[["Questions", "Answers", "category", "score"]]


# ─────────────────────────────────────────────
# GENERATION (HF Inference API)
# ─────────────────────────────────────────────
def build_context(results: pd.DataFrame) -> str:
    chunks = []
    for i, (_, row) in enumerate(results.iterrows(), start=1):
        chunk = f"معلومة {i}:\nالسؤال: {row['Questions']}\nالإجابة: {row['Answers']}"
        chunks.append(chunk)
    return "\n\n".join(chunks)

def build_prompt(query: str, context: str) -> str:
    return f"""
أنت مساعد خدمة عملاء لشركة فودافون مصر.

اعتمد فقط على المعلومات الموجودة في قسم "المعلومات".

القواعد:

- استخدم المعلومات الموجودة فقط.
- لا تستخدم أي معرفة خارجية.
- إذا كانت إجابة السؤال موجودة داخل المعلومات حتى لو بصياغة مختلفة، استخدمها.
- إذا لم تجد أي معلومة تجيب على السؤال، قل فقط:
"للأسف، ملقتش إجابة للسؤال ده في البيانات المتاحة."
- لا تخترع أي معلومة.
- لا تكرر سؤال المستخدم.
- بعد استخراج الإجابة، اكتبها باللهجة المصرية بشكل طبيعي دون تغيير معناها أو إضافة معلومات جديدة.
- إذا كانت الإجابة عبارة عن خطوات، حافظ على ترتيبها.

المعلومات:
{context}

سؤال المستخدم:
{query}

الإجابة:""".strip()

def generate_with_hf(prompt: str, hf_token: str, max_tokens: int = 200) -> str:
    """Generate answer via HuggingFace Inference API."""
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(
            provider="featherless-ai",
            api_key=hf_token
        )

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "أنت مساعد خدمة عملاء لفودافون مصر. أجب بالعربية المصرية بشكل واضح ومختصر واعتمد فقط على المعلومات المقدمة."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.4,
            top_p=0.9,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ خطأ في توليد الإجابة: {e}"


# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────
SEED = 42

@st.cache_data(show_spinner=False)
def load_and_preprocess(csv_path: str):
    df_raw = pd.read_csv(csv_path).rename(columns={'Label': 'category'})
    df_raw['Questions'] = df_raw['Questions'].fillna('')
    df_raw['Answers'] = df_raw['Answers'].fillna('')
    df_raw['q_clean'] = df_raw['Questions'].apply(preprocess)
    df_raw['a_clean'] = df_raw['Answers'].apply(preprocess)
    df_raw['retrieval_text'] = (df_raw['q_clean'] + ' ' + df_raw['a_clean']).str.strip()
    df_raw['q_len'] = df_raw['Questions'].str.len()
    df_raw['a_len'] = df_raw['Answers'].str.len()
    return df_raw

@st.cache_resource(show_spinner=False)
def build_retrievers(csv_path: str):
    df = load_and_preprocess(csv_path)
    df = df[(df['q_clean'].str.strip() != '') & (df['a_clean'].str.strip() != '')].reset_index(drop=True)

    train_df, _ = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['category'])
    train_df = train_df.reset_index(drop=True)

    tfidf_r = TfidfRetriever()
    dense_r = DenseRetriever()
    hybrid_r = HybridRetriever(tfidf_r, dense_r, alpha=0.4)

    # Fit hybrid (fits both internally)
    hybrid_r.fit(train_df, text_col='retrieval_text')

    # Also fit standalone for individual mode
    tfidf_standalone = TfidfRetriever()
    tfidf_standalone.fit(train_df, text_col='retrieval_text')

    return {
        "TF-IDF":  tfidf_standalone,
        "Dense":   dense_r,
        "Hybrid":  hybrid_r,
        "train_df": train_df,
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-size:2.5rem;'>📡</div>
        <div style='font-size:1.1rem; font-weight:900; color:#E60000;'>Vodafone Egypt</div>
        <div style='font-size:0.75rem; color:#666; margin-top:2px;'>Smart Support System</div>
    </div>
    <hr style='border-color:#2E2E2E; margin: 0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ الإعدادات")

    csv_path = st.text_input("مسار ملف البيانات (CSV)", value="vodafone_faq.csv",
                              help="ضع مسار ملف vodafone_faq.csv")

    retriever_mode = st.selectbox(
        "نوع الاسترجاع",
        ["Hybrid (موصى به)", "Dense (دلالي)", "TF-IDF (نصي)"],
        index=0
    )
    mode_map = {
        "Hybrid (موصى به)": "Hybrid",
        "Dense (دلالي)":    "Dense",
        "TF-IDF (نصي)":    "TF-IDF",
    }
    selected_mode = mode_map[retriever_mode]

    top_k = st.slider("عدد المستندات المسترجعة (Top-K)", 1, 10, 3)

    use_generation = st.toggle("تفعيل توليد الإجابة (LLM)", value=False)
    hf_token = ""
    if use_generation:
        hf_token = st.text_input("HuggingFace Token", type="password",
                                  help="احتاجه للاتصال بـ Qwen2.5-3B-Instruct")
        max_tokens = st.slider("أقصى عدد توكنز للإجابة", 50, 300, 150)
    else:
        max_tokens = 150

    st.markdown("<hr style='border-color:#2E2E2E;'>", unsafe_allow_html=True)

    show_preprocessing = st.toggle("عرض تفاصيل المعالجة", value=False)
    show_scores = st.toggle("عرض درجات التشابه", value=True)

    st.markdown("<hr style='border-color:#2E2E2E;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#555; text-align:center; direction:rtl;'>
        مشروع التخرج · جامعة المنصورة<br>
        Arabic NLP · RAG · Vodafone FAQ
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-logo'>📡</div>
    <div>
        <div class='hero-title'>نظام تصنيف تذاكر الدعم الذكي</div>
        <div class='hero-sub'>Intelligent Support Ticket Classification with RAG · Vodafone Egypt FAQ</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA & RETRIEVERS
# ─────────────────────────────────────────────
data_ok = False
retrievers = {}

if os.path.exists(csv_path):
    with st.spinner("⏳ جاري تحميل البيانات وبناء الـ Retrievers ..."):
        try:
            df_data = load_and_preprocess(csv_path)
            retrievers = build_retrievers(csv_path)
            data_ok = True
        except Exception as e:
            st.error(f"❌ خطأ في تحميل البيانات: {e}")
else:
    st.warning(f"⚠️ ملف البيانات غير موجود في المسار: `{csv_path}` — تأكد من رفع `vodafone_faq.csv` في نفس مجلد التطبيق.")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_ask, tab_explore, tab_about = st.tabs(["💬 اسأل المساعد", "📊 استكشاف البيانات", "ℹ️ عن النظام"])


# ══════════════════════════════════════════════
# TAB 1 — ASK
# ══════════════════════════════════════════════
with tab_ask:
    col_main, col_hist = st.columns([3, 1], gap="large")

    with col_main:
        st.markdown("#### ✍️ اكتب سؤالك")
        query_input = st.text_area(
            label="",
            placeholder="مثال: ازاي أعمل إيداع في المحفظة؟  أو  النت عندي بطيء ليه؟",
            height=110,
            label_visibility="collapsed",
        )

        # Example queries
        st.markdown("<div style='direction:rtl; font-size:0.82rem; color:#666; margin-bottom:0.4rem;'>أمثلة سريعة:</div>", unsafe_allow_html=True)
        ex_cols = st.columns(3)
        examples = [
            "إيه أنظمة الكارت الحالية؟",
            "ازاي أشحن محفظة فودافون كاش؟",
            "إيه خدمة روم للتجوال الدولي؟",
        ]
        for i, (col, ex) in enumerate(zip(ex_cols, examples)):
            with col:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    query_input = ex
                    st.rerun()

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
        search_btn = st.button("🔍 ابحث عن إجابة", use_container_width=True)

        if search_btn and query_input.strip():
            if not data_ok:
                st.error("البيانات غير محملة. تحقق من مسار الملف في الشريط الجانبي.")
            else:
                with st.spinner("جارٍ البحث ..."):
                    retriever = retrievers[selected_mode]
                    results = retriever.search(query_input.strip(), top_k=top_k)

                q_clean = preprocess(query_input.strip())

                # Show preprocessing debug
                if show_preprocessing:
                    with st.expander("🔬 تفاصيل المعالجة النصية"):
                        st.markdown(f"**الأصلي:** `{query_input.strip()}`")
                        st.markdown(f"**بعد التطبيع:** `{normalize_arabic(query_input.strip())}`")
                        st.markdown(f"**بعد الـ Preprocessing:** `{q_clean}`")

                # Category classification (majority vote from top results)
                top_cat = results['category'].value_counts().idxmax()

                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

                # Category badge
                st.markdown(f"""
                <div style='direction:rtl; margin-bottom:0.6rem;'>
                    <span style='color:#888; font-size:0.85rem;'>التصنيف المُستنتج: </span>
                    <span class='category-badge'>{top_cat}</span>
                </div>
                """, unsafe_allow_html=True)

                # LLM Generation
                if use_generation and hf_token:
                    with st.spinner("🤖 جارٍ توليد الإجابة ..."):
                        context = build_context(results)
                        prompt = build_prompt(query_input.strip(), context)
                        answer = generate_with_hf(prompt, hf_token, max_tokens)

                    st.markdown("#### 🤖 الإجابة المُولَّدة")
                    st.markdown(f"<div class='answer-card'>{answer}</div>", unsafe_allow_html=True)
                else:
                    # Show top retrieved answer directly
                    st.markdown("#### 💡 أقرب إجابة")
                    top_answer = results.iloc[0]['Answers']
                    st.markdown(f"<div class='answer-card'>{top_answer}</div>", unsafe_allow_html=True)

                # Retrieved docs
                st.markdown(f"#### 📄 المستندات المسترجعة ({selected_mode})")
                for _, row in results.iterrows():
                    score_pct = f"{row['score']*100:.1f}%"
                    score_html = f"<span class='doc-score'>{score_pct}</span>" if show_scores else ""
                    st.markdown(f"""
                    <div class='doc-card'>
                        <div style='font-weight:700; color:#EEE; margin-bottom:4px;'>
                            {score_html} {row['Questions']}
                        </div>
                        <div style='color:#AAA; font-size:0.88rem;'>{row['Answers'][:280]}{'...' if len(row['Answers']) > 280 else ''}</div>
                        <div style='margin-top:6px;'>
                            <span class='category-badge'>{row['category']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Save to session history
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.insert(0, {
                    "query": query_input.strip(),
                    "category": top_cat,
                    "mode": selected_mode,
                })
                st.session_state.history = st.session_state.history[:15]

        elif search_btn:
            st.warning("اكتب سؤالك الأول يا صاحبي 😄")

    # History column
    with col_hist:
        st.markdown("#### 🕓 الأسئلة الأخيرة")
        if 'history' not in st.session_state or not st.session_state.history:
            st.markdown("<div style='color:#555; font-size:0.85rem; direction:rtl;'>لا يوجد تاريخ بعد.</div>", unsafe_allow_html=True)
        else:
            for item in st.session_state.history[:10]:
                st.markdown(f"""
                <div class='history-item'>
                    <div style='font-weight:700; color:#DDD;'>{item['query'][:55]}{'...' if len(item['query'])>55 else ''}</div>
                    <div style='color:#666; font-size:0.75rem; margin-top:3px;'>{item['category']} · {item['mode']}</div>
                </div>
                """, unsafe_allow_html=True)
            if st.button("🗑️ مسح التاريخ", use_container_width=True):
                st.session_state.history = []
                st.rerun()


# ══════════════════════════════════════════════
# TAB 2 — EXPLORE
# ══════════════════════════════════════════════
with tab_explore:
    if not data_ok:
        st.warning("حمّل البيانات أولاً من الشريط الجانبي.")
    else:
        df = df_data

        # Stats row
        cats = df['category'].nunique()
        total = len(df)
        avg_qlen = df['q_len'].mean()
        avg_alen = df['a_len'].mean()

        st.markdown(f"""
        <div class='metric-row'>
            <div class='metric-box'>
                <div class='metric-val'>{total}</div>
                <div class='metric-lbl'>إجمالي الأسئلة</div>
            </div>
            <div class='metric-box'>
                <div class='metric-val'>{cats}</div>
                <div class='metric-lbl'>الفئات</div>
            </div>
            <div class='metric-box'>
                <div class='metric-val'>{avg_qlen:.0f}</div>
                <div class='metric-lbl'>متوسط طول السؤال</div>
            </div>
            <div class='metric-box'>
                <div class='metric-val'>{avg_alen:.0f}</div>
                <div class='metric-lbl'>متوسط طول الإجابة</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown("#### 📊 توزيع الفئات")
            cat_counts = df['category'].value_counts()
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams['font.family'] = 'DejaVu Sans'

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1A1A1A')
            ax.set_facecolor('#1A1A1A')

            colors = ['#E60000', '#FF4444', '#CC0000', '#FF6666',
                      '#991100', '#FF8888', '#AA0000', '#FFAAAA']
            bars = ax.barh(range(len(cat_counts)), cat_counts.values,
                           color=colors[:len(cat_counts)], edgecolor='none')
            ax.set_yticks(range(len(cat_counts)))
            ax.set_yticklabels([ar(x) for x in cat_counts.index.tolist()],
                       fontsize=8, color='#CCC')
            ax.tick_params(colors='#888')
            ax.set_xlabel(ar('عدد الأسئلة'), color='#888')
            ax.set_ylabel(ar('الفئة'), color='#888')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')
            ax.grid(axis='x', alpha=0.2, color='#555')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("#### 📏 توزيع أطوال الأسئلة")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#1A1A1A')
            ax2.set_facecolor('#1A1A1A')
            ax2.hist(df['q_len'], bins=15, color='#E60000', edgecolor='#1A1A1A', alpha=0.85)
            ax2.axvline(df['q_len'].mean(), ls='--',
                color='#FF8888', lw=1.5,
                label=ar(f'المتوسط = {df["q_len"].mean():.0f}'))
            ax2.set_xlabel(ar('عدد الأحرف'), color='#888')
            ax2.set_ylabel(ar('التكرار'), color='#888')
            ax2.tick_params(colors='#888')
            for spine in ax2.spines.values():
                spine.set_edgecolor('#333')
            ax2.grid(alpha=0.2, color='#555')
            ax2.legend(facecolor='#222', labelcolor='#CCC')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### 🗂️ تصفح البيانات الخام")
        cat_filter = st.selectbox("تصفية بالفئة", ["الكل"] + df['category'].unique().tolist())
        if cat_filter == "الكل":
            display_df = df[['Questions', 'Answers', 'category']].head(50)
        else:
            display_df = df[df['category'] == cat_filter][['Questions', 'Answers', 'category']].head(50)

        st.dataframe(
            display_df.rename(columns={'Questions':'السؤال','Answers':'الإجابة','category':'الفئة'}),
            use_container_width=True,
            height=300,
        )


# ══════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div style='direction:rtl; max-width:800px; margin:0 auto; line-height:2;'>

    <h3 style='color:#E60000;'>🎓 عن المشروع</h3>
    <p>
    هذا النظام هو مشروع التخرج لتصنيف تذاكر الدعم الفني الخاصة بفودافون مصر باستخدام تقنية
    <strong>Retrieval-Augmented Generation (RAG)</strong>. النظام يعمل على الإجابة على أسئلة المستخدمين
    بالعربية المصرية بناءً على قاعدة بيانات الأسئلة الشائعة.
    </p>

    <h3 style='color:#E60000;'>🧠 مكونات النظام</h3>
    <ul>
        <li><strong>معالجة النصوص العربية:</strong> تطبيع الألف، إزالة التشكيل والحركات، حذف الـ Stopwords</li>
        <li><strong>TF-IDF Retriever:</strong> Char n-grams (3–5) لمطابقة النصوص</li>
        <li><strong>Dense Retriever:</strong> نموذج <code>intfloat/multilingual-e5-base</code></li>
        <li><strong>Hybrid Retriever:</strong> دمج TF-IDF + Dense بـ MinMax Scaling (alpha=0.4)</li>
        <li><strong>Generator:</strong> <code>Qwen/Qwen2.5-3B-Instruct</code> عبر HuggingFace Inference API</li>
    </ul>

    <h3 style='color:#E60000;'>📈 مقاييس التقييم</h3>
    <ul>
        <li><strong>Recall@K</strong> (K=1,3,5)</li>
        <li><strong>MRR</strong> — Mean Reciprocal Rank</li>
    </ul>

    <h3 style='color:#E60000;'>🗃️ البيانات</h3>
    <p>
    96 سؤال وإجابة مقسمة على 8 فئات رئيسية مأخوذة من موقع فودافون مصر الرسمي:
    الإنترنت الهوائي، الإنترنت الأرضي، فودافون كاش، التجوال الدولي، خدمات الكارت،
    برنامج شكراً، أنظمة Red، الدعم والمساعدة.
    </p>

    <h3 style='color:#E60000;'>🛠️ التقنيات المستخدمة</h3>
    <p>
    Python · scikit-learn · sentence-transformers · HuggingFace Transformers · Streamlit · Pandas · NumPy
    </p>

    </div>
    """, unsafe_allow_html=True)
