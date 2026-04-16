"""
==============================================================================
  ShopSense AI - Shopping Recommendation System
==============================================================================
  A complete AI-powered shopping recommendation system built with Python.

  Algorithms Implemented:
    1. Content-Based Filtering (TF-IDF + Cosine Similarity)
    2. Popularity-Based (Bayesian Weighted Rating / IMDB Formula)
    3. Best Deals (Value Score: discount + rating)
    4. Price-Range Filtering
    5. Hybrid Algorithm (weighted blend of all signals)

  Tech Stack: Python, Flask, pandas, scikit-learn, NumPy

  How to Run:
    1. Install dependencies:  pip install pandas scikit-learn flask numpy
    2. Place 'products.csv' in a 'data/' folder next to this file
    3. Run:  python shopping_recommendation_system.py
    4. Open:  http://localhost:5000

  Mini Project Questions Answered (in the "How It Works" section):
    Q1. Why does this platform use recommendation systems?
    Q2. What type of data does it collect?
    Q3. How does the recommendation system work?
    Q4. What techniques/algorithms are used?
    Q5. How does it personalize content?
    Q6. What challenges does it face?
==============================================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: DATA LOADING & PREPROCESSING                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_and_preprocess(csv_path):
    """Load the product CSV and clean/preprocess all fields."""
    df = pd.read_csv(csv_path)

    # --- Clean price columns: remove currency symbol and commas ---
    def parse_price(price_str):
        if pd.isna(price_str) or price_str == '':
            return np.nan
        price_str = str(price_str).replace('\u20b9', '').replace(',', '').strip()
        try:
            return float(price_str)
        except ValueError:
            return np.nan

    df['discount_price_clean'] = df['discount_price'].apply(parse_price)
    df['actual_price_clean'] = df['actual_price'].apply(parse_price)

    # --- Clean ratings ---
    def parse_rating(r):
        try:
            val = float(r)
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass
        return np.nan

    df['rating_clean'] = df['ratings'].apply(parse_rating)

    # --- Clean number of ratings ---
    def parse_num_ratings(r):
        if pd.isna(r):
            return 0
        r = str(r).replace(',', '').strip()
        try:
            return int(r)
        except ValueError:
            return 0

    df['num_ratings_clean'] = df['no_of_ratings'].apply(parse_num_ratings)

    # --- Calculate discount percentage ---
    df['discount_pct'] = np.where(
        (df['actual_price_clean'] > 0) & (df['discount_price_clean'] > 0),
        ((df['actual_price_clean'] - df['discount_price_clean']) / df['actual_price_clean'] * 100).round(1),
        0
    )

    df['effective_price'] = df['discount_price_clean'].fillna(df['actual_price_clean'])

    # --- Extract brand from product name ---
    def extract_brand(name):
        if pd.isna(name):
            return 'Unknown'
        name = str(name).strip().strip('"')
        brands = ['Samsung', 'OnePlus', 'Redmi', 'boAt', 'Apple', 'Xiaomi', 'Mi ',
                   'realme', 'Fire-Boltt', 'Noise', 'JBL', 'HP', 'Dell', 'Logitech',
                   'SanDisk', 'Oppo', 'iQOO', 'pTron', 'ZEBRONICS', 'Boult',
                   'Ambrane', 'Portronics', 'TP-Link', 'Canon', 'Epson', 'Crucial',
                   'Duracell', 'Lenovo', 'Seagate', 'Nokia', 'Acer', 'STRIFF',
                   'Wayona', 'Tygot', 'Lapster', 'Hammer', 'TAGG', 'Syska',
                   'Classmate', 'Cello', 'Fujifilm', 'Gear', 'Wesley', 'Havells',
                   'Echo', 'Fire TV', 'Amazon']
        for brand in brands:
            if brand.lower() in name.lower():
                return brand.strip()
        return name.split()[0] if name.split() else 'Unknown'

    df['brand'] = df['name'].apply(extract_brand)

    # --- Categorize products ---
    def categorize_product(name):
        if pd.isna(name):
            return 'Other'
        name_lower = str(name).lower()
        categories = {
            'Smartphone': ['phone', 'mobile', '5g', '4g', 'ram, ', 'gb storage', 'gb ram',
                           'redmi', 'oneplus nord ce', 'samsung galaxy m', 'samsung galaxy s',
                           'iphone', 'oppo a', 'oppo f', 'iqoo', 'realme narzo', 'nokia 1',
                           'lava blaze'],
            'Earbuds/TWS': ['airdopes', 'earbuds', 'truly wireless', 'tws', 'buds z',
                            'buds ce', 'buds air', 'bassbuds'],
            'Earphones (Wired)': ['wired earphone', 'wired in ear', 'bassheads', 'in-ear wired',
                                   'wired headphone', 'in ear wired'],
            'Neckband': ['neckband', 'rockerz 255', 'rockerz 330', 'bullets z2',
                         'wireless in ear earphones', 'bluetooth wireless in ear earphone'],
            'Headphones': ['over ear', 'on ear', 'headphone', 'rockerz 450', 'rockerz 550'],
            'Smartwatch': ['smart watch', 'smartwatch', 'watch', 'wave call', 'wave lite',
                           'wave edge', 'ninja', 'pulse', 'phoenix', 'colorfit'],
            'Power Bank': ['power bank', 'mah li', '10000mah', '20000mah'],
            'Charger/Cable': ['charger', 'cable', 'adapter', 'charging', 'usb-c', 'type-c cable',
                              'type c cable', 'micro usb cable', 'lightning'],
            'Storage': ['pen drive', 'flash drive', 'ssd', 'hard drive', 'memory card',
                        'microsd', 'sdxc', 'pendrive', 'hdd'],
            'Mouse': ['mouse', 'wireless mouse', 'gaming mouse'],
            'Keyboard': ['keyboard'],
            'Speaker': ['speaker', 'echo dot', 'sound bomb', 'stone 352', 'stone 620'],
            'TV': ['led tv', 'smart tv', 'android tv', 'fire tv stick'],
            'Camera/Security': ['camera', 'cctv', 'webcam', 'instax'],
            'Trimmer': ['trimmer', 'beard'],
            'Stand/Mount': ['stand', 'tripod', 'mount', 'holder', 'tabletop'],
            'Accessories': ['protector', 'case', 'cover', 'cleaning', 'mousepad', 'mouse pad',
                           'hub', 'otg', 'extension', 'battery', 'batteries',
                           'capo', 'guitar', 'pen ', 'notebook', 'calculator', 'thermometer',
                           'tape', 'paper', 'writing tablet', 'ring light']
        }
        for cat, keywords in categories.items():
            for kw in keywords:
                if kw in name_lower:
                    return cat
        return 'Other'

    df['product_type'] = df['name'].apply(categorize_product)

    # --- Combined text feature for TF-IDF ---
    df['text_features'] = (
        df['name'].fillna('') + ' ' +
        df['brand'].fillna('') + ' ' +
        df['product_type'].fillna('') + ' ' +
        df['main_category'].fillna('') + ' ' +
        df['sub_category'].fillna('')
    )

    df['product_id'] = range(len(df))
    return df


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: RECOMMENDATION ENGINE (5 ALGORITHMS)                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class RecommendationEngine:
    """Multi-algorithm shopping recommendation engine."""

    def __init__(self, csv_path):
        self.df = load_and_preprocess(csv_path)
        self._build_tfidf_matrix()
        self._compute_popularity_scores()
        self._compute_value_scores()

    # ── Algorithm 1: TF-IDF Content-Based Setup ───────────────────────────
    def _build_tfidf_matrix(self):
        """Build TF-IDF matrix for content-based filtering."""
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['text_features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    # ── Algorithm 2: Popularity Score (Bayesian / IMDB Formula) ───────────
    def _compute_popularity_scores(self):
        """WR = (v/(v+m)) * R + (m/(v+m)) * C"""
        C = self.df['rating_clean'].mean()
        m = self.df['num_ratings_clean'].quantile(0.25)

        def weighted_rating(row):
            v = row['num_ratings_clean']
            R = row['rating_clean']
            if pd.isna(R) or v == 0:
                return 0
            return (v / (v + m)) * R + (m / (v + m)) * C

        self.df['popularity_score'] = self.df.apply(weighted_rating, axis=1).round(3)

    # ── Algorithm 3: Value Score (Discount + Rating) ──────────────────────
    def _compute_value_scores(self):
        """value = 0.4 * normalized_discount + 0.6 * normalized_rating"""
        max_discount = self.df['discount_pct'].max()
        norm_discount = self.df['discount_pct'] / max_discount if max_discount > 0 else 0
        norm_rating = self.df['rating_clean'].fillna(0) / 5.0
        self.df['value_score'] = ((norm_discount * 0.4 + norm_rating * 0.6) * 100).round(1)

    def _row_to_dict(self, row, extra=None):
        """Convert a DataFrame row to a JSON-serializable dict."""
        d = {
            'product_id': int(row['product_id']),
            'name': str(row['name']),
            'brand': str(row['brand']),
            'product_type': str(row['product_type']),
            'image': str(row['image']),
            'rating': float(row['rating_clean']) if not pd.isna(row['rating_clean']) else None,
            'num_ratings': int(row['num_ratings_clean']),
            'discount_price': float(row['discount_price_clean']) if not pd.isna(row['discount_price_clean']) else None,
            'actual_price': float(row['actual_price_clean']) if not pd.isna(row['actual_price_clean']) else None,
            'discount_pct': float(row['discount_pct']),
        }
        if extra:
            d.update(extra)
        return d

    # ── Content-Based Recommendations ─────────────────────────────────────
    def content_based(self, product_id, top_n=10):
        if product_id >= len(self.df):
            return []
        sim_scores = sorted(enumerate(self.cosine_sim[product_id]), key=lambda x: x[1], reverse=True)[1:top_n + 1]
        return [self._row_to_dict(self.df.iloc[idx], {'similarity_score': round(float(score), 4), 'algorithm': 'content_based'}) for idx, score in sim_scores]

    # ── Popularity-Based Recommendations ──────────────────────────────────
    def popularity_based(self, top_n=10, product_type=None):
        f = self.df.copy()
        if product_type and product_type != 'All':
            f = f[f['product_type'] == product_type]
        f = f[f['rating_clean'].notna()].sort_values('popularity_score', ascending=False).head(top_n)
        return [self._row_to_dict(row, {'popularity_score': float(row['popularity_score']), 'algorithm': 'popularity'}) for _, row in f.iterrows()]

    # ── Price-Range Recommendations ───────────────────────────────────────
    def price_range_based(self, min_price=0, max_price=100000, product_type=None, top_n=10):
        f = self.df[self.df['effective_price'].notna()].copy()
        f = f[(f['effective_price'] >= min_price) & (f['effective_price'] <= max_price)]
        if product_type and product_type != 'All':
            f = f[f['product_type'] == product_type]
        f = f[f['rating_clean'].notna()].sort_values(['rating_clean', 'num_ratings_clean'], ascending=[False, False]).head(top_n)
        return [self._row_to_dict(row, {'algorithm': 'price_range'}) for _, row in f.iterrows()]

    # ── Best Deals Recommendations ────────────────────────────────────────
    def best_deals(self, top_n=10, product_type=None):
        f = self.df[(self.df['discount_pct'] > 0) & (self.df['rating_clean'].notna()) & (self.df['rating_clean'] >= 3.5)].copy()
        if product_type and product_type != 'All':
            f = f[f['product_type'] == product_type]
        f = f.sort_values('value_score', ascending=False).head(top_n)
        return [self._row_to_dict(row, {'value_score': float(row['value_score']), 'algorithm': 'best_deals'}) for _, row in f.iterrows()]

    # ── Hybrid Recommendations ────────────────────────────────────────────
    def hybrid(self, product_id=None, product_type=None, top_n=10):
        sim = self.cosine_sim[product_id] if product_id is not None and product_id < len(self.df) else np.zeros(len(self.df))
        f = self.df.copy()
        f['content_score'] = sim
        if product_type and product_type != 'All':
            f = f[f['product_type'] == product_type]
        if product_id is not None:
            f = f[f['product_id'] != product_id]
        f = f[f['rating_clean'].notna()]
        for col, src in [('norm_pop', 'popularity_score'), ('norm_val', 'value_score'), ('norm_content', 'content_score')]:
            mx = f[src].max()
            f[col] = f[src] / mx if mx > 0 else 0
        f['hybrid_score'] = (0.40 * f['norm_content'] + 0.35 * f['norm_pop'] + 0.25 * f['norm_val']).round(4)
        f = f.sort_values('hybrid_score', ascending=False).head(top_n)
        return [self._row_to_dict(row, {'hybrid_score': float(row['hybrid_score']), 'algorithm': 'hybrid'}) for _, row in f.iterrows()]

    # ── Analytics ─────────────────────────────────────────────────────────
    def get_stats(self):
        df = self.df
        return {
            'total_products': int(len(df)),
            'total_brands': int(df['brand'].nunique()),
            'total_categories': int(df['product_type'].nunique()),
            'avg_rating': round(float(df['rating_clean'].mean()), 2),
            'avg_discount': round(float(df['discount_pct'].mean()), 1),
            'type_counts': df['product_type'].value_counts().to_dict(),
            'brand_counts': df['brand'].value_counts().head(15).to_dict(),
            'avg_rating_by_type': df.groupby('product_type')['rating_clean'].mean().dropna().round(2).to_dict(),
            'avg_price_by_type': {k: int(v) for k, v in df.groupby('product_type')['effective_price'].mean().dropna().round(0).to_dict().items()},
            'price_ranges': {
                'Under 500': int(len(df[df['effective_price'] < 500])),
                '500-1000': int(len(df[(df['effective_price'] >= 500) & (df['effective_price'] < 1000)])),
                '1000-5000': int(len(df[(df['effective_price'] >= 1000) & (df['effective_price'] < 5000)])),
                '5000-15000': int(len(df[(df['effective_price'] >= 5000) & (df['effective_price'] < 15000)])),
                '15000-30000': int(len(df[(df['effective_price'] >= 15000) & (df['effective_price'] < 30000)])),
                'Over 30000': int(len(df[df['effective_price'] >= 30000])),
            },
        }

    def get_product_types(self):
        return sorted(self.df['product_type'].unique().tolist())

    def get_all_products(self, page=1, per_page=20, product_type=None, search=None):
        f = self.df.copy()
        if product_type and product_type != 'All':
            f = f[f['product_type'] == product_type]
        if search:
            f = f[f['name'].str.contains(search, case=False, na=False)]
        total = len(f)
        page_df = f.iloc[(page - 1) * per_page: page * per_page]
        return {
            'products': [self._row_to_dict(row) for _, row in page_df.iterrows()],
            'total': total, 'page': page, 'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: EMBEDDED FRONTEND (HTML + CSS + JS)                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="description" content="AI-Powered Shopping Recommendation System using TF-IDF, Cosine Similarity, and Hybrid ML algorithms.">
<title>ShopSense AI - Smart Shopping Recommendations</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* === CSS Variables === */
:root{--bg-primary:#FEFCF8;--bg-secondary:#F7F3ED;--bg-card:#FFF;--bg-warm:#FBF8F3;--bg-hero:linear-gradient(165deg,#FBF7F0 0%,#F3EDE3 40%,#EDE5D8 100%);--text-primary:#1B2A4A;--text-secondary:#5A6478;--text-muted:#8E95A4;--accent-1:#1B2A4A;--accent-2:#2D4A7A;--accent-3:#C8963E;--accent-success:#2E7D5B;--accent-warning:#D4920A;--accent-danger:#C7453B;--gradient-primary:linear-gradient(135deg,#1B2A4A,#2D4A7A);--gradient-gold:linear-gradient(135deg,#C8963E,#E8C06A);--gradient-accent:linear-gradient(135deg,#2D4A7A,#4A7AB5);--border-subtle:rgba(27,42,74,.08);--border-medium:rgba(27,42,74,.12);--border-gold:rgba(200,150,62,.3);--radius-sm:8px;--radius-md:12px;--radius-lg:16px;--radius-xl:20px;--radius-round:9999px;--shadow-sm:0 1px 3px rgba(27,42,74,.06),0 1px 2px rgba(27,42,74,.04);--shadow-md:0 4px 12px rgba(27,42,74,.08),0 2px 4px rgba(27,42,74,.04);--shadow-lg:0 10px 30px rgba(27,42,74,.1),0 4px 8px rgba(27,42,74,.04);--shadow-card-hover:0 12px 36px rgba(27,42,74,.12),0 4px 12px rgba(200,150,62,.06);--font-main:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;--font-mono:'JetBrains Mono',monospace;--transition:.3s cubic-bezier(.4,0,.2,1)}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;scrollbar-width:thin;scrollbar-color:var(--accent-3) var(--bg-secondary)}
body{font-family:var(--font-main);background:var(--bg-primary);color:var(--text-primary);line-height:1.6;min-height:100vh;overflow-x:hidden}
::-webkit-scrollbar{width:7px}::-webkit-scrollbar-track{background:var(--bg-secondary)}::-webkit-scrollbar-thumb{background:var(--accent-3);border-radius:4px}

/* Background */
.bg-gradient{position:fixed;top:0;left:0;width:100vw;height:100vh;background:radial-gradient(ellipse at 20% 10%,rgba(200,150,62,.06) 0%,transparent 50%),radial-gradient(ellipse at 80% 90%,rgba(27,42,74,.04) 0%,transparent 50%);pointer-events:none;z-index:0}
.bg-particles{position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;overflow:hidden}
.particle{position:absolute;width:3px;height:3px;background:rgba(200,150,62,.2);border-radius:50%;animation:fp linear infinite}
@keyframes fp{0%{transform:translateY(100vh) scale(0);opacity:0}10%{opacity:.5}90%{opacity:.5}100%{transform:translateY(-10vh) scale(1);opacity:0}}

/* Navbar */
.navbar{position:fixed;top:0;left:0;right:0;z-index:100;display:flex;align-items:center;justify-content:space-between;padding:12px 32px;background:rgba(254,252,248,.85);backdrop-filter:blur(20px);border-bottom:1px solid var(--border-subtle);transition:var(--transition)}
.navbar.scrolled{background:rgba(254,252,248,.96);box-shadow:var(--shadow-md)}
.nav-brand{display:flex;align-items:center;gap:12px}.nav-logo{display:flex}
.nav-title{font-size:1.3rem;font-weight:700;letter-spacing:-.5px;color:var(--accent-1)}
.nav-ai-badge{font-size:.6rem;font-weight:700;background:var(--gradient-gold);color:#fff;padding:2px 8px;border-radius:var(--radius-round);vertical-align:super;letter-spacing:1px}
.nav-links{display:flex;gap:4px}
.nav-link{text-decoration:none;color:var(--text-secondary);font-size:.85rem;font-weight:500;padding:8px 16px;border-radius:var(--radius-sm);transition:var(--transition)}
.nav-link:hover{color:var(--text-primary);background:rgba(27,42,74,.04)}
.nav-link.active{color:var(--accent-1);background:rgba(27,42,74,.07);font-weight:600}

.main-content{position:relative;z-index:1;padding-top:60px}

/* Hero */
.hero{min-height:85vh;display:flex;align-items:center;justify-content:center;padding:60px 32px;text-align:center;background:var(--bg-hero);position:relative}
.hero::after{content:'';position:absolute;bottom:0;left:0;right:0;height:120px;background:linear-gradient(to bottom,transparent,var(--bg-primary));pointer-events:none}
.hero-content{max-width:800px;animation:fu .8s ease-out;position:relative;z-index:1}
@keyframes fu{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}
.hero-badge{display:inline-block;font-size:.78rem;font-weight:600;color:var(--accent-3);background:rgba(200,150,62,.08);border:1px solid rgba(200,150,62,.2);padding:7px 20px;border-radius:var(--radius-round);margin-bottom:24px}
.hero-title{font-size:3.5rem;font-weight:800;line-height:1.12;letter-spacing:-1.5px;margin-bottom:20px;color:var(--accent-1)}
.gradient-text{background:var(--gradient-gold);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero-subtitle{font-size:1.1rem;color:var(--text-secondary);max-width:600px;margin:0 auto 32px;line-height:1.7}
.hero-buttons{display:flex;gap:16px;justify-content:center;margin-bottom:48px}
.hero-stats{display:flex;gap:40px;justify-content:center;flex-wrap:wrap}
.stat-item{text-align:center}.stat-value{font-size:2rem;font-weight:800;color:var(--accent-1);display:block}.stat-label{font-size:.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1.2px;font-weight:600}

/* Buttons */
.btn{display:inline-flex;align-items:center;gap:10px;font-family:var(--font-main);font-size:.9rem;font-weight:600;padding:13px 30px;border-radius:var(--radius-md);border:none;cursor:pointer;transition:var(--transition);text-decoration:none}
.btn-primary{background:var(--gradient-primary);color:#fff;box-shadow:0 4px 14px rgba(27,42,74,.2)}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(27,42,74,.28)}
.btn-ghost{background:transparent;color:var(--text-secondary);border:1.5px solid var(--border-medium)}
.btn-ghost:hover{color:var(--accent-1);background:rgba(27,42,74,.03);border-color:var(--accent-1)}

/* Sections */
.section{padding:80px 32px;max-width:1300px;margin:0 auto}
.section-header{text-align:center;margin-bottom:48px}
.section-title{font-size:2rem;font-weight:800;letter-spacing:-.5px;margin-bottom:8px;color:var(--accent-1)}
.section-desc{font-size:1rem;color:var(--text-secondary)}

/* Dashboard */
.dashboard-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:20px;margin-bottom:40px}
.stat-card{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);padding:24px;position:relative;overflow:hidden;transition:var(--transition);box-shadow:var(--shadow-sm)}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:var(--radius-lg) var(--radius-lg) 0 0}
.stat-card:nth-child(1)::before{background:var(--gradient-primary)}.stat-card:nth-child(2)::before{background:var(--gradient-gold)}.stat-card:nth-child(3)::before{background:var(--gradient-accent)}.stat-card:nth-child(4)::before{background:linear-gradient(135deg,#2E7D5B,#3DA47A)}.stat-card:nth-child(5)::before{background:linear-gradient(135deg,#D4920A,#C7453B)}
.stat-card:hover{transform:translateY(-4px);box-shadow:var(--shadow-card-hover);border-color:var(--border-gold)}
.stat-card .card-icon{font-size:1.8rem;margin-bottom:12px}.stat-card .card-value{font-size:1.8rem;font-weight:800;color:var(--accent-1);margin-bottom:4px}.stat-card .card-label{font-size:.78rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px;font-weight:600}

/* Charts */
.charts-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}
.chart-card{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);padding:24px;transition:var(--transition);box-shadow:var(--shadow-sm)}.chart-card:hover{box-shadow:var(--shadow-md);border-color:var(--border-gold)}
.chart-title{font-size:.9rem;font-weight:600;color:var(--text-secondary);margin-bottom:20px}.chart-container{min-height:180px}
.bar-chart{display:flex;flex-direction:column;gap:10px}.bar-row{display:flex;align-items:center;gap:12px}.bar-label{font-size:.75rem;color:var(--text-secondary);min-width:100px;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar-track{flex:1;height:26px;background:var(--bg-secondary);border-radius:var(--radius-sm);overflow:hidden}.bar-fill{height:100%;border-radius:var(--radius-sm);background:var(--gradient-primary);transition:width 1s ease-out;display:flex;align-items:center;padding-left:10px}
.bar-fill.alt{background:var(--gradient-gold)}.bar-fill.accent{background:var(--gradient-accent)}.bar-fill.green{background:linear-gradient(135deg,#2E7D5B,#3DA47A)}
.bar-value{font-size:.7rem;font-weight:600;color:#fff;white-space:nowrap}

/* Explore */
.explore-controls{display:flex;gap:16px;margin-bottom:32px;flex-wrap:wrap}
.search-box{flex:1;min-width:250px;display:flex;align-items:center;gap:12px;background:var(--bg-card);border:1.5px solid var(--border-medium);border-radius:var(--radius-md);padding:12px 18px;transition:var(--transition);box-shadow:var(--shadow-sm)}
.search-box:focus-within{border-color:var(--accent-3);box-shadow:0 0 0 3px rgba(200,150,62,.1)}
.search-box svg{color:var(--text-muted);flex-shrink:0}
.search-box input{flex:1;background:none;border:none;outline:none;color:var(--text-primary);font-family:var(--font-main);font-size:.9rem}.search-box input::placeholder{color:var(--text-muted)}
.filter-select{background:var(--bg-card);border:1.5px solid var(--border-medium);border-radius:var(--radius-md);padding:12px 18px;color:var(--text-primary);font-family:var(--font-main);font-size:.85rem;cursor:pointer;transition:var(--transition);min-width:180px;box-shadow:var(--shadow-sm)}
.filter-select.wide{min-width:400px;max-width:100%}
.filter-select:focus{outline:none;border-color:var(--accent-3);box-shadow:0 0 0 3px rgba(200,150,62,.1)}
.filter-select option{background:var(--bg-card);color:var(--text-primary)}

/* Product Cards */
.products-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:20px;margin-bottom:32px}
.product-card{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);padding:20px;transition:var(--transition);cursor:pointer;position:relative;overflow:hidden;box-shadow:var(--shadow-sm)}
.product-card:hover{transform:translateY(-6px);box-shadow:var(--shadow-card-hover);border-color:var(--border-gold)}
.product-image-wrap{width:100%;height:160px;background:var(--bg-warm);border-radius:var(--radius-md);display:flex;align-items:center;justify-content:center;margin-bottom:14px;overflow:hidden;border:1px solid var(--border-subtle)}
.product-image-wrap img{max-height:140px;max-width:90%;object-fit:contain;transition:transform .4s ease}.product-card:hover .product-image-wrap img{transform:scale(1.08)}
.product-brand{font-size:.7rem;font-weight:700;color:var(--accent-3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px}
.product-name{font-size:.85rem;font-weight:600;color:var(--text-primary);line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;margin-bottom:10px;min-height:40px}
.product-meta{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.product-rating{display:flex;align-items:center;gap:4px;font-size:.8rem;font-weight:600;color:var(--accent-warning)}.product-rating .star{font-size:.9rem}
.product-reviews{font-size:.7rem;color:var(--text-muted)}
.product-price{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.price-current{font-size:1.05rem;font-weight:700;color:var(--accent-success)}.price-original{font-size:.8rem;color:var(--text-muted);text-decoration:line-through}
.price-discount{font-size:.7rem;font-weight:700;color:var(--accent-danger);background:rgba(199,69,59,.08);padding:2px 8px;border-radius:var(--radius-round)}
.score-badge{position:absolute;top:12px;right:12px;font-size:.7rem;font-weight:700;padding:5px 12px;border-radius:var(--radius-round);color:#fff;z-index:2}
.score-badge.similarity{background:var(--gradient-primary)}.score-badge.popularity{background:var(--gradient-gold)}.score-badge.value{background:linear-gradient(135deg,#2E7D5B,#3DA47A)}.score-badge.hybrid{background:var(--gradient-accent)}

/* Pagination */
.pagination{display:flex;justify-content:center;gap:8px;flex-wrap:wrap}
.page-btn{font-family:var(--font-main);font-size:.8rem;font-weight:600;padding:8px 14px;border-radius:var(--radius-sm);border:1.5px solid var(--border-medium);background:var(--bg-card);color:var(--text-secondary);cursor:pointer;transition:var(--transition)}
.page-btn:hover,.page-btn.active{background:var(--accent-1);color:#fff;border-color:var(--accent-1)}

/* Algo Tabs */
.algo-tabs{display:flex;gap:12px;margin-bottom:28px;flex-wrap:wrap;justify-content:center}
.algo-tab{display:flex;flex-direction:column;align-items:center;gap:4px;padding:16px 24px;background:var(--bg-card);border:1.5px solid var(--border-medium);border-radius:var(--radius-lg);cursor:pointer;transition:var(--transition);font-family:var(--font-main);min-width:130px;box-shadow:var(--shadow-sm)}
.algo-tab:hover{border-color:var(--accent-3);transform:translateY(-3px);box-shadow:var(--shadow-md)}
.algo-tab.active{border-color:var(--accent-1);background:rgba(27,42,74,.04);box-shadow:0 0 0 3px rgba(27,42,74,.06),var(--shadow-md)}
.algo-icon{font-size:1.5rem}.algo-name{font-size:.85rem;font-weight:700;color:var(--text-primary)}.algo-desc{font-size:.65rem;color:var(--text-muted);font-weight:500}
.algo-controls{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);padding:24px;margin-bottom:24px;display:flex;flex-wrap:wrap;gap:20px;align-items:flex-end;box-shadow:var(--shadow-sm)}
.control-group{display:flex;flex-direction:column;gap:8px;flex:1;min-width:200px}.control-group.hidden{display:none}
.control-label{font-size:.8rem;font-weight:600;color:var(--text-secondary)}
.range-inputs{display:flex;align-items:center;gap:12px}.range-field{display:flex;align-items:center;gap:8px;background:var(--bg-warm);border:1.5px solid var(--border-medium);border-radius:var(--radius-sm);padding:8px 12px}
.range-label{font-size:.75rem;color:var(--text-muted);font-weight:600}.range-input{background:none;border:none;outline:none;color:var(--text-primary);font-family:var(--font-main);font-size:.9rem;width:80px}.range-separator{color:var(--text-muted);font-size:1.2rem}
.algo-explanation{background:rgba(27,42,74,.03);border:1px solid rgba(27,42,74,.08);border-left:4px solid var(--accent-1);border-radius:0 var(--radius-lg) var(--radius-lg) 0;padding:20px 24px;margin-bottom:32px;font-size:.85rem;color:var(--text-secondary);line-height:1.7}
.algo-explanation strong{color:var(--accent-1)}.algo-explanation code{font-family:var(--font-mono);font-size:.8rem;background:rgba(27,42,74,.06);padding:2px 8px;border-radius:4px;color:var(--accent-2)}
.rec-results{min-height:300px}.rec-results .products-grid{margin-bottom:0}
.rec-empty{text-align:center;padding:60px 20px;color:var(--text-muted)}.rec-empty svg{width:64px;height:64px;stroke:var(--text-muted);margin-bottom:16px}

/* Theory */
.theory-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:24px}
.theory-card{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:var(--radius-xl);padding:32px;transition:var(--transition);position:relative;box-shadow:var(--shadow-sm)}.theory-card.wide{grid-column:1/-1}.theory-card:hover{box-shadow:var(--shadow-lg);border-color:var(--border-gold)}
.theory-number{font-size:3rem;font-weight:900;color:var(--accent-3);opacity:.2;position:absolute;top:16px;right:24px;line-height:1}
.theory-title{font-size:1.1rem;font-weight:700;margin-bottom:16px;color:var(--accent-1);padding-right:60px}
.theory-content{font-size:.9rem;color:var(--text-secondary);line-height:1.8}.theory-content ul{list-style:none;padding:0}
.theory-content li{padding:6px 0 6px 24px;position:relative}.theory-content li::before{content:'';position:absolute;left:0;top:14px;width:8px;height:8px;border-radius:50%;background:var(--gradient-gold)}
.theory-content strong{color:var(--text-primary)}
.theory-note{background:rgba(200,150,62,.06);border-left:3px solid var(--accent-3);padding:12px 16px;margin-top:12px;border-radius:0 var(--radius-sm) var(--radius-sm) 0;font-size:.85rem}
.theory-flow{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:16px 0;padding:16px;background:var(--bg-warm);border-radius:var(--radius-md);border:1px solid var(--border-subtle)}
.flow-step{display:flex;align-items:center;gap:8px;background:var(--bg-card);border:1px solid var(--border-medium);padding:8px 14px;border-radius:var(--radius-sm);font-size:.8rem;font-weight:500;color:var(--text-primary);box-shadow:var(--shadow-sm)}
.flow-num{width:22px;height:22px;border-radius:50%;background:var(--gradient-primary);color:#fff;display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:700}
.flow-arrow{color:var(--accent-3);font-weight:700;font-size:1.2rem}
.algo-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px;margin-top:16px}
.mini-card{background:var(--bg-warm);border:1px solid var(--border-subtle);border-radius:var(--radius-md);padding:20px;transition:var(--transition)}.mini-card:hover{border-color:var(--accent-3);transform:translateY(-3px);box-shadow:var(--shadow-md)}
.mini-card h4{font-size:.95rem;margin-bottom:10px;color:var(--accent-1)}.mini-card p{font-size:.82rem;color:var(--text-secondary);line-height:1.6;margin-bottom:10px}
.mini-card code{display:block;font-family:var(--font-mono);font-size:.75rem;color:var(--accent-2);background:rgba(27,42,74,.05);padding:8px 12px;border-radius:var(--radius-sm);margin-top:8px;border:1px solid var(--border-subtle)}

/* Footer */
.footer{text-align:center;padding:40px 32px;border-top:1px solid var(--border-subtle);margin-top:60px;background:var(--bg-secondary)}
.footer-content p{font-size:.85rem;color:var(--text-secondary)}.footer-tech{font-size:.75rem!important;color:var(--text-muted)!important;margin-top:4px}

/* Loading */
.loading-skeleton{background:linear-gradient(90deg,var(--bg-secondary) 25%,rgba(255,255,255,.6) 50%,var(--bg-secondary) 75%);background-size:200% 100%;animation:ss 1.5s infinite;border-radius:var(--radius-md)}@keyframes ss{0%{background-position:200% 0}100%{background-position:-200% 0}}
.skeleton-card{height:380px;background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);overflow:hidden}.skeleton-image{width:100%;height:160px;margin:20px 0 0}.skeleton-text{height:14px;margin:16px 20px 0;border-radius:7px}.skeleton-text.short{width:60%}
.animate-in{animation:ci .5s ease-out forwards;opacity:0}@keyframes ci{from{opacity:0;transform:translateY(20px) scale(.97)}to{opacity:1;transform:translateY(0) scale(1)}}

@media(max-width:768px){.navbar{padding:10px 16px}.nav-links{display:none}.hero-title{font-size:2.2rem}.hero{min-height:70vh;padding:40px 16px}.section{padding:60px 16px}.hero-buttons{flex-direction:column}.algo-tabs{gap:8px}.algo-tab{min-width:100px;padding:12px 16px}.products-grid{grid-template-columns:repeat(auto-fill,minmax(200px,1fr))}.theory-grid{grid-template-columns:1fr}.algo-controls{flex-direction:column}.filter-select.wide{min-width:100%}.theory-flow{flex-direction:column}.flow-arrow{transform:rotate(90deg)}}
</style>
</head>
<body>
<div class="bg-gradient"></div>
<div class="bg-particles" id="particles"></div>

<nav class="navbar" id="navbar">
    <div class="nav-brand">
        <div class="nav-logo"><svg width="32" height="32" viewBox="0 0 32 32" fill="none"><rect width="32" height="32" rx="8" fill="url(#lg)"/><path d="M8 12L16 8L24 12V20L16 24L8 20V12Z" stroke="#fff" stroke-width="1.5" fill="none"/><circle cx="16" cy="16" r="3" fill="#fff"/><defs><linearGradient id="lg" x1="0" y1="0" x2="32" y2="32"><stop stop-color="#1B2A4A"/><stop offset="1" stop-color="#2D4A7A"/></linearGradient></defs></svg></div>
        <span class="nav-title">ShopSense <span class="nav-ai-badge">AI</span></span>
    </div>
    <div class="nav-links">
        <a href="#dashboard" class="nav-link active" data-section="dashboard">Dashboard</a>
        <a href="#explore" class="nav-link" data-section="explore">Explore</a>
        <a href="#recommend" class="nav-link" data-section="recommend">Recommend</a>
        <a href="#theory" class="nav-link" data-section="theory">How It Works</a>
    </div>
</nav>

<main class="main-content">
    <section class="hero" id="hero">
        <div class="hero-content">
            <div class="hero-badge">AI-Powered Recommendations</div>
            <h1 class="hero-title">Smart Shopping<br><span class="gradient-text">Recommendations</span></h1>
            <p class="hero-subtitle">Discover the perfect products using advanced machine learning algorithms &mdash; Content-Based Filtering, Popularity Scoring, and Hybrid Intelligence.</p>
            <div class="hero-buttons">
                <button class="btn btn-primary" onclick="scrollToSection('recommend')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>Get Recommendations</button>
                <button class="btn btn-ghost" onclick="scrollToSection('theory')">Learn How It Works &rarr;</button>
            </div>
            <div class="hero-stats" id="heroStats"></div>
        </div>
    </section>

    <section class="section" id="dashboard">
        <div class="section-header"><h2 class="section-title">Analytics Dashboard</h2><p class="section-desc">Real-time insights from our product catalog</p></div>
        <div class="dashboard-grid" id="dashboardGrid"></div>
        <div class="charts-grid">
            <div class="chart-card"><h3 class="chart-title">Product Type Distribution</h3><div class="chart-container" id="typeChart"></div></div>
            <div class="chart-card"><h3 class="chart-title">Top Brands</h3><div class="chart-container" id="brandChart"></div></div>
            <div class="chart-card"><h3 class="chart-title">Price Range Distribution</h3><div class="chart-container" id="priceChart"></div></div>
            <div class="chart-card"><h3 class="chart-title">Average Rating by Category</h3><div class="chart-container" id="ratingChart"></div></div>
        </div>
    </section>

    <section class="section" id="explore">
        <div class="section-header"><h2 class="section-title">Explore Products</h2><p class="section-desc">Browse and search through our catalog</p></div>
        <div class="explore-controls">
            <div class="search-box"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg><input type="text" id="searchInput" placeholder="Search products..." autocomplete="off"></div>
            <select id="typeFilter" class="filter-select"><option value="All">All Categories</option></select>
        </div>
        <div class="products-grid" id="productsGrid"></div>
        <div class="pagination" id="pagination"></div>
    </section>

    <section class="section" id="recommend">
        <div class="section-header"><h2 class="section-title">Recommendation Engine</h2><p class="section-desc">Choose an algorithm and discover personalized product suggestions</p></div>
        <div class="algo-tabs">
            <button class="algo-tab active" data-algo="popular" id="tab-popular"><span class="algo-icon">&#128293;</span><span class="algo-name">Trending</span><span class="algo-desc">Popularity-Based</span></button>
            <button class="algo-tab" data-algo="content" id="tab-content"><span class="algo-icon">&#128279;</span><span class="algo-name">Similar</span><span class="algo-desc">Content-Based (TF-IDF)</span></button>
            <button class="algo-tab" data-algo="deals" id="tab-deals"><span class="algo-icon">&#128176;</span><span class="algo-name">Best Deals</span><span class="algo-desc">Value Score</span></button>
            <button class="algo-tab" data-algo="price" id="tab-price"><span class="algo-icon">&#128178;</span><span class="algo-name">Budget</span><span class="algo-desc">Price-Range Filter</span></button>
            <button class="algo-tab" data-algo="hybrid" id="tab-hybrid"><span class="algo-icon">&#9889;</span><span class="algo-name">Smart Pick</span><span class="algo-desc">Hybrid Algorithm</span></button>
        </div>
        <div class="algo-controls" id="algoControls">
            <div class="control-group hidden" id="contentControls"><label class="control-label">Select a product to find similar items:</label><select id="productSelector" class="filter-select wide"></select></div>
            <div class="control-group hidden" id="priceControls"><label class="control-label">Set your budget range:</label><div class="range-inputs"><div class="range-field"><span class="range-label">Min</span><input type="number" id="minPrice" value="0" min="0" class="range-input"></div><span class="range-separator">&mdash;</span><div class="range-field"><span class="range-label">Max</span><input type="number" id="maxPrice" value="50000" min="0" class="range-input"></div></div></div>
            <div class="control-group" id="categoryControl"><label class="control-label">Filter by category:</label><select id="recTypeFilter" class="filter-select"><option value="All">All Categories</option></select></div>
            <button class="btn btn-primary" id="getRecommendations" onclick="fetchRecommendations()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>Get Recommendations</button>
        </div>
        <div class="algo-explanation" id="algoExplanation"></div>
        <div class="rec-results" id="recResults"></div>
    </section>

    <section class="section" id="theory">
        <div class="section-header"><h2 class="section-title">How It Works</h2><p class="section-desc">Understanding Recommendation Systems in E-Commerce</p></div>
        <div class="theory-grid">
            <div class="theory-card"><div class="theory-number">01</div><h3 class="theory-title">Why does this platform use recommendation systems?</h3><div class="theory-content"><p>E-commerce platforms use recommendation systems to:</p><ul><li><strong>Enhance User Experience</strong> &mdash; Help users discover relevant products from millions of listings</li><li><strong>Increase Revenue</strong> &mdash; Personalized recommendations drive 35% of Amazon's total sales</li><li><strong>Improve Engagement</strong> &mdash; Users spend more time on platforms that understand their preferences</li><li><strong>Reduce Choice Overload</strong> &mdash; With 350M+ products, filtering is essential</li><li><strong>Customer Retention</strong> &mdash; Satisfying recommendations build brand loyalty</li></ul></div></div>
            <div class="theory-card"><div class="theory-number">02</div><h3 class="theory-title">What type of data does it collect?</h3><div class="theory-content"><p>Shopping platforms collect multiple data types:</p><ul><li><strong>Explicit Data</strong> &mdash; Ratings, reviews, wishlists, purchase history</li><li><strong>Implicit Data</strong> &mdash; Browsing history, click patterns, time spent on pages</li><li><strong>Product Metadata</strong> &mdash; Name, category, brand, price, specifications</li><li><strong>User Demographics</strong> &mdash; Age, location, device type, preferences</li><li><strong>Contextual Data</strong> &mdash; Time of day, season, trending events</li></ul><p class="theory-note">Our system uses: <strong>Product names, categories, ratings, number of ratings, and pricing data</strong></p></div></div>
            <div class="theory-card"><div class="theory-number">03</div><h3 class="theory-title">How does the recommendation system work?</h3><div class="theory-content"><p>Our system implements a <strong>multi-layered recommendation pipeline</strong>:</p><div class="theory-flow"><div class="flow-step"><span class="flow-num">1</span><span>Data Ingestion</span></div><div class="flow-arrow">&rarr;</div><div class="flow-step"><span class="flow-num">2</span><span>Feature Extraction (TF-IDF)</span></div><div class="flow-arrow">&rarr;</div><div class="flow-step"><span class="flow-num">3</span><span>Similarity Computation</span></div><div class="flow-arrow">&rarr;</div><div class="flow-step"><span class="flow-num">4</span><span>Score Aggregation</span></div><div class="flow-arrow">&rarr;</div><div class="flow-step"><span class="flow-num">5</span><span>Ranked Results</span></div></div><p>The pipeline processes product metadata, builds feature vectors, computes similarity matrices, and combines multiple scoring signals.</p></div></div>
            <div class="theory-card wide"><div class="theory-number">04</div><h3 class="theory-title">What techniques/algorithms are used?</h3><div class="theory-content"><div class="algo-cards"><div class="mini-card"><h4>Content-Based Filtering</h4><p>Uses <strong>TF-IDF</strong> to convert product text into numerical vectors, then computes <strong>Cosine Similarity</strong> to find related products.</p><code>similarity = cos(A,B) = (A.B) / (||A|| x ||B||)</code></div><div class="mini-card"><h4>Popularity-Based</h4><p>Uses <strong>Bayesian Weighted Rating</strong> (IMDB formula) to rank products, balancing rating value with volume.</p><code>WR = (v/(v+m)) x R + (m/(v+m)) x C</code></div><div class="mini-card"><h4>Value Scoring</h4><p>Combines <strong>normalized discount</strong> with <strong>normalized rating</strong> to find best value-for-money products.</p><code>value = 0.4 x norm_discount + 0.6 x norm_rating</code></div><div class="mini-card"><h4>Hybrid Algorithm</h4><p>Combines all three signals with configurable weights for the most balanced recommendations.</p><code>hybrid = 0.4 x content + 0.35 x pop + 0.25 x value</code></div></div></div></div>
            <div class="theory-card"><div class="theory-number">05</div><h3 class="theory-title">How does it personalize content?</h3><div class="theory-content"><ul><li><strong>Product Similarity</strong> &mdash; When a user selects a product, similar items are surfaced using cosine similarity</li><li><strong>Category Affinity</strong> &mdash; Filtering by the user's preferred product category</li><li><strong>Budget Matching</strong> &mdash; Price-range filtering ensures recommendations fit spending capacity</li><li><strong>Brand Clustering</strong> &mdash; Products from similar brands weighted higher</li><li><strong>Social Proof</strong> &mdash; Popularity scores leverage crowd wisdom</li></ul></div></div>
            <div class="theory-card"><div class="theory-number">06</div><h3 class="theory-title">What challenges does it face?</h3><div class="theory-content"><ul><li><strong>Cold Start Problem</strong> &mdash; New products/users with no history</li><li><strong>Data Sparsity</strong> &mdash; Users interact with a tiny fraction of products</li><li><strong>Scalability</strong> &mdash; Similarity matrices grow quadratically with catalog size</li><li><strong>Filter Bubble</strong> &mdash; Over-personalization traps users in echo chambers</li><li><strong>Dynamic Inventory</strong> &mdash; Products go in/out of stock constantly</li><li><strong>Bias &amp; Fairness</strong> &mdash; Popular brands get disproportionate visibility</li></ul></div></div>
        </div>
    </section>

    <footer class="footer"><div class="footer-content"><p>ShopSense AI &mdash; Shopping Recommendation System</p><p class="footer-tech">Built with Python | Flask | scikit-learn | TF-IDF | Cosine Similarity</p></div></footer>
</main>

<script>
const API='';let currentAlgo='popular',currentPage=1,productTypes=[];
document.addEventListener('DOMContentLoaded',()=>{initParticles();initNavScroll();initNavLinks();initAlgoTabs();initSearch();loadStats();loadProductTypes();loadProducts();updateAlgoExplanation()});
function initParticles(){const c=document.getElementById('particles');for(let i=0;i<25;i++){const p=document.createElement('div');p.classList.add('particle');p.style.left=Math.random()*100+'%';p.style.animationDuration=(8+Math.random()*15)+'s';p.style.animationDelay=(Math.random()*10)+'s';p.style.width=p.style.height=(2+Math.random()*3)+'px';c.appendChild(p)}}
function initNavScroll(){window.addEventListener('scroll',()=>{document.getElementById('navbar').classList.toggle('scrolled',window.scrollY>50)})}
function initNavLinks(){document.querySelectorAll('.nav-link').forEach(l=>{l.addEventListener('click',e=>{e.preventDefault();document.querySelectorAll('.nav-link').forEach(n=>n.classList.remove('active'));l.classList.add('active');scrollToSection(l.dataset.section)})})}
function scrollToSection(id){const el=document.getElementById(id);if(el){window.scrollTo({top:el.getBoundingClientRect().top+window.pageYOffset-80,behavior:'smooth'})}}
function initAlgoTabs(){document.querySelectorAll('.algo-tab').forEach(t=>{t.addEventListener('click',()=>{document.querySelectorAll('.algo-tab').forEach(x=>x.classList.remove('active'));t.classList.add('active');currentAlgo=t.dataset.algo;document.getElementById('contentControls').classList.toggle('hidden',currentAlgo!=='content');document.getElementById('priceControls').classList.toggle('hidden',currentAlgo!=='price');updateAlgoExplanation()})})}
function initSearch(){let d;document.getElementById('searchInput').addEventListener('input',()=>{clearTimeout(d);d=setTimeout(()=>{currentPage=1;loadProducts()},300)});document.getElementById('typeFilter').addEventListener('change',()=>{currentPage=1;loadProducts()})}
async function loadStats(){try{const r=await fetch(API+'/api/stats');const s=await r.json();document.getElementById('heroStats').innerHTML=`<div class="stat-item animate-in" style="animation-delay:.1s"><span class="stat-value">${s.total_products}</span><span class="stat-label">Products</span></div><div class="stat-item animate-in" style="animation-delay:.2s"><span class="stat-value">${s.total_brands}</span><span class="stat-label">Brands</span></div><div class="stat-item animate-in" style="animation-delay:.3s"><span class="stat-value">${s.total_categories}</span><span class="stat-label">Categories</span></div><div class="stat-item animate-in" style="animation-delay:.4s"><span class="stat-value">${s.avg_rating}</span><span class="stat-label">Avg Rating</span></div><div class="stat-item animate-in" style="animation-delay:.5s"><span class="stat-value">${s.avg_discount}%</span><span class="stat-label">Avg Discount</span></div>`;document.getElementById('dashboardGrid').innerHTML=`<div class="stat-card animate-in"><div class="card-icon">&#128230;</div><div class="card-value">${s.total_products}</div><div class="card-label">Total Products</div></div><div class="stat-card animate-in"><div class="card-icon">&#127991;</div><div class="card-value">${s.total_brands}</div><div class="card-label">Brands</div></div><div class="stat-card animate-in"><div class="card-icon">&#128194;</div><div class="card-value">${s.total_categories}</div><div class="card-label">Categories</div></div><div class="stat-card animate-in"><div class="card-icon">&#11088;</div><div class="card-value">${s.avg_rating}</div><div class="card-label">Avg Rating</div></div><div class="stat-card animate-in"><div class="card-icon">&#128184;</div><div class="card-value">${s.avg_discount}%</div><div class="card-label">Avg Discount</div></div>`;renderBarChart('typeChart',s.type_counts,'primary');renderBarChart('brandChart',s.brand_counts,'alt');renderBarChart('priceChart',s.price_ranges,'accent');renderRatingChart('ratingChart',s.avg_rating_by_type)}catch(e){console.error(e)}}
function renderBarChart(id,data,cls){const e=Object.entries(data).sort((a,b)=>b[1]-a[1]).slice(0,8);const mx=Math.max(...e.map(x=>x[1]));document.getElementById(id).innerHTML='<div class="bar-chart">'+e.map(([l,v],i)=>`<div class="bar-row animate-in" style="animation-delay:${i*.05}s"><span class="bar-label" title="${l}">${l}</span><div class="bar-track"><div class="bar-fill ${cls}" style="width:${mx>0?v/mx*100:0}%"><span class="bar-value">${v}</span></div></div></div>`).join('')+'</div>'}
function renderRatingChart(id,data){const e=Object.entries(data).sort((a,b)=>b[1]-a[1]).slice(0,8);document.getElementById(id).innerHTML='<div class="bar-chart">'+e.map(([l,v],i)=>`<div class="bar-row animate-in" style="animation-delay:${i*.05}s"><span class="bar-label" title="${l}">${l}</span><div class="bar-track"><div class="bar-fill green" style="width:${v/5*100}%"><span class="bar-value">${v}</span></div></div></div>`).join('')+'</div>'}
async function loadProductTypes(){try{const r=await fetch(API+'/api/product-types');productTypes=await r.json();['typeFilter','recTypeFilter'].forEach(id=>{const s=document.getElementById(id);productTypes.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;s.appendChild(o)})});const r2=await fetch(API+'/api/products?per_page=500');const d=await r2.json();const s=document.getElementById('productSelector');d.products.forEach(p=>{const o=document.createElement('option');o.value=p.product_id;o.textContent=`[${p.brand}] ${p.name.substring(0,70)}...`;s.appendChild(o)})}catch(e){console.error(e)}}
async function loadProducts(){const g=document.getElementById('productsGrid');const search=document.getElementById('searchInput').value;const type=document.getElementById('typeFilter').value;g.innerHTML=Array(8).fill('<div class="skeleton-card"><div class="skeleton-image loading-skeleton"></div><div class="skeleton-text loading-skeleton"></div><div class="skeleton-text short loading-skeleton"></div></div>').join('');try{let u=API+`/api/products?page=${currentPage}&per_page=20`;if(type&&type!=='All')u+=`&type=${encodeURIComponent(type)}`;if(search)u+=`&search=${encodeURIComponent(search)}`;const r=await fetch(u);const d=await r.json();renderCards(g,d.products);renderPagination(d)}catch(e){g.innerHTML='<div class="rec-empty"><p>Failed to load</p></div>'}}
function renderCards(c,products){if(!products.length){c.innerHTML='<div class="rec-empty" style="grid-column:1/-1"><p>No products found.</p></div>';return}c.innerHTML=products.map((p,i)=>{const rt=p.rating?p.rating:'N/A',rv=p.num_ratings?p.num_ratings.toLocaleString('en-IN'):'0',dp=p.discount_price?`&#8377;${p.discount_price.toLocaleString('en-IN')}`:'',ap=p.actual_price?`&#8377;${p.actual_price.toLocaleString('en-IN')}`:'',dc=p.discount_pct>0?`${p.discount_pct}% off`:'';let sb='';if(p.similarity_score!=null)sb=`<span class="score-badge similarity">${(p.similarity_score*100).toFixed(0)}% match</span>`;else if(p.popularity_score!=null)sb=`<span class="score-badge popularity">${p.popularity_score.toFixed(2)}</span>`;else if(p.value_score!=null)sb=`<span class="score-badge value">${p.value_score.toFixed(0)}</span>`;else if(p.hybrid_score!=null)sb=`<span class="score-badge hybrid">${(p.hybrid_score*100).toFixed(0)}</span>`;return`<div class="product-card animate-in" style="animation-delay:${i*.05}s" onclick="selectProduct(${p.product_id})">${sb}<div class="product-image-wrap"><img src="${p.image}" alt="${p.name}" loading="lazy" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22><rect fill=%22%23f0ebe3%22 width=%22100%22 height=%22100%22/><text fill=%22%23999%22 x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22 font-size=%2212%22>No Image</text></svg>'"></div><div class="product-brand">${p.brand||''}</div><div class="product-name" title="${p.name}">${p.name}</div><div class="product-meta"><span class="product-rating"><span class="star">&#11088;</span> ${rt}</span><span class="product-reviews">${rv} reviews</span></div><div class="product-price">${dp?`<span class="price-current">${dp}</span>`:''}${ap&&dp?`<span class="price-original">${ap}</span>`:!dp&&ap?`<span class="price-current">${ap}</span>`:''}${dc?`<span class="price-discount">${dc}</span>`:''}</div></div>`}).join('')}
function selectProduct(id){document.querySelectorAll('.algo-tab').forEach(t=>t.classList.remove('active'));document.getElementById('tab-content').classList.add('active');currentAlgo='content';document.getElementById('contentControls').classList.remove('hidden');document.getElementById('priceControls').classList.add('hidden');document.getElementById('productSelector').value=id;updateAlgoExplanation();scrollToSection('recommend');setTimeout(()=>fetchRecommendations(),300)}
function renderPagination(d){const el=document.getElementById('pagination');if(d.pages<=1){el.innerHTML='';return}let h='';if(d.page>1)h+=`<button class="page-btn" onclick="goToPage(${d.page-1})">Prev</button>`;for(let i=Math.max(1,d.page-3);i<=Math.min(d.pages,d.page+3);i++)h+=`<button class="page-btn ${i===d.page?'active':''}" onclick="goToPage(${i})">${i}</button>`;if(d.page<d.pages)h+=`<button class="page-btn" onclick="goToPage(${d.page+1})">Next</button>`;el.innerHTML=h}
function goToPage(p){currentPage=p;loadProducts();scrollToSection('explore')}
async function fetchRecommendations(){const res=document.getElementById('recResults');const type=document.getElementById('recTypeFilter').value;res.innerHTML='<div class="products-grid">'+Array(8).fill('<div class="skeleton-card"><div class="skeleton-image loading-skeleton"></div><div class="skeleton-text loading-skeleton"></div><div class="skeleton-text short loading-skeleton"></div></div>').join('')+'</div>';let u='';switch(currentAlgo){case'popular':u=API+'/api/recommend/popular?n=12';if(type!=='All')u+=`&type=${encodeURIComponent(type)}`;break;case'content':u=API+`/api/recommend/content/${document.getElementById('productSelector').value}?n=12`;break;case'deals':u=API+'/api/recommend/deals?n=12';if(type!=='All')u+=`&type=${encodeURIComponent(type)}`;break;case'price':u=API+`/api/recommend/price-range?min=${document.getElementById('minPrice').value||0}&max=${document.getElementById('maxPrice').value||100000}&n=12`;if(type!=='All')u+=`&type=${encodeURIComponent(type)}`;break;case'hybrid':u=API+'/api/recommend/hybrid?n=12';if(type!=='All')u+=`&type=${encodeURIComponent(type)}`;const hp=document.getElementById('productSelector').value;if(hp)u+=`&product_id=${hp}`;break}try{const r=await fetch(u);const d=await r.json();const g=document.createElement('div');g.className='products-grid';res.innerHTML='';res.appendChild(g);renderCards(g,d)}catch(e){res.innerHTML='<div class="rec-empty"><p>Failed to load recommendations.</p></div>'}}
function updateAlgoExplanation(){const ex={popular:'<strong>Popularity-Based Filtering</strong><br>Uses <strong>Bayesian Weighted Rating</strong> (IMDB formula). Prevents products with few but perfect ratings from dominating.<br><code>WR = (v/(v+m)) x R + (m/(v+m)) x C</code>',content:'<strong>Content-Based Filtering (TF-IDF + Cosine Similarity)</strong><br>Converts product text into <strong>TF-IDF vectors</strong>, computes <strong>Cosine Similarity</strong> to find similar products.<br><code>similarity(A,B) = (A.B) / (||A|| x ||B||)</code>',deals:'<strong>Best Deals (Value Score)</strong><br>Combines <strong>discount percentage</strong> with <strong>rating quality</strong>. Only products rated 3.5+ are shown.<br><code>value = 0.4 x normalized_discount + 0.6 x normalized_rating</code>',price:'<strong>Price-Range Filtering</strong><br>Filters within your budget, ranks by <strong>rating</strong> and <strong>review volume</strong>.',hybrid:'<strong>Hybrid Algorithm</strong><br>Combines three signals with weights:<br><code>hybrid = 0.40 x content + 0.35 x popularity + 0.25 x value</code>'};document.getElementById('algoExplanation').innerHTML=ex[currentAlgo]||''}
</script>
</body>
</html>"""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: FLASK WEB SERVER & API ROUTES                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

app = Flask(__name__)

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'products.csv')
engine = RecommendationEngine(CSV_PATH)


@app.route('/')
def index():
    return HTML_TEMPLATE


@app.route('/api/stats')
def api_stats():
    return jsonify(engine.get_stats())


@app.route('/api/product-types')
def api_product_types():
    return jsonify(engine.get_product_types())


@app.route('/api/products')
def api_products():
    return jsonify(engine.get_all_products(
        page=request.args.get('page', 1, type=int),
        per_page=request.args.get('per_page', 20, type=int),
        product_type=request.args.get('type'),
        search=request.args.get('search')
    ))


@app.route('/api/recommend/content/<int:product_id>')
def api_content(product_id):
    return jsonify(engine.content_based(product_id, request.args.get('n', 8, type=int)))


@app.route('/api/recommend/popular')
def api_popular():
    return jsonify(engine.popularity_based(request.args.get('n', 8, type=int), request.args.get('type')))


@app.route('/api/recommend/price-range')
def api_price_range():
    return jsonify(engine.price_range_based(
        request.args.get('min', 0, type=float),
        request.args.get('max', 100000, type=float),
        request.args.get('type'),
        request.args.get('n', 8, type=int)
    ))


@app.route('/api/recommend/deals')
def api_deals():
    return jsonify(engine.best_deals(request.args.get('n', 8, type=int), request.args.get('type')))


@app.route('/api/recommend/hybrid')
def api_hybrid():
    return jsonify(engine.hybrid(
        request.args.get('product_id', None, type=int),
        request.args.get('type'),
        request.args.get('n', 8, type=int)
    ))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: RUN THE APPLICATION                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  ShopSense AI - Shopping Recommendation System")
    print("  Open: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
