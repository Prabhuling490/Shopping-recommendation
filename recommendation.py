"""
Shopping Recommendation System Engine
=====================================
Implements multiple recommendation algorithms:
1. Content-Based Filtering (TF-IDF + Cosine Similarity)
2. Popularity-Based Recommendations
3. Price-Range Based Recommendations
4. Hybrid Recommendation (combining all signals)

Uses: pandas, scikit-learn, numpy
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import os

# ─── Data Loading & Preprocessing ────────────────────────────────────────────

def load_and_preprocess(csv_path):
    """Load the product CSV and clean/preprocess all fields."""
    df = pd.read_csv(csv_path)

    # Clean price columns: remove ₹ and commas, convert to float
    def parse_price(price_str):
        if pd.isna(price_str) or price_str == '':
            return np.nan
        price_str = str(price_str).replace('₹', '').replace(',', '').strip()
        try:
            return float(price_str)
        except ValueError:
            return np.nan

    df['discount_price_clean'] = df['discount_price'].apply(parse_price)
    df['actual_price_clean'] = df['actual_price'].apply(parse_price)

    # Clean ratings
    def parse_rating(r):
        try:
            val = float(r)
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass
        return np.nan

    df['rating_clean'] = df['ratings'].apply(parse_rating)

    # Clean number of ratings
    def parse_num_ratings(r):
        if pd.isna(r):
            return 0
        r = str(r).replace(',', '').strip()
        try:
            return int(r)
        except ValueError:
            return 0

    df['num_ratings_clean'] = df['no_of_ratings'].apply(parse_num_ratings)

    # Calculate discount percentage
    df['discount_pct'] = np.where(
        (df['actual_price_clean'] > 0) & (df['discount_price_clean'] > 0),
        ((df['actual_price_clean'] - df['discount_price_clean']) / df['actual_price_clean'] * 100).round(1),
        0
    )

    # Fill missing prices with actual price
    df['effective_price'] = df['discount_price_clean'].fillna(df['actual_price_clean'])

    # Extract brand from product name
    def extract_brand(name):
        if pd.isna(name):
            return 'Unknown'
        name = str(name).strip().strip('"')
        # Common brand patterns
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
        # Fallback: first word
        return name.split()[0] if name.split() else 'Unknown'

    df['brand'] = df['name'].apply(extract_brand)

    # Categorize products into product types
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

    # Create a combined text feature for TF-IDF
    df['text_features'] = (
        df['name'].fillna('') + ' ' +
        df['brand'].fillna('') + ' ' +
        df['product_type'].fillna('') + ' ' +
        df['main_category'].fillna('') + ' ' +
        df['sub_category'].fillna('')
    )

    # Assign unique IDs
    df['product_id'] = range(len(df))

    return df


# ─── Recommendation Algorithms ───────────────────────────────────────────────

class RecommendationEngine:
    """Multi-algorithm shopping recommendation engine."""

    def __init__(self, csv_path):
        self.df = load_and_preprocess(csv_path)
        self._build_tfidf_matrix()
        self._compute_popularity_scores()
        self._compute_value_scores()

    def _build_tfidf_matrix(self):
        """Build TF-IDF matrix for content-based filtering."""
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['text_features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def _compute_popularity_scores(self):
        """Compute weighted popularity score using Bayesian average."""
        C = self.df['rating_clean'].mean()  # Mean rating across all products
        m = self.df['num_ratings_clean'].quantile(0.25)  # Min ratings threshold

        def weighted_rating(row):
            v = row['num_ratings_clean']
            R = row['rating_clean']
            if pd.isna(R) or v == 0:
                return 0
            return (v / (v + m)) * R + (m / (v + m)) * C

        self.df['popularity_score'] = self.df.apply(weighted_rating, axis=1).round(3)

    def _compute_value_scores(self):
        """Compute value-for-money score based on discount and rating."""
        max_discount = self.df['discount_pct'].max()
        if max_discount > 0:
            norm_discount = self.df['discount_pct'] / max_discount
        else:
            norm_discount = 0

        max_rating = 5.0
        norm_rating = self.df['rating_clean'].fillna(0) / max_rating

        self.df['value_score'] = ((norm_discount * 0.4 + norm_rating * 0.6) * 100).round(1)

    # ── Content-Based Recommendations ──────────────────────────────────────

    def content_based(self, product_id, top_n=10):
        """Get similar products based on TF-IDF cosine similarity."""
        if product_id >= len(self.df):
            return []

        sim_scores = list(enumerate(self.cosine_sim[product_id]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Skip first (itself)
        sim_scores = sim_scores[1:top_n + 1]

        results = []
        for idx, score in sim_scores:
            row = self.df.iloc[idx]
            results.append({
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
                'similarity_score': round(float(score), 4),
                'algorithm': 'content_based'
            })

        return results

    # ── Popularity-Based Recommendations ──────────────────────────────────

    def popularity_based(self, top_n=10, product_type=None):
        """Get most popular products using Bayesian weighted rating."""
        filtered = self.df.copy()
        if product_type and product_type != 'All':
            filtered = filtered[filtered['product_type'] == product_type]

        filtered = filtered[filtered['rating_clean'].notna()]
        filtered = filtered.sort_values('popularity_score', ascending=False).head(top_n)

        results = []
        for _, row in filtered.iterrows():
            results.append({
                'product_id': int(row['product_id']),
                'name': str(row['name']),
                'brand': str(row['brand']),
                'product_type': str(row['product_type']),
                'image': str(row['image']),
                'rating': float(row['rating_clean']),
                'num_ratings': int(row['num_ratings_clean']),
                'discount_price': float(row['discount_price_clean']) if not pd.isna(row['discount_price_clean']) else None,
                'actual_price': float(row['actual_price_clean']) if not pd.isna(row['actual_price_clean']) else None,
                'discount_pct': float(row['discount_pct']),
                'popularity_score': float(row['popularity_score']),
                'algorithm': 'popularity'
            })

        return results

    # ── Price-Range Recommendations ───────────────────────────────────────

    def price_range_based(self, min_price=0, max_price=100000, product_type=None, top_n=10):
        """Get best-rated products within a price range."""
        filtered = self.df.copy()
        filtered = filtered[filtered['effective_price'].notna()]
        filtered = filtered[
            (filtered['effective_price'] >= min_price) &
            (filtered['effective_price'] <= max_price)
        ]

        if product_type and product_type != 'All':
            filtered = filtered[filtered['product_type'] == product_type]

        filtered = filtered[filtered['rating_clean'].notna()]
        filtered = filtered.sort_values(
            ['rating_clean', 'num_ratings_clean'],
            ascending=[False, False]
        ).head(top_n)

        results = []
        for _, row in filtered.iterrows():
            results.append({
                'product_id': int(row['product_id']),
                'name': str(row['name']),
                'brand': str(row['brand']),
                'product_type': str(row['product_type']),
                'image': str(row['image']),
                'rating': float(row['rating_clean']),
                'num_ratings': int(row['num_ratings_clean']),
                'discount_price': float(row['discount_price_clean']) if not pd.isna(row['discount_price_clean']) else None,
                'actual_price': float(row['actual_price_clean']) if not pd.isna(row['actual_price_clean']) else None,
                'discount_pct': float(row['discount_pct']),
                'algorithm': 'price_range'
            })

        return results

    # ── Best Deals Recommendations ────────────────────────────────────────

    def best_deals(self, top_n=10, product_type=None):
        """Get products with highest discount percentage and good ratings."""
        filtered = self.df.copy()
        filtered = filtered[
            (filtered['discount_pct'] > 0) &
            (filtered['rating_clean'].notna()) &
            (filtered['rating_clean'] >= 3.5)
        ]

        if product_type and product_type != 'All':
            filtered = filtered[filtered['product_type'] == product_type]

        filtered = filtered.sort_values('value_score', ascending=False).head(top_n)

        results = []
        for _, row in filtered.iterrows():
            results.append({
                'product_id': int(row['product_id']),
                'name': str(row['name']),
                'brand': str(row['brand']),
                'product_type': str(row['product_type']),
                'image': str(row['image']),
                'rating': float(row['rating_clean']),
                'num_ratings': int(row['num_ratings_clean']),
                'discount_price': float(row['discount_price_clean']) if not pd.isna(row['discount_price_clean']) else None,
                'actual_price': float(row['actual_price_clean']) if not pd.isna(row['actual_price_clean']) else None,
                'discount_pct': float(row['discount_pct']),
                'value_score': float(row['value_score']),
                'algorithm': 'best_deals'
            })

        return results

    # ── Hybrid Recommendations ────────────────────────────────────────────

    def hybrid(self, product_id=None, product_type=None, top_n=10):
        """
        Combine content-based + popularity + value scores for a
        blended recommendation.
        """
        if product_id is not None and product_id < len(self.df):
            # Content-based similarity
            sim_scores = self.cosine_sim[product_id]
        else:
            sim_scores = np.zeros(len(self.df))

        filtered = self.df.copy()
        filtered['content_score'] = sim_scores

        if product_type and product_type != 'All':
            filtered = filtered[filtered['product_type'] == product_type]

        # Remove the source product
        if product_id is not None:
            filtered = filtered[filtered['product_id'] != product_id]

        filtered = filtered[filtered['rating_clean'].notna()]

        # Normalize scores
        max_pop = filtered['popularity_score'].max()
        max_val = filtered['value_score'].max()
        max_content = filtered['content_score'].max()

        if max_pop > 0:
            filtered['norm_pop'] = filtered['popularity_score'] / max_pop
        else:
            filtered['norm_pop'] = 0

        if max_val > 0:
            filtered['norm_val'] = filtered['value_score'] / max_val
        else:
            filtered['norm_val'] = 0

        if max_content > 0:
            filtered['norm_content'] = filtered['content_score'] / max_content
        else:
            filtered['norm_content'] = 0

        # Weighted hybrid score
        w_content = 0.40
        w_pop = 0.35
        w_val = 0.25

        filtered['hybrid_score'] = (
            w_content * filtered['norm_content'] +
            w_pop * filtered['norm_pop'] +
            w_val * filtered['norm_val']
        ).round(4)

        filtered = filtered.sort_values('hybrid_score', ascending=False).head(top_n)

        results = []
        for _, row in filtered.iterrows():
            results.append({
                'product_id': int(row['product_id']),
                'name': str(row['name']),
                'brand': str(row['brand']),
                'product_type': str(row['product_type']),
                'image': str(row['image']),
                'rating': float(row['rating_clean']),
                'num_ratings': int(row['num_ratings_clean']),
                'discount_price': float(row['discount_price_clean']) if not pd.isna(row['discount_price_clean']) else None,
                'actual_price': float(row['actual_price_clean']) if not pd.isna(row['actual_price_clean']) else None,
                'discount_pct': float(row['discount_pct']),
                'hybrid_score': float(row['hybrid_score']),
                'algorithm': 'hybrid'
            })

        return results

    # ── Analytics / Stats ─────────────────────────────────────────────────

    def get_stats(self):
        """Return dataset statistics for the dashboard."""
        df = self.df
        type_counts = df['product_type'].value_counts().to_dict()
        brand_counts = df['brand'].value_counts().head(15).to_dict()

        avg_rating_by_type = (
            df.groupby('product_type')['rating_clean']
            .mean()
            .dropna()
            .round(2)
            .to_dict()
        )

        avg_price_by_type = (
            df.groupby('product_type')['effective_price']
            .mean()
            .dropna()
            .round(0)
            .to_dict()
        )

        price_ranges = {
            'Under ₹500': int(len(df[df['effective_price'] < 500])),
            '₹500 - ₹1000': int(len(df[(df['effective_price'] >= 500) & (df['effective_price'] < 1000)])),
            '₹1000 - ₹5000': int(len(df[(df['effective_price'] >= 1000) & (df['effective_price'] < 5000)])),
            '₹5000 - ₹15000': int(len(df[(df['effective_price'] >= 5000) & (df['effective_price'] < 15000)])),
            '₹15000 - ₹30000': int(len(df[(df['effective_price'] >= 15000) & (df['effective_price'] < 30000)])),
            'Over ₹30000': int(len(df[df['effective_price'] >= 30000])),
        }

        return {
            'total_products': int(len(df)),
            'total_brands': int(df['brand'].nunique()),
            'total_categories': int(df['product_type'].nunique()),
            'avg_rating': round(float(df['rating_clean'].mean()), 2),
            'avg_discount': round(float(df['discount_pct'].mean()), 1),
            'type_counts': type_counts,
            'brand_counts': brand_counts,
            'avg_rating_by_type': avg_rating_by_type,
            'avg_price_by_type': {k: int(v) for k, v in avg_price_by_type.items()},
            'price_ranges': price_ranges,
        }

    def get_product_types(self):
        """Return list of product types."""
        types = sorted(self.df['product_type'].unique().tolist())
        return types

    def get_all_products(self, page=1, per_page=20, product_type=None, search=None):
        """Paginated product listing with optional filtering."""
        filtered = self.df.copy()

        if product_type and product_type != 'All':
            filtered = filtered[filtered['product_type'] == product_type]

        if search:
            mask = filtered['name'].str.contains(search, case=False, na=False)
            filtered = filtered[mask]

        total = len(filtered)
        start = (page - 1) * per_page
        end = start + per_page
        page_df = filtered.iloc[start:end]

        results = []
        for _, row in page_df.iterrows():
            results.append({
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
            })

        return {
            'products': results,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        }


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    engine = RecommendationEngine(os.path.join('data', 'products.csv'))

    print("=== Dataset Stats ===")
    stats = engine.get_stats()
    print(f"Total Products: {stats['total_products']}")
    print(f"Total Brands: {stats['total_brands']}")
    print(f"Avg Rating: {stats['avg_rating']}")
    print(f"Product Types: {stats['type_counts']}")

    print("\n=== Top 5 Popular Products ===")
    for p in engine.popularity_based(top_n=5):
        print(f"  [{p['rating']}*] {p['name'][:60]}... (Score: {p['popularity_score']})")

    print("\n=== Content-Based: Similar to Product 0 ===")
    for p in engine.content_based(0, top_n=5):
        print(f"  [{p['similarity_score']:.3f}] {p['name'][:60]}...")

    print("\n=== Best Deals ===")
    for p in engine.best_deals(top_n=5):
        print(f"  [{p['discount_pct']}% off] {p['name'][:60]}...")

    print("\n=== Hybrid (for product 0) ===")
    for p in engine.hybrid(product_id=0, top_n=5):
        print(f"  [{p['hybrid_score']:.3f}] {p['name'][:60]}...")
