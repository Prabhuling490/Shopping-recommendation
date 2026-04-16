"""
Shopping Recommendation System - Flask Web Server
==================================================
Serves the web UI and exposes REST API endpoints
for the recommendation engine.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import os
from recommendation import RecommendationEngine

app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize the recommendation engine
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'products.csv')
engine = RecommendationEngine(CSV_PATH)

# ─── Web Routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')

# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route('/api/stats')
def api_stats():
    """Return dataset statistics."""
    return jsonify(engine.get_stats())


@app.route('/api/product-types')
def api_product_types():
    """Return all product type categories."""
    return jsonify(engine.get_product_types())


@app.route('/api/products')
def api_products():
    """Paginated product listing."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    product_type = request.args.get('type', None)
    search = request.args.get('search', None)
    return jsonify(engine.get_all_products(page, per_page, product_type, search))


@app.route('/api/recommend/content/<int:product_id>')
def api_content_recommend(product_id):
    """Content-based recommendations for a specific product."""
    top_n = request.args.get('n', 8, type=int)
    results = engine.content_based(product_id, top_n)
    return jsonify(results)


@app.route('/api/recommend/popular')
def api_popular():
    """Popularity-based recommendations."""
    top_n = request.args.get('n', 8, type=int)
    product_type = request.args.get('type', None)
    results = engine.popularity_based(top_n, product_type)
    return jsonify(results)


@app.route('/api/recommend/price-range')
def api_price_range():
    """Price-range based recommendations."""
    min_price = request.args.get('min', 0, type=float)
    max_price = request.args.get('max', 100000, type=float)
    product_type = request.args.get('type', None)
    top_n = request.args.get('n', 8, type=int)
    results = engine.price_range_based(min_price, max_price, product_type, top_n)
    return jsonify(results)


@app.route('/api/recommend/deals')
def api_deals():
    """Best deals recommendations."""
    top_n = request.args.get('n', 8, type=int)
    product_type = request.args.get('type', None)
    results = engine.best_deals(top_n, product_type)
    return jsonify(results)


@app.route('/api/recommend/hybrid')
def api_hybrid():
    """Hybrid recommendations."""
    product_id = request.args.get('product_id', None, type=int)
    product_type = request.args.get('type', None)
    top_n = request.args.get('n', 8, type=int)
    results = engine.hybrid(product_id, product_type, top_n)
    return jsonify(results)


# ─── Run Server ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Shopping Recommendation System")
    print("  Open: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
