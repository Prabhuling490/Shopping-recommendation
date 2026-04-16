/* ═══════════════════════════════════════════════════════════════════════════
   ShopSense AI — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════════════════ */

const API = '';

// ─── State ───────────────────────────────────────────────────────────────────
let currentAlgo = 'popular';
let currentPage = 1;
let productTypes = [];
let allStats = null;

// ─── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initNavScroll();
    initNavLinks();
    initAlgoTabs();
    initSearch();

    // Load data
    loadStats();
    loadProductTypes();
    loadProducts();
    updateAlgoExplanation();
});

// ─── Background Particles ────────────────────────────────────────────────────
function initParticles() {
    const container = document.getElementById('particles');
    for (let i = 0; i < 30; i++) {
        const p = document.createElement('div');
        p.classList.add('particle');
        p.style.left = Math.random() * 100 + '%';
        p.style.animationDuration = (8 + Math.random() * 15) + 's';
        p.style.animationDelay = (Math.random() * 10) + 's';
        p.style.width = p.style.height = (2 + Math.random() * 3) + 'px';
        p.style.opacity = 0.15 + Math.random() * 0.3;
        container.appendChild(p);
    }
}

// ─── Nav Scroll Effect ───────────────────────────────────────────────────────
function initNavScroll() {
    const nav = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        nav.classList.toggle('scrolled', window.scrollY > 50);
    });
}

// ─── Nav Links ───────────────────────────────────────────────────────────────
function initNavLinks() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            const section = link.dataset.section;
            scrollToSection(section);
        });
    });
}

function scrollToSection(id) {
    const el = document.getElementById(id);
    if (el) {
        const offset = 80;
        const y = el.getBoundingClientRect().top + window.pageYOffset - offset;
        window.scrollTo({ top: y, behavior: 'smooth' });
    }
}

// ─── Algorithm Tabs ──────────────────────────────────────────────────────────
function initAlgoTabs() {
    document.querySelectorAll('.algo-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.algo-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentAlgo = tab.dataset.algo;

            // Show/hide controls
            document.getElementById('contentControls').classList.toggle('hidden', currentAlgo !== 'content');
            document.getElementById('priceControls').classList.toggle('hidden', currentAlgo !== 'price');

            updateAlgoExplanation();
        });
    });
}

// ─── Search ──────────────────────────────────────────────────────────────────
function initSearch() {
    let debounce;
    document.getElementById('searchInput').addEventListener('input', (e) => {
        clearTimeout(debounce);
        debounce = setTimeout(() => {
            currentPage = 1;
            loadProducts();
        }, 300);
    });

    document.getElementById('typeFilter').addEventListener('change', () => {
        currentPage = 1;
        loadProducts();
    });
}

// ─── Load Statistics ─────────────────────────────────────────────────────────
async function loadStats() {
    try {
        const res = await fetch(`${API}/api/stats`);
        const stats = await res.json();
        allStats = stats;
        renderHeroStats(stats);
        renderDashboard(stats);
        renderCharts(stats);
    } catch (e) {
        console.error('Failed to load stats:', e);
    }
}

function renderHeroStats(stats) {
    const el = document.getElementById('heroStats');
    el.innerHTML = `
        <div class="stat-item animate-in" style="animation-delay:0.1s">
            <span class="stat-value">${stats.total_products}</span>
            <span class="stat-label">Products</span>
        </div>
        <div class="stat-item animate-in" style="animation-delay:0.2s">
            <span class="stat-value">${stats.total_brands}</span>
            <span class="stat-label">Brands</span>
        </div>
        <div class="stat-item animate-in" style="animation-delay:0.3s">
            <span class="stat-value">${stats.total_categories}</span>
            <span class="stat-label">Categories</span>
        </div>
        <div class="stat-item animate-in" style="animation-delay:0.4s">
            <span class="stat-value">${stats.avg_rating}★</span>
            <span class="stat-label">Avg Rating</span>
        </div>
        <div class="stat-item animate-in" style="animation-delay:0.5s">
            <span class="stat-value">${stats.avg_discount}%</span>
            <span class="stat-label">Avg Discount</span>
        </div>
    `;
}

function renderDashboard(stats) {
    const el = document.getElementById('dashboardGrid');
    el.innerHTML = `
        <div class="stat-card animate-in" style="animation-delay:0.1s">
            <div class="card-icon">📦</div>
            <div class="card-value">${stats.total_products}</div>
            <div class="card-label">Total Products</div>
        </div>
        <div class="stat-card animate-in" style="animation-delay:0.15s">
            <div class="card-icon">🏷️</div>
            <div class="card-value">${stats.total_brands}</div>
            <div class="card-label">Brands</div>
        </div>
        <div class="stat-card animate-in" style="animation-delay:0.2s">
            <div class="card-icon">📂</div>
            <div class="card-value">${stats.total_categories}</div>
            <div class="card-label">Categories</div>
        </div>
        <div class="stat-card animate-in" style="animation-delay:0.25s">
            <div class="card-icon">⭐</div>
            <div class="card-value">${stats.avg_rating}</div>
            <div class="card-label">Avg Rating</div>
        </div>
        <div class="stat-card animate-in" style="animation-delay:0.3s">
            <div class="card-icon">💸</div>
            <div class="card-value">${stats.avg_discount}%</div>
            <div class="card-label">Avg Discount</div>
        </div>
    `;
}

// ─── Charts (Pure CSS bar charts) ────────────────────────────────────────────
function renderCharts(stats) {
    renderBarChart('typeChart', stats.type_counts, 'primary');
    renderBarChart('brandChart', stats.brand_counts, 'alt');
    renderBarChart('priceChart', stats.price_ranges, 'accent');
    renderRatingChart('ratingChart', stats.avg_rating_by_type);
}

function renderBarChart(containerId, data, colorClass) {
    const container = document.getElementById(containerId);
    const entries = Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 8);
    const max = Math.max(...entries.map(e => e[1]));

    let html = '<div class="bar-chart">';
    entries.forEach(([label, value], i) => {
        const pct = max > 0 ? (value / max * 100) : 0;
        html += `
            <div class="bar-row animate-in" style="animation-delay:${i * 0.05}s">
                <span class="bar-label" title="${label}">${label}</span>
                <div class="bar-track">
                    <div class="bar-fill ${colorClass}" style="width:${pct}%">
                        <span class="bar-value">${value}</span>
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';
    container.innerHTML = html;
}

function renderRatingChart(containerId, data) {
    const container = document.getElementById(containerId);
    const entries = Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 8);

    let html = '<div class="bar-chart">';
    entries.forEach(([label, value], i) => {
        const pct = (value / 5 * 100);
        html += `
            <div class="bar-row animate-in" style="animation-delay:${i * 0.05}s">
                <span class="bar-label" title="${label}">${label}</span>
                <div class="bar-track">
                    <div class="bar-fill green" style="width:${pct}%">
                        <span class="bar-value">${value}★</span>
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';
    container.innerHTML = html;
}

// ─── Load Product Types ──────────────────────────────────────────────────────
async function loadProductTypes() {
    try {
        const res = await fetch(`${API}/api/product-types`);
        productTypes = await res.json();

        // Populate all filter dropdowns
        ['typeFilter', 'recTypeFilter'].forEach(id => {
            const sel = document.getElementById(id);
            productTypes.forEach(t => {
                const opt = document.createElement('option');
                opt.value = t;
                opt.textContent = t;
                sel.appendChild(opt);
            });
        });

        // Populate product selector for content-based
        loadProductSelector();
    } catch (e) {
        console.error('Failed to load types:', e);
    }
}

async function loadProductSelector() {
    try {
        const res = await fetch(`${API}/api/products?per_page=500`);
        const data = await res.json();
        const sel = document.getElementById('productSelector');
        data.products.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.product_id;
            opt.textContent = `[${p.brand}] ${p.name.substring(0, 70)}...`;
            sel.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load product selector:', e);
    }
}

// ─── Load Products (Explore) ─────────────────────────────────────────────────
async function loadProducts() {
    const grid = document.getElementById('productsGrid');
    const search = document.getElementById('searchInput').value;
    const type = document.getElementById('typeFilter').value;

    // Show skeleton
    grid.innerHTML = Array(8).fill(`
        <div class="skeleton-card">
            <div class="skeleton-image loading-skeleton"></div>
            <div class="skeleton-text loading-skeleton"></div>
            <div class="skeleton-text short loading-skeleton"></div>
        </div>
    `).join('');

    try {
        let url = `${API}/api/products?page=${currentPage}&per_page=20`;
        if (type && type !== 'All') url += `&type=${encodeURIComponent(type)}`;
        if (search) url += `&search=${encodeURIComponent(search)}`;

        const res = await fetch(url);
        const data = await res.json();

        renderProductCards(grid, data.products);
        renderPagination(data);
    } catch (e) {
        grid.innerHTML = '<div class="rec-empty"><p>Failed to load products</p></div>';
    }
}

function renderProductCards(container, products, algorithm = null) {
    if (!products.length) {
        container.innerHTML = `
            <div class="rec-empty" style="grid-column: 1 / -1">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
                </svg>
                <p>No products found. Try adjusting your filters.</p>
            </div>
        `;
        return;
    }

    container.innerHTML = products.map((p, i) => {
        const rating = p.rating ? `${p.rating}` : 'N/A';
        const reviews = p.num_ratings ? formatNumber(p.num_ratings) : '0';
        const discountPrice = p.discount_price ? `₹${formatNumber(p.discount_price)}` : '';
        const actualPrice = p.actual_price ? `₹${formatNumber(p.actual_price)}` : '';
        const discount = p.discount_pct > 0 ? `${p.discount_pct}% off` : '';

        let scoreBadge = '';
        if (p.similarity_score != null) {
            scoreBadge = `<span class="score-badge similarity">${(p.similarity_score * 100).toFixed(0)}% match</span>`;
        } else if (p.popularity_score != null) {
            scoreBadge = `<span class="score-badge popularity">🔥 ${p.popularity_score.toFixed(2)}</span>`;
        } else if (p.value_score != null) {
            scoreBadge = `<span class="score-badge value">💰 ${p.value_score.toFixed(0)}</span>`;
        } else if (p.hybrid_score != null) {
            scoreBadge = `<span class="score-badge hybrid">⚡ ${(p.hybrid_score * 100).toFixed(0)}</span>`;
        }

        return `
            <div class="product-card animate-in" style="animation-delay:${i * 0.05}s"
                 onclick="selectProduct(${p.product_id})">
                ${scoreBadge}
                <div class="product-image-wrap">
                    <img src="${p.image}" alt="${p.name}" loading="lazy"
                         onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22><rect fill=%22%23222%22 width=%22100%22 height=%22100%22/><text fill=%22%23666%22 x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22>No Image</text></svg>'">
                </div>
                <div class="product-brand">${p.brand || ''}</div>
                <div class="product-name" title="${p.name}">${p.name}</div>
                <div class="product-meta">
                    <span class="product-rating">
                        <span class="star">⭐</span> ${rating}
                    </span>
                    <span class="product-reviews">${reviews} reviews</span>
                </div>
                <div class="product-price">
                    ${discountPrice ? `<span class="price-current">${discountPrice}</span>` : ''}
                    ${actualPrice && discountPrice ? `<span class="price-original">${actualPrice}</span>` : ''}
                    ${!discountPrice && actualPrice ? `<span class="price-current">${actualPrice}</span>` : ''}
                    ${discount ? `<span class="price-discount">${discount}</span>` : ''}
                </div>
            </div>
        `;
    }).join('');
}

function selectProduct(productId) {
    // Switch to content-based tab and set the product
    document.querySelectorAll('.algo-tab').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-content').classList.add('active');
    currentAlgo = 'content';

    document.getElementById('contentControls').classList.remove('hidden');
    document.getElementById('priceControls').classList.add('hidden');

    const selector = document.getElementById('productSelector');
    selector.value = productId;

    updateAlgoExplanation();
    scrollToSection('recommend');

    // Auto-fetch recommendations
    setTimeout(() => fetchRecommendations(), 300);
}

// ─── Pagination ──────────────────────────────────────────────────────────────
function renderPagination(data) {
    const el = document.getElementById('pagination');
    if (data.pages <= 1) { el.innerHTML = ''; return; }

    let html = '';
    const start = Math.max(1, data.page - 3);
    const end = Math.min(data.pages, data.page + 3);

    if (data.page > 1) {
        html += `<button class="page-btn" onclick="goToPage(${data.page - 1})">← Prev</button>`;
    }

    for (let i = start; i <= end; i++) {
        html += `<button class="page-btn ${i === data.page ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
    }

    if (data.page < data.pages) {
        html += `<button class="page-btn" onclick="goToPage(${data.page + 1})">Next →</button>`;
    }

    el.innerHTML = html;
}

function goToPage(page) {
    currentPage = page;
    loadProducts();
    scrollToSection('explore');
}

// ─── Fetch Recommendations ───────────────────────────────────────────────────
async function fetchRecommendations() {
    const results = document.getElementById('recResults');
    const type = document.getElementById('recTypeFilter').value;

    // Show loading
    results.innerHTML = `
        <div class="products-grid">
            ${Array(8).fill(`
                <div class="skeleton-card">
                    <div class="skeleton-image loading-skeleton"></div>
                    <div class="skeleton-text loading-skeleton"></div>
                    <div class="skeleton-text short loading-skeleton"></div>
                </div>
            `).join('')}
        </div>
    `;

    let url = '';

    switch (currentAlgo) {
        case 'popular':
            url = `${API}/api/recommend/popular?n=12`;
            if (type !== 'All') url += `&type=${encodeURIComponent(type)}`;
            break;

        case 'content':
            const productId = document.getElementById('productSelector').value;
            url = `${API}/api/recommend/content/${productId}?n=12`;
            break;

        case 'deals':
            url = `${API}/api/recommend/deals?n=12`;
            if (type !== 'All') url += `&type=${encodeURIComponent(type)}`;
            break;

        case 'price':
            const minP = document.getElementById('minPrice').value || 0;
            const maxP = document.getElementById('maxPrice').value || 100000;
            url = `${API}/api/recommend/price-range?min=${minP}&max=${maxP}&n=12`;
            if (type !== 'All') url += `&type=${encodeURIComponent(type)}`;
            break;

        case 'hybrid':
            url = `${API}/api/recommend/hybrid?n=12`;
            if (type !== 'All') url += `&type=${encodeURIComponent(type)}`;
            const hybridProduct = document.getElementById('productSelector').value;
            if (hybridProduct) url += `&product_id=${hybridProduct}`;
            break;
    }

    try {
        const res = await fetch(url);
        const data = await res.json();

        const grid = document.createElement('div');
        grid.className = 'products-grid';
        results.innerHTML = '';
        results.appendChild(grid);
        renderProductCards(grid, data);
    } catch (e) {
        results.innerHTML = '<div class="rec-empty"><p>Failed to load recommendations. Please try again.</p></div>';
    }
}

// ─── Algorithm Explanations ──────────────────────────────────────────────────
function updateAlgoExplanation() {
    const el = document.getElementById('algoExplanation');
    const explanations = {
        popular: `
            <strong>🔥 Popularity-Based Filtering</strong><br>
            Uses <strong>Bayesian Weighted Rating</strong> (IMDB formula) to rank products.
            This prevents products with very few but perfect ratings from appearing at the top.
            The formula balances the product's average rating with the global mean rating,
            weighted by the number of ratings received.<br>
            <code>WR = (v/(v+m)) × R + (m/(v+m)) × C</code>
            where <strong>v</strong> = number of ratings, <strong>R</strong> = average rating,
            <strong>m</strong> = minimum threshold, <strong>C</strong> = global mean rating.
        `,
        content: `
            <strong>🔗 Content-Based Filtering (TF-IDF + Cosine Similarity)</strong><br>
            Converts product text (name, brand, category) into <strong>TF-IDF vectors</strong>,
            then computes <strong>Cosine Similarity</strong> between vectors to find products
            with similar descriptions. Select a product to find similar items.<br>
            <code>similarity(A,B) = (A·B) / (||A|| × ||B||)</code>
            Higher similarity = more related products in terms of features &amp; description.
        `,
        deals: `
            <strong>💰 Best Deals (Value Score)</strong><br>
            Combines <strong>discount percentage</strong> with <strong>rating quality</strong>
            to surface products offering the best value for money. Only products with
            ratings ≥ 3.5 are considered. <br>
            <code>value_score = 0.4 × normalized_discount + 0.6 × normalized_rating</code>
            This ensures top deals are both well-discounted AND well-reviewed.
        `,
        price: `
            <strong>💲 Price-Range Filtering</strong><br>
            Filters products within your specified budget, then ranks by
            <strong>rating (primary)</strong> and <strong>number of reviews (secondary)</strong>.
            Set your min and max budget to find the best-rated products within your range.
        `,
        hybrid: `
            <strong>⚡ Hybrid Recommendation Algorithm</strong><br>
            Our most sophisticated approach — combines <strong>three signals</strong> with
            configurable weights for balanced recommendations:<br>
            <code>hybrid = 0.40 × content_similarity + 0.35 × popularity + 0.25 × value_score</code><br>
            All scores are normalized to [0,1] before combining, ensuring fair contribution from each signal.
        `,
    };
    el.innerHTML = explanations[currentAlgo] || '';
}

// ─── Utilities ───────────────────────────────────────────────────────────────
function formatNumber(num) {
    if (typeof num === 'number') {
        return num.toLocaleString('en-IN');
    }
    return num;
}
