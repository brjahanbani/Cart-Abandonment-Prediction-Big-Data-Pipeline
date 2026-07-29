# # Stage 1 — Dataset Gathering and Definition
# ## Retail Rocket · Cart Abandonment Prediction · `events.csv` only
# 
# **Objective:** Understand the raw dataset, reconstruct sessions, detect and remove bot traffic,
# engineer session-level features, validate their predictive signal, and export a clean dataset
# ready for the Kafka + Spark pipeline in Stage 2.
# 
# **Sections:**
# 1. Setup & Data Load
# 2. Raw Dataset Inspection
# 3. Session Reconstruction
# 4. Class Imbalance Analysis
# 5. Bot Detection & Removal
# 6. Session Feature Engineering
# 7. Predictive Signal Validation
# 8. Summary Statistics & Export

# ---
# ## 1. Setup & Data Load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.color':       '#e8e8e8',
    'grid.linewidth':   0.6,
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
})
COLORS = {
    'purple': '#7F77DD',
    'teal':   '#1D9E75',
    'amber':  '#EF9F27',
    'coral':  '#D85A30',
    'gray':   '#888780',
}

print('Libraries loaded successfully.')

# ── Load events.csv ──────────────────────────────────────────────────────────
# Adjust path if your file is in a different location
DATA_PATH = 'events.csv'

df_raw = pd.read_csv(
    DATA_PATH,
    header=0,
    names=['timestamp', 'visitorid', 'event', 'itemid', 'transactionid'],
    dtype={
        'timestamp':     'int64',
        'visitorid':     'int64',
        'event':         'category',
        'itemid':        'Int64',    # nullable int — transactionid is sparse
        'transactionid': 'Int64',
    }
)

print(f'Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns')
df_raw.head()

# ---
# ## 2. Raw Dataset Inspection

# ── Basic shape & dtypes ─────────────────────────────────────────────────────
print('=== Shape ===')
print(f'  Rows        : {len(df_raw):>12,}')
print(f'  Columns     : {df_raw.shape[1]:>12}')
print()
print('=== Dtypes ===')
print(df_raw.dtypes.to_string())
print()
print('=== Missing values ===')
print(df_raw.isnull().sum().to_string())
print()
print('Note: transactionid is NULL for all view and addtocart events — expected.')

# ── Convert timestamp to datetime for readability ────────────────────────────
df_raw['event_time'] = pd.to_datetime(df_raw['timestamp'], unit='ms')

ts_min = df_raw['event_time'].min()
ts_max = df_raw['event_time'].max()
span_days = (ts_max - ts_min).days

print(f'Date range  : {ts_min.date()}  →  {ts_max.date()}')
print(f'Span        : {span_days} days  (~{span_days/30:.1f} months)')
print()
print('=== Event type counts ===')
event_counts = df_raw['event'].value_counts()
for etype, cnt in event_counts.items():
    print(f'  {etype:<15}: {cnt:>10,}  ({cnt/len(df_raw)*100:.2f}%)')
print()
print('=== Unique entities ===')
print(f'  Unique visitors  : {df_raw["visitorid"].nunique():>10,}')
print(f'  Unique items     : {df_raw["itemid"].nunique():>10,}')
print(f'  Unique trans.    : {df_raw["transactionid"].nunique():>10,}')

# ── Event volume over time ───────────────────────────────────────────────────
df_raw['date'] = df_raw['event_time'].dt.date
daily = df_raw.groupby(['date', 'event']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(13, 4))
daily['view'].plot(ax=ax, color=COLORS['purple'], label='view', linewidth=1.5)
ax2 = ax.twinx()
if 'addtocart' in daily:
    daily['addtocart'].plot(ax=ax2, color=COLORS['teal'],
                            label='addtocart', linewidth=1.2, linestyle='--')
if 'transaction' in daily:
    daily['transaction'].plot(ax=ax2, color=COLORS['amber'],
                              label='transaction', linewidth=1.2, linestyle=':')

ax.set_ylabel('View events / day', color=COLORS['purple'])
ax2.set_ylabel('Cart / Transaction events / day', color=COLORS['teal'])
ax.set_xlabel('')
ax.set_title('Daily event volume over the observation period')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax2.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('fig1_daily_volume.png', dpi=150, bbox_inches='tight')
plt.show()

# ---
# ## 3. Session Reconstruction
# 
# A **session** is defined as a contiguous sequence of events by the same visitor
# where no two consecutive events are separated by more than **30 minutes**.
# This is the standard e-commerce session timeout used in industry and matches
# the inactivity timeout used in the Spark state store in Stage 2.

# ── Sort by visitor and time — essential before gap computation ──────────────
df = df_raw.sort_values(['visitorid', 'timestamp']).reset_index(drop=True)

SESSION_GAP_MS = 30 * 60 * 1000   # 30 minutes in milliseconds

# Time since previous event for the SAME visitor
df['prev_ts']    = df.groupby('visitorid')['timestamp'].shift(1)
df['gap_ms']     = df['timestamp'] - df['prev_ts']

# A new session starts when:
#   (a) it is the visitor's very first event (gap_ms is NaN), OR
#   (b) the gap to the previous event exceeds 30 minutes
df['new_session'] = (
    df['gap_ms'].isna() |
    (df['gap_ms'] > SESSION_GAP_MS)
).astype(int)

# Cumulative sum gives a unique session counter per visitor
df['session_id'] = (
    df.groupby('visitorid')['new_session'].cumsum()
)

# Composite session key: stable unique identifier
df['session_key'] = df['visitorid'].astype(str) + '_' + df['session_id'].astype(str)

n_sessions = df['session_key'].nunique()
n_visitors = df['visitorid'].nunique()
print(f'Total sessions reconstructed : {n_sessions:>10,}')
print(f'Unique visitors              : {n_visitors:>10,}')
print(f'Avg sessions per visitor     : {n_sessions/n_visitors:>10.2f}')

# ── Session length distribution ──────────────────────────────────────────────
session_lengths = df.groupby('session_key').size()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: full distribution (log scale x-axis to handle long tail)
ax = axes[0]
ax.hist(session_lengths.clip(upper=100), bins=50,
        color=COLORS['purple'], edgecolor='white', linewidth=0.3)
ax.set_xlabel('Events per session (clipped at 100)')
ax.set_ylabel('Number of sessions')
ax.set_title('Session length distribution')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Right: percentile breakdown
ax2 = axes[1]
percentiles = [50, 75, 90, 95, 99, 99.9]
values = [np.percentile(session_lengths, p) for p in percentiles]
bars = ax2.barh([f'p{p}' for p in percentiles], values,
                color=COLORS['teal'], edgecolor='white')
ax2.set_xlabel('Events per session')
ax2.set_title('Session length percentiles')
for bar, val in zip(bars, values):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.0f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('fig2_session_lengths.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Median session length  : {session_lengths.median():.0f} events')
print(f'Mean session length    : {session_lengths.mean():.2f} events')
print(f'Max session length     : {session_lengths.max():,} events')
print(f'Sessions with 1 event  : {(session_lengths == 1).sum():,} ({(session_lengths==1).mean()*100:.1f}%)')

# ---
# ## 4. Class Imbalance Analysis
# 
# The binary label is: **did this session contain at least one `transaction` event?**
# 
# This is the target variable for the classifier. We compute it at the session level.

# ── Compute session-level label ──────────────────────────────────────────────
session_label = df.groupby('session_key').apply(
    lambda g: int((g['event'] == 'transaction').any())
).rename('has_transaction')

label_counts = session_label.value_counts()
n_total      = len(session_label)
n_purchase   = label_counts.get(1, 0)
n_abandon    = label_counts.get(0, 0)

print('=== Class distribution at session level ===')
print(f'  Total sessions    : {n_total:>10,}')
print(f'  Purchase (label=1): {n_purchase:>10,}  ({n_purchase/n_total*100:.2f}%)')
print(f'  Abandon  (label=0): {n_abandon:>10,}  ({n_abandon/n_total*100:.2f}%)')
print(f'  Imbalance ratio   : {n_abandon/n_purchase:.1f} : 1')

# ── Visualise imbalance ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Pie chart
ax = axes[0]
ax.pie(
    [n_abandon, n_purchase],
    labels=['Abandoned\n(no transaction)', 'Purchased'],
    colors=[COLORS['gray'], COLORS['teal']],
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
ax.set_title('Session class distribution')

# Events per session by class
ax2 = axes[1]
df_with_label = df.merge(
    session_label.reset_index(),
    on='session_key'
)
session_summary = df_with_label.groupby(['session_key', 'has_transaction']).size().reset_index(name='n_events')

for label_val, color, name in [(0, COLORS['gray'], 'Abandoned'), (1, COLORS['teal'], 'Purchased')]:
    subset = session_summary[session_summary['has_transaction'] == label_val]['n_events'].clip(upper=50)
    ax2.hist(subset, bins=30, alpha=0.65, color=color, label=name,
             edgecolor='white', linewidth=0.3, density=True)

ax2.set_xlabel('Events per session (clipped at 50)')
ax2.set_ylabel('Density')
ax2.set_title('Session length by class')
ax2.legend()

plt.tight_layout()
plt.savefig('fig3_class_imbalance.png', dpi=150, bbox_inches='tight')
plt.show()

# ---
# ## 5. Bot Detection & Removal
# 
# Retail Rocket's own dataset description warns that up to **40% of browsing data
# can be abnormal traffic**. We identify bots using two independent behavioral signals:
# 
# - **Velocity rule:** A visitor who produces more than `N` events per hour is almost
#   certainly a crawler. Real humans do not sustain high click rates for extended periods.
# - **Regularity rule:** Bots execute requests at machine-precise intervals.
#   The Coefficient of Variation (CV = std / mean) of inter-event times is near 0 for bots;
#   humans show CV > 0.3 due to natural browsing variability.
# 
# We flag a visitor as a bot if **either** signal triggers.

# ── Compute per-visitor behavioral statistics ────────────────────────────────
BOT_VELOCITY_THRESHOLD  = 500   # events per hour — above this → likely bot
BOT_CV_THRESHOLD        = 0.05  # coefficient of variation — below this → likely bot
BOT_MIN_EVENTS          = 20    # need at least 20 events to compute CV reliably

visitor_stats = df.groupby('visitorid').agg(
    total_events     = ('event',     'count'),
    session_span_ms  = ('timestamp', lambda x: x.max() - x.min()),
    n_sessions       = ('session_key', 'nunique'),
).reset_index()

# Events per hour (guard division by zero — visitors with all events at same ms)
visitor_stats['span_hours'] = (visitor_stats['session_span_ms'] / 3_600_000).clip(lower=1/60)
visitor_stats['events_per_hour'] = (
    visitor_stats['total_events'] / visitor_stats['span_hours']
)

# Coefficient of variation of inter-event intervals per visitor
def compute_cv(group):
    intervals = group['timestamp'].sort_values().diff().dropna()
    if len(intervals) < BOT_MIN_EVENTS:
        return np.nan
    mean_iv = intervals.mean()
    if mean_iv == 0:
        return 0.0
    return intervals.std() / mean_iv

print('Computing inter-event CV per visitor (this takes ~1–2 min on 2.7M rows)...')
visitor_cv = df.groupby('visitorid').apply(compute_cv).rename('interval_cv').reset_index()
visitor_stats = visitor_stats.merge(visitor_cv, on='visitorid')
print('Done.')

# ── Apply bot flags ──────────────────────────────────────────────────────────
visitor_stats['is_bot_velocity'] = (
    visitor_stats['events_per_hour'] > BOT_VELOCITY_THRESHOLD
)
visitor_stats['is_bot_cv'] = (
    visitor_stats['interval_cv'].notna() &
    (visitor_stats['interval_cv'] < BOT_CV_THRESHOLD)
)
visitor_stats['is_bot'] = (
    visitor_stats['is_bot_velocity'] | visitor_stats['is_bot_cv']
)

n_bots = visitor_stats['is_bot'].sum()
n_vel_bots = visitor_stats['is_bot_velocity'].sum()
n_cv_bots  = visitor_stats['is_bot_cv'].sum()
n_total_visitors = len(visitor_stats)

print('=== Bot Detection Results ===')
print(f'  Total visitors       : {n_total_visitors:>10,}')
print(f'  Flagged by velocity  : {n_vel_bots:>10,}  ({n_vel_bots/n_total_visitors*100:.2f}%)')
print(f'  Flagged by CV        : {n_cv_bots:>10,}  ({n_cv_bots/n_total_visitors*100:.2f}%)')
print(f'  Total bots (union)   : {n_bots:>10,}  ({n_bots/n_total_visitors*100:.2f}%)')
print()

bot_events = df['visitorid'].isin(
    visitor_stats.loc[visitor_stats['is_bot'], 'visitorid']
).sum()
print(f'  Bot events removed   : {bot_events:>10,}  ({bot_events/len(df)*100:.2f}% of all events)')

# ── Visualise bot signatures ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Events per hour distribution
ax = axes[0]
human_eph  = visitor_stats.loc[~visitor_stats['is_bot'], 'events_per_hour'].clip(upper=600)
bot_eph    = visitor_stats.loc[ visitor_stats['is_bot'], 'events_per_hour'].clip(upper=600)
ax.hist(human_eph, bins=60, color=COLORS['teal'], alpha=0.7,
        label='Human', density=True, edgecolor='white', linewidth=0.2)
ax.hist(bot_eph,   bins=60, color=COLORS['coral'], alpha=0.7,
        label='Bot',   density=True, edgecolor='white', linewidth=0.2)
ax.axvline(BOT_VELOCITY_THRESHOLD, color=COLORS['coral'],
           linestyle='--', linewidth=1.5, label=f'Threshold ({BOT_VELOCITY_THRESHOLD})')
ax.set_xlabel('Events per hour (clipped at 600)')
ax.set_ylabel('Density')
ax.set_title('Velocity: human vs bot distribution')
ax.legend(fontsize=9)

# CV distribution
ax2 = axes[1]
vs_cv = visitor_stats.dropna(subset=['interval_cv'])
human_cv = vs_cv.loc[~vs_cv['is_bot'], 'interval_cv'].clip(upper=3)
bot_cv   = vs_cv.loc[ vs_cv['is_bot'], 'interval_cv'].clip(upper=3)
ax2.hist(human_cv, bins=60, color=COLORS['teal'], alpha=0.7,
         label='Human', density=True, edgecolor='white', linewidth=0.2)
ax2.hist(bot_cv,   bins=60, color=COLORS['coral'], alpha=0.7,
         label='Bot',   density=True, edgecolor='white', linewidth=0.2)
ax2.axvline(BOT_CV_THRESHOLD, color=COLORS['coral'],
            linestyle='--', linewidth=1.5, label=f'Threshold ({BOT_CV_THRESHOLD})')
ax2.set_xlabel('CV of inter-event intervals (clipped at 3)')
ax2.set_ylabel('Density')
ax2.set_title('Timing regularity: human vs bot')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('fig4_bot_detection.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Remove bots from the dataset ─────────────────────────────────────────────
bot_visitors = set(visitor_stats.loc[visitor_stats['is_bot'], 'visitorid'])

df_clean = df[~df['visitorid'].isin(bot_visitors)].copy()

print(f'Events before bot removal : {len(df):>10,}')
print(f'Events after  bot removal : {len(df_clean):>10,}')
print(f'Events removed            : {len(df)-len(df_clean):>10,}  ({(1-len(df_clean)/len(df))*100:.2f}%)')
print()
print(f'Unique visitors before    : {df["visitorid"].nunique():>10,}')
print(f'Unique visitors after     : {df_clean["visitorid"].nunique():>10,}')

# ---
# ## 6. Session Feature Engineering
# 
# We compute 8 session-level features from `events.csv` alone.
# These are the same features the Spark Structured Streaming job will maintain
# as running aggregates in Stage 2 — we compute them here offline to validate
# predictive signal before investing in the pipeline.
# 
# | Feature | Description | Why it matters |
# |---|---|---|
# | `event_count` | Total events in session | Overall engagement |
# | `view_count` | Number of view events | Browsing depth |
# | `addtocart_count` | Number of add-to-cart events | Purchase intent signal |
# | `view_to_cart_ratio` | view_count / addtocart_count | Strongest predictor per Waldmann et al. |
# | `unique_items_count` | Distinct items viewed | Browsing breadth |
# | `session_age_seconds` | Session duration | Engagement duration |
# | `inter_event_interval_mean` | Mean time between events (sec) | Navigation pace |
# | `recency_last_cart_seconds` | Seconds since last cart action | Recency of intent |

# ── Compute session-level features ───────────────────────────────────────────
def build_session_features(group):
    g = group.sort_values('timestamp')

    ts        = g['timestamp'].values
    events    = g['event'].values
    items     = g['itemid'].dropna().values

    view_mask     = events == 'view'
    cart_mask     = events == 'addtocart'
    txn_mask      = events == 'transaction'

    view_cnt  = view_mask.sum()
    cart_cnt  = cart_mask.sum()

    # View-to-cart ratio: high ratio = many views, few carts = browsing without intent
    # Low ratio (approaching 1) = very focused, high intent
    v2c = view_cnt / cart_cnt if cart_cnt > 0 else float(view_cnt)

    # Inter-event intervals
    intervals = np.diff(ts) / 1000.0  # ms → seconds
    mean_interval = intervals.mean() if len(intervals) > 0 else 0.0

    # Session age in seconds
    age_sec = (ts[-1] - ts[0]) / 1000.0 if len(ts) > 1 else 0.0

    # Recency of last cart action
    cart_ts = ts[cart_mask]
    recency_cart = (ts[-1] - cart_ts[-1]) / 1000.0 if len(cart_ts) > 0 else -1.0

    return pd.Series({
        'event_count':                  len(g),
        'view_count':                   int(view_cnt),
        'addtocart_count':              int(cart_cnt),
        'transaction_count':            int(txn_mask.sum()),
        'view_to_cart_ratio':           round(v2c, 4),
        'unique_items_count':           int(pd.Series(items).nunique()),
        'session_age_seconds':          round(age_sec, 2),
        'inter_event_interval_mean':    round(mean_interval, 3),
        'recency_last_cart_seconds':    round(recency_cart, 2),
        'has_transaction':              int(txn_mask.any()),
    })

print('Building session features (this takes 2–4 minutes for 2.7M rows)...')
session_features = df_clean.groupby('session_key').apply(build_session_features).reset_index()
print(f'Done. Shape: {session_features.shape}')
session_features.head()

# ── Feature distributions by class ───────────────────────────────────────────
FEATURES = [
    'event_count', 'view_count', 'addtocart_count',
    'view_to_cart_ratio', 'unique_items_count',
    'session_age_seconds', 'inter_event_interval_mean',
    'recency_last_cart_seconds'
]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

sf_purchase = session_features[session_features['has_transaction'] == 1]
sf_abandon  = session_features[session_features['has_transaction'] == 0]

for ax, feat in zip(axes, FEATURES):
    # Clip extreme outliers for visualisation only
    q99 = session_features[feat].quantile(0.99)
    a_vals = sf_abandon[feat].clip(upper=q99)
    p_vals = sf_purchase[feat].clip(upper=q99)

    ax.hist(a_vals, bins=40, density=True, color=COLORS['gray'],
            alpha=0.65, label='Abandoned', edgecolor='white', linewidth=0.2)
    ax.hist(p_vals, bins=40, density=True, color=COLORS['teal'],
            alpha=0.65, label='Purchased', edgecolor='white', linewidth=0.2)
    ax.set_title(feat.replace('_', ' '), fontsize=9, fontweight='bold')
    ax.set_ylabel('Density', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)

plt.suptitle('Feature distributions by class (abandoned vs purchased)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig5_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Feature correlation heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr = session_features[FEATURES + ['has_transaction']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, ax=ax,
    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
    annot=True, fmt='.2f', annot_kws={'size': 9},
    linewidths=0.4, linecolor='white',
    square=True
)
ax.set_title('Feature correlation matrix (lower triangle)', fontweight='bold')
plt.tight_layout()
plt.savefig('fig6_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

print('Correlation with has_transaction:')
print(corr['has_transaction'].drop('has_transaction').sort_values(ascending=False).to_string())

# ---
# ## 7. Predictive Signal Validation
# 
# Before building the streaming pipeline, we validate that the 8 session features
# derived from `events.csv` alone carry meaningful predictive signal.
# 
# **Method:** Logistic Regression with 5-fold stratified cross-validation.
# This is deliberately simple — if a linear model achieves AUC > 0.70, the features
# are real and worth building a pipeline around. A complex model (XGBoost) will
# improve on this in Stage 3.
# 
# **Note on class imbalance:** We use `class_weight='balanced'` in the Logistic
# Regression instead of SMOTE, consistent with Waldmann et al. (2025) finding that
# resampling methods introduce noise and reduce held-out performance.

# ── Prepare feature matrix ────────────────────────────────────────────────────
# Drop sessions with all-zero recency (no cart event) for this validation only
# In the pipeline, recency = -1 is a valid feature value (no cart in session)
sf_model = session_features.copy()

# Replace -1 (no cart action) with a large value to signal absence
sf_model['recency_last_cart_seconds'] = sf_model['recency_last_cart_seconds'].replace(-1, 9999)

# Cap extreme outliers at 99th percentile — prevents logistic regression instability
for feat in FEATURES:
    q99 = sf_model[feat].quantile(0.99)
    sf_model[feat] = sf_model[feat].clip(upper=q99)

X = sf_model[FEATURES].fillna(0).values
y = sf_model['has_transaction'].values

print(f'Feature matrix: {X.shape}')
print(f'Class balance : {y.mean()*100:.2f}% positive')

# ── 5-fold stratified cross-validation ───────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(
    class_weight='balanced',   # handles imbalance without resampling
    max_iter=1000,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores  = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')
f1_scores   = cross_val_score(lr, X_scaled, y, cv=cv, scoring='f1')
prec_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='precision')
rec_scores  = cross_val_score(lr, X_scaled, y, cv=cv, scoring='recall')

print('=== 5-fold Cross-Validation Results (Logistic Regression) ===')
print(f'  AUC-ROC : {auc_scores.mean():.4f}  ± {auc_scores.std():.4f}')
print(f'  F1      : {f1_scores.mean():.4f}  ± {f1_scores.std():.4f}')
print(f'  Precision: {prec_scores.mean():.4f}  ± {prec_scores.std():.4f}')
print(f'  Recall  : {rec_scores.mean():.4f}  ± {rec_scores.std():.4f}')
print()
if auc_scores.mean() >= 0.70:
    print('✓ AUC >= 0.70 — features have sufficient predictive signal.')
    print('  Proceeding to Stage 2 (Kafka pipeline) is justified.')
else:
    print('✗ AUC < 0.70 — revisit feature engineering before Stage 2.')

# ── Feature importance (logistic regression coefficients) ─────────────────────
lr.fit(X_scaled, y)
coef_df = pd.DataFrame({
    'feature':     FEATURES,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
colors = [COLORS['teal'] if c > 0 else COLORS['coral'] for c in coef_df['coefficient']]
bars = ax.barh(coef_df['feature'], coef_df['coefficient'],
               color=colors, edgecolor='white')
ax.axvline(0, color=COLORS['gray'], linewidth=0.8)
ax.set_xlabel('Logistic regression coefficient (standardised features)')
ax.set_title('Feature importance for purchase prediction\n'
             '(positive = increases purchase probability)')
for bar, val in zip(bars, coef_df['coefficient']):
    ax.text(val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center',
            ha='left' if val >= 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig('fig7_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ── ROC curve ────────────────────────────────────────────────────────────────
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
lr.fit(X_tr, y_tr)
y_prob = lr.predict_proba(X_te)[:, 1]
fpr, tpr, thresholds = roc_curve(y_te, y_prob)
auc_val = roc_auc_score(y_te, y_prob)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=COLORS['purple'], linewidth=2,
        label=f'Logistic Regression (AUC = {auc_val:.4f})')
ax.plot([0, 1], [0, 1], color=COLORS['gray'], linestyle='--',
        linewidth=1, label='Random baseline')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Logistic Regression on session features')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig8_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ---
# ## 8. Summary Statistics & Export
# 
# Final outputs for Stage 1:
# - **Summary statistics table** — for the professor's report
# - **`events_clean.csv`** — bot-removed event log for Stage 2 Kafka producer
# - **`session_features.csv`** — pre-computed features for model development

# ── Dataset summary statistics table ─────────────────────────────────────────
stats_rows = [
    ('Raw events',                    f'{len(df_raw):,}'),
    ('Events after bot removal',      f'{len(df_clean):,}'),
    ('Events removed (bots)',         f'{len(df_raw)-len(df_clean):,}  ({(1-len(df_clean)/len(df_raw))*100:.1f}%)'),
    ('Observation period',            f'{ts_min.date()} → {ts_max.date()} ({span_days} days)'),
    ('Unique visitors (raw)',         f'{df_raw["visitorid"].nunique():,}'),
    ('Unique visitors (clean)',       f'{df_clean["visitorid"].nunique():,}'),
    ('Unique items',                  f'{df_clean["itemid"].nunique():,}'),
    ('Sessions reconstructed',        f'{session_features["session_key"].nunique():,}'),
    ('Session timeout rule',          '30 minutes inactivity'),
    ('Cart sessions total',           f'{(session_features["addtocart_count"]>0).sum():,}'),
    ('Purchase sessions (label=1)',   f'{session_features.query("addtocart_count>0")["has_transaction"].sum():,}  ({session_features.query("addtocart_count>0")["has_transaction"].mean()*100:.2f}%)'),
    ('Abandoned sessions (label=0)',  f'{(session_features.query("addtocart_count>0")["has_transaction"]==0).sum():,}  ({(session_features.query("addtocart_count>0")["has_transaction"]==0).mean()*100:.2f}%)'),
    ('Imbalance ratio',               f'{(session_features.query("addtocart_count>0")["has_transaction"]==0).sum()/session_features.query("addtocart_count>0")["has_transaction"].sum():.1f} : 1'),
    ('Bot detection method',          'Velocity (>500 ev/hr) OR CV of intervals (<0.05)'),
    ('Bot visitors removed',          f'{n_bots:,}  ({n_bots/n_total_visitors*100:.2f}% of visitors)'),
    ('Session features engineered',   '8 (from events.csv alone)'),
    ('Validation model',              'Logistic Regression, 5-fold CV, class_weight=balanced'),
    ('AUC-ROC (CV mean ± std)',       f'{auc_scores.mean():.4f} ± {auc_scores.std():.4f}'),
    ('Imbalance strategy',            'Threshold calibration (not SMOTE) — Waldmann et al. 2025'),
]

summary_df = pd.DataFrame(stats_rows, columns=['Metric', 'Value'])
print('=== STAGE 1 SUMMARY ===')
print(summary_df.to_string(index=False))
summary_df.to_csv('stage1_summary_statistics.csv', index=False)
print('\nSaved: stage1_summary_statistics.csv')

# ── Export clean event log ─────────────────────────────────────────────────────
# This is the file the Stage 2 Kafka producer will read and replay.
# Sorted by timestamp — essential for correct temporal replay.
df_export = df_clean[
    ['timestamp', 'visitorid', 'event', 'itemid', 'transactionid', 'session_key']
].sort_values('timestamp').reset_index(drop=True)

df_export.to_csv('events_clean.csv', index=False)
print(f'Saved: events_clean.csv  ({len(df_export):,} rows)')

# ── Export session features ───────────────────────────────────────────────────
# Add visitorid to session features for traceability
visitor_per_session = df_clean.groupby('session_key')['visitorid'].first().reset_index()
session_features_out = session_features.merge(visitor_per_session, on='session_key')

cols_order = [
    'session_key', 'visitorid',
    'event_count', 'view_count', 'addtocart_count', 'transaction_count',
    'view_to_cart_ratio', 'unique_items_count',
    'session_age_seconds', 'inter_event_interval_mean',
    'recency_last_cart_seconds', 'has_transaction'
]
session_features_out[cols_order].to_csv('session_features.csv', index=False)
print(f'Saved: session_features.csv  ({len(session_features_out):,} sessions)')
print()
print('─── Stage 1 complete ───')
print('Output files:')
print('  events_clean.csv              → input for Stage 2 Kafka producer')
print('  session_features.csv          → input for model training (Stage 3)')
print('  stage1_summary_statistics.csv → for professor report')
print('  fig1_daily_volume.png         → observation period chart')
print('  fig2_session_lengths.png      → session length distribution')
print('  fig3_class_imbalance.png      → class balance chart')
print('  fig4_bot_detection.png        → bot detection visualization')
print('  fig5_feature_distributions.png→ features by class')
print('  fig6_correlation.png          → feature correlation matrix')
print('  fig7_feature_importance.png   → LR coefficients')
print('  fig8_roc_curve.png            → ROC curve')


