import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.dpi': 130, 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'xtick.labelsize': 9,
                     'ytick.labelsize': 9})

# =============================================================================
# 1. LOAD & CLEAN DATA
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Methodology Data.csv")
df = pd.read_csv(file_path)

# ── Short column names ────────────────────────────────────────────────────────
df.columns = [
    'timestamp', 'gender', 'age', 'device',
    'phantom_notifications', 'peak_usage_time',
    'usage_time', 'fomo_scale', 'feelings_scale',
    'immediate_reaction', 'check_frequency',
    'platform_hard_to_quit', 'scrolling_situations',
    'sleep_loss',
    'rank_videos', 'rank_messaging', 'rank_scrolling', 'rank_posting',
    'difficulty_relaxing', 'sadness_comparison',
    'physical_anxiety', 'digital_detox_attempt',
    'detox_impact', 'biggest_challenge'
]

# ── Converting ordinal text into numbers ─────────────────────────────────────

FREQ_MAP = {
    'Never': 1,
    'Once or twice a week': 2,
    'A few times a day': 3,
    'Multiple times an hour': 4
}
df['phantom_num'] = df['phantom_notifications'].map(FREQ_MAP)

SADNESS_MAP = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5}
df['sadness_num'] = df['sadness_comparison'].map(SADNESS_MAP)

ANXIETY_MAP = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5}
df['anxiety_num'] = df['physical_anxiety'].map(ANXIETY_MAP)

RANK_MAP = {'1st': 1, '2nd': 2, '3rd': 3, '4th': 4}
for col in ['rank_videos', 'rank_messaging', 'rank_scrolling', 'rank_posting']:
    df[col + '_num'] = df[col].map(RANK_MAP)

# Quantitative columns used in analysis
QUANT_COLS = ['usage_time', 'fomo_scale', 'feelings_scale',
              'difficulty_relaxing', 'sleep_loss',
              'phantom_num', 'sadness_num', 'anxiety_num']

QUANT_LABELS = {
    'usage_time':          'Daily Usage Time (hrs)',
    'fomo_scale':          'FOMO Scale (1-5)',
    'feelings_scale':      'Feelings Without SM (1-5)',
    'difficulty_relaxing': 'Relaxation Difficulty (1-5)',
    'sleep_loss':          'Sleep Lost (minutes)',
    'phantom_num':         'Phantom Notifications (1-4)',
    'sadness_num':         'Sadness from Comparison (1-5)',
    'anxiety_num':         'Physical Anxiety (1-5)',
}

# Platform short names
PLATFORM_MAP = {
    'Video based platforms like YouTube':                              'YouTube',
    'Short content platforms like Instagram and TikTok':               'Instagram/\nTikTok',
    'Text based platforms like X and Reddit':                          'X/Reddit',
    'Messaging platforms like WhatsApp, Messenger and Telegram':       'WhatsApp'
}
df['platform_short'] = df['platform_hard_to_quit'].map(PLATFORM_MAP)
PLATFORM_ORDER = ['YouTube', 'Instagram/\nTikTok', 'X/Reddit', 'WhatsApp']

print("Data loaded — shape:", df.shape)
print(df[QUANT_COLS].describe().round(2))

# =============================================================================
# 2. DESCRIPTIVE STATISTICS
# =============================================================================

# ── 2A. Qualitative frequency tables ─────────────────────────────────────────
print("\n" + "="*55)
print("QUALITATIVE FREQUENCY TABLES")
print("="*55)
qual_cols = ['gender', 'platform_hard_to_quit', 'check_frequency',
             'digital_detox_attempt', 'immediate_reaction']
for col in qual_cols:
    tbl = df[col].value_counts()
    pct = (tbl / tbl.sum() * 100).round(1)
    result = pd.DataFrame({'Count': tbl, 'Percent(%)': pct})
    print(f"\n── {col} ──\n{result}\n")

# ── 2B. Quantitative frequency tables ────────────────────────────────────────
print("="*55)
print("QUANTITATIVE FREQUENCY TABLES")
print("="*55)

def freq_table(series, bins, labels, name):
    cut = pd.cut(series.dropna(), bins=bins, labels=labels, include_lowest=True)
    tbl = cut.value_counts().sort_index()
    pct = (tbl / tbl.sum() * 100).round(1)
    print(f"\n── {name} ──")
    print(pd.DataFrame({'Count': tbl, 'Percent(%)': pct}))

freq_table(df['fomo_scale'],         [0, 2, 3, 5],   ["Low (1-2)", "Moderate (3)", "High (4-5)"],          "FOMO Scale")
freq_table(df['feelings_scale'],     [0, 2, 3, 5],   ["Negative (1-2)", "Neutral (3)", "Positive (4-5)"],  "Feelings Scale")
freq_table(df['usage_time'],         [0, 2, 4, 6, 15],["0-2 hrs", "2-4 hrs", "4-6 hrs", "6+ hrs"],         "Daily Usage Time")
freq_table(df['difficulty_relaxing'],[0, 2, 3, 5],   ["Low (1-2)", "Moderate (3)", "High (4-5)"],          "Relaxation Difficulty")
freq_table(df['sleep_loss'],         [0, 30, 60, 120, 9999], ["<30 min", "30-60 min", "60-120 min", "120+ min"], "Sleep Lost (minutes)")

# =============================================================================
# 3. UNIVARIATE VISUALIZATIONS
# =============================================================================

# ── 3A. Histogram - FOMO Scale ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['fomo_scale'], bins=5, kde=True, color='steelblue',
             edgecolor='white', ax=ax)
ax.set_title('Univariate A: Distribution of FOMO Scale')
ax.set_xlabel('FOMO Scale (1 = Never, 5 = Always)')
ax.set_ylabel('Frequency')
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.tight_layout()
plt.savefig('plot_01_hist_fomo.png')
plt.show()

# ── 3B. Horizontal Bar - Device Type ─────────────────────────────────────────
individual_devices = df['device'].str.split(', ').explode().str.strip()
device_counts = individual_devices.value_counts()

fig, ax = plt.subplots(figsize=(9, 5))
device_counts.plot(kind='barh', color='steelblue', edgecolor='white', ax=ax)
ax.set_title('Univariate B: Frequency of Each Device Used')
ax.set_xlabel('Number of Respondents')
ax.set_ylabel('Device Type')
ax.invert_yaxis()
for i, v in enumerate(device_counts):
    ax.text(v + 0.3, i, str(v), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('plot_02_bar_device.png')
plt.show()

# ── 3C. Pie - Platform Hard to Quit ──────────────────────────────────────────
platform_counts = df['platform_hard_to_quit'].value_counts()
short_labels = ['Instagram/\nTikTok', 'YouTube\n(Video)', 'WhatsApp/\nMessenger', 'X/Reddit\n(Text)'][:len(platform_counts)]

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    platform_counts,
    labels=short_labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette("pastel"),
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
for at in autotexts:
    at.set_fontsize(10)
ax.set_title('Univariate C: Platforms Difficult to Take a Break From', pad=20)
plt.tight_layout()
plt.savefig('plot_03_pie_platform.png')
plt.show()


# ── 3E. Bar chart - Gender Distribution ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
gender_counts = df['gender'].value_counts()
sns.barplot(x=gender_counts.index, y=gender_counts.values,
            palette='muted', edgecolor='white', ax=ax)
ax.set_title('Univariate E: Gender Distribution')
ax.set_ylabel('Count')
ax.set_xlabel('Gender')
for i, v in enumerate(gender_counts.values):
    ax.text(i, v + 0.3, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('plot_05_bar_gender.png')
plt.show()

# =============================================================================
# 4. BIVARIATE VISUALIZATIONS
# =============================================================================

# ── 4A. Two Qualitative - Stacked Bar (Platform by Gender) ───────────────────
ct = pd.crosstab(df['gender'], df['platform_hard_to_quit'])
ct.columns = ['Instagram/\nTikTok', 'WhatsApp/\nMessenger', 'X/Reddit\n(Text)', 'YouTube\n(Video)']

fig, ax = plt.subplots(figsize=(10, 6))
ct.plot(kind='bar', stacked=True, colormap='Paired', edgecolor='white', ax=ax)
ax.set_title('Bivariate A (2 Qual): Hard-to-Quit Platforms by Gender')
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
ax.legend(title='Platform', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plot_06_stacked_platform_gender.png')
plt.show()

# ── 4B. Two Qualitative - Clustered Bar (Detox Attempt by Gender) ────────────
ct2 = pd.crosstab(df['gender'], df['digital_detox_attempt'])
fig, ax = plt.subplots(figsize=(7, 5))
ct2.plot(kind='bar', colormap='Set2', edgecolor='white', ax=ax)
ax.set_title('Bivariate B (2 Qual): Digital Detox Attempt by Gender')
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
ax.legend(title='Attempted Detox')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plot_07_clustered_detox_gender.png')
plt.show()

# ── 4C. Two Quantitative - Scatter + Correlation (FOMO vs Relaxation) ────────
x = df['fomo_scale'].dropna()
y = df['difficulty_relaxing'].reindex(x.index).dropna()
x, y = x.loc[y.index], y

rho, p = stats.spearmanr(x, y)
fig, ax = plt.subplots(figsize=(8, 6))
np.random.seed(42)
jx = x + np.random.uniform(-0.15, 0.15, len(x))
jy = y + np.random.uniform(-0.15, 0.15, len(y))
ax.scatter(jx, jy, alpha=0.5, color='steelblue', edgecolor='white', s=80)
m, b = np.polyfit(x, y, 1)
xs = np.linspace(1, 5, 100)
ax.plot(xs, m * xs + b, 'r--', linewidth=1.5, label=f'y = {m:.2f}x + {b:.2f}')
sig = 'Significant ✔' if p < 0.05 else 'Not significant ✗'
ax.set_title(f'Bivariate C (2 Quant): FOMO Scale vs Relaxation Difficulty\nSpearman ρ = {rho:.3f}, p = {p:.4f} — {sig}')
ax.set_xlabel('FOMO Scale (1 = Never, 5 = Always)')
ax.set_ylabel('Relaxation Difficulty (1 = Easy, 5 = Very Difficult)')
ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
ax.set_xticks([1, 2, 3, 4, 5]); ax.set_yticks([1, 2, 3, 4, 5])
ax.legend()
plt.tight_layout()
plt.savefig('plot_08_scatter_fomo_relax.png')
plt.show()
print(f"\n  Spearman ρ = {rho:.3f}  (p = {p:.4f}) — {sig} at α=0.05")

# ── 4D. Two Quantitative - Scatter (Usage Time vs Sleep Loss) ────────────────
# sleep_loss is now continuous in minutes; usage_time is continuous in hours
x2 = df['usage_time'].dropna()
y2 = df['sleep_loss'].reindex(x2.index).dropna()
x2, y2 = x2.loc[y2.index], y2

rho2, p2 = stats.spearmanr(x2, y2)
fig, ax = plt.subplots(figsize=(8, 6))
np.random.seed(42)
ax.scatter(x2, y2, alpha=0.5, color='coral', edgecolor='white', s=80)
m2, b2 = np.polyfit(x2, y2, 1)
xs2 = np.linspace(x2.min(), x2.max(), 100)
ax.plot(xs2, m2 * xs2 + b2, 'b--', linewidth=1.5, label=f'y = {m2:.2f}x + {b2:.2f}')
sig2 = 'Significant ✔' if p2 < 0.05 else 'Not significant ✗'
ax.set_title(f'Bivariate D (2 Quant): Daily Usage Time vs Sleep Lost\nSpearman ρ = {rho2:.3f}, p = {p2:.4f} — {sig2}')
ax.set_xlabel('Daily Social Media Usage Time (hours)')
ax.set_ylabel('Sleep Lost (minutes)')
ax.legend()
plt.tight_layout()
plt.savefig('plot_09_scatter_usage_sleep.png')
plt.show()
print(f"\n  Spearman ρ = {rho2:.3f}  (p = {p2:.4f}) — {sig2} at α=0.05")

# ── 4E. 1 Quant + 1 Qual - Multiple Box plots (Usage Time by Gender) ───────────────
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(x='gender', y='usage_time', data=df,
            hue='gender', palette='Set2', width=0.5, linewidth=1.5,
            flierprops=dict(marker=''), legend=False, ax=ax)
sns.stripplot(x='gender', y='usage_time', data=df,
              color='black', alpha=0.35, jitter=True, size=5, ax=ax)
for i, g in enumerate(df['gender'].dropna().unique()):
    n = df[df['gender'] == g].shape[0]
    ax.text(i, 5.35, f'n={n}', ha='center', fontsize=9, color='dimgray')
ax.set_ylim(0.5, 5.6)
ax.set_yticks([2, 4, 6, 8, 10, 12])
ax.set_title('Bivariate E (1 Quant + 1 Qual): Daily Usage Time by Gender')
ax.set_xlabel('Gender')
ax.set_ylabel('Daily Usage Time (hours)')
plt.tight_layout()
plt.savefig('plot_10_box_usage_gender.png')
plt.show()

# =============================================================================
# 5. INFERENTIAL STATISTICS
# =============================================================================
print("\n" + "="*55)
print("INFERENTIAL STATISTICS")
print("="*55)

ALPHA = 0.05

def report(test_name, stat, p, df_=None):
    sig = "Significant" if p < ALPHA else "✗ Not significant"
    df_str = f", df={df_}" if df_ is not None else ""
    print(f"\n  {test_name}: stat={stat:.4f}{df_str}, p={p:.4f} → {sig} (α={ALPHA})")

# ── 5A. DEPENDENT QUANTITATIVE ────────────────────────────────────────────────

# 5A-i. One-sample t-test: Is mean FOMO different from 3 (neutral)?
print("\n── 5A-i. One-sample t-test: Mean FOMO ≠ 3 (neutral) ──")
t_stat, p_val = stats.ttest_1samp(df['fomo_scale'].dropna(), popmean=3)
report("One-sample t-test (FOMO vs 3)", t_stat, p_val)
print(f"         Sample mean FOMO = {df['fomo_scale'].mean():.2f}")

# 5A-ii. One-sample t-test: Is mean sleep loss different from 60 minutes?
print("\n── 5A-ii. One-sample t-test: Mean sleep loss ≠ 60 minutes ──")
t_stat2, p_val2 = stats.ttest_1samp(df['sleep_loss'].dropna(), popmean=60)
report("One-sample t-test (Sleep loss vs 60 min)", t_stat2, p_val2)
print(f"         Sample mean sleep loss = {df['sleep_loss'].mean():.2f} minutes")

# 5A-iii. Two-sample t-test: FOMO by gender
print("\n── 5A-iii. Two-sample t-test: FOMO by Gender ──")
male_fomo   = df[df['gender'] == 'Male']['fomo_scale'].dropna()
female_fomo = df[df['gender'] == 'Female']['fomo_scale'].dropna()
t_stat3, p_val3 = stats.ttest_ind(male_fomo, female_fomo, equal_var=False)
report("Welch t-test (FOMO: Male vs Female)", t_stat3, p_val3)
print(f"         Male mean={male_fomo.mean():.2f}  Female mean={female_fomo.mean():.2f}")

# 5A-iv. Two-sample t-test: Relaxation difficulty by detox attempt
print("\n── 5A-iv. Two-sample t-test: Relaxation Difficulty by Detox Attempt ──")
yes_relax = df[df['digital_detox_attempt'] == 'Yes']['difficulty_relaxing'].dropna()
no_relax  = df[df['digital_detox_attempt'] == 'No']['difficulty_relaxing'].dropna()
t_stat4, p_val4 = stats.ttest_ind(yes_relax, no_relax, equal_var=False)
report("Welch t-test (Relaxation: Detox Yes vs No)", t_stat4, p_val4)
print(f"         Detox Yes mean={yes_relax.mean():.2f}  Detox No mean={no_relax.mean():.2f}")

# 5A-v. One-Way ANOVA: FOMO across platforms (3+ groups)
print("\n── 5A-v. One-Way ANOVA: FOMO Score Across Platforms ──")
groups = [grp['fomo_scale'].dropna().values
          for _, grp in df.groupby('platform_short') if len(grp) > 1]
f_stat, p_anova = stats.f_oneway(*groups)
report("One-Way ANOVA (FOMO by Platform)", f_stat, p_anova,
       df_=f"{len(groups)-1}, {df['fomo_scale'].notna().sum()-len(groups)}")

# ── 5B. DEPENDENT QUALITATIVE / PROPORTIONAL ─────────────────────────────────

# 5B-i. One-sample proportion test: Are >50% attempting a digital detox?
print("\n── 5B-i. One-sample proportion z-test: Detox attempt > 50%? ──")
n_detox = (df['digital_detox_attempt'] == 'Yes').sum()
n_total = df['digital_detox_attempt'].notna().sum()
p_obs = n_detox / n_total
z_stat = (p_obs - 0.5) / np.sqrt(0.5 * 0.5 / n_total)
p_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))
report("One-sample z-test (Detox prop vs 0.5)", z_stat, p_z)
print(f"         Observed proportion = {p_obs:.2%} ({n_detox}/{n_total})")

# 5B-ii. Chi-square test: Association between gender & detox attempt
print("\n── 5B-ii. Chi-square test: Gender × Detox Attempt ──")
ct_chi = pd.crosstab(df['gender'], df['digital_detox_attempt'])
chi2, p_chi, dof, expected = stats.chi2_contingency(ct_chi)
report("Chi-square (Gender × Detox)", chi2, p_chi, df_=dof)
print(f"         Contingency table:\n{ct_chi}")

# 5B-iii. Chi-square test: Platform × Immediate Reaction
print("\n── 5B-iii. Chi-square test: Platform × Immediate Reaction ──")
ct_chi2 = pd.crosstab(df['platform_short'], df['immediate_reaction'])
chi2b, p_chi2b, dof2, _ = stats.chi2_contingency(ct_chi2)
report("Chi-square (Platform × Reaction)", chi2b, p_chi2b, df_=dof2)

# =============================================================================
# 6. ACTIVITY RANKING SUMMARY (Extra)
# =============================================================================
print("\n" + "="*55)
print("ACTIVITY RANKING SUMMARY (lower = more time spent)")
print("="*55)
rank_summary = pd.DataFrame({
    'Activity': ['Watching Videos', 'Messaging', 'Scrolling Feed', 'Posting'],
    'Mean Rank': [
        df['rank_videos_num'].mean(),
        df['rank_messaging_num'].mean(),
        df['rank_scrolling_num'].mean(),
        df['rank_posting_num'].mean(),
    ]
}).sort_values('Mean Rank')
print(rank_summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
colors = sns.color_palette("muted", len(rank_summary))
bars = ax.barh(rank_summary['Activity'], rank_summary['Mean Rank'],
               color=colors, edgecolor='white')
ax.set_xlabel('Mean Rank (1 = Most Time, 4 = Least Time)')
ax.set_title('Activity Ranking: Average Rank by Activity')
ax.invert_yaxis()
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.03, bar.get_y() + bar.get_height()/2,
            f'{w:.2f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('plot_12_activity_ranking.png')
plt.show()