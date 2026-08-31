"""
최종 전략: Jaccard<0.3 + Entry Delay (리드-래그 기반)
- 네트워크 centrality score (daily_scores.csv)
- 리드-래그 래그 기반 entry delay (lead_lag_pairs.csv)
- Jaccard<0.3 적응형 리밸런싱
- ScorePow 가중, 윈저라이징, Amihud 차등비용, NAV 기반
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
import warnings, time
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("  최종 전략: Jaccard<0.3 + Entry Delay (리드-래그 기반)")
print("=" * 70)
print("\n데이터 로딩 중...", flush=True)
t0 = time.time()

# ═══════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════
scores = pd.read_csv("daily_scores.csv", parse_dates=['date'])
pairs = pd.read_csv("lead_lag_pairs.csv")
total_adj_raw = pd.read_csv("total adj close.csv", index_col='Date', parse_dates=True, low_memory=False)
total_adj = total_adj_raw.apply(pd.to_numeric, errors='coerce')
total_adj_monthly = total_adj.resample('ME').last()
returns_monthly = total_adj_monthly.pct_change(fill_method=None)

returns_monthly_wins = returns_monthly.copy()
for col in returns_monthly_wins.columns:
    s = returns_monthly_wins[col].dropna()
    if len(s) > 10:
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        returns_monthly_wins[col] = returns_monthly_wins[col].clip(lo, hi)

prices_raw = pd.read_csv("수정주가.csv", index_col='Date', parse_dates=True, low_memory=False)
prices = prices_raw.apply(pd.to_numeric, errors='coerce')
tv_raw = pd.read_csv("trading value.csv", index_col=0, parse_dates=True, low_memory=False)
tv = tv_raw.apply(pd.to_numeric, errors='coerce')
daily_ret_abs = prices.pct_change().abs()
daily_illiq = daily_ret_abs / tv
amihud_monthly = daily_illiq.resample('ME').mean()

factors_raw = pd.read_csv("회귀분석_코스피.csv", index_col='코드명', parse_dates=True)
for col in factors_raw.columns:
    factors_raw[col] = factors_raw[col].astype(str).str.replace(',', '').astype(float)
kospi_factor_ret = factors_raw['코스피'].pct_change()
hml_ret = factors_raw['Size & Book Value(2X3) HML'].pct_change()
smb_ret = factors_raw['Size & Book Value(2X3) SMB'].pct_change()
mom_ret = factors_raw['Size & Momentum(2X3) Mom'].pct_change()
eco_annual = factors_raw['ECO'] / 100
rf_monthly = (1 + eco_annual) ** (1/12) - 1
mkt_factor = kospi_factor_ret - rf_monthly

ks200_raw = pd.read_csv("KOSPI200.csv", index_col='Date', parse_dates=True)
ks200 = ks200_raw.apply(pd.to_numeric, errors='coerce')
ks200_monthly = ks200.resample('ME').last().dropna()

month_ends = returns_monthly_wins.loc['2010-07-31':'2025-12-31'].index
score_dates = sorted(scores['date'].unique())
rebal_dates_pairs = sorted(pairs['rebal_date'].unique())
HIGH_COST = 0.008; LOW_COST = 0.003; ILLIQ_THRESHOLD = 0.80
THRESHOLD = 0.3; N_QUINTILES = 5

prices_monthly = prices.resample('ME').last()

# 스팩 종목 리스트
all_tickers = scores['ticker'].unique()
spac_set = set(t for t in all_tickers if '스팩' in str(t) or 'SPAC' in str(t))
print(f"  스팩 {len(spac_set)}개 필터링 대상", flush=True)

# 일별 퀀타일/스코어 매핑
daily_quintile = {}; daily_score_map = {}
for d, grp in scores.groupby('date'):
    daily_quintile[d] = dict(zip(grp['ticker'], grp['quintile']))
    daily_score_map[d] = dict(zip(grp['ticker'], grp['composite_score']))

# 기간별 평균 래그
lag_by_period = {}
for rd in rebal_dates_pairs:
    period_pairs = pairs[pairs['rebal_date'] == rd]
    lag_only = period_pairs[period_pairs['lag'] > 0]
    lag_by_period[rd] = int(round(lag_only['lag'].mean())) if len(lag_only) > 0 else 5

print(f"  로딩 완료: {time.time()-t0:.1f}초", flush=True)

# ═══════════════════════════════════════
# 2. Entry Delay 함수
# ═══════════════════════════════════════
def get_stable_q5(q5_set, start, delay_days):
    idx = np.searchsorted(score_dates, start, side='right')
    start_idx = max(idx - delay_days, 0)
    check_dates = score_dates[start_idx:idx]
    if len(check_dates) < delay_days:
        return q5_set
    stable = set()
    for ticker in q5_set:
        consecutive = True
        for d in check_dates[-delay_days:]:
            dq = daily_quintile.get(d, {})
            if dq.get(ticker, 0) != 5:
                consecutive = False; break
        if consecutive:
            stable.add(ticker)
    return stable if len(stable) >= 30 else q5_set

# ═══════════════════════════════════════
# 3. 백테스트: 5분위 전체 + Q5
# ═══════════════════════════════════════
print("\n백테스트 진행 중...", flush=True)

# 5분위 수익률 (수수료전)
q_rets_pre = {q: pd.Series(dtype=float) for q in range(N_QUINTILES)}
for q in range(N_QUINTILES):
    q_rets_pre[q][month_ends[0]] = 0.0

# Q5 수수료후 (최종 전략)
q5_rets = pd.Series(dtype=float)
q5_rets[month_ends[0]] = 0.0
q5_nav = 1.0; q5_prev_pf = pd.Series(dtype=float)
q5_turnover = []; q5_holdings = set()
last_rebal_set = set(); last_rebal_scores = {}

for i in range(len(month_ends) - 1):
    start = month_ends[i]; end = month_ends[i + 1]

    # 적응형 delay
    valid_rd = [d for d in rebal_dates_pairs if d <= str(start.date())]
    current_lag = lag_by_period.get(valid_rd[-1], 5) if valid_rd else 5

    valid_sd = [d for d in score_dates if d <= start]
    if not valid_sd:
        for q in range(N_QUINTILES): q_rets_pre[q][end] = 0.0
        q5_rets[end] = 0.0; continue

    score_date = valid_sd[-1]
    dq = daily_quintile.get(score_date, {}); dsm = daily_score_map.get(score_date, {})
    q5_set = set(t for t, q in dq.items() if q == 5)
    month_score_dates = [d for d in score_dates if d > start and d <= end]

    nr = returns_monthly_wins.loc[end].dropna()

    # 거래정지
    p_s = total_adj_monthly.loc[start] if start in total_adj_monthly.index else None
    p_e = total_adj_monthly.loc[end] if end in total_adj_monthly.index else None
    if p_s is not None and p_e is not None:
        common_idx = nr.index.intersection(p_s.dropna().index).intersection(p_e.dropna().index)
        suspended = p_s.reindex(common_idx)[(p_s.reindex(common_idx) == p_e.reindex(common_idx))].index
        nr = nr.drop(suspended, errors='ignore')

    amihud_now = amihud_monthly.loc[start].dropna() if start in amihud_monthly.index else pd.Series(dtype=float)
    if len(amihud_now) > 0:
        illiquid_set = set(amihud_now[amihud_now >= amihud_now.quantile(ILLIQ_THRESHOLD)].index)
    else: illiquid_set = set()

    # 5분위 수익률 (수수료전)
    month_scores_df = scores[scores['date'] == score_date]
    for q in range(N_QUINTILES):
        q_tickers = month_scores_df[month_scores_df['quintile'] == q+1]['ticker'].values
        q_in_nr = pd.Index(q_tickers).intersection(nr.index)
        q_rets_pre[q][end] = nr[q_in_nr].mean() if len(q_in_nr) > 0 else 0.0

    # Jaccard 체크
    if len(last_rebal_set) == 0: do_rebal = True
    else:
        do_rebal = False
        for sd in month_score_dates:
            dq_mid = daily_quintile.get(sd, {})
            current_q5 = set(t for t, q in dq_mid.items() if q == 5)
            union = last_rebal_set | current_q5
            if len(union) > 0 and len(last_rebal_set & current_q5) / len(union) < THRESHOLD:
                do_rebal = True; break
        if not do_rebal:
            union = last_rebal_set | q5_set
            if len(union) > 0 and len(last_rebal_set & q5_set) / len(union) < THRESHOLD:
                do_rebal = True

    if do_rebal:
        stable_q5 = get_stable_q5(q5_set, start, current_lag)
        # 스팩 제외 + 주가 1000원 미만 제외
        filtered = pd.Index(list(stable_q5)).difference(pd.Index(list(spac_set)))
        pr = prices_monthly.loc[start].dropna() if start in prices_monthly.index else None
        if pr is not None:
            filtered = filtered.intersection(pr[pr >= 1000].index)
        sel_set = set(filtered) if len(filtered) >= 30 else stable_q5  # fallback
        sel_scores = {t: dsm.get(t, 0) for t in sel_set}
        last_rebal_set = q5_set.copy()
        last_rebal_scores = sel_scores.copy()
    else:
        sel_set = q5_holdings
        sel_scores = last_rebal_scores

    sel_tickers = pd.Index(list(sel_set)).intersection(nr.index)
    if len(sel_tickers) == 0:
        q5_rets[end] = 0.0; continue

    # ScorePow 가중
    q5_sc = pd.Series({t: sel_scores.get(t, 0) for t in sel_tickers})
    shifted = q5_sc - q5_sc.min() + 1e-8
    weights = shifted ** 1.5; weights /= weights.sum()

    # NAV 기반 거래비용
    if q5_prev_pf.sum() > 0: prev_weights = q5_prev_pf / q5_prev_pf.sum()
    else: prev_weights = pd.Series(dtype=float)
    all_index = weights.index.union(prev_weights.index)
    target_w = weights.reindex(all_index, fill_value=0)
    prev_w = prev_weights.reindex(all_index, fill_value=0)
    delta_w = target_w - prev_w
    trade_amounts = abs(delta_w) * q5_nav
    cost_rate = np.where(delta_w.index.isin(illiquid_set), HIGH_COST, LOW_COST)
    trade_cost = (trade_amounts * cost_rate).sum()
    NAV_after = q5_nav - trade_cost
    current_pv = weights * NAV_after
    ret_seg = nr.reindex(sel_tickers, fill_value=0)
    next_pv = current_pv * (1 + ret_seg)
    nav_new = next_pv.sum()
    portfolio_ret = nav_new / q5_nav - 1 if q5_nav > 0 else 0.0

    q5_rets[end] = portfolio_ret
    q5_nav = nav_new; q5_prev_pf = next_pv; q5_holdings = set(sel_tickers)
    if q5_prev_pf.sum() > 0:
        q5_turnover.append(trade_amounts.sum() / q5_nav if q5_nav > 0 else 0)

print(f"  백테스트 완료: {time.time()-t0:.1f}초", flush=True)

# ═══════════════════════════════════════
# 4. 성과 계산
# ═══════════════════════════════════════
def get_metrics(ret, nav):
    yr=(nav.index[-1]-nav.index[0]).days/365.25
    cagr=(nav.iloc[-1]/nav.iloc[0])**(1/yr)-1
    vol=ret.std()*np.sqrt(12)
    rfm=(1+0.02)**(1/12)-1
    sharpe=(ret-rfm).mean()/(ret-rfm).std()*np.sqrt(12)
    mdd=(nav/nav.cummax()-1).min()
    return cagr,vol,sharpe,mdd

nav_q5 = (q5_rets + 1).cumprod()
m_q5 = get_metrics(q5_rets, nav_q5)
avg_to = np.mean(q5_turnover) if q5_turnover else 0

# Carhart HAC
fd = mkt_factor.dropna().index
valid_dates = []
for d in month_ends[1:]:
    cl = fd[fd <= d]
    if len(cl) > 0:
        f = cl[-1]
        if f.year == d.year and f.month == d.month:
            valid_dates.append((d, f))
cfd = [f for _, f in valid_dates]

y = np.array([q5_rets.get(d, 0) for d, f in valid_dates])
rf_a = np.array([rf_monthly.loc[d] for d in cfd])
mkt_a = np.array([mkt_factor.loc[d] for d in cfd])
hml_a = np.array([hml_ret.loc[d] for d in cfd])
smb_a = np.array([smb_ret.loc[d] for d in cfd])
mom_a = np.array([mom_ret.loc[d] for d in cfd])
Y_ols = y - rf_a
X_ols = pd.DataFrame({'MKT': mkt_a, 'SMB': smb_a, 'HML': hml_a, 'MOM': mom_a})
X_ols = sm.add_constant(X_ols)
model = sm.OLS(Y_ols, X_ols).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

# 5분위 성과
q_cagrs = []
for q in range(N_QUINTILES):
    nav_q = (q_rets_pre[q] + 1).cumprod()
    m = get_metrics(q_rets_pre[q], nav_q)
    q_cagrs.append(m[0])
mono = all(q_cagrs[j] <= q_cagrs[j+1] for j in range(4))

# ═══════════════════════════════════════
# 5. 출력
# ═══════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  최종 전략 성과")
print(f"{'=' * 70}")
print(f"\n  [5분위 CAGR (수수료전)]")
print(f"  Q1={q_cagrs[0]:.2%}  Q2={q_cagrs[1]:.2%}  Q3={q_cagrs[2]:.2%}  Q4={q_cagrs[3]:.2%}  Q5={q_cagrs[4]:.2%}")
print(f"  단조성: {'O' if mono else 'X'}")

print(f"\n  [Q5 수수료후]")
print(f"  CAGR   = {m_q5[0]:.2%}")
print(f"  Vol    = {m_q5[1]:.1%}")
print(f"  Sharpe = {m_q5[2]:.2f}")
print(f"  MDD    = {m_q5[3]:.1%}")
print(f"  턴오버  = {avg_to:.1%}")
print(f"  α(HAC) = {model.params['const']*12*100:.2f}% (t={model.tvalues['const']:.2f}, p={model.pvalues['const']:.4f})")

print(f"\n  [OLS 결과]")
print(model.summary())

# ═══════════════════════════════════════
# 6. 시각화 (4패널)
# ═══════════════════════════════════════
colors_q = ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#1f77b4']
labels_q = ['Q1 (하위)', 'Q2', 'Q3', 'Q4', 'Q5 (상위)']

fig, axes = plt.subplots(2, 2, figsize=(18, 13))

# Panel 1: 5분위 NAV (수수료전)
ax1 = axes[0, 0]
for q in range(N_QUINTILES):
    nav_q = (q_rets_pre[q] + 1).cumprod()
    m = get_metrics(q_rets_pre[q], nav_q)
    ax1.plot(nav_q.index, nav_q.values, color=colors_q[q], linewidth=1.8,
             label=f'{labels_q[q]}: CAGR={m[0]:.1%}')
ax1.set_title('퀀타일별 NAV (수수료전)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.set_ylabel('NAV'); ax1.grid(True, alpha=0.3); ax1.set_yscale('log')

# Panel 2: Q5 NAV (수수료후) + KOSPI200
ax2 = axes[0, 1]
ax2.plot(nav_q5.index, nav_q5.values, color='#1f77b4', linewidth=2,
         label=f'Q5 수수료후: CAGR={m_q5[0]:.1%}, Sharpe={m_q5[2]:.2f}')
# KOSPI200
ks_common = ks200_monthly.index.intersection(nav_q5.index)
if len(ks_common) > 0:
    ks_start = ks_common[0]
    ks_nav = ks200_monthly.loc[ks_common].iloc[:, 0] / ks200_monthly.loc[ks_start].iloc[0]
    scale = nav_q5.asof(ks_start)
    ks_ret = ks_nav.pct_change().dropna()
    ks_m = get_metrics(ks_ret, ks_nav)
    ax2.plot(ks_nav.index, ks_nav.values * scale, color='gray', linewidth=1.5, linestyle=':',
             label=f'KOSPI200: CAGR={ks_m[0]:.1%}')
ax2.set_title('Q5 수수료후 vs KOSPI200', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.set_ylabel('NAV'); ax2.grid(True, alpha=0.3); ax2.set_yscale('log')

# Panel 3: Log NAV
ax3 = axes[1, 0]
ax3.plot(nav_q5.index, np.log(nav_q5.values), color='#1f77b4', linewidth=2,
         label=f'Q5 Log NAV')
ax3.set_title('Q5 Log NAV (수수료후)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10, loc='upper left')
ax3.set_ylabel('Log(NAV)'); ax3.grid(True, alpha=0.3)

# Panel 4: CAGR 바 차트
ax4 = axes[1, 1]
x = np.arange(N_QUINTILES)
bars = ax4.bar(x, [c*100 for c in q_cagrs], color=colors_q, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, q_cagrs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(['Q1\n(하위)', 'Q2', 'Q3', 'Q4', 'Q5\n(상위)'], fontsize=11)
ax4.set_ylabel('CAGR (%)')
ax4.set_title('퀀타일별 CAGR (수수료전)', fontsize=14, fontweight='bold')
ax4.grid(True, axis='y', alpha=0.3); ax4.axhline(y=0, color='black', linewidth=0.5)

fig.suptitle(
    f'네트워크 기반 적응형 전략 — Jaccard<0.3 + Entry Delay\n'
    f'Q5 수수료후: CAGR={m_q5[0]:.1%}, Sharpe={m_q5[2]:.2f}, '
    f'α={model.params["const"]*12*100:.1f}%(t={model.tvalues["const"]:.2f}), '
    f'MDD={m_q5[3]:.1%}, 턴오버={avg_to:.1%}',
    fontsize=13, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig('final_strategy.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nfinal_strategy.png 저장 완료!")

# 엑셀 저장
df_out = pd.DataFrame({
    'Date': nav_q5.index,
    'NAV': nav_q5.values,
    'Log_NAV': np.log(nav_q5.values),
    'Monthly_Return': q5_rets.values,
})
df_out.to_excel("final_strategy_nav.xlsx", index=False)
print("final_strategy_nav.xlsx 저장 완료!")

print(f"\n총 소요: {time.time()-t0:.1f}초")
