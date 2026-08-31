"""
최종 전략 v2: Z-Score Composite + 월간 리밸런싱
- 새 데이터셋(final 폴더) 사용
- 수정주가_거래대금필터_code: 네트워크용 (코드 기반)
- 현금배당포함: 수익률 계산용 (종목명 기반)
- 시가총액: 시총 (종목명 기반, 쉼표 포함)
- 거래대금: Amihud용 (코드 기반)
- 60_거래대금: 유동성 필터용 (코드 기반)
- 4팩터: OLS용
- 기간: 2000~
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
print("  최종 전략 v2: Z-Score Composite (새 데이터)")
print("=" * 70)
print("\n데이터 로딩 중...", flush=True)
t0 = time.time()

# ═══════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════

# 수정주가 (네트워크용, 코드 기반)
prices_raw = pd.read_csv("수정주가_거래대금필터_code.csv", index_col='Date', parse_dates=True, low_memory=False)
prices = prices_raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce'))
print(f"  수정주가: {prices.shape}", flush=True)

# 현금배당포함 (수익률용, 종목명 기반)
total_raw = pd.read_csv("현금배당포함.csv", index_col='코드명', parse_dates=True, low_memory=False)
total_adj = total_raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce'))
print(f"  현금배당포함: {total_adj.shape}", flush=True)

# 시가총액 (종목명 기반, 쉼표 포함)
mcap_raw = pd.read_csv("시가총액.csv", index_col='코드명', parse_dates=True, low_memory=False)
mcap = mcap_raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.replace('"', '').str.strip(), errors='coerce'))
print(f"  시가총액: {mcap.shape}", flush=True)

# 거래대금 (Amihud용, 코드 기반)
tv_raw = pd.read_csv("거래대금.csv", index_col='Date', parse_dates=True, low_memory=False)
tv = tv_raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce'))
print(f"  거래대금: {tv.shape}", flush=True)

# 60일 거래대금 (유동성 필터, 코드 기반)
tv60_raw = pd.read_csv("60_거래대금.csv", index_col='Date', parse_dates=True, low_memory=False)
tv60 = tv60_raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce'))
print(f"  60_거래대금: {tv60.shape}", flush=True)

# 4팩터
factors_raw = pd.read_csv("4팩터.csv", index_col='Date', parse_dates=True)
for col in factors_raw.columns:
    factors_raw[col] = pd.to_numeric(factors_raw[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')
print(f"  4팩터: {factors_raw.shape}", flush=True)

# 코드↔종목명 매핑 (순서 동일)
code_cols = prices.columns  # A005930, A000660, ...
name_cols = total_adj.columns  # 삼성전자, SK하이닉스, ...
code_to_name = dict(zip(code_cols, name_cols))
name_to_code = dict(zip(name_cols, code_cols))

# 시가총액/현금배당포함을 코드 기반으로 변환
total_adj.columns = [name_to_code.get(c, c) for c in total_adj.columns]
mcap.columns = [name_to_code.get(c, c) for c in mcap.columns]
print(f"  코드 매핑 완료: {len(code_to_name)}개", flush=True)

# 수익률 계산
prices_monthly = prices.resample('ME').last()
total_adj_monthly = total_adj.resample('ME').last()
mcap_monthly = mcap.resample('ME').last()
tv60_monthly = tv60.resample('ME').last()
returns_daily = prices.pct_change()
returns_monthly = total_adj_monthly.pct_change(fill_method=None)

# 윈저라이징
returns_monthly_wins = returns_monthly.copy()
for col in returns_monthly_wins.columns:
    s = returns_monthly_wins[col].dropna()
    if len(s) > 10:
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        returns_monthly_wins[col] = returns_monthly_wins[col].clip(lo, hi)

# Amihud
daily_ret_abs = returns_daily.abs()
daily_illiq = daily_ret_abs / tv
amihud_monthly = daily_illiq.resample('ME').mean()

# 4팩터 파싱 (월말 리샘플링)
factors_monthly = factors_raw.resample('ME').last()
kospi_factor_ret = factors_monthly['KOSPI'].pct_change() if 'KOSPI' in factors_monthly.columns else factors_monthly.iloc[:, 0].pct_change()
hml_ret = factors_monthly['HML'].pct_change() if 'HML' in factors_monthly.columns else factors_monthly.iloc[:, 1].pct_change()
smb_ret = factors_monthly['SMB'].pct_change() if 'SMB' in factors_monthly.columns else factors_monthly.iloc[:, 2].pct_change()
mom_ret = factors_monthly['MOM'].pct_change() if 'MOM' in factors_monthly.columns else factors_monthly.iloc[:, 3].pct_change()
cd91 = factors_monthly['CD91'] if 'CD91' in factors_monthly.columns else factors_monthly.iloc[:, 4]
rf_monthly = cd91 / 100 / 12
mkt_factor = kospi_factor_ret - rf_monthly

# 스팩
spac_set = set(c for c in prices.columns if '스팩' in str(c) or 'SPAC' in str(c))
# 현금배당포함에서도
spac_set_total = set(c for c in total_adj.columns if '스팩' in str(c) or 'SPAC' in str(c))

month_ends = returns_monthly_wins.loc['2009-12-31':'2026-02-28'].index
LOOKBACK = 60; N_QUINTILES = 5
HIGH_COST = 0.008; LOW_COST = 0.003; ILLIQ_THRESHOLD = 0.80

print(f"  로딩 완료: {time.time()-t0:.1f}초", flush=True)
print(f"  기간: {month_ends[0].strftime('%Y-%m')} ~ {month_ends[-1].strftime('%Y-%m')} ({len(month_ends)}개월)")

# ═══════════════════════════════════════
# 2. 함수
# ═══════════════════════════════════════
def compute_cap_scaled_centrality(adj, mcap_vals, beta, max_iter=300, tol=1e-8):
    n=len(adj);x=np.ones(n)/n;mc=mcap_vals.copy()
    mc[mc<=0]=mc[mc>0].min() if (mc>0).any() else 1.0
    penalty=1.0/(mc**beta);penalty/=penalty.max()
    for _ in range(max_iter):
        x_new=penalty*(adj@x);norm=np.linalg.norm(x_new)
        if norm==0:return np.ones(n)/n
        x_new/=norm
        if np.linalg.norm(x_new-x)<tol:break
        x=x_new
    return np.abs(x_new)

def compute_eigenvector_centrality(adj, max_iter=300, tol=1e-8):
    n=len(adj);x=np.ones(n)/n
    for _ in range(max_iter):
        x_new=adj@x;norm=np.linalg.norm(x_new)
        if norm==0:return np.ones(n)/n
        x_new/=norm
        if np.linalg.norm(x_new-x)<tol:break
        x=x_new
    return np.abs(x_new)

def equal_weight_correlation(X):
    T,N=X.shape;Xc=X-X.mean(axis=0);S=(Xc.T@Xc)/T
    std=np.sqrt(np.diag(S));std[std==0]=1
    C=S/np.outer(std,std);np.fill_diagonal(C,1.0);np.clip(C,-1,1,out=C)
    return C

def marchenko_pastur_clipping_full(corr, T):
    N=len(corr);q=N/T;lmax=(1+np.sqrt(q))**2
    ev,evec=np.linalg.eigh(corr)
    idx=np.argsort(ev)[::-1];ev=ev[idx];evec=evec[:,idx]
    ns=max(np.sum(ev>lmax),1);cev=ev.copy()
    if N>ns:cev[ns:]=ev[ns:].mean()
    cev*=N/cev.sum()
    Cc=evec@np.diag(cev)@evec.T
    d=np.sqrt(np.diag(Cc));d[d==0]=1
    Cc/=np.outer(d,d);np.fill_diagonal(Cc,1.0);np.clip(Cc,-1,1,out=Cc)
    return Cc,cev,evec

def get_metrics(ret, nav):
    yr=(nav.index[-1]-nav.index[0]).days/365.25
    cagr=(nav.iloc[-1]/nav.iloc[0])**(1/yr)-1
    vol=ret.std()*np.sqrt(12)
    rfm=(1+0.02)**(1/12)-1
    sharpe=(ret-rfm).mean()/(ret-rfm).std()*np.sqrt(12) if ret.std()>0 else 0
    mdd=(nav/nav.cummax()-1).min()
    return cagr,vol,sharpe,mdd

# ═══════════════════════════════════════
# 3. 코드↔종목명 매핑 (수정주가는 코드, 현금배당포함은 종목명)
# ═══════════════════════════════════════
# 수정주가 컬럼: 코드 (A005930)
# 현금배당포함/시가총액: 종목명 (삼성전자)
# → 네트워크는 코드로 계산, 수익률은 종목명으로 계산
# → 매핑 불필요: 각각 독립적으로 처리

print(f"\n백테스트 진행 중...", flush=True)
t_start = time.time()

# 점수/가중치 저장용
all_score_rows = []

# 5분위 수익률 (등가중, 수수료전)
q_rets_pre = {q: pd.Series(dtype=float) for q in range(N_QUINTILES)}
for q in range(N_QUINTILES):
    q_rets_pre[q][month_ends[0]] = 0.0

# Q5 수수료후
q5_rets = pd.Series(dtype=float)
q5_rets[month_ends[0]] = 0.0
q5_nav = 1.0; q5_prev_pf = pd.Series(dtype=float)
q5_turnover = []; q5_holdings = set()

for i in range(len(month_ends) - 1):
    start = month_ends[i]; end = month_ends[i + 1]
    if (i+1) % 36 == 0:
        print(f"  {start.strftime('%Y-%m')} ({time.time()-t_start:.0f}초)", flush=True)

    # 시총 (종목명 기반)
    mc = mcap_monthly.loc[start].dropna() if start in mcap_monthly.index else pd.Series(dtype=float)
    mc = mc[mc > 0]

    # 거래대금 필터 (코드 기반)
    tv60_now = tv60_monthly.loc[start].dropna() if start in tv60_monthly.index else pd.Series(dtype=float)
    if len(tv60_now) > 0:
        invest_codes = tv60_now[tv60_now > tv60_now.quantile(0.10)].index
    else:
        invest_codes = prices.columns

    # 60일 일별 수익률 (코드 기반)
    dw = returns_daily.loc[:start].tail(LOOKBACK)
    if len(dw) < 40:
        for q in range(N_QUINTILES): q_rets_pre[q][end] = 0.0
        q5_rets[end] = 0.0; continue

    vc = dw.dropna(axis=1, thresh=int(LOOKBACK * 0.8)).columns
    ds = dw[vc].dropna(axis=1)
    if len(ds.columns) < 100:
        for q in range(N_QUINTILES): q_rets_pre[q][end] = 0.0
        q5_rets[end] = 0.0; continue

    cols = ds.columns  # 코드
    Ta = ds.shape[0]

    mc_vals = mc.reindex(cols).fillna(0).values
    investable = cols.intersection(invest_codes)
    mc_invest = mc.reindex(investable).dropna()

    # 상관행렬 + MP Clipping
    corr = equal_weight_correlation(ds.values)
    cc, ev, evec = marchenko_pastur_clipping_full(corr, Ta)
    v1 = np.abs(evec[:, 0]); tr = ev.sum()
    adj = np.maximum(cc, 0); np.fill_diagonal(adj, 0)

    # β 도출
    cent_raw = compute_eigenvector_centrality(adj)
    valid_ll = (mc_vals > 0) & (cent_raw > 0)
    if valid_ll.sum() > 50:
        log_mc = np.log(mc_vals[valid_ll]); log_cent = np.log(cent_raw[valid_ll])
        X_ll = np.column_stack([np.ones(valid_ll.sum()), log_mc])
        b_ll = np.linalg.lstsq(X_ll, log_cent, rcond=None)[0]
        d_beta = abs(b_ll[1]) / 2
    else:
        d_beta = 0.25

    d_w = np.sqrt(ev[1] / ev[0]) if ev[0] > 0 else 0.5
    d_cut = min(np.sqrt(ev[0] / tr), 0.8)

    # Cap-Scaled Centrality
    centrality = compute_cap_scaled_centrality(adj, mc_vals, beta=d_beta)

    # Z-Score Composite
    cent_s = pd.Series(centrality, index=cols)
    lam1_s = pd.Series(v1, index=cols)

    # cutoff: 시총 상위 cutoff% 제외
    if len(mc_invest) > 0:
        exclude = mc_invest[mc_invest >= mc_invest.quantile(1 - d_cut)].index
    else:
        exclude = pd.Index([])

    common = cols.intersection(investable).difference(exclude)

    if len(common) < 50:
        for q in range(N_QUINTILES): q_rets_pre[q][end] = 0.0
        q5_rets[end] = 0.0; continue

    # Z-Score
    cent_common = cent_s[common]
    lam1_common = lam1_s[common]
    cent_z = (cent_common - cent_common.mean()) / cent_common.std() if cent_common.std() > 0 else cent_common * 0
    lam1_z = (lam1_common - lam1_common.mean()) / lam1_common.std() if lam1_common.std() > 0 else lam1_common * 0
    comp_score = cent_z - d_w * lam1_z

    # 퀀타일
    Nc = len(common); npq = Nc // N_QUINTILES
    sorted_idx = np.argsort(comp_score.values)
    tickers = common

    quintile_map = {}
    for q in range(N_QUINTILES):
        si = q * npq; ei = Nc if q == N_QUINTILES - 1 else si + npq
        for idx in sorted_idx[si:ei]:
            quintile_map[tickers[idx]] = q

    # 점수/가중치 저장
    date_str = str(start.date())
    for j, ticker in enumerate(common):
        all_score_rows.append({
            'date': date_str,
            'ticker': ticker,
            'composite_score': float(comp_score.iloc[j]) if j < len(comp_score) else 0,
            'quintile': quintile_map.get(ticker, -1) + 1,
            'centrality': float(cent_s[ticker]),
            'lambda1_loading': float(lam1_s[ticker]),
            'beta': float(d_beta),
            'w': float(d_w),
            'cutoff': float(d_cut),
        })

    # 수익률 (현금배당포함, 이미 코드 기반으로 변환됨)
    nr = returns_monthly_wins.loc[end].dropna() if end in returns_monthly_wins.index else pd.Series(dtype=float)
    if len(nr) == 0:
        for q in range(N_QUINTILES): q_rets_pre[q][end] = 0.0
        q5_rets[end] = 0.0; continue

    # 거래정지 제외
    p_s = total_adj_monthly.loc[start] if start in total_adj_monthly.index else None
    p_e = total_adj_monthly.loc[end] if end in total_adj_monthly.index else None
    if p_s is not None and p_e is not None:
        common_idx = nr.index.intersection(p_s.dropna().index).intersection(p_e.dropna().index)
        suspended = p_s.reindex(common_idx)[(p_s.reindex(common_idx) == p_e.reindex(common_idx))].index
        nr = nr.drop(suspended, errors='ignore')

    # Amihud
    amihud_now = amihud_monthly.loc[start].dropna() if start in amihud_monthly.index else pd.Series(dtype=float)
    if len(amihud_now) > 0:
        illiquid_set = set(amihud_now[amihud_now >= amihud_now.quantile(ILLIQ_THRESHOLD)].index)
    else:
        illiquid_set = set()

    # 5분위 수익률 (등가중, 수수료전)
    for q in range(N_QUINTILES):
        q_tickers = [t for t, qq in quintile_map.items() if qq == q]
        q_in_nr = pd.Index(q_tickers).intersection(nr.index)
        q_rets_pre[q][end] = nr[q_in_nr].mean() if len(q_in_nr) > 0 else 0.0

    # Q5 종목
    q5_codes = [t for t, qq in quintile_map.items() if qq == N_QUINTILES - 1]

    # 스팩/동전주 필터
    q5_filtered = [t for t in q5_codes if t not in spac_set]
    if start in prices_monthly.index:
        pr = prices_monthly.loc[start]
        q5_filtered = [t for t in q5_filtered if t in pr.index and pr[t] >= 1000]

    if len(q5_filtered) < 10:
        q5_filtered = q5_codes  # fallback

    sel_tickers = pd.Index(q5_filtered).intersection(nr.index)
    if len(sel_tickers) == 0:
        q5_rets[end] = 0.0; continue

    # 등가중
    weights = pd.Series(1.0 / len(sel_tickers), index=sel_tickers)

    # NAV 기반 거래비용 (illiquid_set은 위에서 이미 계산됨)
    if q5_prev_pf.sum() > 0:
        prev_weights = q5_prev_pf / q5_prev_pf.sum()
    else:
        prev_weights = pd.Series(dtype=float)

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

elapsed = time.time() - t_start
print(f"  백테스트 완료: {elapsed:.1f}초")

# ═══════════════════════════════════════
# 4. 성과 계산
# ═══════════════════════════════════════
nav_q5 = (q5_rets + 1).cumprod()
m_q5 = get_metrics(q5_rets, nav_q5)
avg_to = np.mean(q5_turnover) if q5_turnover else 0

# 5분위 CAGR
q_cagrs = []
for q in range(N_QUINTILES):
    nav_q = (q_rets_pre[q] + 1).cumprod()
    m = get_metrics(q_rets_pre[q], nav_q)
    q_cagrs.append(m[0])
mono = all(q_cagrs[j] <= q_cagrs[j+1] for j in range(4))

# OLS (HAC) — 직접 날짜 매칭
valid_dates = []
for d in month_ends[1:]:
    if d in mkt_factor.index and not np.isnan(mkt_factor.loc[d]):
        valid_dates.append(d)

if len(valid_dates) > 10:
    y = np.array([q5_rets.get(d, 0) for d in valid_dates])
    rf_a = np.array([rf_monthly.loc[d] if d in rf_monthly.index else 0 for d in valid_dates])
    mkt_a = np.array([mkt_factor.loc[d] for d in valid_dates])
    hml_a = np.array([hml_ret.loc[d] for d in valid_dates])
    smb_a = np.array([smb_ret.loc[d] for d in valid_dates])
    mom_a = np.array([mom_ret.loc[d] for d in valid_dates])

    Y_ols = y - rf_a
    X_ols = pd.DataFrame({'MKT': mkt_a, 'SMB': smb_a, 'HML': hml_a, 'MOM': mom_a})
    X_ols = sm.add_constant(X_ols)
    try:
        model = sm.OLS(Y_ols, X_ols).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        alpha_ann = model.params['const'] * 12 * 100
        alpha_t = model.tvalues['const']
    except:
        alpha_ann = 0; alpha_t = 0; model = None
else:
    alpha_ann = 0; alpha_t = 0; model = None

# ═══════════════════════════════════════
# 5. 출력
# ═══════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  최종 전략 v2 성과 (Z-Score, 수수료후, 2000~)")
print(f"{'=' * 70}")

print(f"\n  [5분위 CAGR (등가중, 수수료전)]")
print(f"  Q1={q_cagrs[0]:.2%}  Q2={q_cagrs[1]:.2%}  Q3={q_cagrs[2]:.2%}  Q4={q_cagrs[3]:.2%}  Q5={q_cagrs[4]:.2%}")
print(f"  단조성: {'O' if mono else 'X'}")

print(f"\n  [Q5 수수료후]")
print(f"  CAGR   = {m_q5[0]:.2%}")
print(f"  Vol    = {m_q5[1]:.1%}")
print(f"  Sharpe = {m_q5[2]:.2f}")
print(f"  MDD    = {m_q5[3]:.1%}")
print(f"  턴오버  = {avg_to:.1%}")
if alpha_ann:
    print(f"  α(HAC) = {alpha_ann:.2f}% (t={alpha_t:.2f})")

if model:
    print(f"\n  [OLS 결과]")
    print(model.summary())

# ═══════════════════════════════════════
# 6. 시각화
# ═══════════════════════════════════════
colors_q = ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#1f77b4']
labels_q = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

fig, axes = plt.subplots(2, 2, figsize=(18, 13))

# Panel 1: 5분위 NAV
ax1 = axes[0, 0]
for q in range(N_QUINTILES):
    nav_q = (q_rets_pre[q] + 1).cumprod()
    m = get_metrics(q_rets_pre[q], nav_q)
    ax1.plot(nav_q.index, nav_q.values, color=colors_q[q], linewidth=1.8,
             label=f'{labels_q[q]}: CAGR={m[0]:.1%}')
ax1.set_title('퀀타일별 NAV (등가중, 수수료전)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.set_ylabel('NAV'); ax1.grid(True, alpha=0.3); ax1.set_yscale('log')

# Panel 2: Q5 NAV
ax2 = axes[0, 1]
ax2.plot(nav_q5.index, nav_q5.values, color='#1f77b4', linewidth=2,
         label=f'Q5 수수료후: CAGR={m_q5[0]:.1%}, Sharpe={m_q5[2]:.2f}')
ax2.set_title('Q5 NAV (수수료후)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.set_ylabel('NAV'); ax2.grid(True, alpha=0.3); ax2.set_yscale('log')

# Panel 3: Log NAV
ax3 = axes[1, 0]
ax3.plot(nav_q5.index, np.log(nav_q5.values), color='#1f77b4', linewidth=2)
ax3.set_title('Q5 Log NAV (수수료후)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Log(NAV)'); ax3.grid(True, alpha=0.3)

# Panel 4: CAGR 바
ax4 = axes[1, 1]
x = np.arange(N_QUINTILES)
bars = ax4.bar(x, [c*100 for c in q_cagrs], color=colors_q, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, q_cagrs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], fontsize=11)
ax4.set_ylabel('CAGR (%)'); ax4.set_title('퀀타일별 CAGR', fontsize=14, fontweight='bold')
ax4.grid(True, axis='y', alpha=0.3); ax4.axhline(y=0, color='black', linewidth=0.5)

fig.suptitle(
    f'네트워크 기반 전략 v2 (Z-Score, 2000~)\n'
    f'Q5: CAGR={m_q5[0]:.1%}, Sharpe={m_q5[2]:.2f}, α={alpha_ann:.1f}%(t={alpha_t:.2f}), MDD={m_q5[3]:.1%}',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('final_strategy_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nfinal_strategy_v2.png 저장 완료!")

# 엑셀
df_out = pd.DataFrame({
    'Date': nav_q5.index, 'NAV': nav_q5.values,
    'Log_NAV': np.log(nav_q5.values), 'Monthly_Return': q5_rets.values,
})
df_out.to_excel("final_strategy_v2_nav.xlsx", index=False)
print("final_strategy_v2_nav.xlsx 저장 완료!")

# 점수 CSV 저장
print("\n점수/가중치 CSV 저장 중...", flush=True)
df_scores = pd.DataFrame(all_score_rows)
df_scores.to_csv("monthly_scores.csv", index=False, encoding='utf-8-sig')
print(f"  monthly_scores.csv 저장 완료 ({len(df_scores):,}행)")

print(f"\n총 소요: {time.time()-t0:.1f}초")
