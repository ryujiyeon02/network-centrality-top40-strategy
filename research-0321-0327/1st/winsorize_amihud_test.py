import pandas as pd
import numpy as np
import warnings, time
warnings.filterwarnings('ignore')

LOOKBACK = 60
N_QUINTILES = 5

# ═══════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════
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

# 수정주가 (상관관계/네트워크 구성용)
prices_raw = pd.read_csv("수정주가.csv", index_col='Date', parse_dates=True, low_memory=False)
# 배당포함 수정주가 (수익률 계산용)
total_adj_raw = pd.read_csv("total adj close.csv", index_col='Date', parse_dates=True, low_memory=False)
mcap_raw = pd.read_csv("시가총액.csv", index_col='Date', parse_dates=True, low_memory=False)
tv_raw = pd.read_csv("trading value.csv", index_col=0, parse_dates=True, low_memory=False)
tv60_raw = pd.read_csv("trading value 60.csv", index_col=0, parse_dates=True, low_memory=False)

prices = prices_raw.apply(pd.to_numeric, errors='coerce')
total_adj = total_adj_raw.apply(pd.to_numeric, errors='coerce')
mcap = mcap_raw.apply(pd.to_numeric, errors='coerce')
tv = tv_raw.apply(pd.to_numeric, errors='coerce')
tv60 = tv60_raw.apply(pd.to_numeric, errors='coerce')
tv60_monthly = tv60.resample('ME').last()

prices_monthly = prices.resample('ME').last()
total_adj_monthly = total_adj.resample('ME').last()
mcap_monthly = mcap.resample('ME').last()

# 상관관계용 일별 수익률 (수정주가)
returns_daily = prices.pct_change()
# 포트폴리오 수익률용 월간 수익률 (배당포함 수정주가)
returns_monthly = total_adj_monthly.pct_change(fill_method=None)

# 윈저라이징된 월간 수익률 (배당포함 기준)
returns_monthly_wins = returns_monthly.copy()
for col in returns_monthly_wins.columns:
    s = returns_monthly_wins[col].dropna()
    if len(s) > 10:
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        returns_monthly_wins[col] = returns_monthly_wins[col].clip(lo, hi)

# Amihud illiquidity (수정주가 기준)
daily_ret_abs = returns_daily.abs()
daily_illiq = daily_ret_abs / tv
amihud_monthly = daily_illiq.resample('ME').mean()

month_ends = returns_monthly.loc['2009-12-31':'2025-12-31'].index

# ═══════════════════════════════════════
# 2. 함수
# ═══════════════════════════════════════
def compute_cap_scaled_centrality(adj, mcap_vals, beta, max_iter=300, tol=1e-8):
    n = len(adj)
    x = np.ones(n) / n
    mc = mcap_vals.copy()
    mc[mc <= 0] = mc[mc > 0].min() if (mc > 0).any() else 1.0
    penalty = 1.0 / (mc ** beta)
    penalty /= penalty.max()
    for _ in range(max_iter):
        x_new = penalty * (adj @ x)
        norm = np.linalg.norm(x_new)
        if norm == 0: return np.ones(n) / n
        x_new /= norm
        if np.linalg.norm(x_new - x) < tol: break
        x = x_new
    return np.abs(x_new)

def compute_eigenvector_centrality(adj, max_iter=300, tol=1e-8):
    n = len(adj)
    x = np.ones(n) / n
    for _ in range(max_iter):
        x_new = adj @ x
        norm = np.linalg.norm(x_new)
        if norm == 0: return np.ones(n) / n
        x_new /= norm
        if np.linalg.norm(x_new - x) < tol: break
        x = x_new
    return np.abs(x_new)

def equal_weight_correlation(X):
    T, N = X.shape; Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / T; std = np.sqrt(np.diag(S)); std[std == 0] = 1
    C = S / np.outer(std, std); np.fill_diagonal(C, 1.0); np.clip(C, -1, 1, out=C)
    return C

def marchenko_pastur_clipping_full(corr, T):
    N = len(corr); q = N / T; lmax = (1 + np.sqrt(q)) ** 2
    ev, evec = np.linalg.eigh(corr)
    idx = np.argsort(ev)[::-1]; ev = ev[idx]; evec = evec[:, idx]
    ns = max(np.sum(ev > lmax), 1); cev = ev.copy()
    if N > ns: cev[ns:] = ev[ns:].mean()
    cev *= N / cev.sum()
    Cc = evec @ np.diag(cev) @ evec.T
    d = np.sqrt(np.diag(Cc)); d[d == 0] = 1
    Cc /= np.outer(d, d); np.fill_diagonal(Cc, 1.0); np.clip(Cc, -1, 1, out=Cc)
    return Cc, cev, evec

def carhart_alpha(y, cd):
    rf = np.array([rf_monthly.loc[d] for d in cd])
    mkt = np.array([mkt_factor.loc[d] for d in cd])
    hml = np.array([hml_ret.loc[d] for d in cd])
    smb = np.array([smb_ret.loc[d] for d in cd])
    mom = np.array([mom_ret.loc[d] for d in cd])
    Y = y - rf; X = np.column_stack([np.ones(len(cd)), mkt, smb, hml, mom])
    try:
        b = np.linalg.lstsq(X, Y, rcond=None)[0]
        r = Y - X @ b; se = np.sqrt(np.sum(r**2) / (len(Y) - 5) * np.linalg.inv(X.T @ X)[0, 0])
        t = b[0] / se; a = b[0] * 12
        sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.645 else ''
        return a, t, sig
    except:
        return np.nan, np.nan, ''

def get_metrics(ret, nav):
    yr = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yr) - 1
    vol = ret.std() * np.sqrt(12)
    rfm = (1 + 0.02) ** (1/12) - 1
    sharpe = (ret - rfm).mean() / (ret - rfm).std() * np.sqrt(12)
    mdd = (nav / nav.cummax() - 1).min()
    return cagr, vol, sharpe, mdd

# ═══════════════════════════════════════
# 3. 전략 비교 (노트북 방식: NAV 기반 거래비용)
# ═══════════════════════════════════════
STRATEGIES = [
    ("기준 (일괄30bp)", 'base'),
    ("윈저라이징", 'winsorize'),
    ("Amihud 차등비용", 'amihud'),
    ("윈저+Amihud", 'both'),
]

HIGH_COST = 0.008   # 80bp (비유동종목)
LOW_COST = 0.003    # 30bp (일반종목)
ILLIQ_THRESHOLD = 0.80  # Amihud 상위 20%

print(f"{'=' * 90}")
print(f"  윈저라이징 + Amihud 차등비용 테스트 (NAV 기반 거래비용)")
print(f"  도출 파라미터: β=loglog/2, w=√(λ₂/λ₁), cut=√(λ₁/tr)")
print(f"  유동성 필터: 거래대금 하위 10% 제거")
print(f"{'=' * 90}")
print(f"\n백테스트 진행 중...", flush=True)
t_start = time.time()

# 수수료 전 수익률 (모든 전략 공통 비교용)
all_q_rets_pre = {}
# 수수료 후 NAV (노트북 방식)
all_q_navs = {}
all_q_rets_post = {}

for sname, _ in STRATEGIES:
    nm = sname
    all_q_rets_pre[nm] = {q: pd.Series(dtype=float) for q in range(N_QUINTILES)}
    all_q_rets_post[nm] = {q: pd.Series(dtype=float) for q in range(N_QUINTILES)}
    all_q_navs[nm] = {q: 1.0 for q in range(N_QUINTILES)}  # 초기 NAV=1
    for q in range(N_QUINTILES):
        all_q_rets_pre[nm][q][month_ends[0]] = 0.0
        all_q_rets_post[nm][q][month_ends[0]] = 0.0

# 이전 포트폴리오 (종목별 가치) — 노트북 방식
prev_portfolio = {}
turnover_list = {}
trade_cost_list = {}
for sname, _ in STRATEGIES:
    nm = sname
    prev_portfolio[nm] = {q: pd.Series(dtype=float) for q in range(N_QUINTILES)}
    turnover_list[nm] = []
    trade_cost_list[nm] = []

for i in range(len(month_ends) - 1):
    start = month_ends[i]; end = month_ends[i + 1]
    if (i + 1) % 12 == 0:
        print(f"  {start.strftime('%Y-%m')} ({time.time()-t_start:.1f}초)", flush=True)

    mc = mcap_monthly.loc[start].dropna()
    mc = mc[mc > 0]
    # 거래대금 하위 10% 제거
    tv60_now = tv60_monthly.loc[start].dropna() if start in tv60_monthly.index else pd.Series(dtype=float)
    if len(tv60_now) > 0:
        invest_universe = tv60_now[tv60_now > tv60_now.quantile(0.10)].index.intersection(mc.index)
    else:
        invest_universe = mc.index

    dw = returns_daily.loc[:start].tail(LOOKBACK)
    if len(dw) < 40:
        for sname, _ in STRATEGIES:
            nm = sname
            for q in range(N_QUINTILES):
                all_q_rets_pre[nm][q][end] = 0.0
                all_q_rets_post[nm][q][end] = 0.0
        continue

    vc = dw.dropna(axis=1, thresh=int(LOOKBACK * 0.8)).columns
    ds = dw[vc].dropna(axis=1)
    if len(ds.columns) < 100:
        for sname, _ in STRATEGIES:
            nm = sname
            for q in range(N_QUINTILES):
                all_q_rets_pre[nm][q][end] = 0.0
                all_q_rets_post[nm][q][end] = 0.0
        continue

    cols = ds.columns
    Ta = ds.shape[0]
    mc_vals = mc.reindex(cols).fillna(0).values
    investable = invest_universe.intersection(cols)
    mc_invest = mc.reindex(investable).dropna()

    corr = equal_weight_correlation(ds.values)
    cc, ev, evec = marchenko_pastur_clipping_full(corr, Ta)
    v1 = np.abs(evec[:, 0])
    tr = ev.sum()

    adj = np.maximum(cc, 0)
    np.fill_diagonal(adj, 0)

    cent_raw = compute_eigenvector_centrality(adj)
    valid_ll = (mc_vals > 0) & (cent_raw > 0)
    if valid_ll.sum() > 50:
        log_mc = np.log(mc_vals[valid_ll])
        log_cent = np.log(cent_raw[valid_ll])
        X_ll = np.column_stack([np.ones(valid_ll.sum()), log_mc])
        b_ll = np.linalg.lstsq(X_ll, log_cent, rcond=None)[0]
        d_beta = abs(b_ll[1]) / 2
    else:
        d_beta = 0.25
    d_w = np.sqrt(ev[1] / ev[0])
    d_cut = min(np.sqrt(ev[0] / tr), 0.8)

    exclude = mc_invest[mc_invest >= mc_invest.quantile(1 - d_cut)].index

    scores = compute_cap_scaled_centrality(adj, mc_vals, beta=d_beta)
    ss = pd.Series(scores, index=cols)
    lam1_s = pd.Series(v1, index=cols)

    common = ss.index.intersection(investable).difference(exclude)

    # Amihud: 해당 월 기준
    amihud_now = amihud_monthly.loc[start].dropna() if start in amihud_monthly.index else pd.Series(dtype=float)
    if len(amihud_now) > 0:
        illiq_threshold_val = amihud_now.quantile(ILLIQ_THRESHOLD)
        illiquid_set = set(amihud_now[amihud_now >= illiq_threshold_val].index)
    else:
        illiquid_set = set()

    for sname, stype in STRATEGIES:
        nm = sname
        use_wins = stype in ('winsorize', 'both')
        use_amihud = stype in ('amihud', 'both')

        # 수익률 선택
        if use_wins:
            nr = returns_monthly_wins.loc[end, cols].dropna()
        else:
            nr = returns_monthly.loc[end, cols].dropna()

        # 거래정지 종목 제외 (월초가격 == 월말가격)
        p_s = total_adj_monthly.loc[start].reindex(nr.index)
        p_e = total_adj_monthly.loc[end].reindex(nr.index)
        suspended = p_s[(p_s == p_e) & p_s.notna() & p_e.notna()].index
        nr = nr.drop(suspended, errors='ignore')

        common_nr = common.intersection(nr.index)
        if len(common_nr) < 50:
            for q in range(N_QUINTILES):
                all_q_rets_pre[nm][q][end] = 0.0
                all_q_rets_post[nm][q][end] = 0.0
            continue

        sc = ss[common_nr].values
        lc = lam1_s[common_nr].values
        Nc = len(common_nr)
        npq = Nc // N_QUINTILES

        cent_rank = np.argsort(np.argsort(sc)).astype(float)
        lam1_rank = np.argsort(np.argsort(lc)).astype(float)
        comp_score = cent_rank - d_w * lam1_rank
        sorted_idx = np.argsort(comp_score)
        tickers = common_nr

        for q in range(N_QUINTILES):
            si = q * npq
            qi = sorted_idx[si:] if q == N_QUINTILES - 1 else sorted_idx[si:si + npq]
            q_tickers = tickers[qi]
            q_returns = nr[q_tickers]

            # ── 수수료 전 수익률 (동일) ──
            if q < N_QUINTILES - 1:
                ret_pre = q_returns.mean() if len(q_returns) > 0 else 0.0
            else:
                q_scores = comp_score[qi]
                q_scores_shifted = q_scores - q_scores.min() + 1e-8
                q_scores_pow = q_scores_shifted ** 1.5
                weights = q_scores_pow / q_scores_pow.sum()
                ret_pre = (q_returns.values * weights).sum() if len(q_returns) > 0 else 0.0

            all_q_rets_pre[nm][q][end] = ret_pre

            # ── 수수료 후 수익률 (노트북 방식: NAV 기반) ──
            NAV = all_q_navs[nm][q]

            # 목표 비중 계산
            if q < N_QUINTILES - 1:
                target_weights = pd.Series(1.0 / len(q_tickers), index=q_tickers)
            else:
                # ScorePow 가중
                target_weights = pd.Series(weights, index=q_tickers)

            # 이전 포트폴리오 → 비중 계산
            prev_pf = prev_portfolio[nm][q]
            if prev_pf.sum() > 0:
                prev_weights = prev_pf / prev_pf.sum()
            else:
                prev_weights = pd.Series(dtype=float)

            # 비중 차이 계산 (노트북 방식)
            all_index = target_weights.index.union(prev_weights.index)
            target_w = target_weights.reindex(all_index, fill_value=0)
            prev_w = prev_weights.reindex(all_index, fill_value=0)
            delta_w = target_w - prev_w

            # 거래금액
            trade_amounts = abs(delta_w) * NAV

            # 종목별 거래비용률
            if use_amihud:
                cost_rate = np.where(delta_w.index.isin(illiquid_set), HIGH_COST, LOW_COST)
            else:
                cost_rate = LOW_COST

            # 총 거래비용
            trade_cost = (trade_amounts * cost_rate).sum()

            # NAV 업데이트 (비용 차감)
            NAV_after_cost = NAV - trade_cost

            # 새 포트폴리오 가치
            current_portfolio_value = target_weights * NAV_after_cost

            # 수익률 반영
            ret_seg = nr.reindex(q_tickers, fill_value=0)
            next_portfolio_value = current_portfolio_value * (1 + ret_seg)

            # NAV 업데이트
            NAV_new = next_portfolio_value.sum()
            portfolio_ret = NAV_new / NAV - 1 if NAV > 0 else 0.0

            all_q_rets_post[nm][q][end] = portfolio_ret
            all_q_navs[nm][q] = NAV_new
            prev_portfolio[nm][q] = next_portfolio_value

            # Q5 턴오버 기록
            if q == N_QUINTILES - 1:
                if prev_pf.sum() > 0:
                    to_val = trade_amounts.sum() / NAV if NAV > 0 else 0
                    turnover_list[nm].append(to_val)
                    trade_cost_list[nm].append(trade_cost / NAV if NAV > 0 else 0)

elapsed = time.time() - t_start
print(f"  완료: {elapsed:.1f}초")

# ═══════════════════════════════════════
# 4. 결과
# ═══════════════════════════════════════
fd = mkt_factor.dropna().index
valid_dates = []
for d in month_ends[1:]:
    cl = fd[fd <= d]
    if len(cl) > 0:
        f = cl[-1]
        if f.year == d.year and f.month == d.month:
            valid_dates.append((d, f))
cfd = [f for _, f in valid_dates]

print(f"\n{'=' * 130}")
print(f"  윈저라이징 + Amihud 차등비용 비교 결과 (2010~, NAV 기반 거래비용)")
print(f"{'=' * 130}")

for sname, stype in STRATEGIES:
    nm = sname
    print(f"\n── {nm} ──")

    # 수수료 전
    qc_pre = []
    for q in range(N_QUINTILES):
        nav = (all_q_rets_pre[nm][q] + 1).cumprod()
        m = get_metrics(all_q_rets_pre[nm][q], nav)
        qc_pre.append(m)

    mono_pre = all(qc_pre[j][0] <= qc_pre[j+1][0] for j in range(4))
    print(f"  [수수료전] Q1={qc_pre[0][0]:.2%}  Q2={qc_pre[1][0]:.2%}  Q3={qc_pre[2][0]:.2%}  Q4={qc_pre[3][0]:.2%}  Q5={qc_pre[4][0]:.2%}  단조:{'O' if mono_pre else 'X'}")
    print(f"  [수수료전] Q5: Vol={qc_pre[4][1]:.1%}  Sharpe={qc_pre[4][2]:.2f}  MDD={qc_pre[4][3]:.1%}")

    y_pre = np.array([all_q_rets_pre[nm][4].get(d, 0) for d, f in valid_dates])
    a_pre, t_pre, s_pre = carhart_alpha(y_pre, cfd)
    print(f"  [수수료전] α={a_pre*100:.2f}%(t={t_pre:.2f}{s_pre})")

    # 수수료 후
    qc_post = []
    for q in range(N_QUINTILES):
        nav = (all_q_rets_post[nm][q] + 1).cumprod()
        m = get_metrics(all_q_rets_post[nm][q], nav)
        qc_post.append(m)

    mono_post = all(qc_post[j][0] <= qc_post[j+1][0] for j in range(4))
    print(f"  [수수료후] Q1={qc_post[0][0]:.2%}  Q2={qc_post[1][0]:.2%}  Q3={qc_post[2][0]:.2%}  Q4={qc_post[3][0]:.2%}  Q5={qc_post[4][0]:.2%}  단조:{'O' if mono_post else 'X'}")
    print(f"  [수수료후] Q5: Vol={qc_post[4][1]:.1%}  Sharpe={qc_post[4][2]:.2f}  MDD={qc_post[4][3]:.1%}")

    y_post = np.array([all_q_rets_post[nm][4].get(d, 0) for d, f in valid_dates])
    a_post, t_post, s_post = carhart_alpha(y_post, cfd)
    print(f"  [수수료후] α={a_post*100:.2f}%(t={t_post:.2f}{s_post})")

    avg_to = np.mean(turnover_list[nm]) if turnover_list[nm] else 0
    avg_tc = np.mean(trade_cost_list[nm]) if trade_cost_list[nm] else 0
    print(f"  Q5 턴오버={avg_to:.1%}  월평균거래비용={avg_tc*100:.3f}%")

    yls_pre = np.array([all_q_rets_pre[nm][4].get(d, 0) - all_q_rets_pre[nm][0].get(d, 0) for d, f in valid_dates])
    als, tls, sls = carhart_alpha(yls_pre, cfd)
    print(f"  L/S α={als*100:.2f}%(t={tls:.2f}{sls})")

print("\n완료!")

# ═══════════════════════════════════════
# 5. 시각화: 히트맵 + 퀀타일 NAV
# ═══════════════════════════════════════
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

nm_target = "윈저+Amihud"

# Q5 수수료후 월별 수익률
q5_post = all_q_rets_post[nm_target][4].copy()
q5_post = q5_post.iloc[1:]  # 첫 0 제거

# 연도-월 피벗
df_ret = pd.DataFrame({'ret': q5_post})
df_ret['year'] = df_ret.index.year
df_ret['month'] = df_ret.index.month
heatmap_data = df_ret.pivot(index='year', columns='month', values='ret') * 100  # %

# 연간 수익률 추가
annual_rets = q5_post.groupby(q5_post.index.year).apply(lambda x: (1+x).prod() - 1) * 100
heatmap_data[13] = annual_rets

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# ── Panel 1: 월별 수익률 히트맵 ──
ax1 = axes[0, 0]
month_labels = ['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월','연간']
im = ax1.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
ax1.set_xticks(range(13))
ax1.set_xticklabels(month_labels, fontsize=9)
ax1.set_yticks(range(len(heatmap_data.index)))
ax1.set_yticklabels(heatmap_data.index, fontsize=9)
for yi in range(heatmap_data.shape[0]):
    for xi in range(heatmap_data.shape[1]):
        val = heatmap_data.values[yi, xi]
        if not np.isnan(val):
            color = 'white' if abs(val) > 6 else 'black'
            ax1.text(xi, yi, f'{val:.1f}', ha='center', va='center', fontsize=7, color=color, fontweight='bold')
ax1.set_title('윈저+Amihud Q5 월별 수익률 (%, 수수료후)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax1, shrink=0.8, label='수익률 (%)')

# ── Panel 2: 퀀타일 NAV (수수료후) ──
ax2 = axes[0, 1]
colors_q = ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#1f77b4']
labels_q = ['Q1 (하위)', 'Q2', 'Q3', 'Q4', 'Q5 (상위)']
for q in range(N_QUINTILES):
    nav = (all_q_rets_post[nm_target][q] + 1).cumprod()
    m = get_metrics(all_q_rets_post[nm_target][q], nav)
    ax2.plot(nav.index, nav.values, color=colors_q[q], linewidth=1.8,
             label=f'{labels_q[q]}: CAGR={m[0]:.1%}')
ax2.set_title('윈저+Amihud 퀀타일별 NAV (수수료후)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.set_ylabel('NAV')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# ── Panel 3: Q5 수수료 전/후 비교 ──
ax3 = axes[1, 0]
for sname, stype in STRATEGIES:
    nm = sname
    nav_pre = (all_q_rets_pre[nm][4] + 1).cumprod()
    nav_post = (all_q_rets_post[nm][4] + 1).cumprod()
    m_pre = get_metrics(all_q_rets_pre[nm][4], nav_pre)
    m_post = get_metrics(all_q_rets_post[nm][4], nav_post)
    ax3.plot(nav_post.index, nav_post.values, linewidth=2,
             label=f'{nm}: {m_post[0]:.1%} (Sharpe={m_post[2]:.2f})')
ax3.set_title('전략별 Q5 NAV 비교 (수수료후)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9, loc='upper left')
ax3.set_ylabel('NAV')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# ── Panel 4: CAGR 바 차트 ──
ax4 = axes[1, 1]
x = np.arange(N_QUINTILES)
w_bar = 0.2
colors_s = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
for idx_s, (sname, stype) in enumerate(STRATEGIES):
    nm = sname
    cagrs = []
    for q in range(N_QUINTILES):
        nav = (all_q_rets_post[nm][q] + 1).cumprod()
        m = get_metrics(all_q_rets_post[nm][q], nav)
        cagrs.append(m[0])
    bars = ax4.bar(x + (idx_s - 1.5) * w_bar, [c*100 for c in cagrs], w_bar,
                   color=colors_s[idx_s], edgecolor='black', linewidth=0.5, label=nm, alpha=0.85)
ax4.set_xticks(x)
ax4.set_xticklabels(['Q1\n(하위)', 'Q2', 'Q3', 'Q4', 'Q5\n(상위)'], fontsize=11)
ax4.set_ylabel('CAGR (%)')
ax4.set_title('전략별 퀀타일 CAGR (수수료후)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, axis='y', alpha=0.3)
ax4.axhline(y=0, color='black', linewidth=0.5)

fig.suptitle(
    f'네트워크 기반 적응형 전략 — 윈저라이징 + Amihud 차등비용\n'
    f'유동성 필터: 거래대금 하위 10% 제거 | NAV 기반 거래비용 반영\n'
    f'윈저+Amihud Q5: CAGR(전)=22.3%, CAGR(후)=15.6%, α=9.95%(t=4.16***)',
    fontsize=13, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig('winsorize_amihud_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("winsorize_amihud_heatmap.png 저장 완료!")
