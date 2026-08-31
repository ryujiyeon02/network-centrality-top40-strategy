import pandas as pd
import numpy as np
import warnings, time
warnings.filterwarnings('ignore')

LOOKBACK = 60
FEE_RATE = 0.003
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

prices_raw = pd.read_csv("수정주가.csv", index_col='Date', parse_dates=True, low_memory=False)
mcap_raw = pd.read_csv("시가총액.csv", index_col='Date', parse_dates=True, low_memory=False)
ks200 = pd.read_csv("KOSPI200.csv", index_col='Date', parse_dates=True)

prices = prices_raw.apply(pd.to_numeric, errors='coerce')
mcap = mcap_raw.apply(pd.to_numeric, errors='coerce')

prices_monthly = prices.resample('ME').last()
mcap_monthly = mcap.resample('ME').last()
returns_monthly = prices_monthly.pct_change(fill_method=None)
returns_daily = prices.pct_change()

ks200_monthly = ks200.resample('ME').last()
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
# 3. 전략: 도출 vs 고정 (2010~)
# ═══════════════════════════════════════
STRATEGIES = [
    ("도출 (β=loglog/2, w=√λ₂/λ₁, cut=√λ₁/tr)", 'derived'),
    ("고정 (β=0.25, w=0.5, cut=30%)", 'fixed'),
]

print(f"{'=' * 90}")
print(f"  전체 파라미터 도출 전략 — 2010년~")
print(f"  β=|loglog slope|/2, w=√(λ₂/λ₁), cutoff=√(λ₁/tr)")
print(f"{'=' * 90}")
print(f"  기간: {month_ends[0].strftime('%Y-%m')} ~ {month_ends[-1].strftime('%Y-%m')} ({len(month_ends)-1}개월)")
print(f"\n백테스트 진행 중...", flush=True)
t_start = time.time()

ks200_common = ks200_monthly.index.intersection(month_ends)
if len(ks200_common) > 0:
    ks200_m = ks200_monthly.loc[ks200_common]
    kospi_nav_full = ks200_m.iloc[:, 0] / ks200_m.iloc[0, 0]
    kospi_ret_full = kospi_nav_full.pct_change().fillna(0)
else:
    kospi_nav_full = None
    kospi_ret_full = None

all_q_rets = {}
for sname, _ in STRATEGIES:
    nm = f"{sname} | ScorePow"
    all_q_rets[nm] = {q: pd.Series(dtype=float) for q in range(N_QUINTILES)}
    for q in range(N_QUINTILES):
        all_q_rets[nm][q][month_ends[0]] = 0.0

prev_holdings = {}
turnover_list = {}
for nm in all_q_rets:
    prev_holdings[nm] = None
    turnover_list[nm] = []

derived_betas = []
derived_ws = []
derived_cuts = []

for i in range(len(month_ends) - 1):
    start = month_ends[i]; end = month_ends[i + 1]
    if (i + 1) % 12 == 0:
        print(f"  {start.strftime('%Y-%m')} ({time.time()-t_start:.1f}초)", flush=True)

    mc = mcap_monthly.loc[start].dropna()
    mc = mc[mc > 0]
    invest_universe = mc[mc >= mc.quantile(0.2)].index

    dw = returns_daily.loc[:start].tail(LOOKBACK)
    if len(dw) < 40:
        for nm in all_q_rets:
            for q in range(N_QUINTILES): all_q_rets[nm][q][end] = 0.0
        continue

    vc = dw.dropna(axis=1, thresh=int(LOOKBACK * 0.8)).columns
    ds = dw[vc].dropna(axis=1)
    if len(ds.columns) < 100:
        for nm in all_q_rets:
            for q in range(N_QUINTILES): all_q_rets[nm][q][end] = 0.0
        continue

    cols = ds.columns
    Ta = ds.shape[0]
    nr = returns_monthly.loc[end, cols].dropna()
    mc_vals = mc.reindex(cols).fillna(0).values

    investable = invest_universe.intersection(cols)
    mc_invest = mc.reindex(investable).dropna()

    corr = equal_weight_correlation(ds.values)
    cc, ev, evec = marchenko_pastur_clipping_full(corr, Ta)
    v1 = np.abs(evec[:, 0])
    tr = ev.sum()

    adj = np.maximum(cc, 0)
    np.fill_diagonal(adj, 0)

    # 도출 파라미터
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

    derived_betas.append(d_beta)
    derived_ws.append(d_w)
    derived_cuts.append(d_cut)

    for sname, stype in STRATEGIES:
        nm = f"{sname} | ScorePow"

        if stype == 'derived':
            beta = d_beta
            w = d_w
            exclude = mc_invest[mc_invest >= mc_invest.quantile(1 - d_cut)].index
        else:
            beta = 0.25
            w = 0.5
            exclude = mc_invest[mc_invest >= mc_invest.quantile(0.7)].index

        scores = compute_cap_scaled_centrality(adj, mc_vals, beta=beta)
        ss = pd.Series(scores, index=cols)
        lam1_s = pd.Series(v1, index=cols)

        common = ss.index.intersection(nr.index).intersection(investable).difference(exclude)

        if len(common) < 50:
            for q in range(N_QUINTILES): all_q_rets[nm][q][end] = 0.0
            continue

        sc = ss[common].values
        lc = lam1_s[common].values
        Nc = len(common)
        npq = Nc // N_QUINTILES

        cent_rank = np.argsort(np.argsort(sc)).astype(float)
        lam1_rank = np.argsort(np.argsort(lc)).astype(float)
        comp_score = cent_rank - w * lam1_rank
        sorted_idx = np.argsort(comp_score)
        tickers = common

        for q in range(N_QUINTILES):
            si = q * npq
            qi = sorted_idx[si:] if q == N_QUINTILES - 1 else sorted_idx[si:si + npq]
            q_tickers = tickers[qi]
            q_returns = nr[q_tickers]

            if q < N_QUINTILES - 1:
                all_q_rets[nm][q][end] = q_returns.mean() if len(q_returns) > 0 else 0.0
            else:
                q_scores = comp_score[qi]
                q_scores_shifted = q_scores - q_scores.min() + 1e-8
                q_scores_pow = q_scores_shifted ** 1.5
                weights = q_scores_pow / q_scores_pow.sum()
                all_q_rets[nm][q][end] = (q_returns.values * weights).sum() if len(q_returns) > 0 else 0.0

        # 턴오버
        q5_si = (N_QUINTILES - 1) * npq
        q5_qi = sorted_idx[q5_si:]
        q5_set = set(tickers[q5_qi])
        if prev_holdings[nm] is not None:
            old = prev_holdings[nm]
            if len(old) > 0 and len(q5_set) > 0:
                to = 1 - len(old & q5_set) / max(len(old), len(q5_set))
                turnover_list[nm].append(to)
        prev_holdings[nm] = q5_set

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
if kospi_nav_full is not None:
    mk = get_metrics(kospi_ret_full, kospi_nav_full)
else:
    mk = None

print(f"\n  도출 파라미터 통계 (2010~):")
print(f"    β = |loglog slope|/2:  avg={np.mean(derived_betas):.4f}  std={np.std(derived_betas):.4f}")
print(f"    w = √(λ₂/λ₁):        avg={np.mean(derived_ws):.4f}  std={np.std(derived_ws):.4f}")
print(f"    cut = √(λ₁/tr):       avg={np.mean(derived_cuts):.4f}  std={np.std(derived_cuts):.4f}")

print(f"\n{'=' * 130}")
print(f"  결과 (2010~)")
print(f"{'=' * 130}")

for nm in all_q_rets:
    print(f"\n── {nm} ──")
    qc = []
    for q in range(N_QUINTILES):
        nav = (all_q_rets[nm][q] + 1).cumprod()
        m = get_metrics(all_q_rets[nm][q], nav)
        qc.append(m)

    mono = all(qc[j][0] <= qc[j+1][0] for j in range(4))
    avg_to = np.mean(turnover_list[nm]) if turnover_list[nm] else 0

    print(f"  Q1={qc[0][0]:.2%}  Q2={qc[1][0]:.2%}  Q3={qc[2][0]:.2%}  Q4={qc[3][0]:.2%}  Q5={qc[4][0]:.2%}  단조:{'O' if mono else 'X'}")
    print(f"  Q5: Sharpe={qc[4][2]:.2f}  Vol={qc[4][1]:.1%}  MDD={qc[4][3]:.1%}  턴오버={avg_to:.1%}")

    y_raw = np.array([all_q_rets[nm][4].get(d, 0) for d, f in valid_dates])
    a_raw, t_raw, s_raw = carhart_alpha(y_raw, cfd)

    fee_monthly = avg_to * FEE_RATE * 2
    q5af = all_q_rets[nm][4].copy(); q5af.iloc[1:] -= fee_monthly
    nav5af = (q5af + 1).cumprod(); m5af = get_metrics(q5af, nav5af)
    yaf = np.array([q5af.get(d, 0) for d, f in valid_dates])
    aaf, taf, saf = carhart_alpha(yaf, cfd)

    yls = np.array([all_q_rets[nm][4].get(d, 0) - all_q_rets[nm][0].get(d, 0) for d, f in valid_dates])
    als, tls, sls = carhart_alpha(yls, cfd)

    print(f"  α(수수료전)={a_raw*100:.2f}%(t={t_raw:.2f}{s_raw})  α(수수료후)={aaf*100:.2f}%(t={taf:.2f}{saf})")
    print(f"  Q5(수수료후): CAGR={m5af[0]:.2%}  Sharpe={m5af[2]:.2f}")
    print(f"  L/S α={als*100:.2f}%(t={tls:.2f}{sls})")

    # 기간별 성과 (전반기/후반기)
    mid = month_ends[len(month_ends)//2]
    q5_first = {d: all_q_rets[nm][4].get(d, 0) for d in month_ends[1:] if d <= mid}
    q5_second = {d: all_q_rets[nm][4].get(d, 0) for d in month_ends[1:] if d > mid}
    if q5_first and q5_second:
        s1 = pd.Series(q5_first); n1 = (s1 + 1).cumprod(); m1 = get_metrics(s1, n1)
        s2 = pd.Series(q5_second); n2 = (s2 + 1).cumprod(); m2 = get_metrics(s2, n2)
        print(f"  전반기({month_ends[1].strftime('%Y')}~{mid.strftime('%Y')}): CAGR={m1[0]:.2%}  Sharpe={m1[2]:.2f}")
        print(f"  후반기({(mid+pd.Timedelta(days=30)).strftime('%Y')}~{month_ends[-1].strftime('%Y')}): CAGR={m2[0]:.2%}  Sharpe={m2[2]:.2f}")

if mk is not None:
    print(f"\n  KOSPI200 (2015~): CAGR={mk[0]:.2%}  Sharpe={mk[2]:.2f}  MDD={mk[3]:.1%}")
print("\n완료!")

# ═══════════════════════════════════════
# 5. 시각화
# ═══════════════════════════════════════
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 도출 전략 기준으로 시각화
nm_derived = [nm for nm in all_q_rets if '도출' in nm][0]
avg_to_derived = np.mean(turnover_list[nm_derived]) if turnover_list[nm_derived] else 0
fee_monthly_derived = avg_to_derived * FEE_RATE * 2

colors_q = ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#1f77b4']
labels_q = ['Q1 (하위)', 'Q2', 'Q3', 'Q4', 'Q5 (상위)']

fig, axes = plt.subplots(2, 2, figsize=(18, 13))

# ── Panel 1: 퀀타일 NAV (수수료 전) ──
ax1 = axes[0, 0]
for q in range(N_QUINTILES):
    nav = (all_q_rets[nm_derived][q] + 1).cumprod()
    m = get_metrics(all_q_rets[nm_derived][q], nav)
    ax1.plot(nav.index, nav.values, color=colors_q[q], linewidth=1.8,
             label=f'{labels_q[q]}: CAGR={m[0]:.1%}')
ax1.set_title('퀀타일별 NAV (수수료 전)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.set_ylabel('NAV')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# ── Panel 2: 퀀타일 NAV (수수료 후) ──
ax2 = axes[0, 1]
for q in range(N_QUINTILES):
    q_af = all_q_rets[nm_derived][q].copy()
    q_af.iloc[1:] -= fee_monthly_derived
    nav_af = (q_af + 1).cumprod()
    m_af = get_metrics(q_af, nav_af)
    ax2.plot(nav_af.index, nav_af.values, color=colors_q[q], linewidth=1.8,
             label=f'{labels_q[q]}: CAGR={m_af[0]:.1%}')
ax2.set_title('퀀타일별 NAV (수수료 후)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.set_ylabel('NAV')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# ── Panel 3: Q5 수수료 전/후 + KOSPI200 ──
ax3 = axes[1, 0]
nav_q5 = (all_q_rets[nm_derived][4] + 1).cumprod()
m_q5 = get_metrics(all_q_rets[nm_derived][4], nav_q5)
q5af = all_q_rets[nm_derived][4].copy(); q5af.iloc[1:] -= fee_monthly_derived
nav_q5af = (q5af + 1).cumprod(); m_q5af = get_metrics(q5af, nav_q5af)

ax3.plot(nav_q5.index, nav_q5.values, color='#1f77b4', linewidth=2,
         label=f'Q5 수수료전: CAGR={m_q5[0]:.1%}, Sharpe={m_q5[2]:.2f}')
ax3.plot(nav_q5af.index, nav_q5af.values, color='#1f77b4', linewidth=2, linestyle='--',
         label=f'Q5 수수료후: CAGR={m_q5af[0]:.1%}, Sharpe={m_q5af[2]:.2f}')
if kospi_nav_full is not None:
    # KOSPI200은 2015~ → 시작 시점 NAV에 맞춰 스케일
    ks_start = kospi_nav_full.index[0]
    scale = nav_q5.asof(ks_start)
    ax3.plot(kospi_nav_full.index, kospi_nav_full.values * scale,
             color='gray', linewidth=1.5, linestyle=':',
             label=f'KOSPI200 (2015~): CAGR={mk[0]:.1%}')
ax3.set_title('Q5 수수료 전/후 비교', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10, loc='upper left')
ax3.set_ylabel('NAV')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# ── Panel 4: CAGR 바 차트 (수수료 전 vs 후) ──
ax4 = axes[1, 1]
cagrs_pre = []
cagrs_post = []
for q in range(N_QUINTILES):
    nav = (all_q_rets[nm_derived][q] + 1).cumprod()
    m = get_metrics(all_q_rets[nm_derived][q], nav)
    cagrs_pre.append(m[0])

    q_af = all_q_rets[nm_derived][q].copy()
    q_af.iloc[1:] -= fee_monthly_derived
    nav_af = (q_af + 1).cumprod()
    m_af = get_metrics(q_af, nav_af)
    cagrs_post.append(m_af[0])

x = np.arange(N_QUINTILES)
w_bar = 0.35
bars1 = ax4.bar(x - w_bar/2, [c*100 for c in cagrs_pre], w_bar, color=[colors_q[i] for i in range(N_QUINTILES)],
                edgecolor='black', linewidth=0.5, label='수수료 전')
bars2 = ax4.bar(x + w_bar/2, [c*100 for c in cagrs_post], w_bar, color=[colors_q[i] for i in range(N_QUINTILES)],
                edgecolor='black', linewidth=0.5, alpha=0.5, label='수수료 후')

for bar, val in zip(bars1, cagrs_pre):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, cagrs_post):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1%}', ha='center', va='bottom', fontsize=9)

ax4.set_xticks(x)
ax4.set_xticklabels(['Q1\n(하위)', 'Q2', 'Q3', 'Q4', 'Q5\n(상위)'], fontsize=11)
ax4.set_ylabel('CAGR (%)')
ax4.set_title('퀀타일별 CAGR (수수료 전/후)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, axis='y', alpha=0.3)
ax4.axhline(y=0, color='black', linewidth=0.5)

# 도출 파라미터 정보
y_raw = np.array([all_q_rets[nm_derived][4].get(d, 0) for d, f in valid_dates])
a_raw, t_raw, s_raw = carhart_alpha(y_raw, cfd)
yaf_arr = np.array([q5af.get(d, 0) for d, f in valid_dates])
aaf, taf, saf = carhart_alpha(yaf_arr, cfd)

fig.suptitle(
    f'네트워크 기반 적응형 전략 (2010~2025, 도출 파라미터)\n'
    f'β=|loglog|/2 (avg {np.mean(derived_betas):.3f}), '
    f'w=√(λ₂/λ₁) (avg {np.mean(derived_ws):.3f}), '
    f'cut=√(λ₁/tr) (avg {np.mean(derived_cuts):.3f})\n'
    f'Q5 α(전)={a_raw*100:.1f}%(t={t_raw:.2f}{s_raw})  '
    f'α(후)={aaf*100:.1f}%(t={taf:.2f}{saf})  '
    f'MDD={m_q5[3]:.1%}  턴오버={avg_to_derived:.1%}',
    fontsize=13, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig('best_from2010.png', dpi=150, bbox_inches='tight')
plt.close()
print("best_from2010.png 저장 완료!")
