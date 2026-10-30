import json, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from lemos import *

plt.rcParams.update({'font.size': 5.6, 'axes.labelsize': 6, 'axes.titlesize': 6.2, 'xtick.labelsize': 5.4, 'ytick.labelsize': 5.4,
                     'legend.fontsize': 5.2, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                     'xtick.major.size': 2, 'ytick.major.size': 2, 'lines.linewidth': 0.9, 'lines.markersize': 2.6,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
C = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#444444']
W1, W2 = 3.4, 7.1
D = '../data/'; F = '../figs/'

toy = json.load(open(D + 'toy_results.json'))
for t in toy:
    t['kind'] = 'split' if t['kind'] == 'split' else 'inert'

# ---------- Fig: exact success vs leakage (3 toy configs)
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.55), sharey=True)
for ax, idx, title in zip(axs, (1, 8), ('split ring: n=2, q=5, k=1, l=4, eta=1', 'inert ring: n=2, q=7, k=1, l=4, eta=2')):
    t = toy[idx]
    lam = np.array(sorted(int(x) for x in t['succ']))
    emp = np.array([t['succ'][str(x)] for x in lam])
    ub = np.minimum(1, 2.0 ** (lam - t['lam0']))
    lb = np.maximum(0, 1 - t['NS'] * t['Ecoll'] / (4 * 2.0 ** lam))
    lbth = np.maximum(0, 1 - t['NS'] * t['q'] ** (-t['n'] * t['k']) * (1 + t['rho']) ** t['n'] / (4 * 2.0 ** lam)) if t['kind'] == 'split' else \
        np.maximum(0, 1 - t['NS'] * (t['q'] ** (-t['n'] * t['k']) + 1 / t['NS']) / (4 * 2.0 ** lam))
    ax.plot(lam, emp, 'o-', color=C[0], label='exact optimal adversary')
    ax.plot(lam, ub, '--', color=C[1], label='Thm. 1 ceiling')
    ax.plot(lam, lbth, ':', color=C[2], label='Thm. 2 floor')
    ax.axvline(t['lam0'], color='#888', lw=0.5)
    ax.set_xlabel('leaked bits $\\lambda$'); ax.set_title(title, fontsize=5.2)
    ax.set_ylim(-0.03, 1.05)
axs[0].set_ylabel('key-recovery probability'); axs[0].legend(loc='center left', frameon=False, bbox_to_anchor=(0.0, 0.62))
fig.tight_layout(pad=0.3); fig.savefig(F + 'fig_window.pdf'); plt.close()

# ---------- Fig: gap bars
fig, ax = plt.subplots(figsize=(W1, 1.45))
x = np.arange(len(toy)); w = 0.27
lab = ['(%d,%d,%d,%d,%d)' % (t['n'], t['q'], t['k'], t['l'], t['eta']) for t in toy]
meas = np.array([max(t['gap_emp'], 1e-4) for t in toy])
bnd = np.array([t['gap_split'] if t['kind'] == 'split' else t['gap_inert'] for t in toy])
ax.bar(x - w / 2, meas, w, color=C[0], label='measured gap $\\log_2(q^{nk}/\\mathbb{E}|\\mathrm{Im}|)$')
ax.bar(x + w / 2, bnd, w, color=C[4], label='Lemma 3 bound')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(lab, rotation=45, ha='right', fontsize=4.6)
ax.set_ylabel('gap (bits)'); ax.set_ylim(5e-5, 2e2); ax.legend(frameon=False, loc='upper left', ncol=2)
fig.tight_layout(pad=0.3); fig.savefig(F + 'fig_gap.pdf'); plt.close()

# ---------- Fig: heatmaps (lambda_max, leakage rate)
ax_ = json.load(open(D + 'heat_axes.json')); H = np.loadtxt(D + 'heat_lambda.csv', delimiter=','); R = np.loadtxt(D + 'heat_rate.csv', delimiter=',')
fig, axs = plt.subplots(1, 2, figsize=(W2, 1.75))
cm = LinearSegmentedColormap.from_list('b', ['#f7fbff', '#6baed6', '#08306b'])
for ax, M, ttl, fmt in ((axs[0], np.maximum(H, 0), 'tolerated leakage $\\lambda_{\\max}$ (bits), $n{=}256$, $k{=}1$, $q{=}7681$', '%.0f'),
                        (axs[1], np.maximum(R, 0), 'leakage rate $\\lambda_0/\\log_2|S|$', '%.2f')):
    im = ax.imshow(M, aspect='auto', cmap=cm, origin='lower')
    ax.set_xticks(range(0, len(ax_['ls']), 2)); ax.set_xticklabels(ax_['ls'][::2]); ax.set_yticks(range(len(ax_['etas']))); ax.set_yticklabels(ax_['etas'])
    ax.set_xlabel('module rank $\\ell$'); ax.set_ylabel('secret bound $\\eta$'); ax.set_title(ttl)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.ax.tick_params(labelsize=4.8, width=0.4)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if (i % 2 == 0 and j % 3 == 0): ax.text(j, i, fmt % M[i, j], ha='center', va='center', fontsize=3.6, color='white' if M[i, j] > 0.55 * M.max() else '#222')
fig.tight_layout(pad=0.3); fig.savefig(F + 'fig_heat.pdf'); plt.close()

# ---------- Fig: box plot latencies
df = pd.read_csv(D + 'perf_raw.csv')
order = ['P0', 'P1', 'P2', 'P3', 'P4', 'Lamport']
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.55), sharey=True)
for ax, op in zip(axs, ('KeyGen', 'Verify')):
    data = [df[(df.scheme == s) & (df.op == op)].usec.values for s in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True, flierprops=dict(markersize=1.2, markeredgewidth=0.3),
                    medianprops=dict(color='k', lw=0.7), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5), boxprops=dict(lw=0.5))
    for patch, c in zip(bp['boxes'], C[:5] + [C[6]]): patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_xticklabels(order, rotation=30); ax.set_title(op); ax.set_yscale('log'); ax.set_yticks([200,300,500,1000,2000]); ax.set_yticklabels(['200','300','500','1000','2000']); ax.minorticks_off()
axs[0].set_ylabel('latency per call ($\\mu$s, log scale)')
fig.tight_layout(pad=0.3); fig.savefig(F + 'fig_box.pdf'); plt.close()

# ---------- Fig: sizes stacked bars
prof = json.load(open(D + 'profiles.json'))
names = ['P0', 'P1', 'P2', 'P3', 'P4']
pk = [prof[n]['pk'] for n in names] + [32 * 512]; sk = [prof[n]['sk'] for n in names] + [32 * 512]; tok = [prof[n]['sk'] for n in names] + [32 * 256]
fig, ax = plt.subplots(figsize=(W1, 1.5))
x = np.arange(6); w = 0.26
ax.bar(x - w, pk, w, color=C[0], label='public key'); ax.bar(x, sk, w, color=C[1], label='secret key'); ax.bar(x + w, tok, w, color=C[2], label='token on the wire')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(names + ['Lamport'], rotation=0); ax.set_ylabel('bytes (log)')
for i, v in enumerate(tok): ax.text(i + w, v * 1.08, '%d' % v, ha='center', fontsize=4.2)
ax.legend(frameon=False, ncol=3, loc='upper left'); ax.set_ylim(100, 6e4)
fig.tight_layout(pad=0.3); fig.savefig(F + 'fig_sizes.pdf'); plt.close()

# ---------- Fig: sk vs lambda target
sv = pd.read_csv(D + 'sk_vs_lambda.csv')
fig, ax = plt.subplots(figsize=(W1, 1.55))
for sec, c, m in zip((128, 192, 256), C[:3], 'osd'):
    d = sv[sv.sec_target == sec]
    ax.plot(d.lam_target, d.sk, marker=m, color=c, label='%d-bit target' % sec)
lam = np.linspace(0, 6000, 50)
ax.plot(lam, 600 + lam / 8 * 1.0, '--', color='#777', lw=0.7, label='1 secret bit per leaked bit')
ax.set_xlabel('required leakage tolerance $\\lambda$ (bits)'); ax.set_ylabel('secret key size (bytes)'); ax.legend(frameon=False, loc='upper left')
fig.tight_layout(pad=0.3); fig.savefig(F + 'fig_skvslam.pdf'); plt.close()

# ---------- Fig: side channel
val = pd.read_csv(D + 'side_validation.csv'); T = pd.read_csv(D + 'side_Tstar.csv')
fig, axs = plt.subplots(1, 2, figsize=(W2, 1.6), gridspec_kw={'width_ratios': [1.05, 1]})
ax = axs[0]
groups = [(e, s) for e in (1, 3) for s in (0.25, 0.5, 1.0, 2.0)]
data = [val[(val.eta == e) & (val.sigma == s)].acc.values for e, s in groups]
bp = ax.boxplot(data, patch_artist=True, widths=0.55, flierprops=dict(markersize=1.2), medianprops=dict(color='k', lw=0.7), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5), boxprops=dict(lw=0.5))
for p, (e, s) in zip(bp['boxes'], groups): p.set_facecolor(C[0] if e == 1 else C[1]); p.set_alpha(0.7)
th = [val[(val.eta == e) & (val.sigma == s)].theory.iloc[0] for e, s in groups]
ax.plot(range(1, 9), th, 'k_', markersize=7, label='closed-form $\\mathbb{E}[\\max\\ \\mathrm{posterior}]$')
ax.set_xticklabels(['%g' % s for e, s in groups]); ax.set_xlabel('noise $\\sigma$ per sample (left: $\\eta{=}1$, right: $\\eta{=}3$)'); ax.set_ylabel('per-coefficient MAP accuracy')
ax.legend(frameon=False, loc='lower left')
ax = axs[1]
for nm, c, m in zip(('P1', 'P2', 'P3', 'P4'), C[:4], 'osd^'):
    d = T[T.profile == nm]
    y = d.Tstar.replace(0, np.nan)
    ax.plot(d.sigma, y, marker=m, color=c, label=nm)
ax.plot([8, 32], [2, 32], ':', color='#777', lw=0.7); ax.text(14, 3.0, 'slope 2', fontsize=4.6, color='#555')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([4,8,16,32]); ax.set_xticklabels(['4','8','16','32']); ax.minorticks_off(); ax.set_xlabel('single-trace noise $\\sigma$'); ax.set_ylabel('safe trace count $T^\\star$')
ax.legend(frameon=False, ncol=2, loc='upper left')
fig.tight_layout(pad=0.3); fig.savefig(F + 'fig_side.pdf'); plt.close()

# ---------- Fig: capability matrix
rows = ['BC key-rev. (2022)', 'TR-AC (2023)', 'LR-AC (2025)', 'Naor-Segev (2009)', 'ADW (2009)', 'AGV (2009)', 'KV (2009)', 'LLW (2024)', 'BRLWE-IoT (2024)', 'LeMoS (ours)']
cols = ['leakage\nobjective', 'explicit\nbound (bits)', 'matching\nlower bound', 'post-quantum\nassumption', 'noisy-leakage\nmodel', 'revocation /\nrotation', 'formal\nproof', 'device-level\nevaluation', 'parameter\nselection']
M = np.array([
 [0, 0, 0, 0, 0, 2, 0, 2, 0],   # BC key revocation
 [0, 0, 0, 0, 0, 2, 2, 2, 0],   # TR-AC
 [2, 1, 0, 0, 1, 0, 2, 2, 0],   # LR-AC
 [2, 2, 1, 0, 0, 0, 2, 0, 0],   # Naor-Segev
 [2, 2, 0, 0, 0, 0, 2, 0, 0],   # ADW
 [2, 2, 0, 2, 0, 0, 2, 0, 0],   # AGV
 [2, 2, 0, 1, 0, 0, 2, 0, 0],   # KV
 [2, 2, 1, 2, 0, 0, 2, 0, 0],   # LLW
 [1, 0, 0, 2, 1, 0, 1, 2, 0],   # BRLWE-IoT
 [2, 2, 2, 2, 2, 2, 2, 1, 2]])  # LeMoS
np.savetxt(D + 'capability_matrix.csv', M, delimiter=',', fmt='%d')
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={'width_ratios': [3.2, 1]})
cmap = LinearSegmentedColormap.from_list('c', ['#f2f2f2', '#9ecae1', '#08519c'])
axs[0].imshow(M, cmap=cmap, aspect='auto', vmin=0, vmax=2)
axs[0].set_xticks(range(len(cols))); axs[0].set_xticklabels(cols, fontsize=4.6); axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels(rows)
axs[0].xaxis.tick_top(); axs[0].tick_params(length=0)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        axs[0].text(j, i, ['--', 'part.', 'yes'][M[i, j]], ha='center', va='center', fontsize=4.2, color='white' if M[i, j] == 2 else '#333')
for s in axs[0].spines.values(): s.set_visible(False)
sc = M.sum(axis=1)
axs[1].barh(range(len(rows)), sc, color=[C[6]] * 9 + [C[1]], height=0.6); axs[1].invert_yaxis(); axs[1].set_yticks([])
for i, v in enumerate(sc): axs[1].text(v + 0.2, i, str(v), va='center', fontsize=4.8)
axs[1].set_xlabel('capability score (max 18)'); axs[1].set_xlim(0, 18)
fig.tight_layout(pad=0.3); fig.savefig(F + 'fig_caps.pdf'); plt.close()
print('scores', dict(zip(rows, sc)))

# ---------- Fig: rotation planning
rs = pd.read_csv(D + 'rotation_summary.csv'); rr = pd.read_csv(D + 'rotation_runs.csv')
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.55), gridspec_kw={'width_ratios': [1, 1.05]})
sub = rs[rs.profile == 'P2']
sig = [8, 16, 32, 64]; rate = [0.1, 1, 10, 100]
M = np.array([[sub[(sub.sigma == s) & (sub.rate == r)].mean_rot_day.iloc[0] for r in rate] for s in sig])
im = axs[0].imshow(np.log10(M + 1e-3), cmap='YlOrRd', aspect='auto', origin='lower', vmin=-3, vmax=2.5)
axs[0].set_xticks(range(4)); axs[0].set_xticklabels(['0.1', '1', '10', '100']); axs[0].set_yticks(range(4)); axs[0].set_yticklabels(sig)
axs[0].set_xlabel('probe traces per hour'); axs[0].set_ylabel('noise $\\sigma$ (P2)')
for i in range(4):
    for j in range(4):
        axs[0].text(j, i, ('%.2f' % M[i, j]) if M[i, j] < 10 else ('%.0f' % M[i, j]), ha='center', va='center', fontsize=4.8, color='white' if np.log10(M[i, j] + 1e-3) > 0.9 else '#222')
axs[0].set_title('rotations per day', fontsize=5.6)
ax = axs[1]
pos = []; data = []; cols = []
for i, s in enumerate(sig):
    for j, nm in enumerate(('P2', 'P4')):
        d = rr[(rr.profile == nm) & (rr.sigma == s) & (rr.rate == 10)].rot_per_day.values
        data.append(d); pos.append(i * 3 + j); cols.append(C[0] if nm == 'P2' else C[1])
bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, flierprops=dict(markersize=1), medianprops=dict(color='k', lw=0.6), whiskerprops=dict(lw=0.4), capprops=dict(lw=0.4), boxprops=dict(lw=0.4))
for p, c in zip(bp['boxes'], cols): p.set_facecolor(c); p.set_alpha(0.75)
ax.set_yscale('log'); ax.set_xticks([i * 3 + 0.5 for i in range(4)]); ax.set_xticklabels(sig); ax.set_xlabel('noise $\\sigma$ (10 traces/h)'); ax.set_ylabel('rotations per day')
ax.legend([bp['boxes'][0], bp['boxes'][1]], ['P2', 'P4'], frameon=False, loc='upper right')
fig.tight_layout(pad=0.3); fig.savefig(F + 'fig_rotate.pdf'); plt.close()

# ---------- Fig: trade-off leakage vs MSIS security
ms = pd.read_csv(D + 'msis_vs_l.csv')
fig, ax = plt.subplots(figsize=(W1, 1.5)); ax2 = ax.twinx()
for eta, c, m in ((1, C[0], 'o'), (3, C[1], 's')):
    d = ms[(ms.eta == eta) & np.isfinite(ms.classical)]
    ax.plot(d.l, d.classical, marker=m, color=c, label='MSIS bits, $\\eta{=}%d$' % eta)
    d2 = ms[ms.eta == eta]
    ax2.plot(d2.l, np.maximum(d2.lam, 0), ls='--', marker=m, color=c, alpha=0.55, markersize=1.8, label='$\\lambda_{\\max}$, $\\eta{=}%d$' % eta)
ax.axhline(128, color='#777', lw=0.5, ls=':'); ax.text(3, 132, '128-bit target', fontsize=4.6, color='#555')
ax.set_xlabel('module rank $\\ell$'); ax.set_ylabel('MSIS core-SVP bits (solid)'); ax2.set_ylabel('tolerated leakage, bits (dashed)')
ax2.spines['right'].set_visible(True); ax.set_xlim(2, 25)
h1, l1 = ax.get_legend_handles_labels(); ax.legend(h1, l1, frameon=False, loc='upper center', ncol=2)
fig.tight_layout(pad=0.3); fig.savefig(F + 'fig_tradeoff.pdf'); plt.close()
