"""Empirical vs analytic Pfa for CA- and OS-CFAR (exponential noise, K=1)."""
import numpy as np

from _style import SERIES, plt, save
from radarlab import dsp

rng = np.random.default_rng(0)
pfas = np.logspace(-1, -4, 7)
emp = {"ca": [], "os": []}
maps = [rng.exponential(1.0, (256, 128)) for _ in range(8)]
for pfa in pfas:
    for k in emp:
        emp[k].append(np.mean([dsp.cfar_2d(P, (2, 2), (6, 4), pfa, k).mask.mean() for P in maps]))
fig, ax = plt.subplots(figsize=(5.6, 4.4))
ax.loglog(pfas, pfas, color="#5f5e58", lw=1, label="design Pfa")
ax.loglog(pfas, emp["ca"], "o-", color=SERIES[0], label="CA-CFAR (empirical)")
ax.loglog(pfas, emp["os"], "s--", color=SERIES[1], label="OS-CFAR k=0.75N (empirical)")
ax.set(xlabel="design Pfa", ylabel="measured Pfa", title="2-D CFAR false-alarm rate")
ax.legend(frameon=False)
save(fig, "cfar_pfa.png")
