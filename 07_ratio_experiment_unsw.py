"""
Augmentation ratio experiment for UNSW-NB15 (x1/x3/x5).
Mirrors 07_ratio_experiment.py; all 3 models.
"""

import math, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sdv.single_table import CTGANSynthesizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")

DATA_DIR    = Path("data/processed_unsw")
CTGAN_DIR   = Path("outputs/ctgan_unsw")
TABDDPM_DIR = Path("outputs/tabddpm_unsw")
DRIFT_DIR   = Path("outputs/drifting_unsw")
OUT_DIR     = Path("outputs/ratio_experiment_unsw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MINORITY_CLS = ["backdoor", "shellcode", "worms"]
LABEL_MAP    = {"backdoor": 1, "shellcode": 2, "worms": 3}
MINORITY_IDX = [1, 2, 3]
BASE_COUNTS  = {"backdoor": 1746, "shellcode": 1133, "worms": 130}
RATIOS       = [1, 3, 5]
MODELS       = ["CTGAN", "TabDDPM", "Drifting"]

RF_SEEDS  = [42, 123, 456, 789, 1024]
RF_PARAMS = dict(n_estimators=100, n_jobs=-1, class_weight="balanced",
                 max_features="sqrt")
COLORS    = {"CTGAN": "steelblue", "TabDDPM": "darkorange", "Drifting": "seagreen"}

feat_cols = pd.read_csv(DATA_DIR / "feature_names.csv")["feature"].tolist()
D = len(feat_cols)


# ── model classes (same as 07_ratio_experiment.py) ────────────────────────────
class Generator(nn.Module):
    def __init__(self, d, hidden=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(d, hidden), nn.SiLU()]
        for _ in range(n_layers-1): layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, d)); self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        h = self.dim//2
        f = torch.exp(-math.log(10_000)*torch.arange(h,device=t.device)/(h-1))
        a = t[:,None].float()*f[None]
        return torch.cat([torch.sin(a),torch.cos(a)],dim=-1)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x+self.net(x))

class MLPDiffusion(nn.Module):
    def __init__(self, d, hidden=512, n_layers=4, t_emb=128):
        super().__init__()
        self.t_embed    = nn.Sequential(SinusoidalEmbedding(t_emb),
                                        nn.Linear(t_emb,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.input_proj = nn.Linear(d, hidden)
        self.blocks     = nn.ModuleList([ResBlock(hidden) for _ in range(n_layers)])
        self.out_proj   = nn.Linear(hidden, d)
    def forward(self, x, t):
        h = self.input_proj(x)+self.t_embed(t)
        for b in self.blocks: h=b(h)
        return self.out_proj(h)

class GaussianDiffusion:
    def __init__(self, T=1000, b0=1e-4, b1=0.02):
        dev = torch.device("cpu")
        betas = torch.linspace(b0, b1, T, device=dev)
        alphas = 1-betas; ab = torch.cumprod(alphas, 0)
        ab_prev = F.pad(ab[:-1],(1,0),value=1.)
        self.T=T; self.betas=betas; self.alphas=alphas; self.alpha_bars=ab
        self.sqrt_ab=ab.sqrt(); self.sqrt_one_ab=(1-ab).sqrt()
        self.post_log_var=torch.log((betas*(1-ab_prev)/(1-ab)).clamp(min=1e-20))

    @torch.no_grad()
    def p_step(self, model, x, t):
        tb = torch.full((x.shape[0],),t,device=x.device,dtype=torch.long)
        ep = model(x,tb)
        x0 = (x-self.sqrt_one_ab[t]*ep)/self.sqrt_ab[t]
        abp = self.alpha_bars[t-1] if t>0 else torch.tensor(1.,device=x.device)
        c1  = abp.sqrt()*self.betas[t]/(1-self.alpha_bars[t])
        c2  = self.alphas[t].sqrt()*(1-abp)/(1-self.alpha_bars[t])
        mean = c1*x0+c2*x
        return mean if t==0 else mean+self.post_log_var[t].exp().sqrt()*torch.randn_like(x)

    @torch.no_grad()
    def sample(self, model, n, d):
        x = torch.randn(n,d,device=next(model.parameters()).device)
        for t in range(self.T-1,-1,-1): x=self.p_step(model,x,t)
        return x


# ── synth file path ────────────────────────────────────────────────────────────
def synth_path(model, cls, ratio):
    dirs = {"CTGAN": CTGAN_DIR, "TabDDPM": TABDDPM_DIR, "Drifting": DRIFT_DIR}
    suf  = "" if ratio==1 else f"_x{ratio}"
    return dirs[model] / f"X_synth_{model.lower()}_standard_{cls}{suf}.npy"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — generate x3 and x5
# ═══════════════════════════════════════════════════════════════════════════════
print("="*55 + "\nSTEP 1 — Generating x3 and x5\n" + "="*55)
diffusion = GaussianDiffusion()

for cls in MINORITY_CLS:
    # CTGAN
    ctgan = CTGANSynthesizer.load(str(CTGAN_DIR/f"ctgan_model_standard_{cls}.pkl"))
    for ratio in [3,5]:
        out = synth_path("CTGAN", cls, ratio)
        if out.exists(): print(f"  [skip] {out.name}"); continue
        n  = BASE_COUNTS[cls]*ratio; t0=time.perf_counter()
        X_s = ctgan.sample(num_rows=n)[feat_cols].values.astype(np.float32)
        np.save(out, X_s)
        print(f"  CTGAN    {cls:<12} x{ratio}  n={n:>5}  {time.perf_counter()-t0:.1f}s")

    # TabDDPM
    net = MLPDiffusion(D)
    net.load_state_dict(torch.load(TABDDPM_DIR/f"tabddpm_model_standard_{cls}.pt", weights_only=True))
    net.eval()
    for ratio in [3,5]:
        out = synth_path("TabDDPM", cls, ratio)
        if out.exists(): print(f"  [skip] {out.name}"); continue
        n  = BASE_COUNTS[cls]*ratio; t0=time.perf_counter()
        X_s = diffusion.sample(net, n, D).numpy().astype(np.float32)
        np.save(out, X_s)
        print(f"  TabDDPM  {cls:<12} x{ratio}  n={n:>5}  {time.perf_counter()-t0:.1f}s")

    # Drifting
    gen = Generator(D)
    gen.load_state_dict(torch.load(DRIFT_DIR/f"drifting_model_standard_{cls}.pt", weights_only=True))
    gen.eval()
    for ratio in [3,5]:
        out = synth_path("Drifting", cls, ratio)
        if out.exists(): print(f"  [skip] {out.name}"); continue
        n  = BASE_COUNTS[cls]*ratio; t0=time.perf_counter()
        with torch.no_grad():
            X_s = gen(torch.randn(n,D)).numpy().astype(np.float32)
        np.save(out, X_s)
        print(f"  Drifting {cls:<12} x{ratio}  n={n:>5}  {time.perf_counter()-t0:.3f}s")

print("\nAll synth files:")
for m in MODELS:
    for c in MINORITY_CLS:
        for r in RATIOS:
            p=synth_path(m,c,r); arr=np.load(p)
            print(f"  {p.name:<58} {arr.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — 5-seed RF
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}\nSTEP 2 — 5-seed RF ({len(RATIOS)} ratios x {len(RF_SEEDS)} seeds x {1+len(MODELS)} RFs)\n{'='*55}")

X_tr = np.load(DATA_DIR/"X_train_standard.npy")
y_tr = np.load(DATA_DIR/"y_train.npy")
X_te = np.load(DATA_DIR/"X_test_standard.npy")
y_te = np.load(DATA_DIR/"y_test.npy")

def build_augmented(X_base, y_base, model, ratio):
    xs,ys=[X_base],[y_base]
    for c in MINORITY_CLS:
        X_s=np.load(synth_path(model,c,ratio))
        xs.append(X_s); ys.append(np.full(len(X_s),LABEL_MAP[c],dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)

def rf_eval(Xtr, ytr, seed):
    rf=RandomForestClassifier(random_state=seed,**RF_PARAMS); rf.fit(Xtr,ytr)
    yp=rf.predict(X_te)
    _,rec,f1,_=precision_recall_fscore_support(y_te,yp,labels=MINORITY_IDX,zero_division=0)
    return {c:{"f1":float(f1[k]),"recall":float(rec[k])} for k,c in enumerate(MINORITY_CLS)}

raw_rows=[]
for seed in RF_SEEDS:
    print(f"  seed={seed}", end="  ", flush=True)
    base_m = rf_eval(X_tr, y_tr, seed)
    print("base done", end="  ", flush=True)
    for ratio in RATIOS:
        for model in MODELS:
            X_aug,y_aug = build_augmented(X_tr,y_tr,model,ratio)
            aug_m = rf_eval(X_aug,y_aug,seed)
            for c in MINORITY_CLS:
                raw_rows.append({"seed":seed,"model":model,"ratio":ratio,"class":c,
                                 "n_synth":BASE_COUNTS[c]*ratio,
                                 "f1_baseline":base_m[c]["f1"],
                                 "f1_augmented":aug_m[c]["f1"],
                                 "f1_delta":aug_m[c]["f1"]-base_m[c]["f1"],
                                 "recall_baseline":base_m[c]["recall"],
                                 "recall_augmented":aug_m[c]["recall"],
                                 "recall_delta":aug_m[c]["recall"]-base_m[c]["recall"]})
    print(flush=True)

df_raw=pd.DataFrame(raw_rows); df_raw.to_csv(OUT_DIR/"raw_results.csv",index=False)

agg=(df_raw.groupby(["model","ratio","class"])
     .agg(f1_delta_mean=("f1_delta","mean"),f1_delta_std=("f1_delta","std"),
          recall_delta_mean=("recall_delta","mean"),recall_delta_std=("recall_delta","std"),
          f1_aug_mean=("f1_augmented","mean"),f1_aug_std=("f1_augmented","std"))
     .round(4).reset_index())
agg.to_csv(OUT_DIR/"summary_aggregated.csv",index=False)

def fmt(m,s): return f"{m:+.4f} +- {s:.4f}"
pivot_rows=[]
for c in MINORITY_CLS:
    for model in MODELS:
        row={"class":c,"model":model}
        for ratio in RATIOS:
            sub=agg[(agg.model==model)&(agg.ratio==ratio)&(agg["class"]==c)]
            row[f"f1_delta_x{ratio}"]=(fmt(sub.f1_delta_mean.iloc[0],sub.f1_delta_std.iloc[0])
                                       if len(sub) else "-")
        pivot_rows.append(row)

df_pivot=pd.DataFrame(pivot_rows)
df_pivot.to_csv(OUT_DIR/"comparison_table.csv",index=False)

print("\n-- F1 delta (mean +- std, 5 seeds) ---")
print(df_pivot[["class","model"]+[f"f1_delta_x{r}" for r in RATIOS]].to_string(index=False))


# ── plots ──────────────────────────────────────────────────────────────────────
x=np.arange(len(MINORITY_CLS))
fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=False)
for ax,c in zip(axes,MINORITY_CLS):
    for model in MODELS:
        sub=agg[(agg.model==model)&(agg["class"]==c)].sort_values("ratio")
        means=sub.f1_delta_mean.values; stds=sub.f1_delta_std.values
        ax.plot(sub.ratio.values,means,marker="o",label=model,color=COLORS[model],linewidth=2)
        ax.fill_between(sub.ratio.values,means-stds,means+stds,alpha=0.15,color=COLORS[model])
    ax.axhline(0,color="black",linewidth=0.8,linestyle="--")
    ax.set_xticks(RATIOS); ax.set_xlabel("Augmentation ratio")
    ax.set_ylabel("F1 delta"); ax.set_title(c); ax.legend(fontsize=8)
plt.suptitle("UNSW-NB15 -- F1 delta vs ratio (mean +- 1std, 5 seeds)",fontsize=12)
plt.tight_layout(); fig.savefig(OUT_DIR/"f1_delta_vs_ratio.png",dpi=150); plt.close()

print(f"\nSaved to: {OUT_DIR.resolve()}")
print("Ratio experiment complete.")
