"""Check, from the logs of a finished (or running) run, that every FedASMU formula was
executed as written in the paper.

There is no official FedASMU code to compare with.  Instead, each run writes
``trace.csv`` (one row per received local update) with the *inputs* and the *outputs*
of every formula, as they were at run time.  This script recomputes every output
from the logged inputs with an independent implementation of the paper's formulas
and reports the largest difference.  It also checks the state hand-over between
updates (a device's parameters after one update are its parameters before the next)
and measures Eq. 1 and Formula 7 on the real model tensors (the realised mixing ratio
|w_new - w_old| / |w_upload - w_old| must equal alpha / beta).

Usage:
  python scripts/verify_fedasmu.py RUN_ID [RUN_ID ...]      # run ids under results/metrics
  python scripts/verify_fedasmu.py --all-profiles appendix   # every FedASMU/FedBuff run of a profile
Writes results/verification/<run_id>/{report.md, checks.json, evidence.txt, verification.png}.
"""
import argparse
import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")
OUT = os.path.join(ROOT, "results", "verification")


# ----------------------------------------------------------------- paper formulas
# Written again here on purpose (not imported from src/) so that the check is independent.
def xi_paper(lam, sig, iota, t, o):  # Eq. 2
    return lam / (np.sqrt(np.maximum(t, 1)) * (t - o + 1) ** sig) + iota


def mix(mu, x):  # Eq. 2 alpha / Formula 8 beta, with x clipped at 0
    x = np.maximum(x, 0.0)
    return mu * x / (1.0 + mu * x)


def phi_paper(gamma, ups, g, o):  # Formula 8
    return gamma / np.sqrt(np.maximum(g, 1)) * (1.0 - ups / np.sqrt(g - o + 1))


def _b(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


class Report:
    def __init__(self, run_id):
        self.run_id, self.rows, self.evidence = run_id, [], []

    def check(self, cid, what, source, n, err=None, tol=None, ok=None, note=""):
        if ok is None:
            ok = n > 0 and (err is None or (np.isfinite(err) and err <= tol))
        self.rows.append(dict(id=cid, what=what, source=source, n=int(n),
                              max_err=None if err is None else float(err), tol=tol, ok=bool(ok), note=note))

    def ev(self, title, lines):
        self.evidence.append((title, lines))


def rel_err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0:
        return float("nan")
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


def verify_fedasmu(rep, tr, cfg):
    f, L = cfg["fedasmu"], cfg["local"]["epochs"]
    mu_a, mu_b = f["mu_alpha"], f["mu_beta"]
    acc = tr[_b(tr.accepted)]
    # ---------------- server: triggering (Algorithm 1 lines 1-6) ----------------
    ev = pd.read_csv(os.path.join(MET, cfg["run_id"], "eval.csv"))
    rep.check("S0", f"at most m' = {cfg['max_concurrency']} devices train at the same time (triggered every T)",
              "Algorithm 1 l.1-6", len(ev), ok=len(ev) > 0 and int(ev.running.max()) <= cfg["max_concurrency"],
              note=f"devices in training at evaluation points: max {int(ev.running.max())}, mean {ev.running.mean():.1f}")
    # ---------------- server: staleness, Eq. 2, Eq. 1, Eq. 3 ----------------
    tau = tr.t - tr.o + 1
    rep.check("S1", "staleness tau = t - o + 1", "Algorithm 1 l.9", len(tr), rel_err(tr.staleness, tau), 0)
    bad = (tr.staleness > cfg["staleness_bound"]) == _b(tr.accepted)
    rep.check("S2", f"update discarded iff tau > {cfg['staleness_bound']}", "Algorithm 1 l.9-10", len(tr),
              ok=not bad.any(), note=f"discarded: {(~_b(tr.accepted)).sum()}")
    xi = xi_paper(acc.lam, acc.sig, acc.iota, acc.t, acc.o)
    rep.check("S3", "xi = lam/(sqrt(t)(t-o+1)^sig) + iota", "Eq. 2", len(acc), rel_err(acc.xi, xi), 1e-9)
    al = mix(mu_a, xi)
    rep.check("S4", "alpha = mu*xi/(1+mu*xi)", "Eq. 2", len(acc), rel_err(acc.alpha, al), 1e-9)
    rep.check("S5", "global model w_{t+1} = (1-alpha) w_t + alpha w_i, measured on the tensors: |w_{t+1}-w_t|/|w_i-w_t| = alpha",
              "Eq. 1", len(acc), float(np.max(np.abs(acc.eq1_ratio - acc.alpha))), 1e-4,
              note=f"max relative residual of the whole parameter vector {acc.eq1_resid.max():.1e}")
    g3 = acc[acc.g_dot_prev.notna()]
    c = mu_a / (1 + mu_a * g3.prev_xi) ** 2
    den = np.sqrt(np.maximum(g3.prev_t, 1)) * g3.prev_tau ** g3.sig_before
    gl, gs, gi = g3.g_dot_prev * c / den, g3.g_dot_prev * c * (-g3.lam_before * np.log(g3.prev_tau) / den), g3.g_dot_prev * c
    e = max(rel_err(g3.grad_lam, gl), rel_err(g3.grad_sig, gs), rel_err(g3.grad_iota, gi))
    rep.check("S6", "dF/dlam, dF/dsig, dF/diota (chain rule through alpha)", "Eq. 3 + appendix A", len(g3), e, 1e-9)
    lam = np.maximum(g3.lam_before - f["eta_lambda"] * g3.grad_lam, 0)
    sig = np.maximum(g3.sig_before - f["eta_sigma"] * g3.grad_sig, 0)
    iota = g3.iota_before - f["eta_iota"] * g3.grad_iota
    e = max(rel_err(g3.lam, lam), rel_err(g3.sig, sig), rel_err(g3.iota, iota))
    rep.check("S7", "lam/sig/iota <- x - eta_x * dF/dx", "Eq. 3", len(g3), e, 1e-9,
              note=f"first upload of a device has no previous update: {len(acc) - len(g3)} rows, parameters unchanged "
                   f"({'OK' if np.allclose(acc[acc.g_dot_prev.isna()].lam, acc[acc.g_dot_prev.isna()].lam_before) else 'MISMATCH'})")
    # state carried over between a device's uploads
    errs = []
    for _, d in acc.sort_values("received").groupby("device"):
        if len(d) > 1:
            errs.append(np.max(np.abs(d.lam_before.values[1:] - d.lam.values[:-1])))
            errs.append(np.max(np.abs(d.sig_before.values[1:] - d.sig.values[:-1])))
    first = acc.sort_values("received").groupby("device").head(1)
    init_ok = np.allclose(first.lam_before, f["lambda0"]) and np.allclose(first.sig_before, f["sigma0"])
    rep.check("S8", "per-device lam/sig: value before an upload == value after its previous upload; start at lam0/sig0",
              "Algorithm 1 (state per device)", len(acc), max(errs) if errs else 0.0, 1e-12, ok=(not errs or max(errs) <= 1e-12) and init_ok)
    # ---------------- device: fresh model request, Formula 7-9, reward ----------------
    req = tr[tr.request_after.notna()]
    rng_ok = req.request_after.between(1, L - 1).all()
    rep.check("D1", f"request slot l* in 1..L-1 (L = {L})", "Algorithm 2 l.4", len(req), ok=rng_ok and len(req) > 0)
    m = tr[_b(tr.merged)]
    rep.check("D2", "fresh model merged only if it is newer (g > o)", "Algorithm 2 l.6-7", len(m),
              ok=len(m) > 0 and bool((m.fresh_version > m.o).all()),
              note=f"requests {len(req)}, merged {len(m)}, no newer model {len(req) - len(m)}")
    phi = phi_paper(m.gamma_before, m.upsilon_before, m.fresh_version, m.o)
    rep.check("D3", "phi = gamma/sqrt(g) * (1 - upsilon/sqrt(g-o+1))", "Formula 8", len(m), rel_err(m.phi, phi), 1e-9)
    rep.check("D4", "beta = mu*phi/(1+mu*phi)", "Formula 8", len(m), rel_err(m.beta, mix(mu_b, phi)), 1e-9)
    rep.check("D5", "local model after merge = (1-beta) w_local + beta w_g, measured on the tensors", "Formula 7", len(m),
              float(np.max(np.abs(m.eq7_ratio - m.beta))), 1e-4)
    s = np.sqrt(m.fresh_version - m.o + 1)
    cc = mu_b / (np.sqrt(np.maximum(m.fresh_version, 1)) * (1 + mu_b * np.maximum(m.phi, 0)) ** 2)
    dg, du = m.grad_dot * cc * (1 - m.upsilon_before / s), -m.grad_dot * cc * m.gamma_before / s
    rep.check("D6", "dF/dgamma, dF/dupsilon", "Formula 9 + appendix A", len(m),
              max(rel_err(m.d_gamma, dg), rel_err(m.d_upsilon, du)), 1e-9)
    e = max(rel_err(m.gamma, m.gamma_before - f["eta_gamma"] * m.d_gamma),
            rel_err(m.upsilon, m.upsilon_before - f["eta_upsilon"] * m.d_upsilon))
    rep.check("D7", "gamma/upsilon <- x - eta_x * dF/dx", "Formula 9", len(m), e, 1e-9)
    errs = []
    for _, d in m.sort_values("received").groupby("device"):
        if len(d) > 1:
            errs.append(np.max(np.abs(d.gamma_before.values[1:] - d.gamma.values[:-1])))
    first = m.sort_values("received").groupby("device").head(1)
    init_ok = np.allclose(first.gamma_before, f["gamma0"]) and np.allclose(first.upsilon_before, f["upsilon0"])
    rep.check("D8", "per-device gamma: value before a merge == value after its previous merge; start at gamma0",
              "Algorithm 2 (device state)", len(m), max(errs) if errs else 0.0, 1e-12, ok=(not errs or max(errs) <= 1e-12) and init_ok)
    rep.check("D9", "reward R = loss(before merge) - loss(after merge), same mini-batch", "Algorithm 2 l.11", len(m),
              rel_err(m.reward, m.loss_before - m.loss_after), 1e-9,
              note=f"R > 0 (merge lowered the loss) in {(m.reward > 0).mean() * 100:.1f}% of merges")
    errs, bad_init = [], False
    for _, d in m.sort_values("received").groupby("device"):
        b_prev = np.concatenate([[0.0], d.b.values[:-1]])
        errs.append(np.max(np.abs(d.b.values - ((1 - f["rho"]) * b_prev + f["rho"] * d.reward.values))))
    rep.check("D10", "device baseline b <- (1-rho) b + rho R (starts at 0)", "Algorithm 2 l.13", len(m),
              max(errs) if errs else float("nan"), 1e-12)
    # ---------------- Q-learning, Eq. 6 ----------------
    q = tr[tr.q_h_after.notna()]
    tgt = q.q_h_before + f["q_phi"] * (q.reward + f["q_psi"] * q.q_max_next - q.q_h_before)
    rep.check("Q1", "H(s,a) <- H + phi (R + psi max_a' H(s',a') - H)", "Eq. 6", len(q), rel_err(q.q_h_after, tgt), 1e-9)
    nxt = np.clip(q.q_prev_state + q.q_action.map({0: 1, 1: 0, 2: -1}), 1, L - 1)
    rep.check("Q2", "action add/stay/minus moves l* by +1/0/-1; device requests at the new l*", "Eq. 6 (actions)", len(q),
              ok=len(q) > 0 and bool((nxt == q.q_next_state).all() and (q.request_after == q.q_next_state).all()))
    gshare = _b(q.q_greedy).mean() if len(q) else float("nan")
    rep.check("Q3", f"epsilon-greedy: greedy choices >= 1 - epsilon = {1 - f['epsilon']:.2f} (minus ties)", "Eq. 6 (policy)", len(q),
              ok=len(q) > 0 and gshare >= 1 - f["epsilon"] - 0.05, note=f"greedy share {gshare * 100:.1f}%")
    # ---------------- LSTM meta model, Eq. 5 ----------------
    mt = tr[tr.meta_slot.notna()]
    first_per_dev = mt.device.is_unique
    rep.check("M1", "LSTM decides the first request slot of each device once; the device uses it", "Eq. 5",
              len(mt), ok=len(mt) > 0 and first_per_dev and bool((mt.meta_slot == mt.request_after).all()) and
              bool((tr[tr.slot_src == "meta"].device.is_unique)),
              note=f"{len(mt)} devices; later trainings use Q-learning ({(tr.slot_src == 'q').sum()} rows)")
    ok_dec = True
    for _, r in mt.iterrows():
        dec, p = json.loads(r.meta_decisions), json.loads(r.meta_probs)
        slot = dec.index(1) + 1 if 1 in dec else L - 1
        ok_dec &= slot == int(r.meta_slot) and abs(sum(p) - 1) < 1e-4
    rep.check("M2", "sampled decisions s_l: first s_l = 1 gives l*; P(l*) sums to 1", "Eq. 5", len(mt), ok=len(mt) > 0 and ok_dec)
    mu = mt[mt.meta_adv.notna()]
    rep.check("M3", "advantage = R - b (moving-average baseline)", "Eq. 5 / REINFORCE", len(mu),
              rel_err(mu.meta_adv, mu.meta_reward - mu.meta_baseline_before), 1e-9)
    nz = mu[mu.meta_adv.abs() > 1e-8]
    same = (np.sign(nz.meta_lp_after - nz.meta_lp_before) == np.sign(nz.meta_adv)).mean() if len(nz) else float("nan")
    rep.check("M4", "policy-gradient step raises log P(chosen slots) when R > b and lowers it when R < b", "Eq. 5 / REINFORCE",
              len(nz), ok=len(nz) > 0 and same >= 0.95, note=f"direction correct in {same * 100:.1f}% of updates")
    # ---------------- evidence: real rows ----------------
    ex = acc[acc.g_dot_prev.notna()].head(1)
    for _, r in ex.iterrows():
        rep.ev("Server, one received update (Eq. 3 -> Eq. 2 -> Eq. 1)", [
            f"received={int(r.received)} device={int(r.device)} t={int(r.t)} o={int(r.o)} staleness={int(r.staleness)}",
            f"Eq.3  lam {r.lam_before:.6g} -> {r.lam:.6g}, sig {r.sig_before:.6g} -> {r.sig:.6g}, iota {r.iota_before:.6g} -> {r.iota:.6g}"
            f"   (g.dprev={r.g_dot_prev:.4g})",
            f"Eq.2  xi = {r.lam:.6g}/(sqrt({int(r.t)})*{int(r.staleness)}^{r.sig:.6g}) + {r.iota:.6g} = {r.xi:.6g};  "
            f"alpha = {mu_a:g}*xi/(1+{mu_a:g}*xi) = {r.alpha:.6f}",
            f"Eq.1  measured |w_t+1 - w_t| / |w_i - w_t| = {r.eq1_ratio:.6f}  (alpha {r.alpha:.6f})"])
    for _, r in m.head(1).iterrows():
        rep.ev("Device, one fresh-model merge (Formula 8 -> 7 -> 9, reward)", [
            f"received={int(r.received)} device={int(r.device)} started from version o={int(r.o)}, request after epoch "
            f"{int(r.request_after)} ({r.slot_src}), received fresh version g={int(r.fresh_version)}",
            f"F8  phi = {r.gamma_before:.6g}/sqrt({int(r.fresh_version)})*(1-{r.upsilon_before:.6g}/sqrt({int(r.fresh_version - r.o + 1)})) "
            f"= {r.phi:.6g};  beta = {r.beta:.6f}",
            f"F7  measured |w_after - w_local| / |w_g - w_local| = {r.eq7_ratio:.6f}  (beta {r.beta:.6f})",
            f"F9  gamma {r.gamma_before:.6g} -> {r.gamma:.6g}, upsilon {r.upsilon_before:.6g} -> {r.upsilon:.6g}",
            f"reward R = {r.loss_before:.4f} - {r.loss_after:.4f} = {r.reward:.4f}"])
    names = ("add", "stay", "minus")
    for _, r in q.head(1).iterrows():
        rep.ev("Device Q-learning, one update (Eq. 6)", [
            f"device={int(r.device)} state l*={int(r.q_prev_state)} action={names[int(r.q_action)]} -> l*={int(r.q_next_state)} "
            f"({'greedy' if str(r.q_greedy).lower() == 'true' else 'exploration'})",
            f"H {r.q_h_before:.5f} -> {r.q_h_after:.5f} = H + {f['q_phi']}*({r.reward:.4f} + {f['q_psi']}*{r.q_max_next:.5f} - H)"])
    for _, r in mu.head(1).iterrows():
        rep.ev("Server LSTM meta model, one REINFORCE update (Eq. 5)", [
            f"device={int(r.device)} features={r.meta_feat} P(l*)={r.meta_probs} decisions={r.meta_decisions} -> l*={int(r.meta_slot)}",
            f"R={r.meta_reward:.4f} b={r.meta_baseline_before:.4f} advantage={r.meta_adv:.4f}; "
            f"log P(decisions) {r.meta_lp_before:.5f} -> {r.meta_lp_after:.5f}"])
    return acc, m, q, mt


def verify_fedbuff(rep, tr, cfg):
    K = cfg["fedbuff"]["K"]
    g = _b(tr.fb_generated)
    n = len(tr)
    rep.check("B1", f"NVFLARE BuffModelManager makes a new global model after every K = {K} updates", "FedBuff Alg. 1",
              n, ok=n > 0 and int(g.sum()) == n // K and bool((tr.received[g] % K == 0).all()),
              note=f"{int(g.sum())} new global models from {n} updates")
    ver = tr.fb_model_version.values
    rep.check("B2", "model version increases by exactly 1 per aggregation", "FedBuff Alg. 1", n,
              ok=n > 0 and bool((np.diff(ver) == g.values[1:].astype(int)).all()))
    rep.ev("FedBuff, first aggregation", [tr[g].head(1).to_string(index=False)])


def figure(path, acc, m, q, mt, cfg):
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    a = ax[0, 0]
    a.scatter(acc.staleness, acc.alpha, s=4, alpha=0.4)
    a.set_xlabel("staleness t-o+1")
    a.set_ylabel("alpha (Eq. 2)")
    a.set_title("Server weight vs staleness")
    a = ax[0, 1]
    a.scatter(acc.alpha, acc.eq1_ratio, s=4, label="global update (Eq. 1)")
    if len(m):
        a.scatter(m.beta, m.eq7_ratio, s=4, label="device merge (Formula 7)")
    lim = [0, 1]
    a.plot(lim, lim, "k:", lw=0.8)
    a.set_xlabel("weight computed by the formula")
    a.set_ylabel("ratio measured on the model tensors")
    a.set_title("Formula weight == real mixing ratio")
    a.legend(fontsize=7)
    a = ax[0, 2]
    a.plot(acc.received, acc.lam, lw=0.6, label="lambda")
    a.plot(acc.received, acc.sig, lw=0.6, label="sigma")
    a.plot(acc.received, acc.iota * 100, lw=0.6, label="iota x100")
    a.set_xlabel("received updates")
    a.set_title("Server control parameters (Eq. 3)")
    a.legend(fontsize=7)
    a = ax[1, 0]
    if len(m):
        a.hist(m.reward, bins=40)
    a.axvline(0, color="k", lw=0.8)
    a.set_xlabel("reward = loss before - after merge")
    a.set_title("Device merge reward")
    a = ax[1, 1]
    L = cfg["local"]["epochs"]
    if len(q):
        qq = q.copy()
        qq["bin"] = pd.cut(qq.received, 10)
        sh = qq.groupby("bin", observed=True).request_after.value_counts(normalize=True).unstack().fillna(0)
        sh.columns = [f"l*={int(c)}" for c in sh.columns]
        sh.index = [f"{int(b.right)}" for b in sh.index]
        sh.plot.area(ax=a, legend=True)
        a.set_xticks(range(len(sh)))
        a.set_xticklabels(sh.index, fontsize=6, rotation=45)
        a.legend(fontsize=7, loc="lower left")
    a.set_xlabel("received updates (bin end)")
    a.set_title(f"Q-learning request slot share (L={L})")
    a = ax[1, 2]
    mu = mt[mt.meta_adv.notna()]
    if len(mu):
        a.scatter(mu.meta_adv, mu.meta_lp_after - mu.meta_lp_before, s=8)
    a.axhline(0, color="k", lw=0.8)
    a.axvline(0, color="k", lw=0.8)
    a.set_xlabel("advantage R - b")
    a.set_ylabel("change of log P(chosen slots)")
    a.set_title("LSTM REINFORCE step direction (Eq. 5)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def verify(run_id):
    d = os.path.join(MET, run_id)
    tp = os.path.join(d, "trace.csv")
    if not os.path.exists(tp):
        print(f"{run_id}: no trace.csv (run made before the verification trace existed)")
        return None
    cfg = json.load(open(os.path.join(d, "manifest.json")))["config"]
    st = json.load(open(os.path.join(d, "status.json"))).get("status")
    tr = pd.read_csv(tp)
    rep = Report(run_id)
    od = os.path.join(OUT, run_id)
    os.makedirs(od, exist_ok=True)
    algo = cfg["algorithm"]
    if algo == "fedasmu":
        acc, m, q, mt = verify_fedasmu(rep, tr, cfg)
        figure(os.path.join(od, "verification.png"), acc, m, q, mt, cfg)
    elif algo.startswith("fedbuff"):
        verify_fedbuff(rep, tr, cfg)
    else:
        print(f"{run_id}: no checks for {algo}")
        return None
    ok = all(r["ok"] for r in rep.rows)
    summ = dict(run_id=run_id, algorithm=algo, run_status=st, rows=len(tr), all_passed=ok, checks=rep.rows)
    json.dump(summ, open(os.path.join(od, "checks.json"), "w"), indent=1)
    with open(os.path.join(od, "evidence.txt"), "w") as fh:
        for title, lines in rep.evidence:
            fh.write(f"## {title}\n" + "\n".join(lines) + "\n\n")
    with open(os.path.join(od, "report.md"), "w") as fh:
        fh.write(f"# Verification of {run_id}\n\nRun status: {st}; {len(tr)} logged updates. "
                 f"Result: **{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}** "
                 f"({sum(r['ok'] for r in rep.rows)}/{len(rep.rows)}).\n\n")
        fh.write("| ID | Check | Paper | Rows | Max error | Tolerance | Result | Note |\n|---|---|---|---|---|---|---|---|\n")
        for r in rep.rows:
            e = "" if r["max_err"] is None else f"{r['max_err']:.1e}"
            t = "" if r["tol"] is None else f"{r['tol']:.0e}"
            fh.write(f"| {r['id']} | {r['what']} | {r['source']} | {r['n']} | {e} | {t} | {'PASS' if r['ok'] else 'FAIL'} | {r['note']} |\n")
        fh.write("\n## Example log rows\n\n```\n" + open(os.path.join(od, "evidence.txt")).read() + "```\n")
    print(f"{run_id}: {'PASS' if ok else 'FAIL'} {sum(r['ok'] for r in rep.rows)}/{len(rep.rows)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all-profiles", nargs="*", default=[])
    a = ap.parse_args()
    runs = list(a.runs)
    for p in a.all_profiles:
        runs += sorted(os.path.basename(x) for x in glob.glob(os.path.join(MET, f"{p}_*"))
                       if os.path.exists(os.path.join(x, "trace.csv")))
    res = [verify(r) for r in runs]
    sys.exit(0 if all(x is not False for x in res) else 1)


if __name__ == "__main__":
    main()
