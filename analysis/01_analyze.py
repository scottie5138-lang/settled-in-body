"""DH2027 candidate-topic computations over the text-free copies in data/.

Writes tables to results/topicN_*.csv and headline numbers to results/key_results.json.
All inputs are candidate or purpose-limited evidence; outputs are exploratory
descriptions authorised by the research principal for this F: workspace only.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
R = os.environ.get("DH2027_RESULTS_DIR", os.path.join(ROOT, "results"))
SEED = 20271128
N_BOOT = 2000
AXES = ["court_institution", "landscape", "temple_reclusion", "route_displacement", "time_void"]
os.makedirs(R, exist_ok=True)
rng = np.random.default_rng(SEED)
key = {}


def save(df, name):
    df.to_csv(f"{R}/{name}.csv", index=False, encoding="utf-8-sig")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cluster_boot(df, cluster, stat):
    """Percentile CI of stat(df) resampling whole clusters (persons)."""
    groups = {k: g for k, g in df.groupby(cluster)}
    ids = np.array(list(groups))
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(ids, size=len(ids), replace=True)
        vals.append(stat(pd.concat([groups[i] for i in pick], ignore_index=True)))
    return np.nanpercentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------- Topic 1: five axes
work = pd.read_parquet(f"{D}/tang_exile/work_terminal.parquet")
person_of = work.set_index("source_record_id")["cbdb_person_id"]
axis = pd.read_parquet(f"{D}/five_axis/axis_view6005_flat.parquet")
axis["value"] = axis["human_status"].fillna(axis["working_status"])
axis["value"] = axis["value"].where(axis["value"].isin(["present", "absent"]))
axis["person"] = axis["source_record_id"].map(person_of)
eff_ids = set(work.loc[work.current_effective_exile_period_work, "source_record_id"])
axis_ids = set(axis.source_record_id)
key["t1_effective_work_level"] = len(eff_ids)
key["t1_axis_works"] = len(axis_ids)
key["t1_effective_not_in_axes"] = sorted(eff_ids - axis_ids)
key["t1_axes_not_effective"] = sorted(axis_ids - eff_ids)
key["t1_axis_works_without_person"] = int(axis.drop_duplicates("source_record_id").person.isna().sum())

rows = []
for ax in AXES:
    a = axis[(axis.field_id == ax) & axis.value.notna()].copy()
    a["y"] = (a.value == "present").astype(float)
    k, n = int(a.y.sum()), len(a)
    lo, hi = cluster_boot(a, "person", lambda x: x.y.mean())
    per_person = a.groupby("person").y.mean()
    ae_lo, ae_hi = np.percentile([rng.choice(per_person.values, len(per_person)).mean() for _ in range(N_BOOT)], [2.5, 97.5])
    rows.append({"axis": ax, "n_works_valued": n, "n_missing": int(((axis.field_id == ax) & axis.value.isna()).sum()),
                 "present": k, "work_level_prevalence": k / n,
                 "person_cluster_ci_low": lo, "person_cluster_ci_high": hi,
                 "author_equal_prevalence": per_person.mean(), "author_equal_ci_low": ae_lo, "author_equal_ci_high": ae_hi,
                 "n_persons": per_person.size})
prev = pd.DataFrame(rows)
save(prev, "topic1_axis_prevalence")

wide = axis.pivot_table(index=["source_record_id", "person"], columns="field_id", values="value", aggfunc="first").reset_index()
complete = wide.dropna(subset=AXES).copy()
for ax in AXES:
    complete[ax] = (complete[ax] == "present").astype(int)
complete["n_present"] = complete[AXES].sum(axis=1)
save(complete.n_present.value_counts().sort_index().rename_axis("n_axes_present").reset_index(name="works"), "topic1_cooccurrence_count")
complete["profile"] = complete[AXES].apply(lambda r: "+".join(a for a in AXES if r[a]) or "(none)", axis=1)
save(complete.profile.value_counts().rename_axis("profile").reset_index(name="works"), "topic1_profiles")
pair = []
for i, a in enumerate(AXES):
    for b in AXES[i + 1:]:
        t = pd.crosstab(complete[a], complete[b])
        phi = np.corrcoef(complete[a], complete[b])[0, 1]
        _, p, _, _ = stats.chi2_contingency(t)
        lift = complete[a].mul(complete[b]).mean() / (complete[a].mean() * complete[b].mean())
        pair.append({"axis_a": a, "axis_b": b, "both_present": int(complete[a].mul(complete[b]).sum()),
                     "phi": phi, "lift": lift, "chi2_p": p})
save(pd.DataFrame(pair), "topic1_pairwise_cooccurrence")
key["t1_complete_works"] = len(complete)
key["t1_share_two_or_more"] = float((complete.n_present >= 2).mean())
key["t1_mean_axes_present"] = float(complete.n_present.mean())

conc = complete.groupby("person").size().sort_values(ascending=False)
key["t1_persons_with_axis_works"] = int(conc.size)
key["t1_top5_share_complete"] = float(conc.head(5).sum() / conc.sum())
loo = []
for p in conc.head(5).index:
    sub = complete[complete.person != p]
    loo.append({"dropped_person_rank": int(list(conc.index).index(p) + 1), "dropped_works": int(conc[p]),
                **{ax: sub[ax].mean() for ax in AXES}})
loo.append({"dropped_person_rank": 0, "dropped_works": 0, **{ax: complete[ax].mean() for ax in AXES}})
save(pd.DataFrame(loo), "topic1_leave_top_author_out")

# validation: blind 50
blind = pd.read_parquet(f"{D}/five_axis/blind50_decisions.parquet")
blind = blind.merge(axis[["source_record_id", "field_id", "working_status", "gpt_suggested_status"]],
                    left_on=["DB_source_record_id", "field_id"], right_on=["source_record_id", "field_id"], how="left")
val = []
for ax in AXES + ["ALL"]:
    b = blind if ax == "ALL" else blind[blind.field_id == ax]
    m = b[b.working_status.isin(["present", "absent"])]
    agree = int((m.human_status == m.working_status).sum())
    lo, hi = wilson(agree, len(m))
    cp_lo = stats.beta.ppf(0.025, agree, len(m) - agree + 1) if agree else 0.0
    kappa = cohen_kappa_score(m.human_status, m.working_status) if m.human_status.nunique() > 1 else np.nan
    val.append({"axis": ax, "positions": len(b), "model_valued": len(m), "agree": agree, "agreement": agree / len(m),
                "wilson_low": lo, "clopper_pearson_low": cp_lo, "cohen_kappa": kappa,
                "human_present_rate": (b.human_status == "present").mean()})
save(pd.DataFrame(val), "topic1_blind50_validation")
key["t1_blind50"] = val[-1]

tgt = axis[axis.human_status.notna()]
conf = pd.crosstab(tgt.human_status, tgt.working_status.fillna("no_model_value"))
save(conf.reset_index(), "topic1_targeted309_confusion")
tv = tgt[tgt.working_status.isin(["present", "absent"])]
key["t1_targeted_positions"] = len(tgt)
key["t1_targeted_model_valued_agreement"] = float((tv.human_status == tv.working_status).mean())
key["t1_targeted_kappa"] = float(cohen_kappa_score(tv.human_status, tv.working_status))
save(tgt.assign(agree=tgt.human_status == tgt.working_status).groupby("field_id")
     .agg(positions=("agree", "size"), agree=("agree", "sum"),
          no_model=("working_status", lambda s: s.isna().sum())).reset_index(), "topic1_targeted309_by_axis")

m20 = pd.read_parquet(f"{D}/five_axis/medium20_resolved.parquet")
key["t1_medium20"] = {"positions": len(m20), "first_agrees_human": int((m20.first_status == m20.human_status).sum()),
                      "second_agrees_human": int((m20.second_status == m20.human_status).sum()),
                      "models_disagree": int((m20.first_status != m20.second_status).sum())}

# exploratory within-author contrast on one shared instrument (existing QTS-route observations)
reuse = pd.read_csv(f"{D}/five_axis/p5_existing_axes_private_reuse_candidate_v01_20261008_axes.csv")
reuse = reuse[reuse.transfer_status == "existing_selected_source_observation"].copy()
reuse["status"] = reuse.source_cells_json.map(lambda s: json.loads(s).get("working_status"))
reuse = reuse[reuse.status.isin(["present", "absent"])]
reuse["y"] = (reuse.status == "present").astype(float)
reuse["person"] = reuse.DB_source_record_id.map(person_of)
reuse["group"] = np.where(reuse.DB_source_record_id.isin(eff_ids), "exile_period", "other_bound_or_unbound")
contrast = []
for ax in AXES:
    a = reuse[reuse.field_id == ax]
    pp = a.groupby(["person", "group"]).y.mean().unstack()
    both = pp.dropna()
    diff = both["exile_period"] - both["other_bound_or_unbound"]
    bs = [rng.choice(diff.values, len(diff)).mean() for _ in range(N_BOOT)]
    wil = stats.wilcoxon(diff) if (diff != 0).any() else None
    contrast.append({"axis": ax, "works_exile": int(a[a.group == "exile_period"].DB_source_record_id.nunique()),
                     "works_other": int(a[a.group != "exile_period"].DB_source_record_id.nunique()),
                     "persons_with_both": len(both),
                     "pooled_exile": a[a.group == "exile_period"].y.mean(), "pooled_other": a[a.group != "exile_period"].y.mean(),
                     "within_author_mean_diff": diff.mean(), "boot_ci_low": np.percentile(bs, 2.5),
                     "boot_ci_high": np.percentile(bs, 97.5), "wilcoxon_p": wil.pvalue if wil else np.nan})
contrast = pd.DataFrame(contrast)
order = contrast.wilcoxon_p.rank(method="first").astype(int)
contrast["holm_p"] = [min(1.0, p * (len(contrast) - r + 1)) for p, r in zip(contrast.wilcoxon_p, order)]
contrast["holm_p"] = contrast.sort_values("wilcoxon_p").holm_p.cummax().reindex(contrast.index)
save(contrast, "topic1_exploratory_within_author_contrast")

# ---------------------------------------------------------------- Topic 2: precision tiers
cls = work.engineering_terminal_class.value_counts()
eff = work[work.current_effective_exile_period_work]
key["t2_tang_funnel"] = {"source_records": len(work), "accepted_binding": int(work.current_accepted_work_binding.sum()),
                         "effective": len(eff), "precise": int(eff.current_precise_exile_period_work.sum()),
                         "proxy": int(eff.current_purpose_limited_proxy_work.sum()),
                         "effective_neither_flag": int((~eff.current_precise_exile_period_work & ~eff.current_purpose_limited_proxy_work).sum())}
persons = pd.read_parquet(f"{D}/tang_exile/person_readback.parquet")
key["t2_person_sum_effective"] = int(persons.effective_exile_work_count.sum())
key["t2_person_sum_proxy"] = int(persons.purpose_limited_proxy_work_count.sum())
episodes = pd.read_parquet(f"{D}/tang_exile/episode_terminal.parquet")
ep_persons = set(episodes.cbdb_person_id)
naive = work[work.current_accepted_work_binding & work.cbdb_person_id.isin(ep_persons)]
key["t2_naive_exile_poet_corpus"] = len(naive)
key["t2_inflation_naive_over_effective"] = len(naive) / len(eff)
key["t2_naive_all_172_bound"] = int(work.current_accepted_work_binding.sum())
eff_by_person = eff.groupby("cbdb_person_id").size().sort_values(ascending=False)
key["t2_persons_with_effective"] = int(eff_by_person.size)
key["t2_top5_share_effective"] = float(eff_by_person.head(5).sum() / len(eff))
x = np.sort(eff_by_person.values)
key["t2_gini_effective_by_person"] = float((2 * np.arange(1, len(x) + 1) - len(x) - 1).dot(x) / (len(x) * x.sum()))
save(persons.engineering_terminal_class.value_counts().rename_axis("person_terminal_class").reset_index(name="persons"), "topic2_tang_person_classes")
save(episodes.groupby(["event_semantics_state", "active_boundary_state"]).size().reset_index(name="episodes"), "topic2_tang_episode_states")

song = pd.read_parquet(f"{D}/song_exile/full_population_m6_153_song_exile_existing_evidence_coverage_v01.parquet")
save(pd.crosstab(song.work_time_precision_tier, song.geographic_precision_tier, margins=True).reset_index(), "topic2_song_time_x_place")
save(pd.crosstab(song.person_event_time_precision_tier, song.work_time_precision_tier, margins=True).reset_index(), "topic2_song_person_event_vs_work_time")
n = len(song)
key["t2_song"] = {"records": n, "persons": int(song.cbdb_person_id.nunique()),
                  "work_time_missing": int((song.work_time_precision_tier == "missing").sum()),
                  "work_time_year_or_better": int(song.work_time_precision_tier.isin(["exact_year_or_closed_range", "exact_subannual"]).sum()),
                  "accepted_composition_place": int((song.geographic_precision_tier == "accepted_composition_place").sum()),
                  "loose_event_place_only": int((song.geographic_precision_tier == "loose_event_place_only").sum()),
                  "person_event_time_known_but_work_time_missing": int(((song.person_event_time_precision_tier != "missing") & (song.work_time_precision_tier == "missing")).sum()),
                  "both_year_and_accepted_place": int((song.work_time_precision_tier.isin(["exact_year_or_closed_range", "exact_subannual"]) & (song.geographic_precision_tier == "accepted_composition_place")).sum())}
sp = song.groupby("cbdb_person_id").agg(records=("source_record_id", "size"),
                                        dated=("work_time_precision_tier", lambda s: s.isin(["exact_year_or_closed_range", "exact_subannual"]).sum()))
sp["dated_share"] = sp.dated / sp.records
key["t2_song_author_equal_dated_share"] = float(sp.dated_share.mean())
key["t2_song_top5_share_records"] = float(sp.records.sort_values(ascending=False).head(5).sum() / n)

# ---------------------------------------------------------------- Topic 3: Literary Map alignment
cand = pd.read_parquet(f"{D}/lm_alignment/full_population_m6_158_tang_lm_body_consistency_candidate_evidence_v01.parquet")
dec = pd.read_parquet(f"{D}/lm_alignment/full_population_m6_159_tang_lm_body_consistency_human_g1_decision_v01.parquet")
j = dec.merge(cand, on="candidate_evidence_id", suffixes=("", "_c"))
j["accepted"] = (j.human_decision == "accepted").astype(int)
save(cand.body_match_type.value_counts().rename_axis("body_match_type").reset_index(name="edges"), "topic3_match_type_all_edges")
by = []
for col in ["body_match_type", "author_exact", "title_exact", "title_base_exact", "short_text_flag", "text_collision_flag", "multiple_strong_conflict"]:
    g = j.groupby(col).accepted.agg(["size", "sum"]).reset_index().rename(columns={col: "level", "size": "edges", "sum": "accepted"})
    g.insert(0, "feature", col)
    g["accept_rate"] = g.accepted / g.edges
    by.append(g)
save(pd.concat(by), "topic3_acceptance_by_feature")
auc = []
for col in ["sequence_similarity", "ngram_jaccard", "title_similarity", "length_ratio"]:
    s = j[[col, "accepted"]].dropna()
    auc.append({"feature": col, "n": len(s), "auc": roc_auc_score(s.accepted, s[col]),
                "median_accepted": s[s.accepted == 1][col].median(), "median_rejected": s[s.accepted == 0][col].median()})
save(pd.DataFrame(auc), "topic3_feature_auc")
j["seq_decile"] = pd.qcut(j.sequence_similarity.rank(method="first"), 10, labels=False) + 1
save(j.groupby("seq_decile").agg(edges=("accepted", "size"), accepted=("accepted", "sum"),
                                 seq_min=("sequence_similarity", "min"), seq_max=("sequence_similarity", "max")).reset_index(), "topic3_acceptance_by_similarity_decile")
prior = pd.read_parquet(f"{D}/lm_alignment/full_population_m6_158_tang_lm_prior_decision_audit_v01.parquet")
key["t3"] = {"candidate_edges": len(cand), "candidate_works": int(cand.source_record_id.nunique()),
             "reviewed_edges": len(dec), "accepted_edges": int(j.accepted.sum()),
             "accept_rate": float(j.accepted.mean()), "accept_rate_ci": wilson(int(j.accepted.sum()), len(j)),
             "prior_decisions": len(prior), "prior_wrong_body_quarantine": int(prior.wrong_local_body_quarantine.sum()),
             "reviewed_by_match_type": j.body_match_type.value_counts().to_dict()}
# 2,402 of the 2,437 rejections are a residual batch rule, not item-level judgments.
save(pd.crosstab([j.decision_source, j.original_export_decision.fillna("none")], j.human_decision).reset_index(), "topic3_decision_source")
item = j[j.decision_source == "human_workbench_local_save"].copy()
item = pd.concat([item, j[(j.decision_source != "human_workbench_local_save") & (j.original_export_decision == "rejected")]])
save(item.groupby(["body_match_type", "human_decision"]).size().unstack(fill_value=0).reset_index(), "topic3_item_level_by_match_type")
works = j.groupby("source_record_id").agg(edges=("accepted", "size"), accepted=("accepted", "sum"),
                                          has_exact=("body_match_type", lambda s: (s == "normalized_full_text_exact").any()))
key["t3"].update({"reviewed_works": len(works), "accepted_works": int((works.accepted > 0).sum()),
                  "item_level_saved_edges": len(item), "item_level_accepted": int(item.accepted.sum()),
                  "residual_batch_rejected": int(((j.decision_source != "human_workbench_local_save") & (j.original_export_decision != "rejected")).sum()),
                  "works_with_exact_edge": int(works.has_exact.sum()),
                  "works_with_exact_edge_accepted": int((works.has_exact & (works.accepted > 0)).sum()),
                  "prior_decision_works_overlap_current": int(len(set(prior.source_record_id) & set(dec.source_record_id)))})

# Where in the workbench order were item-level decisions made, and where do the swept exact edges sit?
queue = pd.read_parquet(f"{D}/lm_alignment/full_population_m6_158_tang_lm_body_consistency_low_confidence_review_queue_v01.parquet",
                        columns=["candidate_evidence_id", "review_order"])
jo = j.merge(queue, on="candidate_evidence_id")
jo["export_state"] = jo.original_export_decision.fillna("none")
last_item = int(jo.loc[jo.export_state != "pending", "review_order"].max())
jo["order_bin"] = pd.cut(jo.review_order, bins=range(0, int(jo.review_order.max()) + 250, 250), right=False)
save(pd.crosstab(jo.order_bin, jo.export_state).reset_index().astype({"order_bin": str}), "topic3_decisions_by_review_order")
swept_exact = jo[(jo.export_state == "pending") & (jo.body_match_type == "normalized_full_text_exact")]
reached_exact = jo[(jo.review_order <= last_item) & (jo.body_match_type == "normalized_full_text_exact")]
flags = ["short_text_flag", "text_collision_flag", "title_exact", "title_base_exact", "suite_cross_unique_injective"]
cmp_ = jo[jo.body_match_type == "normalized_full_text_exact"].groupby("export_state")[flags].mean().T.reset_index(names="flag")
save(cmp_, "topic3_exact_accepted_vs_swept_flags")
key["t3_review_order"] = {"last_item_level_review_order": last_item, "queue_length": int(jo.review_order.max()),
                          "swept_exact_edges": len(swept_exact),
                          "swept_exact_after_last_item_decision": int((swept_exact.review_order > last_item).sum()),
                          "exact_edges_within_reached_range": len(reached_exact),
                          "exact_within_reached_accepted": int((reached_exact.export_state == "accepted").sum()),
                          "exact_within_reached_left_pending": int((reached_exact.export_state == "pending").sum())}

# ---------------------------------------------------------------- Topic 4: LLM routing
s154 = pd.read_parquet(f"{D}/llm_prefill/full_population_m6_154_song_exile_g1_gpt_prefill_v01.parquet")
s155 = pd.read_parquet(f"{D}/llm_prefill/full_population_m6_155_song_exile_multi_candidate_gpt_prefill_v01.parquet")
t157 = pd.read_parquet(f"{D}/llm_prefill/full_population_m6_157_tang_m6129_gpt_prefill_v01.parquet")
save(pd.crosstab(s154.recommended_option, s154.confidence, margins=True).reset_index(), "topic4_song154_option_x_confidence")
save(pd.crosstab(s155.recommended_option, s155.confidence, margins=True).reset_index(), "topic4_song155_option_x_confidence")
save(pd.crosstab([t157.gpt_boundary_case, t157.deterministic_boundary_case], t157.gpt_confidence, margins=True).reset_index(), "topic4_tang157_boundary_x_confidence")
active = t157.active_review.astype(bool)
low = t157.gpt_confidence == "low"
route = pd.DataFrame([
    {"routing_rule": "confidence == low", "flagged": int(low.sum()), "recall_of_594": low[active].mean()},
    {"routing_rule": "GPT boundary flag (equals the active queue)", "flagged": int(t157.gpt_boundary_case.sum()), "recall_of_594": t157.gpt_boundary_case[active].mean()},
    {"routing_rule": "deterministic boundary", "flagged": int(t157.deterministic_boundary_case.sum()), "recall_of_594": t157.deterministic_boundary_case[active].mean()},
    {"routing_rule": "differs from prior rule", "flagged": int(t157.recommendation_differs_from_prior.sum()), "recall_of_594": t157.recommendation_differs_from_prior[active].mean()},
])
save(route, "topic4_tang157_routing_recall")
save(pd.crosstab(t157.lane, t157.gpt_confidence, margins=True).reset_index(), "topic4_tang157_lane_x_confidence")
ax_g = axis[axis.human_status.notna() & axis.gpt_suggested_status.notna()]
save(ax_g.assign(agree=ax_g.gpt_suggested_status == ax_g.human_status).groupby("gpt_confidence").agree.agg(["size", "sum"]).reset_index(), "topic4_axis_gpt_confidence_vs_human")
key["t4"] = {"song_prefill": len(s154) + len(s155), "song_low": int((s154.confidence == "low").sum() + (s155.confidence == "low").sum()),
             "tang_prefill": len(t157), "tang_low": int(low.sum()), "tang_active_review": int(active.sum()),
             "tang_low_recall_of_active": float(low[active].mean()),
             "binding_human_decisions_on_gpt": 0,
             "axis_gpt_vs_human_n": len(ax_g), "axis_gpt_vs_human_agree": int((ax_g.gpt_suggested_status == ax_g.human_status).sum())}

# ---------------------------------------------------------------- Topic 5: governance
cl = pd.read_parquet(f"{D}/governance/full_corpus_source_level_claim_decisions_v01.parquet")
cc = pd.read_parquet(f"{D}/governance/full_corpus_source_level_claim_review_candidates_v01.parquet")
save(cc[["claim_topic", "candidate_wording"]].merge(cl[["claim_topic", "decision", "claim_accepted"]], on="claim_topic"), "topic5_claim_decisions")
el = pd.read_parquet(f"{D}/governance/full_corpus_author_set_downstream_eligibility_v01.parquet")
fun = el.groupby(["downstream_purpose", "corpus_dynasty"]).downstream_input_eligible.agg(["size", "sum"]).reset_index()
fun["eligible_share"] = fun["sum"] / fun["size"]
save(fun.rename(columns={"size": "author_sets", "sum": "eligible"}), "topic5_eligibility_funnel")
save(el[~el.downstream_input_eligible].groupby(["downstream_purpose", "exclusion_reason_code"]).size().reset_index(name="author_sets"), "topic5_exclusion_reasons")
ai = pd.read_parquet(f"{D}/governance/full_corpus_author_set_analysis_input_v01.parquet")
ai["acc"] = ai.has_accepted_identity.astype(bool)
key["t5"] = {"author_sets": len(ai), "accepted_identity": int(ai.acc.sum()),
             "records_total": int(ai.source_record_count.sum()),
             "records_with_accepted_identity_author": int(ai.loc[ai.acc, "source_record_count"].sum()),
             "claims_total": len(cl), "claims_accepted": int(cl.claim_accepted.sum())}
q = ai.groupby("corpus_dynasty").apply(lambda g: pd.Series({
    "author_sets": len(g), "records": g.source_record_count.sum(),
    "top1pct_author_record_share": g.source_record_count.sort_values(ascending=False).head(max(1, len(g) // 100)).sum() / g.source_record_count.sum(),
    "single_record_authors": (g.source_record_count == 1).mean()}), include_groups=False).reset_index()
save(q, "topic5_author_concentration")

with open(f"{R}/key_results.json", "w", encoding="utf-8") as f:
    json.dump(key, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else list(o))
print(json.dumps(key, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else list(o)))
