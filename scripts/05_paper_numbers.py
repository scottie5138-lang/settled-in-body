"""Check selected manuscript values and supporting counts against aggregate result files.

Writes results/paper_numbers_trace.csv and exits non-zero if any recomputed value no
longer rounds to the figure printed in the abstract or the full text. Historical
expectations are preserved in the pre-revision submission archive.
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("DH2027_RESULTS_DIR", os.path.join(ROOT, "results"))


def csv(name):
    return pd.read_csv(f"{R}/{name}.csv")


k1x = json.load(open(f"{R}/topic1x_key_results.json", encoding="utf-8"))
k1y = json.load(open(f"{R}/topic1y_key_results.json", encoding="utf-8"))
blind = csv("topic1_blind50_validation").set_index("axis").loc["ALL"]
joint = csv("topic1y_joint_time_distance_model")
tier = joint[joint.model == "time + distance + dating tier"].set_index(["axis", "term"])
td = joint[joint.model == "time + distance"].set_index(["axis", "term"])
single = csv("topic1x_within_author_slopes").set_index(["covariate", "axis"])
pair = csv("topic1y_pairing_by_time_and_distance").set_index(["covariate", "level"])
cons = csv("topic1y_poet_direction_consistency").set_index("axis")
bai = csv("topic1y_poet_year_profiles").query("poet == '白居易'").set_index("years_in_exile")
dist = csv("topic1y_distance_by_exile_time").set_index("exile_time_bin")
Y, D500 = "per year in exile", "per 500 km from Chang'an"
k1z = json.load(open(f"{R}/topic1z_key_results.json", encoding="utf-8"))
stage = csv("topic1z_within_poet_stage_models").query("model == 'destination vs elsewhere, poets seen in both'").set_index("axis")
rob = csv("topic1z_destination_robustness").query("outcome == 'route_displacement'")
genre = csv("topic1z_title_genre_models").set_index(["model", "outcome", "term"]).sort_index()
phase = csv("topic1z_phase_within_poet").set_index(["outcome", "term"])
lexp = csv("topic1z_lexicon_phase_within_poet").set_index(["outcome", "term"])
lexa = csv("topic1z_lexicon_axis_agreement").set_index("lexicon")
w1 = csv("topic1w_within_poet_contrasts").set_index(["contrast", "outcome"])
amb = csv("topic1w_dwelling_homeward_association").set_index("contrast")
per = csv("topic1v_shang_period_poet_level").set_index("outcome")
coh = csv("topic1v_shang_cohort_749").set_index("outcome")
SEAT, ARR, YSEAT = "seat vs elsewhere", "arrival year vs en route", "per year, poems at the seat"

k1 = json.load(open(f"{R}/key_results.json", encoding="utf-8"))
# counts that come from poem-level files are read from their aggregate (18_public_counts.py)
pc = json.load(open(f"{R}/public_counts.json", encoding="utf-8"))
blind_sample_works = pc["blind_sample_works"]
# Current manuscript and synchronized short abstract; historical expectations are
# no longer kept: the pre-revision archives were deleted on 2026-11-12 after final submission files were made.
counts = pd.Series(pc["stage_counts"])
sensitivity = csv("topic1s_arrival_exchange_sensitivity").set_index("variant")
sk = json.load(open(f"{R}/topic1s_key_results.json", encoding="utf-8"))
coverage = csv("topic1s_temple_phase_coverage").set_index(["variant", "phase"])
claims = [
    # 2026-11-12 abstract rewrite: 8.4, the alternative anchor, -0.61 and 53/54 moved to the full-text checks
    ("poems by poets with exile episodes", k1["t2_naive_exile_poet_corpus"], 10142, 0, "key_results.json"),
    ("confirmed exile-period poems", k1["t2_tang_funnel"]["effective"], 1202, 0, "key_results.json"),
    ("annotated poems", k1x["works"], 1201, 0, "topic1x_key_results.json"),
    ("poets", k1x["coverage"]["persons"], 25, 0, "topic1x_key_results.json"),
    ("classifiable poems", counts.reindex(["destination", "elsewhere", "capital"]).sum(), 700, 0, "public_counts.json"),
    ("blind determinate comparisons in agreement", blind.agree, 247, 0, "topic1_blind50_validation.csv"),
    ("blind sample poems", blind_sample_works, 50, 0, "public_counts.json"),
    ("exchange title contrast (pp)", 100*w1.loc[(ARR,"title_exchange"),"slope"], 33, 0, "topic1w_within_poet_contrasts.csv"),
    ("exile self-naming annual change at seat (pp)", 100*w1.loc[(YSEAT,"lex_exile_self"),"slope"], -3, 0, "topic1w_within_poet_contrasts.csv"),
    ("court seat contrast (pp)", 100*w1.loc[(SEAT,"court_institution"),"slope"], 1, 0, "topic1w_within_poet_contrasts.csv"),
    ("court seat CI low (pp)", 100*w1.loc[(SEAT,"court_institution"),"ci_low"], -11, 0, "topic1w_within_poet_contrasts.csv"),
    ("court seat CI high (pp)", 100*w1.loc[(SEAT,"court_institution"),"ci_high"], 12, 0, "topic1w_within_poet_contrasts.csv"),
    # added with the review M3 abstract revision
    ("route contrast at the seat, evidence-adjusted (pp)", 100*rob.set_index("model").loc["evidence-adjusted seats","slope"], -20, 0, "topic1z_destination_robustness.csv"),
    ("temple next-year estimate rests on Bai Juyi: his share of determinate poems >= 0.95", float((coverage.loc[("all poets","settled +1"),"temple_determinate"] - coverage.loc[("without Bai Juyi","settled +1"),"temple_determinate"]) / coverage.loc[("all poets","settled +1"),"temple_determinate"] >= 0.95), 1, 0, "topic1s_temple_phase_coverage.csv"),
]
funnel = k1["t2_tang_funnel"]
rob = rob.set_index("model")
phase_len = csv("topic1z_phase_within_poet_length_controlled").set_index(["outcome", "term"])
carry = csv("topic1w_court_carriers_and_ambivalence").set_index("stage")
amb_all = amb.loc["all exile-period poems"]
direction = csv("topic1w_poet_direction_seat").set_index("outcome")
loo_key = json.load(open(f"{R}/topic1r_key_results.json", encoding="utf-8"))
rates = csv("topic1w_rates_by_phase").set_index("phase")
blind_axes = csv("topic1_blind50_validation").set_index("axis")
hard = csv("topic1_targeted309_confusion").set_index("human_status").loc[["absent", "present"], ["absent", "present"]]
gpt_conf = csv("topic4_axis_gpt_confidence_vs_human")
seats = csv("topic1z_destination_map").basis.value_counts()
text_claims = [
    *[(f"Tang corpus: {name}", funnel[name], value, 0, "key_results.json") for name,value in
      [("source_records",16701),("accepted_binding",11885),("effective",1202),("precise",325),("proxy",849),("effective_neither_flag",28)]],
    ("poems by poets with exile episodes", k1["t2_naive_exile_poet_corpus"],10142,0,"key_results.json"),
    ("biography / confirmed corpus ratio", k1["t2_inflation_naive_over_effective"], 8.4, 1, "key_results.json"),
    ("capital words correlation with first exile year", per.loc["lex_capital","spearman_start_year"], -0.61, 2, "topic1v_shang_period_poet_level.csv"),
    ("exchange title alternative-anchor contrast (pp)", 100*sensitivity.loc["nonexchange anchor","slope"], 33, 0, "topic1s_arrival_exchange_sensitivity.csv"),
    *[(f"coverage: {name}",k1x["coverage"][name],value,0,"topic1x_key_results.json") for name,value in
      [("episodes",28),("with_year",1051),("with_episode",1133),("with_coordinates",685)]],
    *[(f"corrected place classification: {name}",counts[name],value,0,"public_counts.json") for name,value in
      [("destination",437),("elsewhere",215),("capital",48),("no_reviewed_place",415),("no_episode",68),("destination_unmapped",15),("coarse",3)]],
    ("unclassifiable place",pc["works"]-counts.reindex(["destination","elsewhere","capital"]).sum(),501,0,"public_counts.json"),
    *[(f"mapped destinations: {name}",seats[name],value,0,"topic1z_destination_map.csv") for name,value in
      [("crosswalk_name",9),("admin_code_inference",9),("unmapped",2)]],
    ("classifiable poets",k1z["analysed"]["poets"],18,0,"topic1z_key_results.json"),
    ("classifiable episodes",k1z["analysed"]["episodes"],19,0,"topic1z_key_results.json"),
    ("blind kappa",blind.cohen_kappa,1.0,2,"topic1_blind50_validation.csv"),
    ("blind positions with no determinate model answer",blind.positions-blind.model_valued,3,0,"topic1_blind50_validation.csv"),
    ("targeted positions",k1["t1_targeted_positions"],309,0,"key_results.json"),
    ("targeted agreement percent",100*k1["t1_targeted_model_valued_agreement"],95.7,1,"key_results.json"),
    ("targeted kappa",k1["t1_targeted_kappa"],0.91,2,"key_results.json"),
    *[(f"lexicon hit, matching axis: {lex}",100*lexa.loc[lex,"axis_rate_lexicon_hit"],v,1,"topic1z_lexicon_axis_agreement.csv")
      for lex,v in [("lex_capital",74.1),("lex_road",79.5),("lex_temple",89.9)]],
    *[(f"lexicon miss, matching axis: {lex}",100*lexa.loc[lex,"axis_rate_lexicon_miss"],v,1,"topic1z_lexicon_axis_agreement.csv")
      for lex,v in [("lex_capital",34.7),("lex_road",44.0),("lex_temple",17.2)]],
    *[(f"poets in contrast: {c}",w1.loc[(c,"title_exchange"),"poets"],v,0,"topic1w_within_poet_contrasts.csv")
      for c,v in [(SEAT,13),(ARR,10),(YSEAT,7),("per year, all dated poems",16)]],
    *[(f"route contrast: {name}",100*rob.loc[name,"slope"],v,1,"topic1z_destination_robustness.csv")
      for name,v in [("as coded in 06",-14.8),("evidence-adjusted seats",-19.8),("crosswalk-verified destinations only",-11.5)]],
    ("route leave-one-out smallest decline",100*csv("topic1z_destination_leave_one_poet_out").route_destination_slope.max(),-17.5,1,"topic1z_destination_leave_one_poet_out.csv"),
    ("route leave-one-out largest decline",100*csv("topic1z_destination_leave_one_poet_out").route_destination_slope.min(),-25.5,1,"topic1z_destination_leave_one_poet_out.csv"),
    ("exchange contrast one decimal",100*w1.loc[(ARR,"title_exchange"),"slope"],33.5,1,"topic1w_within_poet_contrasts.csv"),
    ("exchange Holm p",w1.loc[(ARR,"title_exchange"),"holm_p"],0.35,3,"topic1w_within_poet_contrasts.csv"),
    ("self-naming Holm p",w1.loc[(YSEAT,"lex_exile_self"),"holm_p"],0.264,3,"topic1w_within_poet_contrasts.csv"),
    ("alternative anchor count",sk["nonexchange_anchor_episodes"],13,0,"topic1s_key_results.json"),
    ("unchanged anchors",sk["unchanged_anchor_episodes"],13,0,"topic1s_key_results.json"),
    *[(f"alternative arrival contrast: {col}",100*sensitivity.loc["nonexchange anchor",col],v,1,"topic1s_arrival_exchange_sensitivity.csv")
      for col,v in [("slope",33.5),("ci_low",3.1),("ci_high",50.9)]],
    ("alternative arrival poems",sensitivity.loc["nonexchange anchor","works"],147,0,"topic1s_arrival_exchange_sensitivity.csv"),
    ("next year, all poems",sk["original_next_year_all_poems"],71,0,"topic1s_key_results.json"),
    ("next-year phase sample, all poems",coverage.loc[("all poets","settled +1"),"works"],56,0,"topic1s_temple_phase_coverage.csv"),
    ("next-year phase sample, Bai Juyi poems",coverage.loc[("all poets","settled +1"),"bai_works"],55,0,"topic1s_temple_phase_coverage.csv"),
    ("next-year phase sample, without Bai Juyi",coverage.loc[("without Bai Juyi","settled +1"),"works"],1,0,"topic1s_temple_phase_coverage.csv"),
    ("next-year phase sample, determinate temple labels",coverage.loc[("all poets","settled +1"),"temple_determinate"],54,0,"topic1s_temple_phase_coverage.csv"),
    ("next-year phase sample, Bai Juyi determinate temple labels",coverage.loc[("all poets","settled +1"),"temple_determinate"] - coverage.loc[("without Bai Juyi","settled +1"),"temple_determinate"],53,0,"topic1s_temple_phase_coverage.csv"),
    ("temple next-year point estimate",100*phase.loc[("temple_reclusion","settled +1"),"slope"],24.5,1,"topic1z_phase_within_poet.csv"),
    ("temple next-year length-controlled",100*phase_len.loc[("temple_reclusion","settled +1"),"slope"],24.1,1,"topic1z_phase_within_poet_length_controlled.csv"),
    *[(f"self-naming annual: {col}",100*w1.loc[(YSEAT,"lex_exile_self"),col],v,1,"topic1w_within_poet_contrasts.csv")
      for col,v in [("slope",-3.3),("ci_low",-5.2),("ci_high",-1.5)]],
    *[(f"court at seat: {col}",100*w1.loc[(SEAT,"court_institution"),col],v,1,"topic1w_within_poet_contrasts.csv")
      for col,v in [("slope",1.2),("ci_low",-10.8),("ci_high",11.6)]],
    *[(f"court on arrival: {col}",100*w1.loc[(ARR,"court_institution"),col],v,1,"topic1w_within_poet_contrasts.csv")
      for col,v in [("slope",-5.0),("ci_low",-27.2),("ci_high",4.9)]],
    *[(f"return vocabulary at seat: {col}",100*w1.loc[(SEAT,"lex_homeward"),col],v,1,"topic1w_within_poet_contrasts.csv")
      for col,v in [("slope",-1.6),("ci_low",-12.6),("ci_high",12.7)]],
    *[(f"court annual at seat: {col}",100*w1.loc[(YSEAT,"court_institution"),col],v,1,"topic1w_within_poet_contrasts.csv")
      for col,v in [("slope",-4.8),("ci_low",-9.1),("ci_high",2.5)]],
    *[(f"court poems carrying exchanges: {stage}",100*carry.loc[stage,"court_share_exchange_title"],v,0,"topic1w_court_carriers_and_ambivalence.csv")
      for stage,v in [("destination",35),("elsewhere",25),("capital",26)]],
    *[(f"return and dwelling co-occurrence: {col}",100*amb_all[col],v,1,"topic1w_dwelling_homeward_association.csv")
      for col,v in [("slope",2.3),("ci_low",-3.2),("ci_high",8.5)]],
    ("capital correlation p",per.loc["lex_capital","spearman_perm_p"],0.003,3,"topic1v_shang_period_poet_level.csv"),
    *[(f"capital words in {period} Tang",100*per.loc["lex_capital",f"{period}_adjusted"],v,0,"topic1v_shang_period_poet_level.csv")
      for period,v in zip(["early","high","mid","late"],[20,11,7,4])],
    *[(f"poets in {period} Tang",per.loc["lex_capital",f"{period}_poets"],v,0,"topic1v_shang_period_poet_level.csv")
      for period,v in zip(["early","high","mid","late"],[3,8,7,4])],
    ("court axis correlation",per.loc["court_institution","spearman_start_year"],-0.47,2,"topic1v_shang_period_poet_level.csv"),
    ("exchange titles correlation",per.loc["title_exchange","spearman_start_year"],0.43,2,"topic1v_shang_period_poet_level.csv"),
    ("exchange titles correlation p",per.loc["title_exchange","spearman_perm_p"],0.051,3,"topic1v_shang_period_poet_level.csv"),
    *[(f"exchange titles {name}",coh.loc["title_exchange",col]*factor,v,digits,"topic1v_shang_cohort_749.csv")
      for name,col,factor,v,digits in [("before 749","mean_before_749",100,8,0),("from 749","mean_from_749",100,25,0),
      ("poets before 749","poets_before_749",1,5,0),("poets from 749","poets_from_749",1,17,0),
      ("correlation from 749","spearman_within_from_749",1,-0.04,2)]],
    ("Bai Jiangzhou poems",bai.works.sum(),206,0,"topic1y_poet_year_profiles.csv"),
    *[(f"Bai route year {year}",100*bai.loc[year,"route_displacement"],v,1,"topic1y_poet_year_profiles.csv") for year,v in [(0,55.6),(3,31.2)]],
    *[(f"Bai temple year {year}",100*bai.loc[year,"temple_reclusion"],v,1,"topic1y_poet_year_profiles.csv") for year,v in [(0,18.0),(1,30.2),(2,34.2)]],
    # added with the review M2/M4 revision of section 5.3
    *[(f"{label}: {col}",100*w1.loc[(c,o),col],v,1,"topic1w_within_poet_contrasts.csv")
      for label,c,o,vals in [("capital words at seat",SEAT,"lex_capital",(-3.3,-9.5,3.5)),
                             ("court annual, all dated","per year, all dated poems","court_institution",(-3.2,-6.0,0.3)),
                             ("capital words annual at seat",YSEAT,"lex_capital",(2.1,0.8,5.8)),
                             ("grief words at seat",SEAT,"lex_grief",(-9.3,-19.1,0.9)),
                             ("grief words annual at seat",YSEAT,"lex_grief",(-0.6,-2.8,0.7)),
                             ("dwelling words at seat",SEAT,"lex_dwelling",(3.4,-2.4,13.4)),
                             ("self-naming at seat",SEAT,"lex_exile_self",(3.2,-2.3,7.8)),
                             ("titles to central officials at seat",SEAT,"to_central_office",(1.3,-4.2,6.4))]
      for col,v in zip(("slope","ci_low","ci_high"),vals)],
    ("capital words annual at seat: Holm p",w1.loc[(YSEAT,"lex_capital"),"holm_p"],0.143,3,"topic1w_within_poet_contrasts.csv"),
    ("dwelling words higher at seat: poets",direction.loc["lex_dwelling","higher_at_seat"],7,0,"topic1w_poet_direction_seat.csv"),
    ("dwelling words, poets compared",direction.loc["lex_dwelling","poets"],7,0,"topic1w_poet_direction_seat.csv"),
    # added with review M5-M10
    *[(f"leave one poet out, {k}: {lab}",100*loo_key[k][f],v,1,"topic1r_key_results.json")
      for k,vals in [("exchange titles, arrival-proxy year vs en route",(16.2,41.1,34.1)),("exile self-naming, per year at the seat",(-4.1,-2.4,-3.3))]
      for (lab,f),v in zip([("min","min"),("max","max"),("without Bai Juyi","without_bai_juyi")],vals)],
    *[(f"exchange titles raw share: {ph}",100*rates.loc[ph,"title_exchange"],v,0,"topic1w_rates_by_phase.csv")
      for ph,v in [("capital",31),("en route",6),("arrival year",28)]],
    *[(f"exchange titles at seat: {col}",100*w1.loc[(SEAT,"title_exchange"),col],v,1,"topic1w_within_poet_contrasts.csv")
      for col,v in [("slope",18.1),("ci_low",-1.4),("ci_high",38.2)]],
    ("exchange titles higher at seat: poets",direction.loc["title_exchange","higher_at_seat"],5,0,"topic1w_poet_direction_seat.csv"),
    *[(f"human present rate: {ax}",100*blind_axes.loc[ax,"human_present_rate"],v,0,"topic1_blind50_validation.csv")
      for ax,v in [("court_institution",40),("temple_reclusion",26),("landscape",70),("route_displacement",60),("time_void",50)]],
    ("hard cases determinate",hard.values.sum(),277,0,"topic1_targeted309_confusion.csv"),
    ("hard cases agreed",hard.values.trace(),265,0,"topic1_targeted309_confusion.csv"),
    ("medium-confidence positions",k1["t1_medium20"]["positions"],20,0,"key_results.json"),
    ("medium-confidence model disagreements",k1["t1_medium20"]["models_disagree"],13,0,"key_results.json"),
    ("first-pass candidates matching reviewer",gpt_conf["sum"].sum(),18,0,"topic4_axis_gpt_confidence_vs_human.csv"),
    ("first-pass candidates checked",gpt_conf["size"].sum(),31,0,"topic4_axis_gpt_confidence_vs_human.csv"),
    ("poems four or more years into exile",pc["works_four_or_more_years"],198,0,"public_counts.json"),
    ("of which annual-proxy dated",pc["works_four_or_more_years_proxy_dated"],196,0,"public_counts.json"),
]
rows = [{"where":where,"claim":c,"computed":float(v),"printed":p,"matches":round(float(v),digits)==p,
         "source_file":source if source.startswith("data/") else f"results/{source}"}
        for where,items in [("abstract",claims),("full text",text_claims)] for c,v,p,digits,source in items]
out=pd.DataFrame(rows)
out.to_csv(f"{R}/paper_numbers_trace.csv",index=False,encoding="utf-8-sig")
for where,label in [("abstract","abstract"),("full text","full-text")]:
    part=out[out["where"]==where]
    print(f"{part.matches.sum()}/{len(part)} {label} numerical checks passed")
for row in out[~out.matches].itertuples():
    print("MISMATCH",row.where,row.claim,row.computed,row.printed)
sys.exit(0 if out.matches.all() else 1)
