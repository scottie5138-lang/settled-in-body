# Settled in Body, Not in Mind? — minimal reproduction package

Aggregate results, tables, figures and code for the DH2027 long paper *Settled in Body, Not in Mind? Linking Tang
Exile Poems to When and Where They Were Written*.

## What this package reproduces

From the aggregate files in `results/`, a reviewer can

- check selected abstract and full-text estimates and supporting counts with
  `python scripts/05_paper_numbers.py`. It reports the checks passed and writes `results/paper_numbers_trace.csv`;
- rebuild manuscript Table 1 with `python scripts/15_paper_tables.py`, then use its linkage and validation CSVs
  to rebuild Figure 1 with `python scripts/20_process_figure.py`;
- rebuild Figure 2 with `python scripts/14_phase_figure.py` and Figure 3 with `python scripts/12_period_figure.py`.

The audit covers the entries in the trace. The manuscript's blind-sampling frame (1,063 poems) and prior-review
count (111 poems) are reported from a restricted sampling record. The package includes aggregate blind-review
results for the 50 sampled poems.

`MANIFEST_SHA256.csv` lists the SHA-256 of every released file.

## What it does not contain

The poem-level linkage (composition dates, places and exile episodes), the five-axis annotations and the database
snapshot belong to an ongoing governed database project and are not publicly released. The scripts in `analysis/`
show how every measure and model was computed (word lists, title markers, place classification, arrival proxy,
within-poet models, poet bootstrap, Holm correction, leave-one-poet-out checks); they need those restricted inputs
and cannot be run from this package. Poem texts come from the MIT-licensed
[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) corpus. No poem text, model prompt or model
rationale is included.

## Layout

| Path | Content |
|---|---|
| `scripts/` | Runnable from `results/`: number check (05), Figure 3 (12), Figure 2 (14), Table 1 and Figure 1 inputs (15), Figure 1 (20) |
| `analysis/` | Analysis code, for reading: 01, 03, 04, 06 (corpus, validation, time and distance models), 07 (place classification, arrival proxy, word lists, phase models), 08 (within-poet contrasts), 09 and 11 (between-poet comparisons), 16 (arrival-anchor and temple sensitivity), 17 (leave-one-poet-out), 18 (aggregate counts used by 05, 14, 15) |
| `results/` | Aggregate result files (coefficients, intervals, poet and poem counts); `tables/` contains manuscript Table 1 and the linkage and validation inputs for Figure 1. `figures/` contains manuscript Figures 1–3 in order: `fig0_process`, `fig1_phase_axes`, `fig2_period_trend` |

## Notes

- Two exchange-title flags were removed after a manual review of the 24 titles whose only exchange marker is 和:
  in `src_bbecbe9e0a9c95fa4023` and `src_d6b74aacf21fbe1ef853` (Yuan Zhen, 810) 歸和 / 杜歸和 is a personal name.
- 1,202 poems are confirmed as written in exile but 1,201 carry five-axis annotations: `src_e1445c42b66ca63481f2`
  was confirmed after the annotations were frozen (`results/key_results.json`).
- At +1 year, 56 poems (55 by Bai Juyi) meet the within-poet phase criteria. The temple estimate uses
  54 determinate labels, 53 of them from Bai Juyi.
- Analyses ran in Python 3.11.9 with numpy 2.3.2, pandas 2.3.1, scipy 1.16.1 and scikit-learn 1.7.2; the
  rebuild scripts need pandas and matplotlib (`requirements.txt`). Fixed seeds are defined in the analysis scripts.
- Figures use Microsoft JhengHei when available and fall back to DejaVu Sans.

## Licence

Code (`scripts/`, `analysis/`) is released under the MIT Licence (`LICENSE`). Aggregate results, tables and
figures (`results/`) are released under Creative Commons Attribution 4.0 International (CC BY 4.0,
`LICENSE-results.md`). The restricted poem-level linkage, annotations and database snapshot are available from the
author on reasonable request.

## Citation

Lee, Chihung. 2027. "Settled in Body, Not in Mind? Linking Tang Exile Poems to When and Where They Were Written."
Long paper, DH2027, University of Galway.
