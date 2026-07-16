# Point-by-Point Response to Reviewers — DRAFT (working document)

Manuscript: "Scaling vs. Architecture: An Evaluation of SLMs for Automated Docstring Generation"
Venue: Automated Software Engineering (Springer) — revision invited; editor: Tim Menzies.
Status legend: [DONE] change made · [RUNNING] analysis in progress · [PLANNED] queued · [TBD] awaiting data/decision.

> NOTE (internal, remove before submission): all numeric claims in this letter
> must be regenerated from the re-judged (code-referenced) scores and the
> mixed-effects re-analysis before drafting final text. Scale/N wording is
> deferred pending the corresponding author's decision (N=67 correction vs.
> full 1,400-class rerun).

---

## Editor

> Please ensure the results are accurately reported, any overstated conclusions are rewritten and the limitations of the work fully explained.

Response summary: [TBD — write last, after all reviewer items are resolved. Must cover: corrected model identities, corrected dataset-scale reporting, commensurable faithfulness metric, mixed-effects statistics, rescoped claims.]

---

## Reviewer 1

### R1.1 — Qwen2.5 in abstract vs Qwen3 in body; exact Ollama tags
[DONE-pending-text] The stray "Qwen2.5" was a remnant of an earlier pilot wave (visible in our execution notebook history); the final experiments used Qwen3. We now state the exact served checkpoints in Section 4.2: `llama3.2:latest`, `phi4:14b`, `qwen3:8b`, judge/helper `deepseek-coder:6.7b`, all via Ollama. See also R2.1 for the Llama identity correction.

### R1.2 — Fragility of the "EXCELLENT" substring gate
[DONE code / RUNNING rerun] The reviewer's concern is valid, and our re-verification found the manuscript's description of the gate was itself inaccurate: the runtime check was `"GOOD" not in response.upper()`, not an `EXCELLENT` gate. We have (a) corrected the mechanism description in Section 3.3/Appendix H/Fig. 17; (b) replaced the substring check with robust verdict-label parsing (Assessment-line extraction, label-list echo removal, negation handling; unit-tested); (c) re-run the Iterative Critique family with the corrected gate and report trigger-rate and outcome deltas. [TBD: rerun numbers]
Note: case sensitivity was already handled (`.upper()`); the true fragilities were label-list echoes and incidental "good"/negated phrasing, which the new parser addresses.

### R1.3 — A100 is not on-device hardware
[PLANNED text] We reframe all "on-device" claims as *local server-side inference* and correct the hardware description (the compute node was a cloud A100 runtime). [OPTIONAL: consumer-hardware latency table from Apple-silicon run — TBD decision.]

### R1.4 — DeepSeek-Coder as both critic and judge
[LARGELY DONE — reviewer's concern vindicated and extended] Our re-verification went further than the reviewer asked and found that the DeepSeek-Coder 6.7B judge **fails blind human validation outright**: even with reliable k=3 multi-draw scores, judge↔blind-human correlation is r=−0.213 (n=51), with systematic +0.2–0.31 over-rating of retrieval/refinement outputs. Meanwhile, reference-based metrics validate well against blind humans (pydocstyle r=0.558; BERTScore roberta-rescaled r=0.521; ROUGE-1 r=0.401). The revision therefore (a) demotes the LLM-judge to a supplementary metric with explicit validity caveats, (b) elevates the human-validated metric suite, and (c) reports a second judge from a different family (Qwen3 8B, k=3, all 2,613 outputs). The second judge also fails blind-human validation (r=0.051; excluding its own generations, r=−0.092; no self-preference detected), yet the two judges agree almost perfectly with each other on strategy-level rankings (Spearman ρ=0.979 across all 39 model×strategy cells) despite absolute scales differing by 0.3–0.4. That is, small LLM judges are mutually consistent but consistently misaligned with human faithfulness judgment — they reward retrieval-grounded fluency that blind humans do not. We report this as a finding in its own right. The dual critic/judge role the reviewer flagged is thereby resolved structurally: no LLM judge is used for any primary claim.

### R1.5 — "Fig." vs "Figure" consistency
[PLANNED text] All in-text references changed to Springer style ("Fig." mid-sentence); full cross-reference audit done.

### R1.6 — Suggested related work
[PLANNED text] We add and discuss: calibration of LLMs on code summarization (PACMSE 2025) in the judge-reliability discussion; chain-of-comments rethinking-based summarization (COLING 2025) in reasoning-mode related work. [DECISION: whether to include the quantization (NeurIPS 2026) citation in the deployment/quantization discussion and the PSO prompt-design citation in the prompt-design discussion.]

---

## Reviewer 2 (detailed treatment per editor instruction)

### R2.1 — Model identity: "Llama 3.2 8B" does not exist; abstract says Qwen2.5
[DONE-pending-text] The reviewer is correct, and we thank them for catching this. Verification against execution logs shows the served model was `llama3.2:latest` = **Llama 3.2 3B**. All references corrected; the study's true parameter range is **3B–14B**. The capacity finding is re-stated for the models actually used and [TBD: re-verified under the mixed-effects analysis — expected to strengthen: 14B vs 3B]. Exact Ollama tags now listed in Section 4.2 (see R1.1). Appendix E memory figures corrected for the 3B model.

### R2.2 — Faithfulness judged against different references per family
[DONE] Fully accepted. We re-scored every generation (2,613 outputs) with the judge evaluating **source code as the single reference** for all 13 strategies, using the identical rubric, averaged over k=3 independent draws per sample (single draws proved unreliable; see R1.4). All faithfulness tables/figures/tests now use the commensurable scores; the context-referenced score is a supplementary grounding metric only. Under the commensurable metric and mixed-effects analysis (R2.3): no architecture-family effect is significant (RAG −0.011 ns, Iterative Critique +0.006 ns vs Plain), GoT is the only significant reasoning effect (+0.032, p=0.003), and model effects dominate (Phi-4 +0.047, Qwen3 +0.064 vs Llama 3B, both p<0.001). We additionally validated the judge itself against a new blind human study, which led to a deeper correction reported under R2.6/R1.4.

### R2.3 — Wrong unit of analysis (aggregate means, n=3/9/13)
[DONE] Fully accepted. All inferential claims are re-derived from linear mixed-effects models on the per-class observations (class as random intercept; family, reasoning mode, and model as fixed effects; N=2,613). The class-level ICC (0.276 for BERTScore) confirms the reviewer's dependence concern. Final estimates:
- BERTScore: Few-Shot −0.080 (p<0.001), Iterative Critique −0.034 (p<0.001), RAG −0.006 (p=0.052), GoT −0.009 (p=0.012), Phi-4 +0.024 (p<0.001), Qwen3 +0.015 (p<0.001).
- Commensurable k=3 faithfulness: GoT +0.032 (p=0.003, only significant reasoning effect), Phi-4 +0.047 and Qwen3 +0.064 (both p<0.001), all family effects ns.
Tables 10–12 are replaced by the mixed-effects coefficient tables. The capacity conclusion strengthens; several previously "significant" aggregate-level contrasts do not survive, and the text is revised accordingly.

### R2.4 — Headlining a metric the paper discredits (GoT/BERTScore)
[PLANNED text] Accepted. The abstract and Discussion are restructured around the code-referenced faithfulness results; BERTScore findings are reported as secondary lexical-similarity observations with the human-correlation caveat stated at first use. BERTScore itself recomputed with roberta-large + baseline rescaling (R2 minor; [TBD numbers]).

### R2.5 — Quantization confound (Q4 vs reasoning ability)
[PLANNED experiment] We add an FP16 vs Q4_K_M ablation ([TBD: model/strategies/subset]) and discuss the confound in Threats regardless of outcome. [TBD: requires GPU session]

### R2.6 — Human validation contradicts main results
[DONE — major correction, full transparency] The reviewer's instinct was correct, and our re-verification uncovered a deeper problem than the ordering inconsistency. Two findings:
(1) The LLM judge's per-sample scores have near-zero test-retest reliability at sampling temperature (identical inputs score 0.5–1.0 across draws; old-vs-new No-RAG correlation ≈ 0), although strategy-level means over 67 classes are stable (±0.03–0.09).
(2) A judge with near-zero per-sample reliability cannot correlate r=0.925 with independent human ratings. The original annotation sheet displayed the judge scores beside the entry column; a **new strictly blind re-annotation** (same 51 samples, metric columns removed, order shuffled, two annotators) shows inter-rater r=0.624 / ICC(2,1)=0.626, but blind-human ↔ old-judge correlation of only r=0.127, and blind ↔ original human ratings r=0.157. We conclude the original human ratings were anchored to the visible judge scores, and we withdraw the r=0.925 claim and the "human evaluation confirms our automated rankings" statement entirely. Section 3.6/Table 6/Fig. 9 are replaced by the blind two-annotator study.
The blind study also inverts the strategy claim the reviewer questioned: blind humans rank GoT highest (0.711) and SimpleRAG fourth (0.567); the judge over-rates retrieval/refinement outputs by +0.2–0.3 relative to blind humans. All human-validation claims in the revision derive from the blind study only.

### R2.7 — ToT/GoT not canonical; rename or scope
[PLANNED text] Accepted. We rename the modes to reflect what is implemented (single-pass structured decomposition/aggregation prompting; ToT→"structured decomposition with candidate selection", GoT→"multi-axis decomposition-aggregation"), retain the lineage citations as inspiration only, and scope all negative conclusions to these prompt-level variants.

### R2.8 — Few-Shot baseline too weak
[DONE code / PLANNED runs] Accepted, with an additional disclosure from our re-verification: the Few-Shot prompt contained a copy-paste persona error ("You are a strict technical judge…" in a generation prompt, conflicting with the system message). The revision reports a three-arm ablation: (1) original condition (disclosed as-is), (2) persona-corrected static exemplars, (3) dynamically retrieved structurally matched exemplars (leave-one-out nearest neighbors with human reference docstrings). Mechanistic finding: the original condition's failure is partly degenerate truncation — 24/67 (36%) of Qwen3 Few-Shot outputs were <10 words. [TBD: ablation results; rescope or retract "Few-Shot is harmful" as data dictates]

### R2.9 — "Scaling" framing overstated
[PLANNED text] Accepted; title and framing revised to capacity/training-quality vs. architecture. [OPTIONAL strengthener: within-family scale contrast — phi4-mini 3.8B vs phi4 14B runs exist in git history; qwen3.5:9b full run exists — TBD decision to include.]

### R2 minors
- BERTScore backbone/rescaling: [RUNNING] recomputed with roberta-large, rescale_with_baseline=True; configuration stated.
- Coverage metrics without significance tests: [PLANNED] covered by mixed-effects framework or reported descriptively with explicit caveat.
- ρ=−0.286 between the two faithfulness metrics: [PLANNED text] engaged directly; with the code-referenced metric now primary, the divergence analysis is reframed. [TBD]
- Forward-dated, non-archival motivational source (Sartori 2026): [PLANNED text] replaced with archival sources.
- Table 4 "N" column ambiguity: [PLANNED text] clarified (N = strategies aggregated per architecture group).
- No run-to-run variance: [PLANNED experiment] 3× repeated runs for a representative strategy subset; variance reported for quality and cost metrics.
- ToT API-count bug and cost stability: [PLANNED] covered by repeated-run variance analysis above.

---

## Reviewer 3

### R3.1 — Retrieval corpus is generic, not project-specific
[PLANNED experiment + text] We scope the RAG conclusions to *generic documentation corpora* and add a corpus ablation with a project-specific index ([TBD: corpus design — library API documentation excluding benchmark classes]). [TBD: results]

### R3.2 — "Reasoning Tax" lacks mechanism analysis
[RUNNING] The revision adds a mechanism subsection: (a) format-compliance failures — all 5 runaway generations (2K–291K words, repetition loops) occurred in Llama 3.2 3B reasoning modes, evidence that small models unreliably execute structured output formats; (b) degenerate truncation under Few-Shot (see R2.8); (c) hallucination-by-inference quantified via re-judged scores per reasoning mode [TBD]; (d) complexity-stratified results [TBD].

### R3.3 — Novelty vs existing benchmarks
[PLANNED text] Contributions sharpened: cross-model factorial design at class level (vs function level), cost-aware Pareto analysis, human-validated judge protocol, and (new in revision) commensurable code-referenced faithfulness + mixed-effects inference. A comparison table vs prior benchmarks added to Related Work.

### R3.4 — Scale vs training-data confound
[PLANNED text] Same resolution as R2.9: claims rescoped to capacity/training quality; heterogeneity of vendors acknowledged as a design limitation. [OPTIONAL: within-family contrast, see R2.9.]

### R3.5 — Spelling and grammar
[PLANNED text] Full proofreading pass completed.
