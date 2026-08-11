# AI Scientists in Health

This is the official repository for the 2026 Perspective *From AI Agents to AI Scientists in Health: Emerging Landscape, Challenges, and the Path Forward*. It provides an evidence-oriented map of systems participating in scientific inquiry, comparing their objectives, human–AI division of work across four recurring research functions, and reported evaluation.

## AI Scientist research patterns

**System and objective** identifies the scientific problem a system was designed to address and the primary output it is intended to produce.

**Functional workflow** maps the reported contributions of AI and human investigators across four recurring research functions:

1. **Grounding and formulation** links existing evidence, data, or researcher input to questions, hypotheses, or candidate directions.
2. **Design and execution** specifies and performs computational analyses, simulations, or physical experiments.
3. **Analysis and interpretation** assesses results, statistical evidence, biological meaning, and uncertainty.
4. **Iteration and adaptation** uses findings or feedback to revise subsequent analyses, experiments, hypotheses, or search directions.

These functions may connect sequentially, recur iteratively, or operate in parallel. Responsibility may also shift between AI and human investigators during the same study.

**Reported evaluation** summarizes how the original study assessed its outputs or claims, such as through computational benchmarks, expert review, retrospective or prospective data, independent replication, or wet-lab experiments. It records the evidence reported by each paper rather than assigning a common quality score.

## Representative systems

The evidence map begins with the peer-reviewed systems compared in the current manuscript and extends them with newly audited health-domain systems. Emerging preprints are listed separately from peer-reviewed work. Systems are assigned by their primary demonstrated application, although some cross domain boundaries. **AI** and **Human** labels describe reported contributions, not relative importance.

### General scientific and AI/ML research

These systems provide broader context for agentic scientific workflows but were not demonstrated primarily on health research.

| System and objective | Grounding and formulation | Design and execution | Analysis and interpretation | Iteration and adaptation | Reported evaluation |
| --- | --- | --- | --- | --- | --- |
| **The AI Scientist** (2026, *Nature*)<br>End-to-end machine-learning studies<br>[Paper](https://doi.org/10.1038/s41586-026-10265-5) · [Code](https://github.com/SakanaAI/AI-Scientist-v2) | **Human:** specifies the research subfield.<br>**AI:** generates ideas and searches the literature. | **Human:** provides starting code in template-based mode.<br>**AI:** modifies or generates code and executes experiments. | **AI:** produces results and figures and writes the manuscript. | **AI:** uses outcomes to guide later experiments or search nodes. | Automated-review benchmark and blind workshop review of three selected manuscripts; one passed the first review round. |
| **ERA** (2026, *Nature*)<br>Empirical scientific-software optimization<br>[Paper](https://doi.org/10.1038/s41586-026-10658-6) · [Code](https://github.com/google-research/era) | **Human:** defines the task and metric.<br>**AI:** derives and recombines ideas from the literature. | **AI:** generates and runs code in a sandbox. | **AI:** scores outputs against the metric.<br>**Human:** checks selected implementations for fidelity. | **AI:** mutates, branches, and backtracks based on scores. | Six scientific benchmarks with held-out or public comparisons. |
| **Agentomics** (2026, *Bioinformatics*)<br>Biomedical machine-learning model development<br>[Paper](https://doi.org/10.1093/bioinformatics/btag250) · [Code](https://github.com/BioGeMT/Agentomics-ML) | **Human:** supplies a dataset, metric, and resource limits.<br>**AI:** formulates modeling strategies. | **AI:** builds and runs models through validated development steps. | **AI:** selects by validation score and evaluates the selected model on an isolated test set. | **AI:** revises and recombines strategies using prior results. | Triplicate runs on 20 benchmarks; best-of-three produced a new reported state of the art on 11/20 datasets (19/60 runs). |

### Preclinical biomedical research

These systems support biomedical discovery before or alongside clinical testing, including hypothesis development, molecular design, experimental analysis, and evidence synthesis.

| System and objective | Grounding and formulation | Design and execution | Analysis and interpretation | Iteration and adaptation | Reported evaluation |
| --- | --- | --- | --- | --- | --- |
| **Virtual Lab** (2025, *Nature*)<br>SARS-CoV-2 nanobody design<br>[Paper](https://doi.org/10.1038/s41586-025-09442-9) · [Code](https://github.com/zou-group/virtual_lab) | **Human:** defines the project and writes meeting agendas.<br>**AI:** selects the design approach and tools. | **AI:** implements the ESM–AlphaFold–Rosetta pipeline.<br>**Human:** runs code and ELISA experiments. | **AI:** combines model scores to rank candidates. | **AI:** uses top-ranked mutants to seed four computational design rounds. | ELISA testing of 92 mutants, with two candidates highlighted. |
| **AI Co-Scientist** (2026, *Nature*)<br>Hypothesis generation and prioritization<br>[Paper](https://doi.org/10.1038/s41586-026-10644-y) · [Project](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/) | **Human:** specifies the goal and constraints.<br>**AI:** generates literature-grounded hypotheses. | **AI:** proposes experimental plans.<br>**Human:** selects and performs wet-lab tests. | **Human:** interprets experimental results. | **AI:** refines hypotheses through debate and ranking feedback. | Expert ratings and expert-selected wet-lab follow-up. |
| **Robin** (2026, *Nature*)<br>Dry age-related macular degeneration drug discovery<br>[Paper](https://doi.org/10.1038/s41586-026-10652-y) · [Code](https://github.com/Future-House/robin) | **Human:** defines the disease objective.<br>**AI:** synthesizes literature and prioritizes candidates. | **AI:** proposes assays and candidates.<br>**Human:** modifies assays and runs experiments. | **AI:** analyzes assay and RNA-seq data.<br>**Human:** performs reference analyses. | **AI:** uses experimental results to guide follow-up assays and a second candidate round. | Primary-cell replication and a same-assay Deep Research comparator. |
| **CellVoyager** (2026, *Nature Methods*)<br>Single-cell RNA-sequencing exploration<br>[Paper](https://doi.org/10.1038/s41592-026-03029-6) · [Code](https://github.com/zou-group/CellVoyager) | **Human:** supplies research context and data.<br>**AI:** generates literature-informed hypotheses and plans. | **AI:** writes and executes notebook analyses. | **AI:** interprets results.<br>**Human:** reviews case-study analyses. | **AI:** replans from results and post-analysis expert feedback. | CellBench (76 studies), three expert-rated cases, and selected replication in independent datasets. |
| **Biomni** (2026, *Science*)<br>General biomedical task execution<br>[Paper](https://doi.org/10.1126/science.adz4351) · [Code](https://github.com/snap-stanford/biomni) | **Human:** submits a query.<br>**AI:** retrieves tools and databases and generates a plan. | **AI:** executes code and tools and generates laboratory-automation protocols.<br>**Human:** performs the cloning experiment. | **AI:** synthesizes tool outputs into an answer or report. | **AI:** replans from observations across three protein-design cycles. | A 443-query benchmark and five biomedical case studies. |
| **DeepEvidence** (2026, *Nature Machine Intelligence*)<br>Biomedical evidence exploration and synthesis<br>[Paper](https://doi.org/10.1038/s42256-026-01266-0) · [Code](https://github.com/RyanWangZf/BioDSA/tree/main/biodsa/agents/deepevidence) | **Human:** supplies a research question.<br>**AI:** searches 17 biomedical sources and builds an evidence graph. | **AI:** plans breadth- and depth-oriented searches and executes database queries and code. | **AI:** synthesizes evidence with provenance.<br>**Human:** rates selected open-ended outputs. | **AI:** expands and refines searches and the evidence graph. | Four open benchmarks, curated biomedical tasks, and expert-rated open-ended challenges. |
| **DiscoVerse** (2026, *Frontiers in Artificial Intelligence*)<br>Pharmaceutical archive search and reverse translation<br>[Paper](https://doi.org/10.3389/frai.2026.1808378) | **Human:** defines archival questions and output schemas.<br>**AI:** decomposes queries and retrieves evidence. | **AI:** searches and extracts from preclinical, clinical, and strategic records. | **AI:** produces source-linked syntheses.<br>**Human:** adjudicates context and meaning. | **AI:** refines retrieval when review agents identify evidence gaps. | Seven expert-labelled questions over records for 180 molecules (recall ≥0.986; precision 0.71–0.91) and three retrospective use cases. |
| **GenExp** (2026, *Journal of the Royal Society Interface*)<br>Systems-biology hypothesis testing<br>[Paper](https://doi.org/10.1098/rsif.2026.0043) · [Code](https://github.com/DanielBrunnsaker/GenExp) | **Human:** defines the scope.<br>**AI:** generates and prioritizes structured hypotheses. | **AI:** designs protocols and laboratory automation executes them.<br>**Human:** reviews safety and performs remaining physical steps. | **AI:** analyzes growth and metabolomics data.<br>**Human:** interprets the scientific findings. | **AI:** uses one experiment's metabolomics results to generate and test a follow-up hypothesis. | Controlled yeast experiments with negative controls and up to ten biological replicates per condition; 976 metabolomics injections, including 336 study-quality-control injections. |
| **MacAma** (2026, *Smart Medicine*)<br>Systematic review and meta-analysis<br>[Paper](https://doi.org/10.1002/smmd.70045) · [Code](https://github.com/YilinYuan/MacAma) | **Human:** defines the question, search, and PICOS criteria.<br>**AI:** retrieves and screens studies. | **AI:** extracts structured text and runs analysis on verified inputs.<br>**Human:** verifies chart data and selects statistical parameters. | **AI:** summarizes analyses and drafts text.<br>**Human:** interprets heterogeneity and revises the manuscript. | **Human:** inspects, revises, or overrides agent decisions. | Human-reference screening benchmarks (354 and 599 records; initial-screen recall 56.6%) and one 5,598-record case study. |

### Clinical research

These systems investigate clinical questions using patient-derived data, pathology, electronic health records, or trial-design evidence.

| System and objective | Grounding and formulation | Design and execution | Analysis and interpretation | Iteration and adaptation | Reported evaluation |
| --- | --- | --- | --- | --- | --- |
| **SPARK** (2026, *Nature Medicine*)<br>Cancer-pathology biomarker discovery<br>[Paper](https://doi.org/10.1038/s41591-026-04357-y) · [Code](https://github.com/cpath-ukk/SPARK) | **Human:** specifies the task and available data.<br>**AI:** generates and refines biological concepts. | **AI:** converts concepts to code and runs cohort analyses. | **AI:** verifies parameters and performs statistical analyses.<br>**Human:** reviews selected findings. | **AI:** repairs code but does not redirect research using empirical results. | Eighteen retrospective cohorts (>5,400 patients; five cancers), with false-discovery-rate control and independent test cohorts. |
| **EmulatRx** (2026, *Nature Communications*)<br>Real-world-evidence-informed clinical trial design<br>[Paper](https://doi.org/10.1038/s41467-026-74501-2) · [Code](https://github.com/TrialLab/EmulatRx) | **Human:** defines the trial question and provides expert feedback.<br>**AI:** retrieves trials and literature and drafts a protocol. | **AI:** maps criteria to EHR data, constructs cohorts, and runs causal analyses. | **AI:** generates trial-design reports.<br>**Human and AI:** review clinical and methodological plausibility. | **AI:** revises criteria, covariates, or analyses when missingness, sparsity, or imbalance is detected. | Twenty trial protocols with manual reference labels, SQL review, synthetic causal ground truth, and retrospective EHR and published-trial comparisons. |

### Population health

These systems address population-level evidence synthesis, disease modeling, or forecasting. The strongest current evidence comes from prospective forecasting and expert-grounded evidence-synthesis evaluation; the second and third entries below are preprints and should be interpreted accordingly.

| System and objective | Grounding and formulation | Design and execution | Analysis and interpretation | Iteration and adaptation | Reported evaluation |
| --- | --- | --- | --- | --- | --- |
| **Martinson et al.** (2026, preprint)<br>Prospective multi-pathogen forecasting<br>[Paper](https://arxiv.org/abs/2605.16238) · [Code](https://github.com/google-research/google-research/tree/master/epi_forecasts) | **Human:** defines the forecasting task and seeds method descriptions.<br>**AI:** formulates and combines forecasting approaches. | **AI:** generates and evaluates more than 207,500 candidate models. | **AI:** scores validation and retrospective test performance.<br>**Human:** selects a diverse internal pool and 19 ensemble components. | **AI:** uses scores to branch, revise, and recombine model code. | Time-stamped prospective submissions to CDC forecasting hubs for influenza, COVID-19, and RSV. |
| **AgentSLR** (2026, preprint)<br>Epidemiological systematic-review workflow<br>[Paper](https://arxiv.org/abs/2603.22327) | **Human:** defines review questions, criteria, and reference annotations.<br>**AI:** retrieves candidate studies. | **AI:** screens articles, converts PDFs, and extracts structured evidence. | **AI:** creates correctable structured records.<br>**Human:** reviews outputs; final report synthesis was not evaluated. | **Not demonstrated:** the workflow uses fixed stages; human abstract screening improves evidence retention. | Stage-isolated evaluation on 16,248 expert-annotated records; no model exceeded average field-level extraction F1 of 0.67. |
| **EPIAGENT** (2026, preprint)<br>Epidemiological simulator synthesis<br>[Paper](https://arxiv.org/abs/2602.00299) | **Human:** supplies scenarios, data, constraints, and an execution scaffold.<br>**AI:** retrieves epidemiological knowledge and builds flow graphs. | **AI:** generates, calibrates, and runs mechanistic simulators. | **AI:** checks structural, numerical, and scenario consistency. | **AI:** revises graphs and code from verification and performance feedback. | Structural ablations, retrospective COVID-19 calibration, and scenario-consistency tests; no prospective forecast evaluation. |

## Interpreting the evidence

Current systems demonstrate that AI can contribute to multiple connected parts of scientific inquiry. Several systems link computational reasoning with laboratory experiments, retrospective cohorts, independent datasets, or externally scored benchmarks.

The evidence also limits stronger conclusions:

- Current studies support **hybrid human–AI research workflows** more directly than broadly autonomous scientific discovery.
- Validation of selected candidates or analyses does not establish the reliability of an entire ranking process or research workflow.
- Retrospective, benchmark, wet-lab, expert, and peer-review evaluations support different claims and should not be treated as interchangeable.
- Functional breadth alone does not establish scientific rigor, reproducibility, or significance.

## Contributing and expansion criteria

Contributions to expand the evidence map are welcome. A system need not perform all four research functions, but each proposed entry should:

1. cite a primary paper describing participation in at least one substantive research function, rather than only a standalone model, benchmark, review, or care-delivery task;
2. state the scientific objective and output while distinguishing AI contributions from researcher input, selection, execution, interpretation, and oversight;
3. summarize the reported evaluation precisely enough to connect the evidence to the system's claims, including negative results and human dependencies when reported;
4. identify whether the primary evidence is peer-reviewed or a preprint and provide stable paper and official code or project links when available; and
5. use neutral wording without treating functional breadth as a measure of autonomy, maturity, or scientific quality.

Please submit additions or corrections through a pull request following this schema.

## Citation

Chen, Qingyu, Xin Wang, Rong Zhou, Hyunjae Kim, Shuai Wang, Wenjun Zhao, Tianyu Liu, Irbaz Riaz, Lifang He, Yize Zhao, Carlos Oliver, Gunjan Tiyyagura, Mark Iscoe, Fares Alahdab, Hongyu Zhao, Hua Xu, and Zhiyong Lu. “From AI Agents to AI Scientists in Health: Emerging Landscape, Challenges, and the Path Forward.” June 6, 2026. Available at [SSRN](https://ssrn.com/abstract=6889798) or [https://doi.org/10.2139/ssrn.6889798](https://doi.org/10.2139/ssrn.6889798).
