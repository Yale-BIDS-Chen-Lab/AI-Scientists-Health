# AI-Scientists-Health

AI Scientists are autonomous systems designed to conduct research end-to-end, including generating hypotheses, designing and executing evaluations, interpreting results, and iteratively refining research directions. As a paradigm that extends beyond large language models and conventional AI agents, they hold transformative potential for health, where heterogeneous multimodal data, stringent evidence requirements, and demanding governance expectations make this both the most consequential and the most underserved frontier for the emerging paradigm.

This repository curates the representative work cited in the perspective paper *From AI Agents to AI Scientists in Health: Emerging Landscape, Challenges, and the Path Forward* (2026). Sections mirror the paper's structure, surveying the emerging landscape (Section 2), enumerating health-specific opportunities (Section 3), and addressing the six interlocking challenges of the **Workflow-driven, Evidence-grounded, Cross-modal, Accountable, Reproducible, and Embodied (WE CARE)** roadmap (Sections 4–5). 

**Note: I haven't validate the paper, code, and resource links.** 

## Table of Contents
- [Representative AI Scientist Systems](#representative-ai-scientist-systems)
- [Health-Specific Opportunities and Translational Applications](#health-specific-opportunities-and-translational-applications)
- [Workflow-Driven Evaluation: Benchmarks for AI Scientists in Health](#workflow-driven-evaluation-benchmarks-for-ai-scientists-in-health)
- [Evidence Grounding, Cross-Modal Robustness, and Failure Modes](#evidence-grounding-cross-modal-robustness-and-failure-modes)
- [Reproducible and Accountable Governance: Reporting Standards and Policy](#reproducible-and-accountable-governance-reporting-standards-and-policy)
- [Contributing](#contributing)
- [Citation](#citation)

## Representative AI Scientist Systems
*Section 2: The Emerging Landscape of AI Scientists.* The thirteen systems characterized in Table 1 of the perspective paper, ordered chronologically.

| Title | Year | Paper | Code | Notes |
| --- | --- | --- | --- | --- |
| Agent Laboratory | 2025 | [arXiv](https://arxiv.org/abs/2501.04227) | [GitHub](https://github.com/SamuelSchmidgall/AgentLaboratory) | Academic role hierarchy (PhD, Postdoc, Prof., ML/SW Eng., Reviewers) for end-to-end ML research |
| Towards an AI Co-Scientist | 2025 | [arXiv](https://arxiv.org/abs/2502.18864) | — | Gemini-2.0 multi-agent system with Elo-tournament hypothesis evolution; validated drug repurposing for AML and liver fibrosis |
| Robin | 2025 | [arXiv](https://arxiv.org/abs/2505.13400) | [GitHub](https://github.com/Future-House/robin) | Three-agent (Crow, Falcon, Finch) lab-in-the-loop system; identified ripasudil as a novel dAMD candidate |
| AI-Researcher | 2025 | [arXiv](https://arxiv.org/abs/2505.18705) | [GitHub](https://github.com/HKUDS/AI-Researcher) | Six-agent pipeline for autonomous scientific innovation; introduces Scientist-Bench |
| AlphaEvolve | 2025 | [arXiv](https://arxiv.org/abs/2506.13131) | — | Coding agent for scientific and algorithmic discovery; evolutionary search over a program database |
| STELLA | 2025 | [arXiv](https://arxiv.org/abs/2507.02004) | [GitHub](https://github.com/zaixizhang/STELLA) | Self-evolving biomedical agent with Template Library and dynamic Tool Ocean |
| DeepScientist | 2025 | [arXiv](https://arxiv.org/abs/2509.26603) | [GitHub](https://github.com/ResearAI/DeepScientist) | Bayesian-Optimization-driven discovery with cumulative Findings Memory; surpasses human SOTA on three AI tasks |
| Virtual Lab | 2025 | [Nature](https://doi.org/10.1038/s41586-025-09442-9) | [GitHub](https://github.com/zou-group/virtual-lab) | LLM PI dynamically instantiates a scientist team; designed and validated 92 SARS-CoV-2 nanobodies |
| ASCollab | 2025 | [arXiv](https://arxiv.org/abs/2510.08619) | — | Heterogeneous research agents in evolving collaboration networks; applied to TCGA cancer cohorts |
| Kosmos | 2025 | [arXiv](https://arxiv.org/abs/2511.02824) | — | 12-hour autonomous campaigns coordinated by a structured world model; ~42K lines of code and ~1,500 papers per run |
| SAGA | 2025 | [arXiv](https://arxiv.org/abs/2512.21782) | — | Bi-level architecture that evolves objective functions themselves, not just hypotheses |
| The AI Scientist | 2026 | [Nature](https://doi.org/10.1038/s41586-026-10265-5) | [GitHub](https://github.com/SakanaAI/AI-Scientist-v2) | End-to-end automation of AI research; first AI-generated paper to pass workshop peer review |
| EvoScientist | 2026 | [arXiv](https://arxiv.org/abs/2603.08127) | — | Multi-agent evolving AI Scientist with dual ideation + experimentation memory stores |

## Health-Specific Opportunities and Translational Applications
*Section 3: Potential of AI Scientists in Health.* Methodologies and applications across the five health-specific directions identified in the perspective: rare and neglected diseases, continuous evidence synthesis, closed-loop validation, clinical operations, and population health surveillance.

| Title | Year | Paper | Notes |
| --- | --- | --- | --- |
| HealthMap | 2008 | [PLoS Medicine](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0050151) | Internet-based emerging-infectious-disease intelligence |
| Living Systematic Reviews | 2014 | [PLoS Medicine](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1001603) | Foundational paper on continuously updated evidence synthesis |
| Using Big Data to Emulate a Target Trial (Hernán & Robins) | 2016 | [Am. J. Epidemiology](https://academic.oup.com/aje/article/183/8/758/1739732) | Foundational target-trial-emulation framework |
| Science of Learning Health Systems | 2016 | [Learning Health Systems](https://onlinelibrary.wiley.com/doi/10.1002/lrh2.10020) | Foundational framing of learning health systems |
| Nextstrain | 2018 | [Bioinformatics](https://academic.oup.com/bioinformatics/article/34/23/4121/5001388) | Real-time pathogen-evolution tracking |
| AI in COVID-19 Drug Repurposing | 2020 | [Lancet Digital Health](https://doi.org/10.1016/S2589-7500(20)30192-8) | Foundational work on AI-driven drug repurposing during COVID |
| AI in COVID-19 with NLP | 2021 | [Annu. Rev. Biomed. Data Sci.](https://www.annualreviews.org/doi/10.1146/annurev-biodatasci-021821-061544) | Reviews NLP-based AI responses to COVID-19 |
| Why 90% of Clinical Drug Development Fails | 2022 | [APSB](https://www.sciencedirect.com/science/article/pii/S2211383522000521) | Quantifies the cost-of-failure problem AI Scientists target |
| Predictive Validity in Drug Discovery | 2022 | [Nat. Rev. Drug Discov.](https://www.nature.com/articles/s41573-022-00552-x) | Predictive-validity framework for drug-discovery pipelines |
| Empowering Biomedical Discovery with AI Agents | 2024 | [Cell](https://www.cell.com/cell/fulltext/S0092-8674(24)01070-5) | Vision paper on AI agents as biomedical research collaborators |
| AI-Driven Predictive Biomarker Discovery with Contrastive Learning | 2025 | [Cancer Cell](https://www.cell.com/cancer-cell/abstract/S1535-6108(25)00125-3) | Contrastive-learning approach to oncology biomarker discovery |
| AI and Network Medicine: Path to Precision Medicine | 2025 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIra2401229) | Reviews AI-network-medicine integration |
| AI in Rare and Intractable Diseases | 2025 | [IRDR](https://www.jstage.jst.go.jp/article/irdr/14/2/14_2025.01010/_article) | Survey of AI for rare diseases |
| AI-Powered Drug Discovery for Neglected Diseases | 2025 | [J. Global Health](https://jogh.org/2025/jogh-15-03002) | Reviews AI applications for neglected-disease drug discovery |
| Living Guideline Recommendations | 2025 | [ZEFQ](https://www.sciencedirect.com/science/article/pii/S1865921725000388) | Systematic evaluation of living-guideline implementations |
| LLMs in Clinical Trials | 2025 | [BMC Medicine](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-025-04317-2) | Reviews LLM applications in clinical trials |
| Limitations of Fine-Tuning LLMs with Updated Medical Knowledge | 2025 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIcs2401155) | Empirical limits of medical-LLM fine-tuning |
| Trial Emulation, Simulation, and Augmentation Using EHR and Generative AI | 2025 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIe2500894) | Connects target-trial emulation to generative AI |

## Workflow-Driven Evaluation: Benchmarks for AI Scientists in Health
*WE CARE pillar: **W**orkflow-driven (Challenge 1 / Section 5.1).* Benchmarks and evaluation frameworks for LLMs and AI Scientists in health, ranging from QA tasks to interactive clinical-environment simulation.

| Name | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| MedQA | 2021 | [Applied Sciences](https://www.mdpi.com/2076-3417/11/14/6421) | [GitHub](https://github.com/jind11/MedQA) | Open-domain medical QA from medical licensing exams |
| LAB-Bench | 2024 | [arXiv](https://arxiv.org/abs/2407.10362) | [GitHub](https://github.com/Future-House/LAB-Bench) | Foundational biology-research capabilities benchmark (DBQA, LitQA) |
| Humanity's Last Exam | 2025 | [arXiv](https://arxiv.org/abs/2501.14249) | [Website](https://lastexam.ai/) | Frontier-difficulty benchmark with a Biomedicine subset |
| Benchmarking LLMs for Biomedical NLP | 2025 | [Nature Communications](https://www.nature.com/articles/s41467-025-56989-2) | [GitHub](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks) | Recommendations for biomedical NLP benchmarking |
| BRIDGE | 2025 | [arXiv](https://arxiv.org/abs/2504.19467) | [GitHub](https://github.com/YLab-Open/BRIDGE) · [HF Dataset](https://huggingface.co/datasets/YLab-Open/BRIDGE-Open) · [HF Leaderboard](https://huggingface.co/spaces/YLab-Open/BRIDGE-Medical-Leaderboard) | Benchmark for understanding real-world clinical practice text |
| Knowledge–Practice Performance Gap in Clinical LLMs | 2025 | [JMIR](https://www.jmir.org/2025/1/e84120/) | — | Systematic review of 39 clinical LLM benchmarks |
| LMOD+ | 2025 | [ACM Trans. Comput. Healthc.](https://dl.acm.org/doi/10.1145/3801746) | [Project Page](https://kfzyqin.github.io/lmod_plus) | Multimodal ophthalmology benchmark for medical MLLMs |
| MedAgentBench | 2025 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIdbp2500144) | [GitHub](https://github.com/stanfordmlgroup/MedAgentBench) · [Project Page](https://stanfordmlgroup.github.io/projects/medagentbench/) | Virtual EHR environment to benchmark medical LLM agents |
| Testing and Evaluation of Healthcare LLM Applications | 2025 | [JAMA](https://jamanetwork.com/journals/jama/fullarticle/2829061) | — | Systematic review of LLM evaluation in healthcare |
| Clinical Environment Simulator | 2026 | [Nature Medicine](https://www.nature.com/articles/s41591-026-04252-6) | — | Dynamic AI evaluation through interactive clinical-encounter simulation |
| MedHELM | 2026 | [Nature Medicine](https://www.nature.com/articles/s41591-026-03476-w) | [GitHub](https://github.com/stanford-crfm/helm) · [Leaderboard](https://crfm.stanford.edu/helm/medhelm/latest/) | Holistic LLM evaluation for medical tasks |

## Evidence Grounding, Cross-Modal Robustness, and Failure Modes
*WE CARE pillars: **E**vidence-grounded and **C**ross-modal (Challenges 2–3 / Sections 5.2–5.3).* Hallucination, memorization, modality gaps, data poisoning, model collapse, causal-reasoning limits, and other failure modes that AI Scientists in health must defend against.

| Title | Year | Paper | Notes |
| --- | --- | --- | --- |
| Biases in EHR Data Due to Healthcare Processes | 2018 | [BMJ](https://www.bmj.com/content/361/bmj.k1479) | Foundational evidence of EHR-process biases |
| Mind the Gap: Modality Gap in Multi-Modal Contrastive Learning | 2022 | [NeurIPS](https://arxiv.org/abs/2203.02053) | Foundational evidence of representational modality gaps |
| Survey of Hallucination in Natural Language Generation | 2023 | [ACM Computing Surveys](https://dl.acm.org/doi/10.1145/3571730) | Foundational hallucination survey |
| The Value of Standards for Health Datasets in AI | 2023 | [Nature Medicine](https://www.nature.com/articles/s41591-023-02608-w) | Argues for dataset standards in clinical AI |
| AI Models Collapse When Trained on Recursively Generated Data | 2024 | [Nature](https://www.nature.com/articles/s41586-024-07566-y) | Foundational evidence on synthetic-data feedback collapse |
| Causal Reasoning and Large Language Models | 2024 | [arXiv](https://arxiv.org/abs/2305.00050) | Frontier review of LLMs for causal reasoning |
| Hallucination of Multimodal Large Language Models: A Survey | 2025 | [arXiv](https://arxiv.org/abs/2404.18930) | Survey of multimodal hallucination |
| An Automated Framework for Assessing How LLMs Cite Medical References | 2025 | [Nature Communications](https://www.nature.com/articles/s41467-025-58943-8) | Evaluates citation faithfulness in medical LLMs |
| Medical LLMs Are Vulnerable to Data-Poisoning Attacks | 2025 | [Nature Medicine](https://www.nature.com/articles/s41591-024-03445-1) | Demonstrates poisoning of medical LLMs |
| Memorization in LLMs in Medicine | 2025 | [arXiv](https://arxiv.org/abs/2509.08604) | Prevalence and implications of medical-LLM memorization |
| Rethinking RAG for Medicine | 2025 | [arXiv](https://arxiv.org/abs/2511.06738) | Large-scale expert evaluation of medical RAG |
| LLM-Assisted Systematic Review of LLMs in Clinical Medicine | 2026 | [Nature Medicine](https://www.nature.com/articles/s41591-026-03460-8) | Meta-review of clinical LLM evidence base |
| Bias and Equity in LLM Applications for Healthcare | 2026 | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6053515) | Scoping review of LLM-induced healthcare inequities |
| AI-Generated Data Contamination Erodes Pathological Variability | 2026 | [arXiv](https://arxiv.org/abs/2601.12946) | Quantifies AI-data-contamination harm in pathology |

## Reproducible and Accountable Governance: Reporting Standards and Policy
*WE CARE pillars: **R**eproducible, **A**ccountable, and the governance facets of **E**mbodied (Challenges 4–6 / Sections 5.4–5.6).* Reporting frameworks and governance instruments referenced in the WE CARE roadmap as complements to AI-Scientist-specific reporting.

| Title | Year | Paper | Notes |
| --- | --- | --- | --- |
| MI-CLAIM Checklist | 2020 | [Nature Medicine](https://www.nature.com/articles/s41591-020-1041-y) | Minimum-information standard for clinical AI modeling |
| MINIMAR | 2020 | [JAMIA](https://academic.oup.com/jamia/article/27/12/2011/5905479) | Minimum information for medical-AI reporting |
| CONSORT-AI Extension | 2020 | [Lancet Digital Health](https://doi.org/10.1016/S2589-7500(20)30218-1) | Reporting guidelines for clinical trials of AI interventions |
| SPIRIT-AI Extension | 2020 | [Lancet Digital Health](https://doi.org/10.1016/S2589-7500(20)30219-3) | Reporting guidelines for clinical-trial protocols with AI interventions |
| The Medical Algorithmic Audit | 2022 | [Lancet Digital Health](https://doi.org/10.1016/S2589-7500(22)00003-6) | Framework for clinical algorithm auditing |
| Reporting Standards for LLM-Linked Chatbots in Health | 2023 | [Nature Medicine](https://www.nature.com/articles/s41591-023-02556-5) | Reporting guidance for LLM-linked health chatbots |
| TRIPOD+AI Statement | 2024 | [BMJ](https://www.bmj.com/content/385/bmj-2023-078378) | Updated TRIPOD guidance for clinical prediction models with ML |
| CHAI Assurance Standards Guide | 2024 | [PDF](https://chai.org/assurance-standards-guide/) | Coalition for Health AI quality and ethics standards |
| Ethical and Regulatory Challenges of LLMs in Medicine | 2024 | [Lancet Digital Health](https://doi.org/10.1016/S2589-7500(24)00061-X) | Surveys ethics/regulation issues for medical LLMs |
| Safety Challenges of AI in Medicine in the Era of LLMs | 2024 | [arXiv](https://arxiv.org/abs/2409.18968) | Surveys safety challenges for medical AI in the LLM era |
| TRIPOD-LLM Reporting Guideline | 2025 | [Nature Medicine](https://www.nature.com/articles/s41591-024-03425-5) | Reporting guideline for studies using LLMs |
| GAMER Statement | 2025 | [BMJ EBM](https://ebm.bmj.com/content/30/6/390) | Reporting guideline for generative AI tools in medical research |
| ICMJE Recommendations | 2026 | [Website](https://www.icmje.org/recommendations/) | Authorship and accountability framework for medical-journal publications |

## Contributing
Contributions are welcome. Please open a PR and:
1. Add entries in the most relevant section.
2. Keep descriptions brief and neutral.
3. Use stable links (conference page, DOI, arXiv, or official repository).
4. Avoid duplicates and maintain chronological ordering within each section.

## Citation
If you use this repository in your work, please cite:
```bibtex
@misc{ai_scientists_health_awesome,
  title  = {AI-Scientists-Health: An Awesome-Style Survey for AI in Health},
  author = {Contributors},
  year   = {2026},
  url    = {https://github.com/Yale-BIDS-Chen-Lab/AI-Scientists-Health}
}
```
