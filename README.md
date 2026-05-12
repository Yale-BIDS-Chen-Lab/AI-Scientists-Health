# AI-Scientists-Health

AI Scientists are autonomous systems designed to conduct research end-to-end, including generating hypotheses, designing and executing evaluations, interpreting results, and iteratively refining research directions. As a paradigm that extends beyond large language models and conventional AI agents, they hold transformative potential for health, where heterogeneous multimodal data, stringent evidence requirements, and demanding governance expectations make this both the most consequential and the most underserved frontier for the emerging paradigm.

This repository curates representative work cited in or relevant to **Section 2 (The Emerging Landscape of AI Scientists)** of the perspective paper *From AI Agents to AI Scientists in Health: Emerging Landscape, Challenges, and the Path Forward* (2026). Entries are organized along two axes: the three paradigms (LLMs, AI Agents, AI Scientists) and three health subdomains (Biomedical, Clinical, Public Health).

**Categorization rules.** Each entry is placed in exactly one cell based on strict criteria:
- **LLM**: a language model itself (text, sequence, or multimodal foundation model with language as a primary modality). Evaluation studies of LLMs, perspectives, and benchmarks are excluded.
- **AI Agent**: a system that does multi-step reasoning with tool use, planning, and memory, but operates on bounded tasks. Benchmarks that evaluate agents are excluded; only the agent systems themselves are listed.
- **AI Scientist**: a system that satisfies all four properties defined in Section 2.1 of the paper, namely (1) multi-stage with varying autonomy, (2) sustained and persistent, (3) self-evolving, and (4) knowledge-accessing and tool-using. Reviews, perspectives, simulators, and partial pipelines that miss any of these properties are excluded or reclassified.
- **Subdomain**: an entry must have a demonstrated application or evaluation in the corresponding subdomain. Paradigm-defining general-purpose AI Scientist systems from ML or algorithmic domains (e.g., The AI Scientist, AlphaEvolve, DeepScientist, EvoScientist, AI-Researcher, Agent Laboratory) are foundational to Section 2 but are not listed here because they do not meet the subdomain requirement; readers should consult Table 1 of the perspective paper for the broader landscape.

**Note:** paper, code, and resource links have not been individually validated.

## Table of Contents
- [Biomedical](#biomedical) (preclinical discovery: drug discovery, omics, protein and molecule modeling, biomedical literature)
  - [Biomedical LLMs](#biomedical-llms)
  - [Biomedical AI Agents](#biomedical-ai-agents)
  - [Biomedical AI Scientists](#biomedical-ai-scientists)
- [Clinical](#clinical) (clinical research and healthcare delivery: diagnosis, treatment, EHR, trials, clinical imaging)
  - [Clinical LLMs](#clinical-llms)
  - [Clinical AI Agents](#clinical-ai-agents)
  - [Clinical AI Scientists](#clinical-ai-scientists)
- [Public Health](#public-health) (population-level health: epidemiology, surveillance, outbreak intelligence, digital epidemiology)
  - [Public Health LLMs](#public-health-llms)
  - [Public Health AI Agents](#public-health-ai-agents)
  - [Public Health AI Scientists](#public-health-ai-scientists)
- [Contributing](#contributing)
- [Citation](#citation)

Within each cell, entries are sorted chronologically (oldest first).

---

## Biomedical

### Biomedical LLMs
Domain-pretrained or fine-tuned language models, including text, protein, nucleic acid, single-cell, and multimodal foundation models with language as a primary modality.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| SciBERT | 2019 | [EMNLP](https://aclanthology.org/D19-1371/) | [GitHub](https://github.com/allenai/scibert) | BERT pretrained on 1.14M scientific papers; canonical scientific-text encoder |
| BioBERT | 2020 | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btz682) | [GitHub](https://github.com/dmis-lab/biobert) | First widely adopted BERT pretrained on PubMed; baseline for biomedical NER, RE, QA |
| ProtTrans (ProtBERT, ProtT5) | 2021 | [IEEE TPAMI](https://doi.org/10.1109/TPAMI.2021.3095381) | [GitHub](https://github.com/agemagician/ProtTrans) | Family of protein language models trained on UniRef and BFD |
| PubMedBERT | 2021 | [ACM Trans. Comput. Healthc.](https://doi.org/10.1145/3458754) | [HF](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) | From-scratch PubMed pretraining; introduced the BLURB benchmark |
| BioGPT | 2022 | [Briefings in Bioinformatics](https://doi.org/10.1093/bib/bbac409) | [GitHub](https://github.com/microsoft/BioGPT) | First widely used generative biomedical LLM; SOTA on PubMedQA at release |
| Galactica | 2022 | [arXiv](https://arxiv.org/abs/2211.09085) | [Project page](https://galactica.org/) | 120B-parameter LLM trained on curated scientific corpus with SMILES, AA, and LaTeX tokens |
| ESM-2 / ESMFold | 2023 | [Science](https://doi.org/10.1126/science.ade2574) | [GitHub](https://github.com/facebookresearch/esm) | 15B-param protein LM; enabled the ESM Metagenomic Atlas |
| ProGen | 2023 | [Nature Biotechnology](https://doi.org/10.1038/s41587-022-01618-2) | — | Generated wet-lab-validated artificial lysozymes; first controllable protein generation by an LM |
| Geneformer | 2023 | [Nature](https://doi.org/10.1038/s41586-023-06139-9) | [HF](https://huggingface.co/ctheodoris/Geneformer) | Pretrained on 30M single-cell transcriptomes; identified validated cardiomyopathy targets |
| BioMedGPT | 2023 | [arXiv](https://arxiv.org/abs/2308.09442) | [GitHub](https://github.com/PharMolix/OpenBioMed) | Open 10B-param multimodal biomedical LLM unifying molecule, protein, and natural language |
| DNABERT-2 | 2023 | [arXiv](https://arxiv.org/abs/2306.15006) | [GitHub](https://github.com/MAGICS-LAB/DNABERT_2) | BPE-tokenized genomic LM matching SOTA with 21x fewer parameters; introduced the GUE benchmark |
| InstructProtein | 2023 | [arXiv](https://arxiv.org/abs/2310.03269) | — | Bidirectional protein-language instruction tuning over UniProt knowledge graph |
| BioMedLM | 2024 | [arXiv](https://arxiv.org/abs/2403.18421) | [HF](https://huggingface.co/stanford-crfm/BioMedLM) | Compact 2.7B GPT trained from scratch on PubMed; runs on a single A100 |
| scGPT | 2024 | [Nature Methods](https://doi.org/10.1038/s41592-024-02201-0) | [GitHub](https://github.com/bowang-lab/scGPT) | Generative transformer pretrained on 33M cells; SOTA on cell-type annotation and perturbation |
| scFoundation | 2024 | [Nature Methods](https://doi.org/10.1038/s41592-024-02305-7) | [GitHub](https://github.com/biomap-research/scFoundation) | 100M-param model on 50M cells; SOTA on drug response and perturbation tasks |
| BioMistral | 2024 | [ACL Findings](https://aclanthology.org/2024.findings-acl.348/) | [GitHub](https://github.com/BioMistral/BioMistral) | Mistral-7B continually pretrained on PubMed Central; multilingual biomedical evaluation |
| ChatNT | 2024 | [bioRxiv](https://doi.org/10.1101/2024.04.30.591835) | [GitHub](https://github.com/instadeepai/ChatNT) | Conversational nucleic-acid foundation model handling 27 genomics tasks via natural-language prompts |
| ESM3 | 2024 | [bioRxiv](https://doi.org/10.1101/2024.07.01.600583) | [GitHub](https://github.com/evolutionaryscale/esm) | 98B-param multimodal protein LM over sequence, structure, and function tokens |
| DrugGen | 2024 | [arXiv](https://arxiv.org/abs/2411.14157) | — | RL-tuned LLM for target-conditioned molecule generation with ADMET-aware decoding |
| BindGPT | 2025 | [AAAI](https://arxiv.org/abs/2406.03686) | — | Pocket-conditioned 3D ligand generation by a GPT-style LM with RL fine-tuning |
| TxGemma | 2025 | [arXiv](https://arxiv.org/abs/2504.06196) | [HF](https://huggingface.co/google/txgemma-9b-chat) | Gemma-based open models specialized for therapeutic-discovery tasks |
| BioReason | 2025 | [arXiv](https://arxiv.org/abs/2505.23579) | — | Reinforcement-learning reasoning training over a DNA + text LM |

### Biomedical AI Agents
Multi-step systems with tool use, planning, and bounded task scope. Systems that approach AI Scientist criteria but lack a self-evolving mechanism are listed here.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| ChemCrow | 2023 | [Nature Machine Intelligence](https://doi.org/10.1038/s42256-024-00832-8) ([arXiv](https://arxiv.org/abs/2304.05376)) | [GitHub](https://github.com/ur-whitelab/chemcrow-public) | Seminal LLM agent equipped with 18 chemistry tools; synthesized organocatalysts and DEET |
| Coscientist (Boiko) | 2023 | [Nature](https://doi.org/10.1038/s41586-023-06792-0) | — | GPT-4 agent that planned, coded, and ran palladium-catalyzed cross-coupling on a robotic platform |
| ProtAgents | 2024 | [Digital Discovery](https://doi.org/10.1039/D4DD00013G) | — | Multi-agent de novo protein design combining MD and physics simulators with LLM planners |
| CRISPR-GPT | 2024 | [arXiv](https://arxiv.org/abs/2404.18021) | — | Tool-augmented agent guiding gRNA design and protocol selection across CRISPR editing systems |
| GeneAgent | 2024 | [arXiv](https://arxiv.org/abs/2405.16205) | [GitHub](https://github.com/ncbi-nlp/GeneAgent) | Self-verification agent grounding gene-set claims via NCBI and Enrichr APIs |
| BioDiscoveryAgent | 2024 | [arXiv](https://arxiv.org/abs/2405.17631) | [GitHub](https://github.com/snap-stanford/BioDiscoveryAgent) | Iteratively selects CRISPR perturbations; outperforms Bayesian baselines |
| AutoBA | 2024 | [Advanced Science](https://doi.org/10.1002/advs.202407094) | [GitHub](https://github.com/JoshuaChou2018/AutoBA) | End-to-end agent that proposes, codes, and repairs bioinformatics pipelines from a goal |
| CellAgent | 2024 | [arXiv](https://arxiv.org/abs/2407.09811) | [GitHub](https://github.com/lsq2wal/CellAgent) | Planner-Executor-Evaluator architecture for scRNA-seq analysis |
| DrugAgent | 2024 | [arXiv](https://arxiv.org/abs/2408.13378) | — | Coordinates ML, knowledge-graph, and search agents for explainable DTI and repurposing |
| AtomAgents | 2025 | [PNAS](https://doi.org/10.1073/pnas.2414074122) | — | Multi-agent system coupling LLMs to physics simulators for materials and biomaterials design |
| AutoProteinEngine | 2025 | [COLING Industry](https://aclanthology.org/2025.coling-industry.10/) | — | LLM agent automating protein-engineering AutoML pipelines via tool calling |
| ESCARGOT | 2025 | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btaf031) | — | Graph-of-Thoughts agent over biomedical knowledge graphs for multi-hop reasoning |
| Biomni | 2025 | [bioRxiv](https://doi.org/10.1101/2025.05.30.656746) | [GitHub](https://github.com/snap-stanford/Biomni) | General biomedical agent integrating ~150 tools and 100+ databases across 25 tasks |

### Biomedical AI Scientists
Systems demonstrated on biomedical research that satisfy all four AI Scientist properties: multi-stage with varying autonomy, sustained and persistent, self-evolving, and knowledge-accessing and tool-using.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| AI Co-Scientist | 2025 | [arXiv](https://arxiv.org/abs/2502.18864) | — | Gemini-2.0 multi-agent generate-debate-evolve system with Elo-tournament hypothesis evolution; validated drug repurposing for AML and liver fibrosis; subsequently shown to independently recapitulate the cf-PICI bacterial-evolution mechanism ([Cell](https://doi.org/10.1016/j.cell.2025.08.018)) |
| Robin | 2025 | [arXiv](https://arxiv.org/abs/2505.13400) | [GitHub](https://github.com/Future-House/robin) | Three-agent (Crow, Falcon, Finch) lab-in-the-loop system; identified ripasudil as a novel dAMD candidate |
| STELLA | 2025 | [arXiv](https://arxiv.org/abs/2507.02004) | [GitHub](https://github.com/zaixizhang/STELLA) | Self-evolving biomedical agent with Template Library and dynamic Tool Ocean; SOTA on HLE Biomedicine |
| GenExp | 2025 | [bioRxiv](https://doi.org/10.1101/2025.06.24.661378) | — | Multi-agent platform extending Adam/Eve robot scientists for closed-loop yeast systems biology |
| NeuroDISK | 2025 | [bioRxiv](https://doi.org/10.1101/2025.02.10.637567) | — | Continuous inquiry-driven discovery system over ENIGMA neuroimaging-genetics data |
| OmniCellAgent | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.21.665802) | — | Co-scientist tailored to single-cell precision-medicine discovery loops in cancer and neurodegeneration |
| Virtual Lab | 2025 | [Nature](https://doi.org/10.1038/s41586-025-09442-9) | [GitHub](https://github.com/zou-group/virtual-lab) | LLM PI dynamically instantiates a scientist team; designed and validated 92 SARS-CoV-2 nanobodies |
| ASCollab | 2025 | [arXiv](https://arxiv.org/abs/2510.08619) | — | Heterogeneous research agents in evolving collaboration networks; applied to TCGA cancer cohorts |
| Kosmos | 2025 | [arXiv](https://arxiv.org/abs/2511.02824) | — | 12-hour autonomous campaigns coordinated by a structured world model; ~42K lines of code and ~1,500 papers per run |
| SAGA | 2025 | [arXiv](https://arxiv.org/abs/2512.21782) | — | Bi-level architecture that evolves objective functions themselves; demonstrated for antibiotic and DNA design |

---

## Clinical

### Clinical LLMs
Clinical and medical language models targeting clinical NLP, EHR, diagnostic reasoning, clinical QA, and medical imaging in clinical contexts.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| ClinicalBERT (Alsentzer) | 2019 | [ClinicalNLP / arXiv](https://arxiv.org/abs/1904.03323) | [GitHub](https://github.com/EmilyAlsentzer/clinicalBERT) | First publicly released BERT pretrained on MIMIC-III notes |
| ClinicalBERT (Huang) | 2019 | [arXiv](https://arxiv.org/abs/1904.05342) | [GitHub](https://github.com/kexinhuang12345/clinicalBERT) | Parallel ClinicalBERT focused on 30-day readmission prediction |
| GatorTron | 2022 | [npj Digital Medicine](https://doi.org/10.1038/s41746-022-00742-2) | [HF](https://huggingface.co/UFNLP/gatortron-base) | 8.9B-param clinical LM trained on 90B words of UF Health notes |
| Clinical-Longformer | 2023 | [arXiv](https://arxiv.org/abs/2301.11847) | [GitHub](https://github.com/luoyuanlab/Clinical-Longformer) | Long-context (4096-token) clinical encoder for note-level reasoning |
| ChatDoctor | 2023 | [arXiv](https://arxiv.org/abs/2303.14070) | [GitHub](https://github.com/Kent0n-Li/ChatDoctor) | Early LLaMA-based patient-doctor chatbot trained on HealthCareMagic-100k |
| MedAlpaca | 2023 | [arXiv](https://arxiv.org/abs/2304.08247) | [GitHub](https://github.com/kbressem/medAlpaca) | Open instruction-tuned medical LLaMA; widely used baseline |
| HuatuoGPT | 2023 | [Findings of EMNLP](https://aclanthology.org/2023.findings-emnlp.725/) | [GitHub](https://github.com/FreedomIntelligence/HuatuoGPT) | Major Chinese-language clinical chat LLM combining distilled and real doctor data |
| Clinical Camel | 2023 | [arXiv](https://arxiv.org/abs/2305.12031) | [GitHub](https://github.com/bowang-lab/clinical-camel) | LLaMA-2 fine-tune surpassing GPT-3.5 on USMLE and MedQA |
| NYUTron | 2023 | [Nature](https://doi.org/10.1038/s41586-023-06160-y) | — | LLM pretrained on 10 years of NYU Langone notes; predicts readmission, mortality, LOS |
| Med-PaLM | 2023 | [Nature](https://doi.org/10.1038/s41586-023-06291-2) | — | Introduced MultiMedQA and Med-PaLM; first medical LLM to clear the USMLE pass mark |
| Asclepius | 2023 | [arXiv](https://arxiv.org/abs/2309.00237) | [HF](https://huggingface.co/starmpcc/Asclepius-13B) | Shows synthetic notes can substitute MIMIC for shareable clinical LLM training |
| MEDITRON-70B | 2023 | [arXiv](https://arxiv.org/abs/2311.16079) | [GitHub](https://github.com/epfLLM/meditron) | EPFL open Llama-2-based medical LLM outperforming GPT-3.5 and Med-PaLM |
| BiomedGPT (generalist VLM) | 2024 | [Nature Medicine](https://doi.org/10.1038/s41591-024-03185-2) | [GitHub](https://github.com/taokz/BiomedGPT) | Lightweight open biomedical vision-language foundation model across 25 tasks |
| Med-PaLM M | 2024 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2300138) | — | Multimodal Med-PaLM unifying text, imaging, and genomics on a PaLM-E backbone |
| BiMediX | 2024 | [arXiv](https://arxiv.org/abs/2402.13253) | [GitHub](https://github.com/mbzuai-oryx/BiMediX) | First Arabic-English mixture-of-experts medical LLM on Mixtral-8x7B |
| Apollo | 2024 | [arXiv](https://arxiv.org/abs/2403.03640) | [GitHub](https://github.com/FreedomIntelligence/Apollo) | Multilingual medical LLM across the 6 most-spoken languages with XMedBench |
| MedFound | 2025 | [Nature Medicine](https://www.nature.com/articles/s41591-025-03520-1) | [GitHub](https://github.com/medfound/medfound) | 176B-param medical LLM with self-bootstrapped diagnostic reasoning across specialties |
| Med-PaLM 2 | 2025 | [Nature Medicine](https://doi.org/10.1038/s41591-024-03423-7) | — | Med-PaLM 2 reaches expert-level MedQA accuracy (>86%) |

### Clinical AI Agents
Agents for clinical reasoning, EHR query, trial matching, and clinical decision support. Benchmarks that evaluate these agents (e.g., MedAgentBench, AgentClinic) are not listed here because they are evaluation frameworks rather than agent systems.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| Almanac | 2024 | [NEJM AI](https://ai.nejm.org/doi/abs/10.1056/AIoa2300068) | [arXiv](https://arxiv.org/abs/2303.01229) | First clinical RAG system rigorously evaluated by clinicians across 9 specialties |
| AMIE | 2024 | [arXiv](https://arxiv.org/abs/2401.05654) | — | Google DeepMind self-play history-taking diagnostic agent matching or exceeding PCPs |
| MedAgents | 2024 | [ACL Findings](https://aclanthology.org/2024.findings-acl.33/) | [GitHub](https://github.com/gersteinlab/MedAgents) | Multi-disciplinary role-play collaboration framework for zero-shot medical reasoning |
| Agent Hospital | 2024 | [arXiv](https://arxiv.org/abs/2405.02957) | — | Simulated hospital where doctor agents self-evolve via virtual patients |
| TriageAgent | 2024 | [Findings of EMNLP](https://aclanthology.org/2024.findings-emnlp.886/) | — | Multi-agent debate framework specialized for emergency triage decisions |
| EHRAgent | 2024 | [EMNLP](https://arxiv.org/abs/2401.07128) | [GitHub](https://github.com/wshi83/EhrAgent) | Code-generating agent for structured EHR query and reasoning across MIMIC and eICU |
| ClinicalAgent | 2024 | [ACM BCB](https://doi.org/10.1145/3698587.3701359) | — | Multi-agent system for trial-outcome prediction with role-specialized agents |
| MDAgents | 2024 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/90d1fc07f46e31387978b88e7e057a31-Abstract-Conference.html) | [GitHub](https://github.com/mitmedialab/MDAgents) | Dynamically scales agent collaboration to case complexity |
| TrialGPT | 2024 | [Nature Communications](https://doi.org/10.1038/s41467-024-53081-z) | [GitHub](https://github.com/ncbi-nlp/TrialGPT) | NIH zero-shot patient-trial matching system; 42.6% screening-time reduction |
| MMed-RAG | 2024 | [arXiv](https://arxiv.org/abs/2410.13085) | [GitHub](https://github.com/richard-peng-xia/MMed-RAG) | Multimodal medical RAG with domain-aware retrieval across imaging modalities |
| ColaCare | 2025 | [WWW](https://arxiv.org/abs/2410.02551) | [GitHub](https://github.com/PKU-AICare/ColaCare) | Multidisciplinary-team-style DoctorAgents plus MetaAgent for EHR prediction |
| Zero-Shot Trial Matching (Wornow) | 2025 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIcs2400360) | — | Stanford zero-shot LLM trial matching pipeline |
| MedRAX | 2025 | [ICML](https://arxiv.org/abs/2502.02673) | [GitHub](https://github.com/bowang-lab/MedRAX) | Tool-using chest X-ray agent integrating segmentation, classification, and VQA |
| AMIE (multimodal) | 2025 | [arXiv](https://arxiv.org/abs/2505.04653) | — | Multimodal AMIE on Gemini 2.0 Flash; matches PCPs in OSCE chats with images |
| AMIE (management) | 2025 | [arXiv](https://arxiv.org/abs/2503.06074) | — | Extends AMIE to longitudinal management with Dialogue plus Mx reasoning agents |
| AgentMD | 2025 | [Nature Communications](https://doi.org/10.1038/s41467-025-64430-x) | — | Auto-curates 2,164 clinical calculators; 87.7% vs GPT-4 40.9% risk-prediction accuracy |
| TxAgent | 2025 | [arXiv](https://arxiv.org/abs/2503.10970) | [GitHub](https://github.com/mims-harvard/TxAgent) | Harvard Zitnik-lab agent with 211 tools for personalized therapeutic reasoning |
| TeamPath | 2025 | [arXiv](https://arxiv.org/abs/2511.17652) | — | Multi-agent pathology copilot for diagnostic image and report reasoning |

### Clinical AI Scientists
*This bucket is intentionally empty.* The perspective paper's thesis is that AI Scientists in clinical research and healthcare delivery are a near-term gap, not a populated category. No published system in the clinical subdomain currently satisfies all four AI Scientist properties (multi-stage with varying autonomy, sustained and persistent, self-evolving, and knowledge-accessing and tool-using). Several existing works approach this frontier but fall short:

- Trial-emulation and continuously-learning AI frameworks (e.g., Dahabreh 2025, Lin 2025, Rosenthal 2025) are perspectives and analytic frameworks rather than deployed systems.
- The clinical perspective on AI co-scientist (Nature Medicine 2026) is a viewpoint piece, not a new system.
- Clinical environment simulators (Nature Medicine 2026) are evaluation infrastructure, not AI Scientist systems.
- Diagnostic and therapeutic agents (AMIE, TxAgent, TrialGPT) operate on bounded tasks rather than self-directed clinical research cycles, so they are listed under [Clinical AI Agents](#clinical-ai-agents).

---

## Public Health

### Public Health LLMs
Language models applied to population-level health: epidemiology, surveillance, infodemiology, vaccine sentiment, social determinants of health.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| CT-BERT | 2020 | [arXiv](https://arxiv.org/abs/2005.07503) ([Front. AI](https://doi.org/10.3389/frai.2023.1023281)) | [GitHub](https://github.com/digitalepidemiologylab/covid-twitter-bert) | COVID-domain Twitter LM trained on 160M tweets; reused for vaccine sentiment and infodemiology |
| MentalBERT | 2022 | [LREC](https://arxiv.org/abs/2110.15621) | [HF](https://huggingface.co/mental/mental-bert-base-uncased) | Domain-pretrained LM on Reddit mental-health text; backbone for depression and suicidal-ideation surveillance |
| PHS-BERT | 2022 | [NLPPower @ ACL](https://aclanthology.org/2022.nlppower-1.3/) | [HF](https://huggingface.co/publichealthsurveillance/PHS-BERT) | Public-health-domain BERT benchmarked across 25 public-health surveillance tasks |
| PandemicLLM | 2024 | [Nature Computational Science](https://doi.org/10.1038/s43588-025-00798-6) ([arXiv](https://arxiv.org/abs/2404.06962)) | — | Multimodal LLM reformulating outbreak forecasting as text reasoning over policy, genomics, and spatial signals |
| SDoH-GPT | 2024 | [arXiv](https://arxiv.org/abs/2407.17126) | — | LLM pipeline extracting social determinants of health at population scale |
| EpiLLM | 2025 | [arXiv](https://arxiv.org/abs/2505.12738) | — | Dual-branch LLM aligning case counts and human-mobility tokens for spatio-temporal forecasting |
| PH-LLM (Infoveillance) | 2025 | [medRxiv](https://www.medrxiv.org/content/10.1101/2025.02.08.25321587v1) | [GitHub](https://github.com/luoyuanlab/PH-LLM) | Open multilingual Qwen-2.5-based suite (0.5B-32B) for real-time public-health infoveillance |

### Public Health AI Agents
Agents with tool use, planning, and retrieval for outbreak detection, epidemiological modeling, evidence synthesis, and policy reasoning.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| AD-AutoGPT | 2023 | [PLOS Global Public Health](https://doi.org/10.1371/journal.pgph.0004383) ([arXiv](https://arxiv.org/abs/2306.10095)) | — | Early autonomous LLM agent scraping AD news and visualizing spatio-temporal infodemiology trends |
| Epidemic Modeling with Generative Agents | 2023 | [arXiv](https://arxiv.org/abs/2307.04986) | — | ChatGPT-driven generative agents reproducing quarantine, isolation, and multi-wave epidemic dynamics |
| Human-AI Evidence Synthesis | 2024 | [Cell Reports Sustainability](https://doi.org/10.1016/j.crsus.2024.100132) | — | Tool-using LLM workflow for evidence-synthesis screening reused in public-health reviews |
| LLM Data Extraction for Evidence Synthesis | 2024 | [Research Synthesis Methods](https://doi.org/10.1002/jrsm.1710) | — | LLM-as-extractor agent for public-health evidence synthesis at scale |
| LLMs for AMR Policy Development | 2024 | [Environmental Science & Technology](https://pubs.acs.org/doi/10.1021/acs.est.4c07842) | — | RAG-based LLM agent integrating multisectoral AMR policy documents across 146 countries |
| Epidemic Information Extraction with LLMs | 2024 | [arXiv](https://arxiv.org/abs/2408.14277) | — | LLM ensemble extracting structured outbreak signals from ProMED and WHO DON |
| BEACON | 2025 | [Journal of Infectious Diseases](https://doi.org/10.1093/infdis/jiaf642) | [Platform](https://beacon.bu.edu/) | Deployed event-based surveillance pairing a domain-adapted PandemIQ-Llama agent with analyst-in-the-loop review |
| LLM Agentic Framework for Cholera Risk | 2025 | [Springer LNCS](https://doi.org/10.1007/978-3-032-11733-5_31) | — | LLM agent reasoning over feature importance and regression artifacts for policy-grade cholera risk |
| EpidemIQs | 2025 | [arXiv](https://arxiv.org/abs/2510.00024) | [GitHub](https://github.com/KsuNetse/EpidemIQs) | Multi-agent "scientist + task-expert" framework running prompt-to-paper epidemic modeling pipelines |

### Public Health AI Scientists
*This bucket is intentionally empty.* No published system in the public-health subdomain currently satisfies all four AI Scientist properties. The closest contenders fall short on the self-evolving criterion:

- EpidemIQs (arXiv 2510.00024) automates a prompt-to-paper epidemic-modeling pipeline but operates as a fixed workflow without explicit self-evolution; it is therefore listed under [Public Health AI Agents](#public-health-ai-agents).
- The CEPI Pandemic Preparedness Engine (PPX) is an announced platform under construction rather than a deployed system with published capabilities.

---

## Contributing
Contributions are welcome. Please open a PR and:
1. Add entries in the cell that matches both paradigm (LLM, AI Agent, AI Scientist) and subdomain (Biomedical, Clinical, Public Health), following the strict definitions stated at the top of this document.
2. Keep notes brief, neutral, and under ~30 words.
3. Use stable links (DOI, arXiv, official publisher page, or canonical repository).
4. Avoid duplicates and maintain chronological ordering within each cell.
5. Do not add reviews, perspective articles, benchmarks, evaluation studies, or simulators; only the LLM, AI Agent, or AI Scientist system itself.

## Citation
If you use this repository in your work, please cite:
```bibtex
@misc{ai_scientists_health_awesome,
  title  = {AI-Scientists-Health: An Awesome-Style Survey of LLMs, AI Agents, and AI Scientists across Biomedical, Clinical, and Public Health},
  author = {Contributors},
  year   = {2026},
  url    = {https://github.com/Yale-BIDS-Chen-Lab/AI-Scientists-Health}
}
```
