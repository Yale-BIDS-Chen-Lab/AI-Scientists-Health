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
Text-based biomedical language models and multimodal LMs in which natural language is a primary modality (e.g., text-and-molecule, text-and-DNA, text-and-protein hybrids). Pure biological-sequence foundation models (e.g., ESM, Geneformer, scGPT, scFoundation, DNABERT-2, ProtTrans, ProGen, ESM3) are excluded; the perspective paper references them as tools used by AI Scientists rather than as LLMs in the language-model lineage that becomes Agents and AI Scientists.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| SciBERT | 2019 | [EMNLP](https://aclanthology.org/D19-1371/) | [GitHub](https://github.com/allenai/scibert) | BERT pretrained on 1.14M scientific papers; canonical scientific-text encoder |
| BioBERT | 2020 | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btz682) | [GitHub](https://github.com/dmis-lab/biobert) | First widely adopted BERT pretrained on PubMed; baseline for biomedical NER, RE, QA |
| PubMedBERT | 2021 | [ACM Trans. Comput. Healthc.](https://doi.org/10.1145/3458754) | [HF](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) | From-scratch PubMed pretraining; introduced the BLURB benchmark |
| BioGPT | 2022 | [Briefings in Bioinformatics](https://doi.org/10.1093/bib/bbac409) | [GitHub](https://github.com/microsoft/BioGPT) | First widely used generative biomedical LLM; SOTA on PubMedQA at release |
| Galactica | 2022 | [arXiv](https://arxiv.org/abs/2211.09085) | [Project page](https://galactica.org/) | 120B-parameter LLM trained on curated scientific corpus with SMILES, AA, and LaTeX tokens |
| BioMedGPT | 2023 | [arXiv](https://arxiv.org/abs/2308.09442) | [GitHub](https://github.com/PharMolix/OpenBioMed) | Open 10B-param multimodal biomedical LLM unifying molecule, protein, and natural language |
| Mol-Instructions | 2024 | [ICLR](https://arxiv.org/abs/2306.08018) | [GitHub](https://github.com/zjunlp/Mol-Instructions) | Large-scale biomolecular instruction dataset and LLaMA-tuned models across molecule, protein, and biomolecule-text tasks |
| ChemLLM | 2024 | [arXiv](https://arxiv.org/abs/2402.06852) | [HF](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat) | First chemistry-specific LLM with ChemData instruction tuning and ChemBench evaluation |
| BioMistral | 2024 | [ACL Findings](https://aclanthology.org/2024.findings-acl.348/) | [GitHub](https://github.com/BioMistral/BioMistral) | Mistral-7B continually pretrained on PubMed Central; multilingual biomedical evaluation |
| BioMedLM | 2024 | [arXiv](https://arxiv.org/abs/2403.18421) | [HF](https://huggingface.co/stanford-crfm/BioMedLM) | Compact 2.7B GPT trained from scratch on PubMed; runs on a single A100 |
| ChatNT | 2024 | [bioRxiv](https://doi.org/10.1101/2024.04.30.591835) | [GitHub](https://github.com/instadeepai/ChatNT) | Conversational nucleic-acid foundation model controlled by natural-language prompts across 27 genomics tasks |
| Tx-LLM | 2024 | [arXiv](https://arxiv.org/abs/2406.06316) | — | PaLM-2-tuned LLM jointly handling text and molecular/protein/disease entities across 66 therapeutic-development tasks; predecessor to TxGemma |
| MAMMAL | 2024 | [arXiv](https://arxiv.org/abs/2410.22367) | [GitHub](https://github.com/BiomedSciAI/biomed-multi-alignment) | IBM text-instructed multimodal foundation model unifying protein, small-molecule, and single-cell modalities; SOTA on 9/11 drug-discovery tasks |
| DrugGen | 2024 | [arXiv](https://arxiv.org/abs/2411.14157) | — | RL-tuned LLM for target-conditioned molecule (SMILES) generation with ADMET-aware decoding |
| NatureLM | 2025 | [arXiv](https://arxiv.org/abs/2502.07527) | [HF](https://huggingface.co/microsoft/NatureLM-8x7B) | Microsoft 8B-46.7B sequence LM unifying small molecules, proteins, DNA, RNA, and materials, controllable via text instructions |
| BindGPT | 2025 | [AAAI](https://arxiv.org/abs/2406.03686) | — | Pocket-conditioned 3D ligand generation by a GPT-style LM with RL fine-tuning |
| TxGemma | 2025 | [arXiv](https://arxiv.org/abs/2504.06196) | [HF](https://huggingface.co/google/txgemma-9b-chat) | Gemma-based open models specialized for therapeutic-discovery tasks |
| BioReason | 2025 | [arXiv](https://arxiv.org/abs/2505.23579) | [GitHub](https://github.com/bowang-lab/BioReason) | DNA-LLM with RL-based reasoning training; +15% on variant effect, 98% on KEGG disease pathway prediction |
| BioReason-Pro | 2026 | [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1) | [GitHub](https://github.com/bowang-lab/BioReason-Pro) | First multimodal reasoning LLM for protein function (Qwen3-4B + ESM3 + GO graph); human experts prefer its annotations over UniProt ground truth in 79% of cases |

### Biomedical AI Agents
Multi-step systems with tool use, planning, and bounded task scope. Systems that approach AI Scientist criteria but lack a self-evolving mechanism are listed here.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| ChemCrow | 2023 | [Nature Machine Intelligence](https://doi.org/10.1038/s42256-024-00832-8) ([arXiv](https://arxiv.org/abs/2304.05376)) | [GitHub](https://github.com/ur-whitelab/chemcrow-public) | Seminal LLM agent equipped with 18 chemistry tools; synthesized organocatalysts and DEET |
| Coscientist (Boiko) | 2023 | [Nature](https://doi.org/10.1038/s41586-023-06792-0) | — | GPT-4 agent that planned, coded, and ran palladium-catalyzed cross-coupling on a robotic platform |
| BioPlanner | 2024 | [arXiv](https://arxiv.org/abs/2310.10632) | [GitHub](https://github.com/bioplanner/bioplanner) | FutureHouse LLM agent that converts natural-language biology protocols into evaluable pseudocode |
| GeneGPT | 2024 | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btae075) | [GitHub](https://github.com/ncbi/GeneGPT) | NCBI tool-augmented LLM that calls BLAST, Gene, dbSNP, and OMIM APIs to answer genomics questions |
| ProtAgents | 2024 | [Digital Discovery](https://doi.org/10.1039/D4DD00013G) | — | Multi-agent de novo protein design combining MD and physics simulators with LLM planners |
| CRISPR-GPT | 2024 | [arXiv](https://arxiv.org/abs/2404.18021) | — | Tool-augmented agent guiding gRNA design and protocol selection across CRISPR editing systems |
| GeneAgent | 2024 | [arXiv](https://arxiv.org/abs/2405.16205) | [GitHub](https://github.com/ncbi-nlp/GeneAgent) | Self-verification agent grounding gene-set claims via NCBI and Enrichr APIs |
| BioDiscoveryAgent | 2024 | [arXiv](https://arxiv.org/abs/2405.17631) | [GitHub](https://github.com/snap-stanford/BioDiscoveryAgent) | Iteratively selects CRISPR perturbations; outperforms Bayesian baselines |
| AutoBA | 2024 | [Advanced Science](https://doi.org/10.1002/advs.202407094) | [GitHub](https://github.com/JoshuaChou2018/AutoBA) | End-to-end agent that proposes, codes, and repairs bioinformatics pipelines from a goal |
| CellAgent | 2024 | [arXiv](https://arxiv.org/abs/2407.09811) | [GitHub](https://github.com/lsq2wal/CellAgent) | Planner-Executor-Evaluator architecture for scRNA-seq analysis |
| DrugAgent | 2024 | [arXiv](https://arxiv.org/abs/2408.13378) | — | Coordinates ML, knowledge-graph, and search agents for explainable DTI and repurposing |
| PaperQA2 | 2024 | [arXiv](https://arxiv.org/abs/2409.13740) | [GitHub](https://github.com/Future-House/paper-qa) | FutureHouse RAG agent that decomposes scientific literature QA into search, summarize, and answer-revision tools; outperforms PhD biologists on LitQA2 |
| AtomAgents | 2025 | [PNAS](https://doi.org/10.1073/pnas.2414074122) | — | Multi-agent system coupling LLMs to physics simulators for materials and biomaterials design |
| BioMaster | 2025 | [bioRxiv](https://doi.org/10.1101/2025.01.23.634608) | — | Role-based multi-agent RAG system that automates RNA-seq, ChIP-seq, scRNA-seq, and Hi-C workflows |
| AutoProteinEngine | 2025 | [COLING Industry](https://aclanthology.org/2025.coling-industry.10/) | — | LLM agent automating protein-engineering AutoML pipelines via tool calling |
| LIDDiA | 2025 | [arXiv](https://arxiv.org/abs/2502.13959) | [GitHub](https://github.com/ninglab/LIDDiA) | Reasoner-Executor-Evaluator-Memory drug-discovery agent that hits pharmaceutical criteria on >70% of 30 clinical targets in silico |
| PharmAgents | 2025 | [arXiv](https://arxiv.org/abs/2503.22164) | — | Virtual-pharma multi-agent system covering target discovery, lead identification, optimization, and preclinical evaluation in silico |
| ESCARGOT | 2025 | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btaf031) | — | Graph-of-Thoughts agent over biomedical knowledge graphs for multi-hop reasoning |
| Biomni | 2025 | [bioRxiv](https://doi.org/10.1101/2025.05.30.656746) | [GitHub](https://github.com/snap-stanford/Biomni) | General biomedical agent integrating ~150 tools and 100+ databases across 25 tasks |
| GenoMAS | 2025 | [arXiv](https://arxiv.org/abs/2507.21035) | [GitHub](https://github.com/Liu-Hy/GenoMAS) | Six-agent code-driven gene-expression-analysis framework; +10.6% and +16.9% over prior art on GenoTEX |
| BioMARS | 2025 | [arXiv](https://arxiv.org/abs/2507.01485) | — | LLM+VLM+robotics platform with Biologist/Technician/Inspector agents for autonomous cell-culture experiments |

### Biomedical AI Scientists
Systems demonstrated on biomedical research that satisfy all four AI Scientist properties: multi-stage with varying autonomy, sustained and persistent, self-evolving, and knowledge-accessing and tool-using.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| BioResearcher | 2024 | [arXiv](https://arxiv.org/abs/2412.09429) | — | Hierarchical multi-agent dry-lab biomedical research system (search, literature, design, programming) with iterative refinement; 63% execution success on eight unmet objectives |
| AI Co-Scientist | 2025 | [arXiv](https://arxiv.org/abs/2502.18864) | — | Gemini-2.0 multi-agent generate-debate-evolve system with Elo-tournament hypothesis evolution; validated drug repurposing for AML and liver fibrosis; subsequently shown to independently recapitulate the cf-PICI bacterial-evolution mechanism ([Cell](https://doi.org/10.1016/j.cell.2025.08.018)) |
| Robin | 2025 | [arXiv](https://arxiv.org/abs/2505.13400) | [GitHub](https://github.com/Future-House/robin) | Three-agent (Crow, Falcon, Finch) lab-in-the-loop system; identified ripasudil as a novel dAMD candidate |
| CellVoyager | 2025 | [bioRxiv](https://doi.org/10.1101/2025.06.03.657517) | — | Autonomous junior-scientist agent that reads scRNA-seq papers, runs live-coded analyses, hypothesizes follow-ups, and iteratively revises plans; surfaced novel findings in COVID-19, brain aging, and the menstrual cycle |
| STELLA | 2025 | [arXiv](https://arxiv.org/abs/2507.02004) | [GitHub](https://github.com/zaixizhang/STELLA) | Self-evolving biomedical agent with Template Library and dynamic Tool Ocean; SOTA on HLE Biomedicine |
| GenExp | 2025 | [bioRxiv](https://doi.org/10.1101/2025.06.24.661378) | — | Multi-agent platform extending Adam/Eve robot scientists for closed-loop yeast systems biology |
| NeuroDISK | 2025 | [bioRxiv](https://doi.org/10.1101/2025.02.10.637567) | — | Continuous inquiry-driven discovery system over ENIGMA neuroimaging-genetics data |
| OmniCellAgent | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.21.665802) | — | Co-scientist tailored to single-cell precision-medicine discovery loops in cancer and neurodegeneration |
| BioLab | 2025 | [bioRxiv](https://doi.org/10.1101/2025.09.03.674085) | — | Eight-agent (Planner/Reasoner/Critic/Memory) system orchestrating 219 xBio-Tools across the xTrimo FM universe; designed optimized PD-1 antibodies with sub-0.02 nM IC50 |
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
| Polaris | 2024 | [arXiv](https://arxiv.org/abs/2403.13313) | [Hippocratic AI](https://hippocraticai.com/polaris-3/) | Multi-LLM "constellation" stack purpose-built for safety-focused patient-facing voice conversations |
| Me-LLaMA | 2024 | [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01533-1) | [PhysioNet](https://physionet.org/content/me-llama/1.0.0/) | LLaMA-2 continually pretrained on 129B medical tokens; outperforms ChatGPT on 7/8 datasets after instruction tuning |
| Meerkat | 2024 | [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01653-8) | [arXiv](https://arxiv.org/abs/2404.00376) | Small medical LLM that learns reasoning from textbook-derived CoT; designed for single-GPU on-premises clinical deployment |
| BiomedGPT (generalist VLM) | 2024 | [Nature Medicine](https://doi.org/10.1038/s41591-024-03185-2) | [GitHub](https://github.com/taokz/BiomedGPT) | Lightweight open biomedical vision-language foundation model across 25 tasks |
| Med-PaLM M | 2024 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2300138) | — | Multimodal Med-PaLM unifying text, imaging, and genomics on a PaLM-E backbone |
| BiMediX | 2024 | [arXiv](https://arxiv.org/abs/2402.13253) | [GitHub](https://github.com/mbzuai-oryx/BiMediX) | First Arabic-English mixture-of-experts medical LLM on Mixtral-8x7B |
| Apollo | 2024 | [arXiv](https://arxiv.org/abs/2403.03640) | [GitHub](https://github.com/FreedomIntelligence/Apollo) | Multilingual medical LLM across the 6 most-spoken languages with XMedBench |
| HuatuoGPT-Vision | 2024 | [arXiv](https://arxiv.org/abs/2406.19280) | [GitHub](https://github.com/FreedomIntelligence/HuatuoGPT-Vision) | Yi-1.5 + CLIP medical multimodal LLM continually trained on 1.3M PubMedVision image-QA pairs |
| UltraMedical | 2024 | [arXiv](https://arxiv.org/abs/2406.03949) | [GitHub](https://github.com/TsinghuaC3I/UltraMedical) | 410K mixed synthetic/manual biomedical instructions + DPO; Llama-3-70B-UltraMedical hits 86.5 on MedQA-USMLE |
| PathChat | 2024 | [Nature](https://doi.org/10.1038/s41586-024-07618-3) | [Lab page](https://comp-path.bwh.harvard.edu/pathchat/) | Mahmood-Lab vision-language pathology copilot fine-tuned on ~456K visual-language instructions |
| BiMediX2 | 2024 | [arXiv](https://arxiv.org/abs/2412.07769) | [GitHub](https://github.com/mbzuai-oryx/BiMediX2) | First bilingual Arabic-English medical LMM (Llama-3.1 backbone) covering text and medical imaging |
| HuatuoGPT-o1 | 2024 | [arXiv](https://arxiv.org/abs/2412.18925) | [GitHub](https://github.com/FreedomIntelligence/HuatuoGPT-o1) | Chinese/English medical reasoning LLM trained with verifier-guided SFT + PPO on 40K verifiable problems |
| MedFound | 2025 | [Nature Medicine](https://www.nature.com/articles/s41591-025-03520-1) | [GitHub](https://github.com/medfound/medfound) | 176B-param medical LLM with self-bootstrapped diagnostic reasoning across specialties |
| Med-PaLM 2 | 2025 | [Nature Medicine](https://doi.org/10.1038/s41591-024-03423-7) | — | Med-PaLM 2 reaches expert-level MedQA accuracy (>86%) |
| Baichuan-M1 | 2025 | [arXiv](https://arxiv.org/abs/2502.12671) | — | Chinese-English medical LLM trained from scratch on 20T tokens with explicit medical curriculum |
| Preferred-MedLLM-Qwen-72B | 2025 | [arXiv](https://arxiv.org/abs/2504.18080) | [HF](https://huggingface.co/pfnet/Preferred-MedLLM-Qwen-72B) | Qwen-2.5-72B with continued Japanese medical pretraining and reasoning preference optimization; SOTA on IgakuQA |
| Lingshu | 2025 | [arXiv](https://arxiv.org/abs/2506.07044) | [HF](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B) | Qwen2.5-VL-based generalist medical multimodal LLM covering 12+ imaging modalities |
| SNUH KMed.ai | 2025 | [SNUH announcement](http://www.snuh.org/global/en/about/newsView.do?bbs_no=7022) | — | Korean clinical LLM trained on 38M de-identified Seoul National University Hospital clinical texts; 96.4% on KMLE |
| AyurParam | 2025 | [arXiv](https://arxiv.org/abs/2511.02374) | [HF](https://huggingface.co/bharatgenai/AyurParam) | Bilingual Hindi-English instruction-tuned LLM specialized for Ayurveda and Indian traditional medicine |

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
| RadioRAG | 2024 | [Radiology: Artificial Intelligence](https://pubs.rsna.org/doi/10.1148/ryai.240476) ([arXiv](https://arxiv.org/abs/2407.15621)) | — | Online retrieval-augmented radiology QA agent pulling live data from authoritative radiologic sources |
| MMedAgent | 2024 | [EMNLP Findings](https://aclanthology.org/2024.findings-emnlp.510/) | — | Multimodal medical agent that learns to invoke specialized tools (segmentation, classification, grounding) across imaging modalities |
| SurgeryLLM | 2024 | [npj Digital Medicine](https://www.nature.com/articles/s41746-024-01391-3) | — | RAG-based LLM framework for surgical decision support using current evidence-based guidelines |
| MMed-RAG | 2024 | [arXiv](https://arxiv.org/abs/2410.13085) | [GitHub](https://github.com/richard-peng-xia/MMed-RAG) | Multimodal medical RAG with domain-aware retrieval across imaging modalities |
| ColaCare | 2025 | [WWW](https://arxiv.org/abs/2410.02551) | [GitHub](https://github.com/PKU-AICare/ColaCare) | Multidisciplinary-team-style DoctorAgents plus MetaAgent for EHR prediction |
| Zero-Shot Trial Matching (Wornow) | 2025 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIcs2400360) | — | Stanford zero-shot LLM trial matching pipeline |
| MedRAX | 2025 | [ICML](https://arxiv.org/abs/2502.02673) | [GitHub](https://github.com/bowang-lab/MedRAX) | Tool-using chest X-ray agent integrating segmentation, classification, and VQA |
| MEDDxAgent | 2025 | [ACL](https://arxiv.org/abs/2502.19175) | — | Modular differential-diagnosis agent with DDxDriver orchestrator, history-taker, and retrieval/strategy agents |
| MedAgent-Pro | 2025 | [arXiv](https://arxiv.org/abs/2503.18968) | — | Evidence-based diagnostic agent decomposing diagnosis into guideline-RAG planning plus multimodal reasoning |
| AMIE (multimodal) | 2025 | [arXiv](https://arxiv.org/abs/2505.04653) | — | Multimodal AMIE on Gemini 2.0 Flash; matches PCPs in OSCE chats with images |
| AMIE (management) | 2025 | [arXiv](https://arxiv.org/abs/2503.06074) | — | Extends AMIE to longitudinal management with Dialogue plus Mx reasoning agents |
| MAI-DxO | 2025 | [arXiv](https://arxiv.org/abs/2506.22405) | — | Microsoft sequential-diagnosis multi-agent orchestrator over NEJM CPCs; reaches 80% diagnostic accuracy with o3 under cost constraints |
| TrialGenie | 2025 | [medRxiv](https://www.medrxiv.org/content/10.1101/2025.04.17.25326033v1) | — | Five-agent clinical-trial-design system (Supervisor/Trialist/Informatician/Clinician/Statistician) autonomously refining acute-disease trial protocols |
| DrugGPT | 2025 | [Nature Biomedical Engineering](https://www.nature.com/articles/s41551-025-01471-z) | — | Knowledge-grounded collaborative LLM agent for clinical drug analysis (prescribing, interactions, contraindications) |
| Ferber autonomous oncology agent | 2025 | [Nature Cancer](https://www.nature.com/articles/s43018-025-00991-6) | — | GPT-4-based oncology agent orchestrating vision transformers, MedSAM, OncoKB, and PubMed; 87.5% tool-use accuracy |
| WiseMind (MARiA) | 2025 | [npj Digital Medicine](https://www.nature.com/articles/s41746-026-02559-9) ([arXiv](https://arxiv.org/abs/2502.20689)) | — | DBT-inspired Reasonable-Mind + Emotional-Mind agents with DSM-5 KG for empathetic psychiatric diagnosis; 84.2% accuracy |
| COMPOSER-LLM | 2025 | [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01689-w) | — | Sepsis-prediction system using an LLM to extract context for high-uncertainty cases; prospectively deployed |
| PEACH | 2025 | [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01858-x) | — | Approved Class A perioperative AI chatbot deployed at Singapore General Hospital; integrates 35 institution-specific guidelines |
| EvoMDT | 2025 | [npj Digital Medicine](https://www.nature.com/articles/s41746-025-02304-8) | — | Five-agent coordinator-mediated multi-cancer MDT system with per-case self-evolution loop that updates prompts/consensus weights from physician feedback |
| AgentMD | 2025 | [Nature Communications](https://doi.org/10.1038/s41467-025-64430-x) | — | Auto-curates 2,164 clinical calculators; 87.7% vs GPT-4 40.9% risk-prediction accuracy |
| TxAgent | 2025 | [arXiv](https://arxiv.org/abs/2503.10970) | [GitHub](https://github.com/mims-harvard/TxAgent) | Harvard Zitnik-lab agent with 211 tools for personalized therapeutic reasoning |
| PathAgent | 2025 | [arXiv](https://arxiv.org/abs/2511.17052) | — | Training-free WSI pathology agent with Navigator/Perceptor/Executor modules that iteratively zoom and locate ROIs |
| TeamPath | 2025 | [arXiv](https://arxiv.org/abs/2511.17652) | — | Multi-agent pathology copilot for diagnostic image and report reasoning |

### Clinical AI Scientists
Systems that satisfy all four AI Scientist properties and operate on clinical data or in clinical research workflows. This bucket remained empty in early curation rounds because the perspective paper's thesis is that clinical AI Scientists are a near-term gap; the first qualifying systems are now appearing.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| HealthFlow | 2025 | [arXiv](https://arxiv.org/abs/2508.02621) | [GitHub](https://github.com/yhzhu99/HealthFlow) | Self-evolving multi-agent system for autonomous EHR-analysis research with meta-agent planner, executor, evaluator, and reflector; builds a durable strategy-heuristic knowledge base reused across sessions |
| SPARK | 2026 | [Nature Medicine](https://www.nature.com/articles/s41591-026-04357-y) | — | Mahmood-Lab agentic AI that autonomously generates biological hypotheses from histology, instantiates analytical tools without retraining, runs them, and refines across iterations; evaluated on 18 cohorts, 5 cancer types, >5,400 patients with prognostic/predictive endpoints |

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
| PandemIQ Llama | 2025 | [AAAI](https://doi.org/10.1609/aaai.v40i46.41301) | [GitHub](https://github.com/noc-lab/PandemIQ-Llama) | Llama-3.1-8B continually pretrained on 5.8B tokens of curated pandemic corpus; powers the BEACON surveillance platform |

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
| AI-VaxGuide | 2025 | [arXiv](https://arxiv.org/abs/2507.03493) | — | Agentic RAG over WHO and national immunization protocols for point-of-care vaccination guidance |
| Multi-Agent Counterspeech for Health Misinformation | 2025 | [COLM](https://arxiv.org/abs/2507.07307) | — | Multi-LLM agent system coordinating retrieval, evidence enhancement, and response refinement for health-misinformation counterspeech |
| Tobacco Misinformation Multi-Agent Pipeline | 2025 | [Frontiers in Artificial Intelligence](https://doi.org/10.3389/frai.2025.1659861) | — | Three-agent CrewAI pipeline producing 0-100 credibility scores for tobacco-misinformation claims in under 7 seconds |
| EpidemIQs | 2025 | [arXiv](https://arxiv.org/abs/2510.00024) | [GitHub](https://github.com/KsuNetse/EpidemIQs) | Multi-agent "scientist + task-expert" framework running prompt-to-paper epidemic modeling pipelines |
| EpiPlanAgent | 2025 | [arXiv](https://arxiv.org/abs/2512.10313) | — | Multi-agent epidemic response planning with task decomposition, RAG over emergency standards, and scenario simulation; cuts plan-generation time by 93.9% |
| EpiAgent | 2026 | [arXiv](https://arxiv.org/abs/2602.00299) | — | Iterative program-synthesis agent that auto-builds and recalibrates compartmental epidemic simulators via retrieval-augmented flow-graph synthesis |

### Public Health AI Scientists
*This bucket is intentionally empty.* No published system in the public-health subdomain currently satisfies all four AI Scientist properties. The closest contenders fall short on the self-evolving criterion:

- EpidemIQs (arXiv 2510.00024) automates a prompt-to-paper epidemic-modeling pipeline but operates as a fixed workflow without explicit self-evolution; it is therefore listed under [Public Health AI Agents](#public-health-ai-agents).
- EpiAgent (Datta 2026) has multi-stage autonomy with verification and refinement loops, but is single-session and does not maintain cross-session memory or evolve strategies from prior failures.
- PandemIQ Llama / BEACON is a domain-adapted LLM plus a human-in-the-loop pipeline rather than a self-evolving research agent.
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
