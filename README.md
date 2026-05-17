# AI-Scientists-Health

This repository is centered on **AI Scientists in health** and is intended to serve as a public, regularly updated companion resource to the paper *From AI Agents to AI Scientists in Health: Emerging Landscape, Challenges, and the Path Forward* (2026).

AI Scientists are systems that pursue scientific questions through multi-stage, sustained, and adaptive inquiry, including generating hypotheses, designing and executing evaluations, interpreting results, and iteratively refining research directions. In health, this emerging paradigm is especially consequential because heterogeneous multimodal data, stringent evidence requirements, and demanding governance expectations make the path from capability to trustworthy deployment particularly challenging.

To situate AI Scientists within their broader technical lineage, this repository also includes **selected** large language models (LLMs) and AI Agents. These entries are included only to document the paradigm transition, clarify boundary cases, and provide subdomain-specific context. This repository is therefore **not** intended to be an exhaustive catalog of all health-related LLMs or AI agents. Instead, it is designed to support transparent and evolving curation of systems most relevant to interpreting AI Scientists in health as an emerging paradigm.

Entries are organized along two axes: three paradigms (**LLMs**, **AI Agents**, **AI Scientists**) and three health subdomains (**Biomedical**, **Clinical**, **Public Health**).

## Categorization rules

Each entry is placed in exactly one cell based on the following criteria:

- **LLM**: a language model itself (text, sequence, or multimodal foundation model with language as a primary modality). Evaluation studies of LLMs, perspectives, and benchmarks are excluded.
- **AI Agent**: a system that performs multi-step reasoning with tool use, planning, and memory, but operates on bounded tasks. Benchmarks that evaluate agents are excluded; only the agent systems themselves are listed.
- **AI Scientist**: a system that satisfies all four properties defined in Section 2.1 of the paper:  
  (1) multi-stage with varying autonomy,  
  (2) sustained and persistent,  
  (3) self-evolving, and  
  (4) knowledge-accessing and tool-using.  
  Reviews, perspectives, simulators, and partial pipelines that do not satisfy all four properties are excluded or reclassified.
- **Subdomain**: an entry must have a demonstrated application or evaluation in the corresponding subdomain.

Paradigm-defining general-purpose AI Scientist systems from machine learning or algorithmic domains (for example, *The AI Scientist*, *AlphaEvolve*, *DeepScientist*, *EvoScientist*, *AI-Researcher*, and *Agent Laboratory*) are foundational for understanding the broader landscape, but are not listed here because they do not satisfy the subdomain requirement. Readers should consult Table 1 of the paper for the broader cross-domain view.

## Notes column format

Each entry in the Notes column follows a single shared structure: `[architecture or approach descriptor] for [primary task or domain]; [optional one distinctive mechanism]`. Notes are limited to roughly 25 words and one sentence.

- **Include**: backbone or architectural noun (e.g., "multi-agent", "RAG", "self-evolving"); primary domain or modality (e.g., "EHR", "scRNA-seq", "histology", "outbreak forecasting"); primary task or capability (e.g., "differential diagnosis", "trial matching"); and one distinctive mechanism only if it is defining (e.g., "Elo-tournament hypothesis evolution", "world-model coordination").
- **Exclude**: institutional or lab names (e.g., "FutureHouse", "Mahmood Lab", "NCBI"); superlatives and chronology adjectives (e.g., "first", "seminal", "canonical", "early"); benchmark numbers, SOTA claims, and comparisons (e.g., "+15% on Y", "outperforms GPT-X"); case-study findings (e.g., specific disease names, drug names); subsequent or historical context (e.g., "predecessor to X", "subsequently shown to..."); and parameter counts unless they are the sole differentiator.

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

Within each cell, entries are sorted chronologically (oldest first).

---

## Biomedical

### Biomedical LLMs
Text-based biomedical language models and multimodal LMs in which natural language is a primary modality (e.g., text-and-molecule, text-and-DNA, text-and-protein hybrids). Pure biological-sequence foundation models (e.g., ESM, Geneformer, scGPT, scFoundation, DNABERT-2, ProtTrans, ProGen, ESM3) are excluded; the perspective paper references them as tools used by AI Scientists rather than as LLMs in the language-model lineage that becomes Agents and AI Scientists.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| BioGPT | 2022 | [Briefings in Bioinformatics](https://doi.org/10.1093/bib/bbac409) | [GitHub](https://github.com/microsoft/BioGPT) | Generative biomedical LLM pretrained on PubMed |
| Galactica | 2022 | [arXiv](https://arxiv.org/abs/2211.09085) | [Project page](https://galactica.org/) | LLM trained on a curated scientific corpus with SMILES, amino-acid, and LaTeX tokens |
| BioMedGPT | 2023 | [arXiv](https://arxiv.org/abs/2308.09442) | [GitHub](https://github.com/PharMolix/OpenBioMed) | Multimodal biomedical LLM unifying molecule, protein, and natural-language inputs |
| Mol-Instructions | 2024 | [ICLR](https://arxiv.org/abs/2306.08018) | [GitHub](https://github.com/zjunlp/Mol-Instructions) | LLaMA-tuned models on a biomolecular instruction dataset spanning molecule, protein, and biomolecule-text tasks |
| ChemLLM | 2024 | [arXiv](https://arxiv.org/abs/2402.06852) | [HF](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat) | Chemistry-specific LLM with ChemData instruction tuning |
| BioMistral | 2024 | [ACL Findings](https://aclanthology.org/2024.findings-acl.348/) | [GitHub](https://github.com/BioMistral/BioMistral) | Mistral-7B continually pretrained on PubMed Central with multilingual biomedical evaluation |
| BioMedLM | 2024 | [arXiv](https://arxiv.org/abs/2403.18421) | [HF](https://huggingface.co/stanford-crfm/BioMedLM) | Compact GPT trained from scratch on PubMed |
| ChatNT | 2024 | [bioRxiv](https://doi.org/10.1101/2024.04.30.591835) | [GitHub](https://github.com/instadeepai/ChatNT) | Conversational nucleic-acid foundation model controlled by natural-language prompts across genomics tasks |
| Tx-LLM | 2024 | [arXiv](https://arxiv.org/abs/2406.06316) | — | PaLM-2-tuned LLM jointly handling text and molecular, protein, and disease entities for therapeutic-development tasks |
| MAMMAL | 2024 | [arXiv](https://arxiv.org/abs/2410.22367) | [GitHub](https://github.com/BiomedSciAI/biomed-multi-alignment) | Text-instructed multimodal foundation model unifying protein, small-molecule, and single-cell modalities for drug discovery |
| DrugGen | 2024 | [arXiv](https://arxiv.org/abs/2411.14157) | — | RL-tuned LLM for target-conditioned SMILES generation with ADMET-aware decoding |
| NatureLM | 2025 | [arXiv](https://arxiv.org/abs/2502.07527) | [HF](https://huggingface.co/microsoft/NatureLM-8x7B) | Sequence LM unifying small molecules, proteins, DNA, RNA, and materials, controllable via text instructions |
| BindGPT | 2025 | [AAAI](https://arxiv.org/abs/2406.03686) | — | GPT-style LM for pocket-conditioned 3D ligand generation with RL fine-tuning |
| TxGemma | 2025 | [arXiv](https://arxiv.org/abs/2504.06196) | [HF](https://huggingface.co/google/txgemma-9b-chat) | Gemma-based open models specialized for therapeutic-discovery tasks |
| BioReason | 2025 | [arXiv](https://arxiv.org/abs/2505.23579) | [GitHub](https://github.com/bowang-lab/BioReason) | DNA-LLM with RL-based reasoning training for variant-effect and pathway prediction |
| BioReason-Pro | 2026 | [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1) | [GitHub](https://github.com/bowang-lab/BioReason-Pro) | Multimodal reasoning LLM coupling a Qwen backbone with ESM3 and a GO graph for protein function prediction |

### Biomedical AI Agents
Multi-step systems with tool use, planning, and bounded task scope. Systems that approach AI Scientist criteria but lack a self-evolving mechanism are listed here.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| ChemCrow | 2023 | [Nature Machine Intelligence](https://doi.org/10.1038/s42256-024-00832-8) ([arXiv](https://arxiv.org/abs/2304.05376)) | [GitHub](https://github.com/ur-whitelab/chemcrow-public) | LLM agent equipped with chemistry tools for organic synthesis planning |
| Coscientist (Boiko) | 2023 | [Nature](https://doi.org/10.1038/s41586-023-06792-0) | — | GPT-4 agent that plans, codes, and runs chemistry experiments on a robotic platform |
| BioPlanner | 2024 | [arXiv](https://arxiv.org/abs/2310.10632) | [GitHub](https://github.com/bioplanner/bioplanner) | LLM agent converting natural-language biology protocols into evaluable pseudocode |
| GeneGPT | 2024 | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btae075) | [GitHub](https://github.com/ncbi/GeneGPT) | Tool-augmented LLM calling BLAST, Gene, dbSNP, and OMIM APIs to answer genomics questions |
| ProtAgents | 2024 | [Digital Discovery](https://doi.org/10.1039/D4DD00013G) | — | Multi-agent system for de novo protein design combining MD and physics simulators with LLM planners |
| CRISPR-GPT | 2024 | [arXiv](https://arxiv.org/abs/2404.18021) | — | Tool-augmented agent for gRNA design and protocol selection across CRISPR editing systems |
| GeneAgent | 2024 | [arXiv](https://arxiv.org/abs/2405.16205) | [GitHub](https://github.com/ncbi-nlp/GeneAgent) | Self-verification agent grounding gene-set claims via database APIs |
| BioDiscoveryAgent | 2024 | [arXiv](https://arxiv.org/abs/2405.17631) | [GitHub](https://github.com/snap-stanford/BioDiscoveryAgent) | LLM agent for iterative selection of CRISPR perturbation experiments |
| AutoBA | 2024 | [Advanced Science](https://doi.org/10.1002/advs.202407094) | [GitHub](https://github.com/JoshuaChou2018/AutoBA) | End-to-end agent proposing, coding, and repairing bioinformatics pipelines from natural-language goals |
| CellAgent | 2024 | [arXiv](https://arxiv.org/abs/2407.09811) | [GitHub](https://github.com/lsq2wal/CellAgent) | Planner-Executor-Evaluator agent for scRNA-seq analysis |
| DrugAgent | 2024 | [arXiv](https://arxiv.org/abs/2408.13378) | — | Multi-agent system combining ML, knowledge-graph, and search agents for drug-target interaction and repurposing |
| PaperQA2 | 2024 | [arXiv](https://arxiv.org/abs/2409.13740) | [GitHub](https://github.com/Future-House/paper-qa) | RAG agent decomposing scientific-literature QA into search, summarize, and answer-revision tools |
| AtomAgents | 2025 | [PNAS](https://doi.org/10.1073/pnas.2414074122) | — | Multi-agent system coupling LLMs to physics simulators for materials and biomaterials design |
| BioMaster | 2025 | [bioRxiv](https://doi.org/10.1101/2025.01.23.634608) | — | Role-based multi-agent RAG system automating RNA-seq, ChIP-seq, scRNA-seq, and Hi-C workflows |
| AutoProteinEngine | 2025 | [COLING Industry](https://aclanthology.org/2025.coling-industry.10/) | — | LLM agent automating protein-engineering AutoML pipelines via tool calling |
| LIDDiA | 2025 | [arXiv](https://arxiv.org/abs/2502.13959) | [GitHub](https://github.com/ninglab/LIDDiA) | Reasoner-Executor-Evaluator-Memory agent for in-silico drug discovery |
| PharmAgents | 2025 | [arXiv](https://arxiv.org/abs/2503.22164) | — | Multi-agent system covering target discovery, lead identification, optimization, and preclinical evaluation in silico |
| ESCARGOT | 2025 | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btaf031) | — | Graph-of-Thoughts agent over biomedical knowledge graphs for multi-hop reasoning |
| Biomni | 2025 | [bioRxiv](https://doi.org/10.1101/2025.05.30.656746) | [GitHub](https://github.com/snap-stanford/Biomni) | General-purpose biomedical agent integrating broad tool and database access across biomedical research tasks |
| GenoMAS | 2025 | [arXiv](https://arxiv.org/abs/2507.21035) | [GitHub](https://github.com/Liu-Hy/GenoMAS) | Six-agent code-driven framework for gene-expression analysis |
| BioMARS | 2025 | [arXiv](https://arxiv.org/abs/2507.01485) | — | LLM, VLM, and robotics platform with Biologist, Technician, and Inspector agents for autonomous cell-culture experiments |

### Biomedical AI Scientists
Systems demonstrated on biomedical research that satisfy all four AI Scientist properties: multi-stage with varying autonomy, sustained and persistent, self-evolving, and knowledge-accessing and tool-using.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| BioResearcher | 2024 | [arXiv](https://arxiv.org/abs/2412.09429) | — | Hierarchical multi-agent system for dry-lab biomedical research with iterative refinement across search, literature, design, and programming |
| AI Co-Scientist | 2025 | [arXiv](https://arxiv.org/abs/2502.18864) | — | Multi-agent generate-debate-evolve system with Elo-tournament hypothesis evolution for biomedical research |
| Robin | 2025 | [arXiv](https://arxiv.org/abs/2505.13400) | [GitHub](https://github.com/Future-House/robin) | Three-agent lab-in-the-loop system for therapeutic-candidate discovery |
| CellVoyager | 2025 | [bioRxiv](https://doi.org/10.1101/2025.06.03.657517) | — | Autonomous agent reading scRNA-seq papers, running live-coded analyses, and iteratively revising hypotheses |
| STELLA | 2025 | [arXiv](https://arxiv.org/abs/2507.02004) | [GitHub](https://github.com/zaixizhang/STELLA) | Self-evolving biomedical agent with a persistent Template Library and a dynamic Tool Ocean |
| GenExp | 2025 | [bioRxiv](https://doi.org/10.1101/2025.06.24.661378) | — | Multi-agent platform for closed-loop yeast systems biology, extending the Adam and Eve robot-scientist line |
| NeuroDISK | 2025 | [bioRxiv](https://doi.org/10.1101/2025.02.10.637567) | — | Continuous inquiry-driven discovery system over neuroimaging-genetics data |
| OmniCellAgent | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.21.665802) | — | Co-scientist for single-cell precision-medicine discovery loops |
| BioLab | 2025 | [bioRxiv](https://doi.org/10.1101/2025.09.03.674085) | — | Eight-agent Planner-Reasoner-Critic-Memory system orchestrating biomolecular tools for antibody and biomolecule design |
| Virtual Lab | 2025 | [Nature](https://doi.org/10.1038/s41586-025-09442-9) | [GitHub](https://github.com/zou-group/virtual-lab) | LLM principal-investigator dynamically instantiating a scientist team for biomolecule design |
| ASCollab | 2025 | [arXiv](https://arxiv.org/abs/2510.08619) | — | Heterogeneous research agents in self-evolving collaboration networks for cancer-omics discovery |
| Kosmos | 2025 | [arXiv](https://arxiv.org/abs/2511.02824) | — | Long-horizon autonomous research campaigns coordinated by a structured world model |
| SAGA | 2025 | [arXiv](https://arxiv.org/abs/2512.21782) | — | Bi-level architecture that evolves objective functions themselves for biomolecule design |

---

## Clinical

### Clinical LLMs
Clinical and medical language models targeting clinical NLP, EHR, diagnostic reasoning, clinical QA, and medical imaging in clinical contexts.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| ChatDoctor | 2023 | [arXiv](https://arxiv.org/abs/2303.14070) | [GitHub](https://github.com/Kent0n-Li/ChatDoctor) | LLaMA fine-tuned on patient-doctor dialogue for clinical chat |
| MedAlpaca | 2023 | [arXiv](https://arxiv.org/abs/2304.08247) | [GitHub](https://github.com/kbressem/medAlpaca) | Instruction-tuned medical LLaMA |
| HuatuoGPT | 2023 | [Findings of EMNLP](https://aclanthology.org/2023.findings-emnlp.725/) | [GitHub](https://github.com/FreedomIntelligence/HuatuoGPT) | Chinese-language clinical chat LLM combining distilled and real doctor data |
| Clinical Camel | 2023 | [arXiv](https://arxiv.org/abs/2305.12031) | [GitHub](https://github.com/bowang-lab/clinical-camel) | LLaMA-2 fine-tune for medical question answering |
| NYUTron | 2023 | [Nature](https://doi.org/10.1038/s41586-023-06160-y) | — | Clinical LLM pretrained on hospital-system notes for readmission, mortality, and length-of-stay prediction |
| Med-PaLM | 2023 | [Nature](https://doi.org/10.1038/s41586-023-06291-2) | — | Medical LLM evaluated on the MultiMedQA suite |
| Asclepius | 2023 | [arXiv](https://arxiv.org/abs/2309.00237) | [HF](https://huggingface.co/starmpcc/Asclepius-13B) | Clinical LLM trained on synthetic notes as a shareable substitute for MIMIC |
| MEDITRON-70B | 2023 | [arXiv](https://arxiv.org/abs/2311.16079) | [GitHub](https://github.com/epfLLM/meditron) | Open Llama-2-based medical LLM |
| Polaris | 2024 | [arXiv](https://arxiv.org/abs/2403.13313) | [Hippocratic AI](https://hippocraticai.com/polaris-3/) | Multi-LLM stack for safety-focused patient-facing voice conversations |
| Me-LLaMA | 2024 | [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01533-1) | [PhysioNet](https://physionet.org/content/me-llama/1.0.0/) | LLaMA-2 continually pretrained on medical tokens with downstream instruction tuning |
| Meerkat | 2024 | [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01653-8) | [arXiv](https://arxiv.org/abs/2404.00376) | Compact medical LLM learning reasoning from textbook-derived chain-of-thought for on-premises deployment |
| BiomedGPT (generalist VLM) | 2024 | [Nature Medicine](https://doi.org/10.1038/s41591-024-03185-2) | [GitHub](https://github.com/taokz/BiomedGPT) | Open biomedical vision-language foundation model spanning clinical tasks |
| Med-PaLM M | 2024 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2300138) | — | Multimodal Med-PaLM unifying text, imaging, and genomics on a PaLM-E backbone |
| BiMediX | 2024 | [arXiv](https://arxiv.org/abs/2402.13253) | [GitHub](https://github.com/mbzuai-oryx/BiMediX) | Arabic-English mixture-of-experts medical LLM on Mixtral-8x7B |
| Apollo | 2024 | [arXiv](https://arxiv.org/abs/2403.03640) | [GitHub](https://github.com/FreedomIntelligence/Apollo) | Multilingual medical LLM across six widely spoken languages |
| HuatuoGPT-Vision | 2024 | [arXiv](https://arxiv.org/abs/2406.19280) | [GitHub](https://github.com/FreedomIntelligence/HuatuoGPT-Vision) | Medical multimodal LLM trained on PubMedVision image-QA pairs |
| UltraMedical | 2024 | [arXiv](https://arxiv.org/abs/2406.03949) | [GitHub](https://github.com/TsinghuaC3I/UltraMedical) | Llama-3 medical LLM trained on mixed synthetic and manual biomedical instructions with DPO |
| PathChat | 2024 | [Nature](https://doi.org/10.1038/s41586-024-07618-3) | [Lab page](https://comp-path.bwh.harvard.edu/pathchat/) | Vision-language pathology copilot fine-tuned on visual-language instructions |
| BiMediX2 | 2024 | [arXiv](https://arxiv.org/abs/2412.07769) | [GitHub](https://github.com/mbzuai-oryx/BiMediX2) | Bilingual Arabic-English medical multimodal LLM on a Llama-3.1 backbone covering text and medical imaging |
| HuatuoGPT-o1 | 2024 | [arXiv](https://arxiv.org/abs/2412.18925) | [GitHub](https://github.com/FreedomIntelligence/HuatuoGPT-o1) | Chinese-English medical reasoning LLM trained with verifier-guided SFT and PPO |
| MedFound | 2025 | [Nature Medicine](https://www.nature.com/articles/s41591-025-03520-1) | [GitHub](https://github.com/medfound/medfound) | Medical LLM with self-bootstrapped diagnostic reasoning across specialties |
| Med-PaLM 2 | 2025 | [Nature Medicine](https://doi.org/10.1038/s41591-024-03423-7) | — | Medical LLM extending Med-PaLM for medical question answering |
| Baichuan-M1 | 2025 | [arXiv](https://arxiv.org/abs/2502.12671) | — | Chinese-English medical LLM trained from scratch with an explicit medical curriculum |
| Preferred-MedLLM-Qwen-72B | 2025 | [arXiv](https://arxiv.org/abs/2504.18080) | [HF](https://huggingface.co/pfnet/Preferred-MedLLM-Qwen-72B) | Qwen-2.5 with continued Japanese medical pretraining and reasoning preference optimization |
| Lingshu | 2025 | [arXiv](https://arxiv.org/abs/2506.07044) | [HF](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B) | Qwen2.5-VL-based generalist medical multimodal LLM covering multiple imaging modalities |
| SNUH KMed.ai | 2025 | [SNUH announcement](http://www.snuh.org/global/en/about/newsView.do?bbs_no=7022) | — | Korean clinical LLM trained on de-identified hospital clinical texts |
| AyurParam | 2025 | [arXiv](https://arxiv.org/abs/2511.02374) | [HF](https://huggingface.co/bharatgenai/AyurParam) | Bilingual Hindi-English instruction-tuned LLM specialized for Ayurveda and Indian traditional medicine |

### Clinical AI Agents
Agents for clinical reasoning, EHR query, trial matching, and clinical decision support. Benchmarks that evaluate these agents (e.g., MedAgentBench, AgentClinic) are not listed here because they are evaluation frameworks rather than agent systems.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| Almanac | 2024 | [NEJM AI](https://ai.nejm.org/doi/abs/10.1056/AIoa2300068) | [arXiv](https://arxiv.org/abs/2303.01229) | Clinical RAG system evaluated by clinicians across multiple specialties |
| AMIE | 2024 | [arXiv](https://arxiv.org/abs/2401.05654) | — | Self-play history-taking diagnostic agent for primary-care consultations |
| MedAgents | 2024 | [ACL Findings](https://aclanthology.org/2024.findings-acl.33/) | [GitHub](https://github.com/gersteinlab/MedAgents) | Multi-disciplinary role-play collaboration framework for zero-shot medical reasoning |
| Agent Hospital | 2024 | [arXiv](https://arxiv.org/abs/2405.02957) | — | Simulated-hospital environment where doctor agents self-evolve via virtual patients |
| TriageAgent | 2024 | [Findings of EMNLP](https://aclanthology.org/2024.findings-emnlp.886/) | — | Multi-agent debate framework for emergency triage decisions |
| EHRAgent | 2024 | [EMNLP](https://arxiv.org/abs/2401.07128) | [GitHub](https://github.com/wshi83/EhrAgent) | Code-generating agent for structured EHR query and reasoning |
| ClinicalAgent | 2024 | [ACM BCB](https://doi.org/10.1145/3698587.3701359) | — | Multi-agent system with role-specialized agents for clinical-trial outcome prediction |
| MDAgents | 2024 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/90d1fc07f46e31387978b88e7e057a31-Abstract-Conference.html) | [GitHub](https://github.com/mitmedialab/MDAgents) | Adaptive multi-agent framework that scales collaboration to case complexity |
| TrialGPT | 2024 | [Nature Communications](https://doi.org/10.1038/s41467-024-53081-z) | [GitHub](https://github.com/ncbi-nlp/TrialGPT) | Zero-shot LLM agent for patient-to-trial matching |
| RadioRAG | 2024 | [Radiology: Artificial Intelligence](https://pubs.rsna.org/doi/10.1148/ryai.240476) ([arXiv](https://arxiv.org/abs/2407.15621)) | — | Online retrieval-augmented agent for radiology question answering |
| MMedAgent | 2024 | [EMNLP Findings](https://aclanthology.org/2024.findings-emnlp.510/) | — | Multimodal medical agent invoking specialized imaging tools for segmentation, classification, and grounding |
| SurgeryLLM | 2024 | [npj Digital Medicine](https://www.nature.com/articles/s41746-024-01391-3) | — | RAG-based framework for surgical decision support grounded in evidence-based guidelines |
| MMed-RAG | 2024 | [arXiv](https://arxiv.org/abs/2410.13085) | [GitHub](https://github.com/richard-peng-xia/MMed-RAG) | Multimodal medical RAG with domain-aware retrieval across imaging modalities |
| ColaCare | 2025 | [WWW](https://arxiv.org/abs/2410.02551) | [GitHub](https://github.com/PKU-AICare/ColaCare) | Multidisciplinary-team-style DoctorAgents combined with a MetaAgent for EHR prediction |
| Zero-Shot Trial Matching (Wornow) | 2025 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIcs2400360) | — | Zero-shot LLM pipeline for clinical-trial matching |
| MedRAX | 2025 | [ICML](https://arxiv.org/abs/2502.02673) | [GitHub](https://github.com/bowang-lab/MedRAX) | Tool-using chest-X-ray agent integrating segmentation, classification, and VQA |
| MEDDxAgent | 2025 | [ACL](https://arxiv.org/abs/2502.19175) | — | Modular differential-diagnosis agent with orchestrator, history-taker, and retrieval-strategy modules |
| MedAgent-Pro | 2025 | [arXiv](https://arxiv.org/abs/2503.18968) | — | Evidence-based diagnostic agent combining guideline-RAG planning with multimodal reasoning |
| AMIE (multimodal) | 2025 | [arXiv](https://arxiv.org/abs/2505.04653) | — | Multimodal extension of AMIE for image-grounded clinical consultations |
| AMIE (management) | 2025 | [arXiv](https://arxiv.org/abs/2503.06074) | — | Extension of AMIE for longitudinal disease management with dialogue and treatment-reasoning agents |
| MAI-DxO | 2025 | [arXiv](https://arxiv.org/abs/2506.22405) | — | Sequential-diagnosis multi-agent orchestrator for clinical-pathological case reasoning |
| TrialGenie | 2025 | [medRxiv](https://www.medrxiv.org/content/10.1101/2025.04.17.25326033v1) | — | Five-agent system for autonomous clinical-trial protocol design and refinement |
| DrugGPT | 2025 | [Nature Biomedical Engineering](https://www.nature.com/articles/s41551-025-01471-z) | — | Knowledge-grounded collaborative agent for clinical drug analysis |
| Ferber autonomous oncology agent | 2025 | [Nature Cancer](https://www.nature.com/articles/s43018-025-00991-6) | — | GPT-4 oncology agent orchestrating imaging, segmentation, ontology, and literature tools |
| WiseMind (MARiA) | 2025 | [npj Digital Medicine](https://www.nature.com/articles/s41746-026-02559-9) ([arXiv](https://arxiv.org/abs/2502.20689)) | — | Dual-mind agent architecture with a DSM-5 knowledge graph for psychiatric diagnosis |
| COMPOSER-LLM | 2025 | [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01689-w) | — | Sepsis-prediction system using an LLM to extract context for high-uncertainty cases |
| PEACH | 2025 | [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01858-x) | — | Perioperative chatbot integrating institution-specific guidelines as a regulated decision-support tool |
| EvoMDT | 2025 | [npj Digital Medicine](https://www.nature.com/articles/s41746-025-02304-8) | — | Coordinator-mediated multi-cancer MDT system with per-case self-evolution from physician feedback |
| AgentMD | 2025 | [Nature Communications](https://doi.org/10.1038/s41467-025-64430-x) | — | Agent that auto-curates clinical calculators and applies them for patient-level risk prediction |
| TxAgent | 2025 | [arXiv](https://arxiv.org/abs/2503.10970) | [GitHub](https://github.com/mims-harvard/TxAgent) | Tool-using agent integrating broad therapeutic-reasoning tools for personalized treatment |
| PathAgent | 2025 | [arXiv](https://arxiv.org/abs/2511.17052) | — | Training-free whole-slide pathology agent with Navigator, Perceptor, and Executor modules |
| TeamPath | 2025 | [arXiv](https://arxiv.org/abs/2511.17652) | — | Multi-agent pathology copilot for diagnostic image and report reasoning |

### Clinical AI Scientists
Systems that satisfy all four AI Scientist properties and operate on clinical data or in clinical research workflows. This bucket remained empty in early curation rounds because the perspective paper's thesis is that clinical AI Scientists are a near-term gap; the first qualifying systems are now appearing.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| HealthFlow | 2025 | [arXiv](https://arxiv.org/abs/2508.02621) | [GitHub](https://github.com/yhzhu99/HealthFlow) | Self-evolving multi-agent system for autonomous EHR-analysis research with persistent strategy-heuristic memory across sessions |
| SPARK | 2026 | [Nature Medicine](https://www.nature.com/articles/s41591-026-04357-y) | — | Agentic system generating biological hypotheses from histology, instantiating analytical tools, and refining across iterations for cancer prognosis |

---

## Public Health

### Public Health LLMs
Language models applied to population-level health: epidemiology, surveillance, infodemiology, vaccine sentiment, social determinants of health.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| PandemicLLM | 2024 | [Nature Computational Science](https://doi.org/10.1038/s43588-025-00798-6) ([arXiv](https://arxiv.org/abs/2404.06962)) | — | Multimodal LLM for outbreak forecasting as text reasoning over policy, genomic, and spatial signals |
| SDoH-GPT | 2024 | [arXiv](https://arxiv.org/abs/2407.17126) | — | LLM pipeline extracting social determinants of health at population scale |
| EpiLLM | 2025 | [arXiv](https://arxiv.org/abs/2505.12738) | — | Dual-branch LLM aligning case-count and human-mobility tokens for spatio-temporal epidemic forecasting |
| PH-LLM (Infoveillance) | 2025 | [medRxiv](https://www.medrxiv.org/content/10.1101/2025.02.08.25321587v1) | [GitHub](https://github.com/luoyuanlab/PH-LLM) | Open multilingual Qwen-2.5 suite for real-time public-health infoveillance |
| PandemIQ Llama | 2025 | [AAAI](https://doi.org/10.1609/aaai.v40i46.41301) | [GitHub](https://github.com/noc-lab/PandemIQ-Llama) | Llama-3.1 continually pretrained on a curated pandemic corpus for outbreak surveillance |

### Public Health AI Agents
Agents with tool use, planning, and retrieval for outbreak detection, epidemiological modeling, evidence synthesis, and policy reasoning.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| AD-AutoGPT | 2023 | [PLOS Global Public Health](https://doi.org/10.1371/journal.pgph.0004383) ([arXiv](https://arxiv.org/abs/2306.10095)) | — | Autonomous LLM agent scraping disease news and visualizing spatio-temporal infodemiology trends |
| Epidemic Modeling with Generative Agents | 2023 | [arXiv](https://arxiv.org/abs/2307.04986) | — | Generative agents reproducing quarantine, isolation, and multi-wave epidemic dynamics |
| Human-AI Evidence Synthesis | 2024 | [Cell Reports Sustainability](https://doi.org/10.1016/j.crsus.2024.100132) | — | Tool-using LLM workflow for evidence-synthesis screening in public-health reviews |
| LLM Data Extraction for Evidence Synthesis | 2024 | [Research Synthesis Methods](https://doi.org/10.1002/jrsm.1710) | — | LLM-as-extractor agent for public-health evidence synthesis at scale |
| LLMs for AMR Policy Development | 2024 | [Environmental Science & Technology](https://pubs.acs.org/doi/10.1021/acs.est.4c07842) | — | RAG-based agent integrating multisectoral antimicrobial-resistance policy documents |
| Epidemic Information Extraction with LLMs | 2024 | [arXiv](https://arxiv.org/abs/2408.14277) | — | LLM-ensemble extractor producing structured outbreak signals from ProMED and WHO outbreak reports |
| BEACON | 2025 | [Journal of Infectious Diseases](https://doi.org/10.1093/infdis/jiaf642) | [Platform](https://beacon.bu.edu/) | Event-based surveillance pairing a domain-adapted LLM agent with analyst-in-the-loop review |
| LLM Agentic Framework for Cholera Risk | 2025 | [Springer LNCS](https://doi.org/10.1007/978-3-032-11733-5_31) | — | LLM agent reasoning over feature-importance and regression artifacts for cholera-risk policy guidance |
| AI-VaxGuide | 2025 | [arXiv](https://arxiv.org/abs/2507.03493) | — | Agentic RAG over immunization protocols for point-of-care vaccination guidance |
| Multi-Agent Counterspeech for Health Misinformation | 2025 | [COLM](https://arxiv.org/abs/2507.07307) | — | Multi-agent system coordinating retrieval, evidence enhancement, and response refinement for health-misinformation counterspeech |
| Tobacco Misinformation Multi-Agent Pipeline | 2025 | [Frontiers in Artificial Intelligence](https://doi.org/10.3389/frai.2025.1659861) | — | Three-agent pipeline producing credibility scores for tobacco-misinformation claims |
| EpidemIQs | 2025 | [arXiv](https://arxiv.org/abs/2510.00024) | [GitHub](https://github.com/KsuNetse/EpidemIQs) | Multi-agent scientist-plus-task-expert framework running prompt-to-paper epidemic-modeling pipelines |
| EpiPlanAgent | 2025 | [arXiv](https://arxiv.org/abs/2512.10313) | — | Multi-agent system for epidemic-response planning with task decomposition, RAG over emergency standards, and scenario simulation |
| EpiAgent | 2026 | [arXiv](https://arxiv.org/abs/2602.00299) | — | Iterative program-synthesis agent that auto-builds and recalibrates compartmental epidemic simulators |

### Public Health AI Scientists
*This bucket is intentionally empty.* No published system in the public-health subdomain currently satisfies all four AI Scientist properties. The closest contenders fall short on the self-evolving criterion:

- EpidemIQs (arXiv 2510.00024) automates a prompt-to-paper epidemic-modeling pipeline but operates as a fixed workflow without explicit self-evolution; it is therefore listed under [Public Health AI Agents](#public-health-ai-agents).
- EpiAgent (Datta 2026) has multi-stage autonomy with verification and refinement loops, but is single-session and does not maintain cross-session memory or evolve strategies from prior failures.
- PandemIQ Llama and the BEACON platform constitute a domain-adapted LLM plus a human-in-the-loop pipeline rather than a self-evolving research agent.
- The CEPI Pandemic Preparedness Engine (PPX) is an announced platform under construction rather than a deployed system with published capabilities.

---

## Contributing
Contributions are welcome. Please open a PR and:
1. Add entries in the cell that matches both paradigm (LLM, AI Agent, AI Scientist) and subdomain (Biomedical, Clinical, Public Health), following the strict definitions stated at the top of this document.
2. Use stable links (DOI, arXiv, official publisher page, or canonical repository).
3. Avoid duplicates and maintain chronological ordering within each cell.
4. Do not add reviews, perspective articles, benchmarks, evaluation studies, or simulators; only the LLM, AI Agent, or AI Scientist system itself.
5. Write the Notes column according to the format described in [Notes column format](#notes-column-format): one sentence, around 25 words, focused on architecture or approach and primary task or domain, and free of institutional names, superlatives, benchmark numbers, comparisons, case-study findings, and historical context.
