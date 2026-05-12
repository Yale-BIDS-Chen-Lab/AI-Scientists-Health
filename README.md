# AI-Scientists-Health

AI Scientists are autonomous systems designed to conduct research end-to-end, including generating hypotheses, designing and executing evaluations, interpreting results, and iteratively refining research directions. As a paradigm that extends beyond large language models and conventional AI agents, they hold transformative potential for health, where heterogeneous multimodal data, stringent evidence requirements, and demanding governance expectations make this both the most consequential and the most underserved frontier for the emerging paradigm.

This repository curates representative work cited in or relevant to **Section 2 (The Emerging Landscape of AI Scientists)** of the perspective paper *From AI Agents to AI Scientists in Health: Emerging Landscape, Challenges, and the Path Forward* (2026). Entries are organized along two axes: the three paradigms (LLMs, AI Agents, AI Scientists) and three health subdomains (Biomedical, Clinical, Public Health).

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

Within each cell, entries are sorted chronologically (oldest first). Entries that span subdomains are placed in the bucket where the primary demonstration sits.

---

## Biomedical

### Biomedical LLMs
Domain-pretrained or fine-tuned foundation and language models for biomedical text, molecules, proteins, and omics.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| SciBERT | 2019 | [EMNLP](https://aclanthology.org/D19-1371/) | [GitHub](https://github.com/allenai/scibert) | BERT pretrained on 1.14M scientific papers; canonical scientific-text encoder |
| BioBERT | 2020 | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btz682) | [GitHub](https://github.com/dmis-lab/biobert) | First widely adopted BERT pretrained on PubMed; baseline for biomedical NER, RE, QA |
| AlphaFold | 2021 | [Nature](https://doi.org/10.1038/s41586-021-03819-7) | [GitHub](https://github.com/google-deepmind/alphafold) | Solved the 50-year protein structure prediction problem; paradigm-defining biological foundation model |
| ProtTrans (ProtBERT, ProtT5) | 2021 | [IEEE TPAMI](https://doi.org/10.1109/TPAMI.2021.3095381) | [GitHub](https://github.com/agemagician/ProtTrans) | Family of protein language models trained on UniRef and BFD; MSA-free protein prediction |
| PubMedBERT | 2021 | [ACM Trans. Comput. Healthc.](https://doi.org/10.1145/3458754) | [HF](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) | From-scratch PubMed pretraining; introduced the BLURB benchmark |
| BioGPT | 2022 | [Briefings in Bioinformatics](https://doi.org/10.1093/bib/bbac409) | [GitHub](https://github.com/microsoft/BioGPT) | First widely used generative biomedical LLM; SOTA on PubMedQA at release |
| Galactica | 2022 | [arXiv](https://arxiv.org/abs/2211.09085) | [Project page](https://galactica.org/) | 120B-parameter LLM trained on curated scientific corpus with SMILES, AA, and LaTeX tokens |
| ESM-2 / ESMFold | 2023 | [Science](https://doi.org/10.1126/science.ade2574) | [GitHub](https://github.com/facebookresearch/esm) | 15B-param protein LM enabling 60x faster MSA-free folding; ESM Metagenomic Atlas |
| ProGen | 2023 | [Nature Biotechnology](https://doi.org/10.1038/s41587-022-01618-2) | — | Generated wet-lab-validated artificial lysozymes; first controllable protein generation by an LM |
| Geneformer | 2023 | [Nature](https://doi.org/10.1038/s41586-023-06139-9) | [HF](https://huggingface.co/ctheodoris/Geneformer) | Pretrained on 30M single-cell transcriptomes; identified validated cardiomyopathy targets |
| BioMedGPT | 2023 | [arXiv](https://arxiv.org/abs/2308.09442) | [GitHub](https://github.com/PharMolix/OpenBioMed) | Open 10B-param multimodal biomedical LLM unifying molecule, protein, and natural language |
| DNABERT-2 | 2023 | [arXiv](https://arxiv.org/abs/2306.15006) | [GitHub](https://github.com/MAGICS-LAB/DNABERT_2) | BPE-tokenized genomic LM matching SOTA with 21x fewer parameters; introduced the GUE benchmark |
| MolFM | 2023 | [arXiv](https://arxiv.org/abs/2307.09484) | — | Joint pretraining over molecular graphs, SMILES, and biomedical text |
| InstructProtein | 2023 | [arXiv](https://arxiv.org/abs/2310.03269) | — | Bidirectional protein-language instruction tuning over UniProt knowledge graph |
| BioMedLM | 2024 | [arXiv](https://arxiv.org/abs/2403.18421) | [HF](https://huggingface.co/stanford-crfm/BioMedLM) | Compact 2.7B GPT trained from scratch on PubMed; runs on a single A100 |
| scGPT | 2024 | [Nature Methods](https://doi.org/10.1038/s41592-024-02201-0) | [GitHub](https://github.com/bowang-lab/scGPT) | Generative transformer pretrained on 33M cells; SOTA on cell-type annotation and perturbation |
| scFoundation | 2024 | [Nature Methods](https://doi.org/10.1038/s41592-024-02305-7) | [GitHub](https://github.com/biomap-research/scFoundation) | 100M-param model on 50M cells; SOTA on drug response and perturbation tasks |
| BiomedCLIP | 2024 | [NEJM AI](https://ai.nejm.org/doi/10.1056/AIoa2400640) ([arXiv](https://arxiv.org/abs/2303.00915)) | [HF](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | CLIP trained on PMC-15M; SOTA biomedical image-text foundation model |
| BioMistral | 2024 | [ACL Findings](https://aclanthology.org/2024.findings-acl.348/) | [GitHub](https://github.com/BioMistral/BioMistral) | Mistral-7B continually pretrained on PubMed Central; multilingual biomedical evaluation |
| ChatNT | 2024 | [bioRxiv](https://doi.org/10.1101/2024.04.30.591835) | [GitHub](https://github.com/instadeepai/ChatNT) | Conversational nucleic-acid foundation model handling 27 genomics tasks via natural-language prompts |
| ESM3 | 2024 | [bioRxiv](https://doi.org/10.1101/2024.07.01.600583) | [GitHub](https://github.com/evolutionaryscale/esm) | 98B-param multimodal protein LM over sequence, structure, and function tokens |
| AlphaFold 3 | 2024 | [Nature](https://doi.org/10.1038/s41586-024-07487-w) | [Server](https://alphafoldserver.com/) | Diffusion-based unified model for proteins, nucleic acids, ligands, and ions |
| TxGemma | 2025 | [arXiv](https://arxiv.org/abs/2504.06196) | [HF](https://huggingface.co/google/txgemma-9b-chat) | Gemma-based open models specialized for therapeutic-discovery tasks with agentic tool use |
| BioReason | 2025 | [arXiv](https://arxiv.org/abs/2505.23579) | — | RL reasoning training over a DNA + text LM for variant-effect and pathway tasks |

### Biomedical AI Agents
Agents with tool use, planning, and multi-step reasoning for biomedical research tasks.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| ChemCrow | 2023 | [Nature Machine Intelligence](https://doi.org/10.1038/s42256-024-00832-8) ([arXiv](https://arxiv.org/abs/2304.05376)) | [GitHub](https://github.com/ur-whitelab/chemcrow-public) | Seminal LLM agent equipped with 18 chemistry tools; synthesized organocatalysts and DEET |
| Coscientist | 2023 | [Nature](https://doi.org/10.1038/s41586-023-06792-0) | — | GPT-4 agent that planned, coded, and ran palladium-catalyzed cross-coupling on a robotic platform |
| ProtAgents | 2024 | [Digital Discovery](https://doi.org/10.1039/D4DD00013G) | — | Multi-agent de novo protein design combining MD and physics simulators with LLM planners |
| CRISPR-GPT | 2024 | [arXiv](https://arxiv.org/abs/2404.18021) | — | Tool-augmented agent guiding gRNA design and protocol selection across CRISPR editing systems |
| GeneAgent | 2024 | [arXiv](https://arxiv.org/abs/2405.16205) | [GitHub](https://github.com/ncbi-nlp/GeneAgent) | LLM agent grounding gene-set claims via NCBI and Enrichr APIs to reduce hallucination |
| BioDiscoveryAgent | 2024 | [arXiv](https://arxiv.org/abs/2405.17631) | [GitHub](https://github.com/snap-stanford/BioDiscoveryAgent) | Iteratively selects CRISPR perturbations; outperforms Bayesian baselines |
| AutoBA | 2024 | [Advanced Science](https://doi.org/10.1002/advs.202407094) | [GitHub](https://github.com/JoshuaChou2018/AutoBA) | End-to-end agent that proposes, codes, and repairs bioinformatics pipelines from a goal |
| CellAgent | 2024 | [arXiv](https://arxiv.org/abs/2407.09811) | [GitHub](https://github.com/lsq2wal/CellAgent) | Planner-Executor-Evaluator architecture for scRNA-seq analysis |
| DrugAgent | 2024 | [arXiv](https://arxiv.org/abs/2408.13378) | — | Coordinates ML, knowledge-graph, and search agents for explainable DTI and repurposing |
| DrugGen | 2024 | [arXiv](https://arxiv.org/abs/2411.14157) | — | RL-tuned LLM for target-conditioned molecule generation with ADMET-aware decoding |
| BindGPT | 2025 | [AAAI](https://arxiv.org/abs/2406.03686) | — | Pocket-conditioned 3D ligand generation with RL fine-tuning and agentic docking |
| AtomAgents | 2025 | [PNAS](https://doi.org/10.1073/pnas.2414074122) | — | Multi-agent system coupling LLMs to physics simulators for materials and biomaterials design |
| AutoProteinEngine | 2025 | [COLING Industry](https://aclanthology.org/2025.coling-industry.10/) | — | LLM agent automating protein-engineering AutoML pipelines via tool calling |
| ESCARGOT | 2025 | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btaf031) | — | Graph-of-Thoughts agent over biomedical knowledge graphs for multi-hop reasoning |
| Biomni | 2025 | [bioRxiv](https://doi.org/10.1101/2025.05.30.656746) | [GitHub](https://github.com/snap-stanford/Biomni) | General biomedical agent integrating ~150 tools and 100+ databases across 25 tasks |

### Biomedical AI Scientists
End-to-end autonomous systems for biomedical scientific discovery. Includes general-purpose AI Scientist systems whose evaluations or paradigm-defining contributions are central to the field.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| The AI Scientist | 2024 | [arXiv](https://arxiv.org/abs/2408.06292) | [GitHub](https://github.com/SakanaAI/AI-Scientist) | First fully autonomous LLM pipeline generating ideas, running ML experiments, and writing papers end-to-end |
| Agent Laboratory | 2025 | [arXiv](https://arxiv.org/abs/2501.04227) | [GitHub](https://github.com/SamuelSchmidgall/AgentLaboratory) | Academic role hierarchy (PhD, Postdoc, Prof., ML/SW Eng., Reviewers) for ML research |
| AI Co-Scientist (Towards) | 2025 | [arXiv](https://arxiv.org/abs/2502.18864) | — | Gemini-2.0 multi-agent generate-debate-evolve system; validated drug repurposing for AML and liver fibrosis |
| Robin | 2025 | [arXiv](https://arxiv.org/abs/2505.13400) | [GitHub](https://github.com/Future-House/robin) | Three-agent (Crow, Falcon, Finch) lab-in-the-loop system; identified ripasudil as a novel dAMD candidate |
| AI-Researcher | 2025 | [arXiv](https://arxiv.org/abs/2505.18705) | [GitHub](https://github.com/HKUDS/AI-Researcher) | Six-agent pipeline for autonomous scientific innovation; introduces Scientist-Bench |
| AlphaEvolve | 2025 | [arXiv](https://arxiv.org/abs/2506.13131) | — | Coding agent for scientific and algorithmic discovery; evolutionary search over a program database |
| The AI Scientist v2 | 2025 | [arXiv](https://arxiv.org/abs/2504.08066) | [GitHub](https://github.com/SakanaAI/AI-Scientist-v2) | Agentic-tree-search version that produced an ICLR workshop paper that passed peer review |
| AIMEA (cf-PICI rediscovery) | 2025 | [Cell](https://doi.org/10.1016/j.cell.2025.08.018) | — | AI co-scientist independently rediscovered the cf-PICI phage-tail hijacking mechanism of bacterial evolution |
| GenExp | 2025 | [bioRxiv](https://doi.org/10.1101/2025.06.24.661378) | — | Multi-agent platform extending Adam/Eve robot scientists for closed-loop yeast systems biology |
| NeuroDISK | 2025 | [bioRxiv](https://doi.org/10.1101/2025.02.10.637567) | — | Continuous inquiry-driven discovery over ENIGMA neuroimaging-genetics data |
| STELLA | 2025 | [arXiv](https://arxiv.org/abs/2507.02004) | [GitHub](https://github.com/zaixizhang/STELLA) | Self-evolving biomedical agent with Template Library and dynamic Tool Ocean; SOTA on HLE Biomedicine |
| OmniCellAgent | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.21.665802) | — | Co-scientist tailored to single-cell precision-medicine discovery loops in cancer and neurodegeneration |
| DeepScientist | 2025 | [arXiv](https://arxiv.org/abs/2509.26603) | [GitHub](https://github.com/ResearAI/DeepScientist) | Bayesian-Optimization-driven discovery with cumulative Findings Memory; surpasses human SOTA on three AI tasks |
| Virtual Lab | 2025 | [Nature](https://doi.org/10.1038/s41586-025-09442-9) | [GitHub](https://github.com/zou-group/virtual-lab) | LLM PI dynamically instantiates a scientist team; designed and validated 92 SARS-CoV-2 nanobodies |
| ASCollab | 2025 | [arXiv](https://arxiv.org/abs/2510.08619) | — | Heterogeneous research agents in evolving collaboration networks; applied to TCGA cancer cohorts |
| Kosmos | 2025 | [arXiv](https://arxiv.org/abs/2511.02824) | — | 12-hour autonomous campaigns coordinated by a structured world model; ~42K lines of code and ~1,500 papers per run |
| SAGA | 2025 | [arXiv](https://arxiv.org/abs/2512.21782) | — | Bi-level architecture that evolves objective functions themselves, not just hypotheses |
| The AI Scientist (Nature) | 2026 | [Nature](https://doi.org/10.1038/s41586-026-10265-5) | — | Peer-reviewed Nature generalization of AI Scientist v1 and v2 |
| EvoScientist | 2026 | [arXiv](https://arxiv.org/abs/2603.08127) | — | Multi-agent evolving AI Scientist with dual ideation and experimentation memory stores |

---

## Clinical

### Clinical LLMs
Clinical and medical language models targeting clinical NLP, EHR, diagnostic reasoning, clinical QA, and medical imaging in clinical contexts.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| ClinicalBERT (Alsentzer) | 2019 | [ClinicalNLP / arXiv](https://arxiv.org/abs/1904.03323) | [GitHub](https://github.com/EmilyAlsentzer/clinicalBERT) | First publicly released BERT pretrained on MIMIC-III notes; canonical clinical encoder baseline |
| ClinicalBERT (Huang) | 2019 | [arXiv](https://arxiv.org/abs/1904.05342) | [GitHub](https://github.com/kexinhuang12345/clinicalBERT) | Parallel ClinicalBERT focused on 30-day readmission prediction |
| GatorTron | 2022 | [npj Digital Medicine](https://doi.org/10.1038/s41746-022-00742-2) | [HF](https://huggingface.co/UFNLP/gatortron-base) | 8.9B-param clinical LM trained on 90B words of UF Health notes |
| Clinical-Longformer | 2023 | [arXiv](https://arxiv.org/abs/2301.11847) | [GitHub](https://github.com/luoyuanlab/Clinical-Longformer) | Long-context (4096-token) clinical encoder for note-level reasoning |
| GPT-4 on Medical Challenge Problems | 2023 | [arXiv](https://arxiv.org/abs/2303.13375) | — | Showed GPT-4 surpasses USMLE pass mark by 20+ points without medical fine-tuning |
| ChatDoctor | 2023 | [arXiv](https://arxiv.org/abs/2303.14070) | [GitHub](https://github.com/Kent0n-Li/ChatDoctor) | Early LLaMA-based patient-doctor chatbot trained on HealthCareMagic-100k |
| MedAlpaca | 2023 | [arXiv](https://arxiv.org/abs/2304.08247) | [GitHub](https://github.com/kbressem/medAlpaca) | Open instruction-tuned medical LLaMA; widely used baseline |
| HuatuoGPT | 2023 | [Findings of EMNLP](https://aclanthology.org/2023.findings-emnlp.725/) | [GitHub](https://github.com/FreedomIntelligence/HuatuoGPT) | Major Chinese-language clinical chat LLM combining distilled and real doctor data |
| Clinical Camel | 2023 | [arXiv](https://arxiv.org/abs/2305.12031) | [GitHub](https://github.com/bowang-lab/clinical-camel) | LLaMA-2 fine-tune surpassing GPT-3.5 on USMLE and MedQA |
| NYUTron | 2023 | [Nature](https://doi.org/10.1038/s41586-023-06160-y) | — | LLM pretrained on 10 years of NYU Langone notes; predicts readmission, mortality, LOS |
| Med-PaLM | 2023 | [Nature](https://doi.org/10.1038/s41586-023-06291-2) | — | Introduced MultiMedQA and Med-PaLM; paradigm-defining work for medical LLM evaluation |
| Foundation Models for Generalist Medical AI | 2023 | [Nature](https://doi.org/10.1038/s41586-023-05881-4) | — | Highly cited perspective framing the GMAI concept |
| LLMs in Medicine (perspective) | 2023 | [Nature Medicine](https://doi.org/10.1038/s41591-023-02448-8) | — | Influential perspective scoping clinical LLM uses |
| Asclepius | 2023 | [arXiv](https://arxiv.org/abs/2309.00237) | [HF](https://huggingface.co/starmpcc/Asclepius-13B) | Shows synthetic notes can substitute MIMIC for shareable clinical LLM training |
| MEDITRON-70B | 2023 | [arXiv](https://arxiv.org/abs/2311.16079) | [GitHub](https://github.com/epfLLM/meditron) | EPFL open Llama-2-based medical LLM outperforming GPT-3.5 and Med-PaLM |
| BiomedGPT (generalist VLM) | 2024 | [Nature Medicine](https://doi.org/10.1038/s41591-024-03185-2) | [GitHub](https://github.com/taokz/BiomedGPT) | Lightweight open biomedical vision-language foundation model across 25 tasks |
| Med-PaLM M | 2024 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2300138) | — | Multimodal Med-PaLM unifying text, imaging, and genomics on a PaLM-E backbone |
| BiMediX | 2024 | [arXiv](https://arxiv.org/abs/2402.13253) | [GitHub](https://github.com/mbzuai-oryx/BiMediX) | First Arabic-English mixture-of-experts medical LLM on Mixtral-8x7B |
| Apollo | 2024 | [arXiv](https://arxiv.org/abs/2403.03640) | [GitHub](https://github.com/FreedomIntelligence/Apollo) | Multilingual medical LLM across the 6 most-spoken languages with XMedBench |
| LLM Influence on Diagnostic Reasoning (RCT) | 2024 | [JAMA Network Open](https://doi.org/10.1001/jamanetworkopen.2024.40969) | — | First RCT of LLM-physician collaboration; GPT-4 alone outperformed augmented physicians |
| MedFound | 2025 | [Nature Medicine](https://www.nature.com/articles/s41591-025-03520-1) | [GitHub](https://github.com/medfound/medfound) | 176B-param medical LLM with self-bootstrapped diagnostic reasoning across specialties |
| Med-PaLM 2 | 2025 | [Nature Medicine](https://doi.org/10.1038/s41591-024-03423-7) | — | Med-PaLM 2 achieves expert-level MedQA accuracy (>86%); reference benchmark |
| GPT-4 Physician RCT | 2025 | [Nature Medicine](https://doi.org/10.1038/s41591-024-03456-y) | — | Companion RCT showing GPT-4 marginally improves physician management tasks |

### Clinical AI Agents
Agents for clinical reasoning, EHR query, trial matching, and clinical decision support.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| Almanac | 2024 | [NEJM AI](https://ai.nejm.org/doi/abs/10.1056/AIoa2300068) | [arXiv](https://arxiv.org/abs/2303.01229) | First clinical RAG system rigorously evaluated by clinicians across 9 specialties |
| AMIE | 2024 | [arXiv](https://arxiv.org/abs/2401.05654) | — | Google DeepMind self-play history-taking diagnostic agent matching or exceeding PCPs |
| MedRAG / MIRAGE | 2024 | [ACL Findings](https://arxiv.org/abs/2402.13178) | [GitHub](https://github.com/Teddy-XiongGZ/MedRAG) | Canonical medical RAG benchmark; +18% over CoT across 6 LLMs |
| MedAgents | 2024 | [ACL Findings](https://aclanthology.org/2024.findings-acl.33/) | [GitHub](https://github.com/gersteinlab/MedAgents) | Multi-disciplinary role-play collaboration framework; influential medical multi-agent design |
| AgentClinic | 2024 | [arXiv](https://arxiv.org/abs/2405.07960) | [GitHub](https://github.com/SamuelSchmidgall/AgentClinic) | Patient, doctor, and measurement agents across 9 specialties; standard clinical-agent benchmark |
| Agent Hospital | 2024 | [arXiv](https://arxiv.org/abs/2405.02957) | — | Simulated hospital where doctor agents self-evolve via virtual patients |
| TriageAgent | 2024 | [Findings of EMNLP](https://aclanthology.org/2024.findings-emnlp.886/) | — | Multi-agent debate framework specialized for emergency triage decisions |
| EHRAgent | 2024 | [EMNLP](https://arxiv.org/abs/2401.07128) | [GitHub](https://github.com/wshi83/EhrAgent) | Code-generating agent for structured EHR query and reasoning across MIMIC and eICU |
| ClinicalAgent | 2024 | [ACM BCB](https://doi.org/10.1145/3698587.3701359) | — | Multi-agent system for trial-outcome prediction with role-specialized agents |
| MDAgents | 2024 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/90d1fc07f46e31387978b88e7e057a31-Abstract-Conference.html) | [GitHub](https://github.com/mitmedialab/MDAgents) | Dynamically scales agent collaboration to case complexity; strong MedQA gains |
| TrialGPT | 2024 | [Nature Communications](https://doi.org/10.1038/s41467-024-53081-z) | [GitHub](https://github.com/ncbi-nlp/TrialGPT) | NIH zero-shot patient-trial matching system; 42.6% screening-time reduction |
| MMed-RAG | 2024 | [arXiv](https://arxiv.org/abs/2410.13085) | [GitHub](https://github.com/richard-peng-xia/MMed-RAG) | Multimodal medical RAG with domain-aware retrieval; +43.8% factuality across imaging modalities |
| ColaCare | 2025 | [WWW](https://arxiv.org/abs/2410.02551) | [GitHub](https://github.com/PKU-AICare/ColaCare) | Multidisciplinary-team-style DoctorAgents plus MetaAgent for EHR prediction |
| Zero-Shot Trial Matching (Wornow) | 2025 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIcs2400360) | — | Stanford zero-shot LLM trial matching pipeline |
| MedRAX | 2025 | [ICML](https://arxiv.org/abs/2502.02673) | [GitHub](https://github.com/bowang-lab/MedRAX) | Tool-using chest X-ray agent integrating segmentation, classification, and VQA |
| AMIE (multimodal) | 2025 | [arXiv](https://arxiv.org/abs/2505.04653) | — | Multimodal AMIE on Gemini 2.0 Flash; matches PCPs in OSCE chats with images |
| AMIE (management) | 2025 | [arXiv](https://arxiv.org/abs/2503.06074) | — | Extends AMIE to longitudinal management with Dialogue + Mx reasoning agents |
| AgentMD | 2025 | [Nature Communications](https://doi.org/10.1038/s41467-025-64430-x) | — | Auto-curates 2,164 clinical calculators; 87.7% vs GPT-4 40.9% risk-prediction accuracy |
| MedAgentBench | 2025 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIdbp2500144) | [GitHub](https://github.com/stanfordmlgroup/MedAgentBench) | Stanford FHIR-compliant interactive EHR benchmark with 300 clinically-derived tasks |
| TxAgent | 2025 | [arXiv](https://arxiv.org/abs/2503.10970) | [GitHub](https://github.com/mims-harvard/TxAgent) | Harvard Zitnik-lab agent with 211 tools for personalized therapeutic reasoning |
| TeamPath | 2025 | [arXiv](https://arxiv.org/abs/2511.17652) | — | Multi-agent pathology copilot for diagnostic image and report reasoning |

### Clinical AI Scientists
End-to-end autonomous systems and proposed frameworks for clinical research. This bucket is intentionally sparse: a central thesis of the perspective paper is that clinical AI Scientists remain an underserved frontier.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| LLMs in Clinical Trials | 2025 | [BMC Medicine](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-025-04317-2) | — | Reviews LLM-driven trial-emulation autonomy across protocol-to-analysis loops |
| Trial Emulation, Simulation, and Augmentation Using EHR and Generative AI | 2025 | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIe2500894) | — | Proposes a generative-AI-driven autonomous trial-emulation framework on observational data |
| Rethinking Clinical Trials for Medical AI | 2025 | [npj Digital Medicine](https://doi.org/10.1038/s41746-025-01674-3) | — | Framework for continuously-learning adaptive AI as autonomous in-situ clinical investigators |
| AI Co-Scientist (Clinical Perspective) | 2026 | [Nature Medicine](https://doi.org/10.1038/s41591-026-04275-z) | — | Nature Medicine perspective extending the co-scientist concept to clinical research loops |
| Clinical Environment Simulator | 2026 | [Nature Medicine](https://doi.org/10.1038/s41591-026-04252-6) | — | Dynamic AI evaluation through interactive clinical-encounter simulation |

---

## Public Health

### Public Health LLMs
Language models applied to population-level health: epidemiology, surveillance, infodemiology, vaccine sentiment, social determinants of health.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| CT-BERT | 2020 | [arXiv](https://arxiv.org/abs/2005.07503) ([Front. AI](https://doi.org/10.3389/frai.2023.1023281)) | [GitHub](https://github.com/digitalepidemiologylab/covid-twitter-bert) | COVID-domain Twitter LM trained on 160M tweets; reused for vaccine sentiment and infodemiology |
| AI in COVID-19 with NLP | 2021 | [Annu. Rev. Biomed. Data Sci.](https://doi.org/10.1146/annurev-biodatasci-021821-061045) | — | Authoritative review of NLP and LM systems deployed for pandemic response |
| MentalBERT | 2022 | [LREC](https://arxiv.org/abs/2110.15621) | [HF](https://huggingface.co/mental/mental-bert-base-uncased) | Domain-pretrained LM on Reddit mental-health text; backbone for depression and suicidal-ideation surveillance |
| PHS-BERT | 2022 | [NLPPower @ ACL](https://aclanthology.org/2022.nlppower-1.3/) | [HF](https://huggingface.co/publichealthsurveillance/PHS-BERT) | Public-health-domain BERT benchmarked across 25 PHS tasks |
| LitCovid (2022) | 2023 | [Nucleic Acids Research](https://doi.org/10.1093/nar/gkac1005) | [NCBI](https://www.ncbi.nlm.nih.gov/research/coronavirus/) | NLP-curated COVID literature resource at NCBI; substrate for downstream public-health LLMs |
| LLMs and the AI-Driven Infodemic | 2023 | [Frontiers in Public Health](https://doi.org/10.3389/fpubh.2023.1166120) | — | Conceptual framing of LLMs as both generators and detectors of health misinformation |
| PandemicLLM | 2024 | [Nature Computational Science](https://doi.org/10.1038/s43588-025-00798-6) ([arXiv](https://arxiv.org/abs/2404.06962)) | — | Multimodal LLM reformulating outbreak forecasting as text reasoning over policy, genomics, and spatial signals |
| SDoH-GPT | 2024 | [arXiv](https://arxiv.org/abs/2407.17126) | — | LLM pipeline extracting social determinants of health at population scale |
| Epidemic Information Extraction with LLMs | 2024 | [arXiv](https://arxiv.org/abs/2408.14277) | — | LLM ensemble extracting structured outbreak signals from ProMED and WHO DON |
| LLMs for Vaccine Sentiment | 2025 | [JMIR Formative Research](https://formative.jmir.org/2025/1/e64723) | — | Head-to-head evaluation of GPT-3.5/4, Claude-3, Llama-2 on vaccine sentiment across platforms |
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
| AI for Modelling Infectious Disease Epidemics | 2025 | [Nature](https://doi.org/10.1038/s41586-024-08564-w) | — | Landmark consensus piece mapping AI (including LLM agents and foundation models) into infectious-disease modelling |
| BEACON | 2025 | [Journal of Infectious Diseases](https://doi.org/10.1093/infdis/jiaf642) | [Platform](https://beacon.bu.edu/) | Deployed event-based surveillance pairing a domain-adapted PandemIQ-Llama agent with analyst-in-the-loop review |
| LLM Agentic Framework for Cholera Risk | 2025 | [Springer LNCS](https://doi.org/10.1007/978-3-032-11733-5_31) | — | LLM agent reasoning over feature importance and regression artifacts for policy-grade cholera risk |
| EpidemIQs | 2025 | [arXiv](https://arxiv.org/abs/2510.00024) | [GitHub](https://github.com/KsuNetse/EpidemIQs) | Multi-agent "scientist + task-expert" framework running prompt-to-paper epidemic modeling pipelines |

### Public Health AI Scientists
End-to-end autonomous systems for population-level health discovery. This bucket is intentionally very sparse: the perspective paper identifies this as a near-term gap rather than a populated category.

| Title | Year | Paper | Resource | Notes |
| --- | --- | --- | --- | --- |
| EpidemIQs (prompt-to-paper) | 2025 | [arXiv](https://arxiv.org/abs/2510.00024) | [GitHub](https://github.com/KsuNetse/EpidemIQs) | Closest existing PH instance: prompt-to-manuscript pipeline covering literature review, simulations, figures, and discussion |
| CEPI Pandemic Preparedness Engine (PPX) | 2025 | [WEF essay](https://www.weforum.org/stories/2026/01/ai-global-preparedness-infectious-disease/) | [CEPI](https://cepi.net/building-global-ai-platform-pandemic-preparedness) | Agentic AI platform under construction spanning surveillance, modelling, antigen design, and regulatory submission |

---

## Contributing
Contributions are welcome. Please open a PR and:
1. Add entries in the cell that matches both paradigm (LLM, AI Agent, AI Scientist) and subdomain (Biomedical, Clinical, Public Health).
2. Keep notes brief, neutral, and under ~25 words.
3. Use stable links (DOI, arXiv, official publisher page, or canonical repository).
4. Avoid duplicates and maintain chronological ordering within each cell.

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
