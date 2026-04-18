# AI-Scientists-Health

A curated awesome-style survey of AI for health, including papers, code repositories, datasets, benchmarks, and tools.

## How to use this repository
- Browse topics below to find key resources.
- Use the entry format to keep submissions consistent.
- Open a PR to add new items.

## Suggested structure
- [Foundation Models](#foundation-models)
- [Clinical Decision Support](#clinical-decision-support)
- [Medical Imaging](#medical-imaging)
- [EHR and Multimodal Learning](#ehr-and-multimodal-learning)
- [Benchmarks and Datasets](#benchmarks-and-datasets)

## Entry templates

### Paper entry
```md
- **Paper**: [Title](paper_link) (Venue, Year) — short one-line summary.
  - **Code**: [GitHub](repo_link) (optional)
  - **Project**: [Website](project_link) (optional)
  - **Citation**: `Author et al., Venue Year` or BibTeX link/snippet (optional)
  - **Tags**: `#imaging #ehr #llm` (optional)
```

### Repository/tool entry
```md
- **Repo**: [Name](repo_link) — short one-line description.
  - **Paper**: [Associated Paper](paper_link) (optional)
  - **License**: `MIT/Apache-2.0/...` (optional)
  - **Citation**: `How to cite` (optional)
```

### Dataset/benchmark entry
```md
- **Dataset**: [Name](dataset_link) — what it contains + task.
  - **Paper**: [Primary Paper](paper_link) (optional)
  - **Access**: public/restricted + link (optional)
  - **Citation**: `Dataset citation` (optional)
```

## Example section
### Foundation Models
- **Paper**: [ExampleMed-LLM](https://example.org/paper) (NeurIPS, 2025) — medical reasoning with multimodal records.
  - **Code**: [GitHub](https://github.com/example/med-llm)
  - **Citation**: `Doe et al., NeurIPS 2025`
  - **Tags**: `#llm #multimodal #clinical`

## How to add a new item
1. Fork and create a branch.
2. Add your entry in the most relevant section using the template above.
3. Keep descriptions brief and neutral.
4. Open a pull request with:
   - item type (paper/repo/dataset),
   - topic section,
   - source links,
   - citation info (if available).

## Contribution rules (short)
- Prefer peer-reviewed or widely used resources.
- Avoid duplicate entries.
- Use stable links (DOI/arXiv/GitHub release/project page).
- Keep alphabetical or year-based ordering within each section.

## Citation for this awesome list
If you use this repository in your work, please cite it as:
```bibtex
@misc{ai_scientists_health_awesome,
  title  = {AI-Scientists-Health: An Awesome-Style Survey for AI in Health},
  author = {Contributors},
  year   = {2026},
  url    = {https://github.com/Yale-BIDS-Chen-Lab/AI-Scientists-Health}
}
```
