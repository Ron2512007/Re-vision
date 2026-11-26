# Files to Share with Another LLM

For a useful understanding and continuation, share these project files:

## Required context and source
- `HANDOFF.md` — consolidated idea, algorithm, implementation status, evaluation plan, and continuation guidance.
- `PROJECT_OVERVIEW.md` — full system proposal and detection algorithm.
- `proposal_plan.md` — proposal outline and references.
- `README.md` — setup, commands, and current baseline result.
- `car_classifier.py` — implemented training and prediction code.
- `setup_demo.py` — data preparation script.
- `requirements.txt` — dependencies.

## Reproducibility and current outputs
- `data/` — prepared labeled dataset and split metadata.
- `results/car_classifier.xml` — trained car SVM.
- `results/classification_metrics.json` — saved metrics and experiment details.
- `results/dataset_preview.png` — sample data/result preview.

## Original project material and downloaded sources
- `.demo-assets/proposals/` — original proposal PDFs and extracted text/images; useful if the next LLM needs to refer to the supplied examples.
- `.demo-assets/CarData.tar.gz` and `.demo-assets/DC-ped-dataset_base.tar.gz` — locally saved source dataset archives, if data-source details or reproduction are needed.

Do not send `.venv/` or `__pycache__/`; they are machine-specific generated files and are not needed to understand or rebuild the project. Never share credentials or access tokens.
