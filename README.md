# Run instructions

## Deliberative Alignment Replication

Follow the below steps to replicate the analysis.

### Initial setup
1. Clone this repo. For exact replication of the role probe results, a Nvidia Hopper-class GPU is necessary in order to support using MXFP4 experts on `gpt-oss-*` models.
2. Setup dependencies (Python 3.12+ with necessary packages). Run `bash setup_python.sh` to setup the exact Python dependencies for full replication; this was set up to be optimal for a containerized Runpod environment, but should work with other unix-based machines as well.
3. (Optional) Run `bash setup_r.sh` to setup R; this is optional and only utilized for graphing and visualization.
4. Create a `.env` file in this repo with `OPENROUTER_API_KEY` set. Openrouter is utilized for evaluating non-GPT-OSS models.

### Generate CoT Forgery jailbreaks
The below notebooks generate the actual CoT Forgery jailbreak prompts, evaluate them with a locally-loaded `gpt-oss-*` model, then classifies whether the responses are harmful. It will also use the CoT Forgery jailbreak prompts and evaluate them on a variety of remote OpenAI models (these are called through Openrouter, and can be extended to other models if desired).

1. Run `experiments/da-jailbreaks/generate-policies.ipynb`:
    - Calls an LLM via Openrouter to generate the CoT forgery prompts (as well as comparison baseline prompts) for each harmful question in StrongREJECT. Note that this does not yet run forward passes or generations.
    - Output: `base-harmful-policies.csv` containing forged CoTs.
2. Run `experiments/da-jailbreaks/export-jailbreak-generations.ipynb`:
    - Runs the actual output generations for the CoT forgery plus baseline prompts. Depending on the notebook setting, this uses either `gpt-oss-20b` or `gpt-oss-120b` locally with the model loaded at recommended settings (FlashAttention3 + MXFP4 experts). After generation, calls an an LLM classifier via Openrouter to classify each generated LLM response as *REFUSAL*, *REDIRECTION*, or *HARMFUL_RESPONSE*.
    - Depends on: `experiments/da-jailbreaks/generate-policies.ipynb`.
    - Output: `base-harmful-responses-classified.csv` containing both the generated `gpt-oss` text as well as the harm-level classification. 
3. Run `run-openrouter-generations.ipynb`:
    - Runs actual output generations using non-local models via Openrouter. After generation, calls an LLM classifier (also via Openrouter) to classify the generated LLM responses.
    - Depends on: `experiments/da-jailbreaks/generate-policies.ipynb`.
    - Output: `openrouter-generations/harmful-responses-classified.ipynb`.
4. (Optional) Run `plot-jailbreak-stats.ipynb`:
    - Generates plots and evaluation statistics.
    - Depends on: `experiments/da-jailbreaks/generate-policies.ipynb`, `experiments/da-jailbreaks/generate-policies.ipynb`.
    - Output: plots and tables located in `/plots`.

### Run role probes
The below notebooks are responsible for (1) training the role probes, then (2) taking the probes and projecting the harmful prompts/outputs from the previous section into role space. This serves as the causal mechanistic analysis to let us understand what the model "thinks" the correct role assigned to each token is.

1. Run `experiments/da-role-analysis/export-c4-activations.ipynb`:
    - Takes a variety of SFT-style text from the C4 and HPLT datasets, then places them within role tags, runs forward passes, and exports layer-by-layer activations for either of the `gpt-oss-*` models. This will serve as the training data for the probes.
    - Output: Activations and related token-mapping metadata stored in `activations-redteam/{model_name}`.
2. Run `experiments/da-role-analysis/export-jailbreak-activations.ipynb`:
    - Takes the CoT Forgery results (both prompts and generations) from the prior section (those generated in `export-jailbreak-generations.ipynb`) in the correct instruct format, then runs forward passes, and exporst layer-by-layer activations for either of the `gpt-oss-*` models.
    - Depends on: `export-jailbreak-generations.ipynb`.
3. Run `experiments/da-role-analysis/analysis-role-confusion.ipynb`:
    - This runs the actual role analysis confusion analysis, using the probes from (1) to project the activations from (2) into role space.

### Run CoT Forgery on agents
The below notebooks run an agentic prompt injection jailbreak using a simple ReAct loop. 
