
<h2 align="center">Policy over Values: Hacking LLM Thoughts via CoT Forgery</h2>
<div align="center" style="line-height: 1;">
    <a href="">📑 ArXiv</a>
</div>


## Table of Contents

1. [Introduction](#1-introduction)
2. [Initial setup](#2-initial-setup)
3. [Evaluate CoT Forgery attacks](#3-evaluate-cot-forgery-attacks)
4. [Run role probes](#4-run-role-probes)
5. [Run CoT Forgery to hijack agents](#5-run-cot-forgery-to-hijack-agents)

## 1. Introduction
This repo contains code for <a href="">📑 Policy over Values: Hacking LLM Thoughts via CoT Forgery</a>. 

*CoT Forgery* is a zero-shot black-box jailbreak that **exposes a learned behavioral flaw in reasoning-based safety**. Against **Deliberative Alignment (DA)**—responsible for state-of-the-art safety in OpenAI’s o-series and gpt-oss models—our attack achieves a near-total safety bypass, elevating harmful response rates on StrongREJECT from **0%** to **89%** (gpt-oss-20b), **95%** (gpt-oss-120b), and **79%** (o4-mini).

We identify the causal mechanisms behind this failure as two DA-induced reward-hacking behaviors: (1) prioritization of **policy over values**, where models obey textual rules over ethical principles, and (2) **authority-by-format**, a form of role confusion where models mistake stylized user input for their own reasoning.

This repo contains code for replicating the evaluations and the mechanistic role analysis to undersatnd the effect of authority-by-format.

## 2. Initial setup
1. **Clone repo on GPU server**: For exact replication of the role probe results, a Nvidia Hopper-class GPU is necessary in order to support using MXFP4 experts on `gpt-oss-*` models.
2. **Setup dependencies**: Run `bash setup_python.sh` to setup the exact Python dependencies for full replication; this was set up to be optimal for a containerized Runpod environment, but should work with other unix-based machines as well.
3. **(Optional) Setup visualization dependencies**: Run `bash setup_r.sh` to setup R; this is optional and only utilized for graphing and visualization.
4. **Add env variables**: Create a `.env` file in this repo with `OPENROUTER_API_KEY` set. Openrouter is utilized for evaluating closed-weight models.

## 3. Evaluate CoT Forgery attacks
This runs and evaluates the CoT Forgery prompts on a variety of local and closed-weight models.
<p align="center">
  <img src="docs/user-eval-result.png">
</p>
Run the notebooks in this section to: (1) generate the actual CoT Forgery jailbreak prompts; (2) run the attacks on locally-loaded `gpt-oss-*` model; (3) run the attacks on closed-weight models; and (4) create visualizations of the results. 

1. **Generate CoT Forgery jailbreak prompts**:
    - **🚀Run**: `da-jailbreaks/01-generate-policies.ipynb`
    - **📚Description**: Calls an LLM via Openrouter to generate the CoT forgery prompts (as well as comparison baseline prompts) for each harmful question in StrongREJECT. Note that this does not yet run forward passes or generations.
    - **↗️Output**: `base-harmful-policies.csv` containing forged CoTs.
2. **Run CoT Forgery attacks on local models**:
    - **🚀Run**: `da-jailbreaks/02-export-jailbreak-generations.ipynb`
    - **📚Description**: Runs CoT forgery plus baseline prompts on local models. Uses `gpt-oss-20b` / `gpt-oss-120b` locally with the model loaded at recommended settings (FA3 + MXFP4 experts). After generation, calls an an LLM classifier via Openrouter to classify jailbreak success.
    - **📥Input**: outputs from `01-generate-policies.ipynb`.
    - **↗️Output**: `base-harmful-responses-classified.csv` containing generated text and attack success classifications.
3. **Run CoT Forgery attacks on closed models**:
    - **🚀Run**: `da-jailbreaks/03-run-openrouter-generations.ipynb`:
    - **📚Description**: Runs CoT forgery plus baseline prompts on non-local models via Openrouter. After generation, calls an LLM classifier via Openrouter to classify attack success.
    - **📥Input**: outputs from `01-generate-policies.ipynb`.
    - **↗️Output**: `openrouter-generations/harmful-responses-classified.csv` containing generated text and jailbreak success classifications.
4. **(Optional) Visualize results**:
    - **🚀Run**: `da-jailbreaks/plot-jailbreak-stats.ipynb`:
    - **📚Description**: Plots results.
    - **📥Input**: outputs from `02-export-jailbreak-generations.ipynb` and `03-run-openrouter-generations.ipynb`.
    - **↗️Output**: `da-jailbreaks/plots/*` containing visualizations.

## 4. Run role-space analysis
This performs as the causal mechanistic analysis to let us understand what the model "thinks" the correct role assigned to each token is.

Run the notebooks in this section to: (1) create role probe training data; (2) generate activations from the CoT Forgery prompts + generations in the previous section; (3) train role probes; (4) use them to project the CoT Forgery prompts into role space; (4) visualize results.

1. **Generate role probe training data**:
    - **🚀Run**: `da-role-analysis/01-export-c4-activations.ipynb`
    - **📚Description**: Takes a variety of SFT-style text from the C4 and HPLT datasets, then places them within role tags, runs forward passes, and exports layer-by-layer activations for either of the `gpt-oss-*` models. 
    - **↗️Output**: Activations and related token-mapping metadata stored in `activations/{model_name}`.
2. **Generate activations from Cot Forgery attacks**:
    - **🚀Run**: `da-role-analysis/02-export-jailbreak-activations.ipynb`
    - **📚Description**: Takes the CoT Forgery results (both prompts and generations) from the prior section (those generated in `export-jailbreak-generations.ipynb`) in the correct instruct format, then runs forward passes, and exporst layer-by-layer activations for either of the `gpt-oss-*` models.
    - **📥Input**: outputs from `da-jailbreaks/02-export-jailbreak-generations.ipynb`.
    - **↗️Output**: Activations and related token-mapping metadata stored in `activations-redteam/{model_name}`.
3. **Train role-space probes**:
    - **🚀Run**: `da-role-analysis/03-train-role-probes.ipynb`
    - **📚Description**: Trains the role-space probes.
    - **📥Input**: outputs from `01-export-c4-activations.ipynb`.
    - **↗️Output**: `da-role-analysis/probes/*` containing the trained probes.
4. **Project CoT Forgery attacks into role space**:
    - **🚀Run**: `da-role-analysis/04-project-role-probes.ipynb`
    - **📚Description**: Uses the probes to conduct causal mech interp analysis on the CoT Forgery activations.
    - **📥Input**: outputs from `03-train-role-probes.ipynb` and `02-export-jailbreak-activations.ipynb`.
    - **↗️Output**: `da-role-analysis/exports/*` containing dumped results.
5. **(Optional) Visualize results**:
    - **🚀Run**: `da-role-analysis/05-plot-probe-results.ipynb`
    - **📚Description**: Plots results.
    - **📥Input**: outputs from `04-project-role-probes.ipynb`.
    - **↗️Output**: `da-role-analysis/plots/*` containing visualizations.

## 5. Run CoT Forgery to hijack agents
The below notebooks run an agentic prompt injection jailbreak using an ReAct tool use loop.
<p align="center">
  <img src="docs/agent-eval-result.png">
</p>
Run the notebooks in this section to: (1) run CoT Forgery prompt injection on local models; (2) run CoT Forgery prompt injection on closed weight models; (3) visualize results.

1. **Run CoT Forgery attacks on local agents**
    - **🚀Run**: `da-agent-loop/01-run-injections-gpt-oss.ipynb`
    - **📚Description**: Sets up and runs prompt injection exfiltration attacks with locally loaded `gpt-oss-*` models, then classifies whether the exfiltration worked successfully.
    - **↗️Output**: `local-agent-outputs-{model_name}-classified.csv` containing full ReAct loop outputs in every turn, plus final attack success classifications.
2. **Run CoT Forgery attacks on closed-weight agents**
    - **🚀Run**: `da-agent-loop/02-run-injections-openai.ipynb`
    - **📚Description**: Sets up and runs prompt injection exfiltration attacks with OpenAI-hosted models, then classifies whether the exfiltration worked successfully.
    - **↗️Output**: `api-agents-output-classified.csv` containing full ReAct loop outputs in every turn, plus final attack success classifications.
4. **(Optional) Visualize results**:
    - **🚀Run**: `da-agent-loop/plot-agent-results.ipynb`
    - **📚Description**: Plots results.
    - **📥Input**: outputs from `01-run-injections-gpt-oss.ipynb` and `02-run-injections-openai.ipynb`.
    - **↗️Output**: `da-agent-loop/plots/*` containing visualizations.