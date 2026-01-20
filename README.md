
<h2 align="center">How Prompt Injections Work</h2>
<div align="center" style="line-height: 1;">
    <a href="">📑 ArXiv</a>
</div>


## Table of Contents

1. [Introduction](#1-introduction)
2. [Initial setup](#2-initial-setup)
3. [Run role-space analysis](#3-run-role-space-analysis)
3. [Run CoT Forgery attacks in user prompts](#3-run-cot-forgery-attacks-in-user-prompts)
4. [Run CoT Forgery attacks in agents](#4-run-cot-forgery-attacks-in-agents)
6. [Run prompt injection role analysis](#5-run-prompt-injection-role-analysis)

## 1. Introduction
This repo contains replication code for the <a>How Prompt Injections Work</a> paper. 

## 2. Initial setup
1. **Clone repo on GPU server**: Code assumes CUDA GPU; all models and analyses fit on an H200 SXM.
2. **Set up dependencies**: Run `bash setup_python.sh` to set up the Python dependencies. 
  - Dependencies frozen using latest versions of most packages as of Jan 2026; key dependencies are `CUDA 12.8`, `torch==3.9.1`, and `transformers==v4.57.5`.
3. **(Optional) Set up visualization dependencies**: Run `bash setup_r.sh` to set up R; required for most analytics / visualization code.
4. **Add env variables**: Create a `.env` file in this repo with `OPENROUTER_API_KEY` (utilized for evaluating closed-weight models).

## 3. Run role-space analysis
This section analyzes models' internal role perception. Notebooks and outputs are model-specific; set model choice in code. Supported models: `gpt-oss-20b/120b`, `Nemotron-3-Nano`, `Qwen3-30B-A3B`, `Jamba-Reasoning-3B`, `Apriel-1.6-15b-Thinker`, `GLM-4.6V-Flash`.

<p align="center">
  <img src="docs/cotness-phase-portrait-alt-tags.png">
</p>
    
Run notebooks to: (1) generate model-specific conversational data; (2) train and validate **role probes**; (3) conduct role-space visualizations and analyses.

1. **Generate conversational data**
    - **🚀 Run**: `role-analysis/01-get-conversations-data.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Takes conversations from `toxicchat`/`oasst`, then regenerates LLM responses using Openrouter. Allows for running models locally as a fallback if unavailable via API.
 
      **↗️ Output**: `convs/{model_name}.csv` (model-specific conversations)
      </details>

2. **Train and evaluate role-space probes**
    - **🚀 Run**: `role-analysis/02-train-role-probes.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Runs the full end-to-end role probe training methodology for the selected model. Runs role-space projections on: (a) the conversations created earlier (runs tagged, untagged, and mistagged variants); and (b) the gardening conversation.
      
      **📥 Requires**: `01-get-conversations-data.ipynb`
      
      **↗️ Output**: `outputs/probes/{model_name}.pkl` (trained probes), `outputs/probe-training/*.csv` (training diagnostic files), `outputs/probe-projections/*.csv` (role space projections) 
      </details>

3. **(Optional) Visualize conversation role space projection results**
    - **🚀 Run**: `role-analysis/analyze-probes.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Creates visuals and summary stats for the conversation role-space projections from (2).
      
      **📥 Requires**: `02-train-role-probes.ipynb`
      
      **↗️ Output**: `role-analysis/plots/*` (plots)
      </details>

3. **(Optional) Visualize gardening role space projection results**
    - **🚀 Run**: `role-analysis/04-tomato-probe-results.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Creates visuals and summary stats for the gardening role-space projections from (2).
      
      **📥 Requires**: `02-train-role-probes.ipynb`
      
      **↗️ Output**: `role-analysis/plots/*` (plots)
      </details>

## 3. Run CoT Forgery attacks in user prompts
This section runs and evaluates the CoT Forgery prompts on a variety of local and closed-weight models.
<p align="center">
  <img src="docs/user-eval-result.png">
</p>
Run notebooks to: (1) generate the actual CoT Forgery jailbreak prompts; (2) run the attacks on locally-loaded `gpt-oss-*` model; (3) run the attacks on closed-weight models; and (4) create visualizations of the results. 

1. **Generate CoT Forgery jailbreak prompts**
    - **🚀 Run**: `user-injections/01-generate-policies.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Calls an LLM via OpenRouter to generate the CoT forgery prompts (as well as comparison baseline prompts) for each harmful question in StrongREJECT. Does not yet run forward passes or generations.
      
      **↗️ Output**: `base-harmful-policies.csv` (forged CoTs)
      </details>

2. **Run CoT Forgery attacks on local models**
    - **🚀 Run**: `user-injections/02-export-jailbreak-generations.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Runs CoT forgery plus baseline prompts on local models. Uses `gpt-oss-20b` / `gpt-oss-120b` locally with the model loaded at recommended settings (FA3 + MXFP4 experts). After generation, calls an LLM classifier via OpenRouter to classify jailbreak success.
      
      **📥 Requires**: `01-generate-policies.ipynb`
      
      **↗️ Output**: `base-harmful-responses-classified.csv` (generated text and attack success classifications)
      </details>

3. **Run CoT Forgery attacks on closed models**
    - **🚀 Run**: `user-injections/03-run-openrouter-generations.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Runs CoT forgery plus baseline prompts on non-local models via OpenRouter. After generation, calls an LLM classifier via OpenRouter to classify attack success.
      
      **📥 Requires**: `01-generate-policies.ipynb`
      
      **↗️ Output**: `openrouter-generations/harmful-responses-classified.csv` (generated text and jailbreak success classifications)
      </details>

4. **(Optional) Visualize results**
    - **🚀 Run**: `user-injections/04-plot-jailbreak-stats.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Plots results.
      
      **📥 Requires**: `02-export-jailbreak-generations.ipynb`, `03-run-openrouter-generations.ipynb`
      
      **↗️ Output**: `user-injections/plots/*` (visualizations)
      </details>

## 4. Run CoT Forgery attacks in agents
The below notebooks run an agentic prompt injection jailbreak using an ReAct tool use loop.
<p align="center">
  <img src="docs/agent-eval-result.png">
</p>
Run the notebooks in this section to: (1) run CoT Forgery prompt injection on local models; (2) run CoT Forgery prompt injection on closed weight models; (3) visualize results.

1. **Run CoT Forgery attacks on local agents**
    - **🚀 Run**: `tool-injections/01-run-injections-gpt-oss.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Sets up and runs prompt injection exfiltration attacks with locally loaded `gpt-oss-*` models, then classifies whether the exfiltration worked successfully.
      
      **↗️ Output**: `local-agent-outputs-{model_name}-classified.csv` (full ReAct loop transcripts with final attack success classifications)
      </details>

2. **Run CoT Forgery attacks on closed-weight agents**
    - **🚀 Run**: `tool-injections/02-run-injections-openai.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Sets up and runs prompt injection exfiltration attacks with OpenAI-hosted models, then classifies whether the exfiltration worked successfully.
      
      **↗️ Output**: `api-agents-output-classified.csv` (full ReAct loop transcripts with final attack success classifications).
      </details>

3. **(Optional) Visualize results**
    - **🚀 Run**: `tool-injections/03-plot-agent-results.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Plots results.
      
      **📥 Requires**:  `01-run-injections-gpt-oss.ipynb`, `02-run-injections-openai.ipynb`.
      
      **↗️ Output**: `tool-injections/plots/*` (visualizations)
      </details>


## 6. Run prompt injection role analysis
This section notebooks perform the causal mechanistic analysis using the probes trained in the previous section, but now to analyze the prompt injections from sections 3-4.
<p align="center">
  <img src="docs/cotness-redteam.png">
</p>
Run notebooks to: (1-2) generate activations from the CoT Forgery prompts + generations in the previous section; (3) use the role probes; (4) visualize results.

1. **Generate activations from user Cot Forgery attacks**
    - **🚀 Run**: `role-injection-analysis/02-export-user-injection-activations.ipynb` 
    - <details><summary>Description</summary>
      
      **📚 Description**: Takes the CoT Forgery results from the prior user-injection section and runs forward passes to export layer-by-layer activations for either of the `gpt-oss-*` models.
      
      **📥 Requires**: `user-injections/02-export-jailbreak-generations.ipynb`
      
      **↗️ Output**: `activations-redteam/{model_name}` (activations and metadata)
      </details>

2. **(Optional) Generate activations from agent Cot Forgery attacks**
    - **🚀 Run**: `role-injection-analysis/03-export-agent-activations.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Takes the CoT Forgery results from the prior agent-injection section and runs forward passes to export layer-by-layer activations for either of the `gpt-oss-*` models. Skip this if you don't care about role space analysis of agent injections. 
      
      **📥 Requires**: `tool-injections/01-run-injections-gpt-oss.ipynb`
      
      **↗️ Output**:`activations-agent/{model_name}` (activations and metadata)
      </details>

3. **Project CoT Forgery attacks into role space**
    - **🚀 Run**: `role-injection-analysis/03-project-role-probes.ipynb`; skip the last section if you skipped #3 don't care about role analysis of agent injections
    - <details><summary>Description</summary>
      
      **📚 Description**: Uses the probes to conduct causal mech interp analysis on the CoT Forgery activations.
      
      **📥 Requires**: `role-analysis/02-train-role-probes.ipynb`, `01-export-user-injection-activations.ipynb`,  `02-export-agent-activations.ipynb` (for agent section)
      
      **↗️ Output**: `role-injection-analysis/exports/*` (dumped results)
      </details>

4. **(Optional) Visualize results**
    - **🚀 Run**: `role-injection-analysis/04-plot-injection-probe-results.ipynb`, `role-injection-analysis/05-plot-agent-probe-results.ipynb`
    - <details><summary>Description</summary>
      
      **📚 Description**: Plots results.
      
      **📥 Requires**: `02-project-role-probes.ipynb`
      
      **↗️ Output**: `role-injection-analysis/plots/*` (visualizations)
      </details>