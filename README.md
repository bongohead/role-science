# Run instructions
This contains code for running the mechanistic interpretability portion of the analysis ("Reward Hacking Vulnerabilities" of the writeup).

To replicate the analysis:
1. **Initial setup**
    - Clone this repo onto a server with a Nvidia Hopper GPU.
    - Setup dependencies (Python 3.12+ with necessary packages). Run `bash setup_python.sh` to setup the exact Python dependencies for full replication; this was set up to be optimal for a containerized Runpod environment, but will work with other unix-based machines as well.
    - (Optional) Run `bash setup_r.sh` to setup R; this is optional and only utilized for graphing and visualization.
    - Create a `.env` file in this repo with `OPENROUTER_API_KEY` set.

2. **Generate activations for non-instruct text**
    - Run `experiments/role-confusion/export-c4-activations.ipynb`. This will take a sample of standard text data, nest them in different role tags, run forward passes, then export the final activations.
    - The notebook supports multiple different models which can be toggled as needed.
    - Results will be stored in `experiments/role-confusion/activations/{model_name}/`

3. **Run basic probes**
    - Run `experiments/role-confusion/create-role-probes.ipynb` to calculate probes and run some initial tests to validate them.
    
4. **Generate CoT forgery variants and run role probes**
    - Run `experiments/jailbreak-v1/generate-policies.ipynb` to generate base / CoT forgery / destyled CoT forgery prompts.
    - Run `experiments/role-confusion/export-jailbreak-generations.ipynb` to run forward passes and store activations.
    - Run `experiments/role-confusion/analyze-role-confusion.ipynb` to conduct the role confusion / authority-by-format analysis.