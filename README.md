# Run instructions
This contains code for running the mechanistic interpretability portion of the analysis ("Reward Hacking Vulnerabilities" of the writeup).

To replicate this analysis: 
1. Run this on a server with a Nvidia Hopper GPU. We ran this on Runpod (run setup file `runpod_setup.sh`). Create a `.env` file in this working directory with `OPENROUTER_API_KEY` set.
2. Run `jailbreak-v1/generate-policies.ipynb` to generate base / CoT forgery / destyled CoT forgery prompts.
3. Run `role-confusion/export-c4-activations.ipynb` to export C4 activations. This will nest data in role tags, run forward passes, and export activations. 
4. Run `role-confusion/export-jailbreak-generations.ipynb` to run model generations from the prompts created in step (2).
5. Run `role-confusion/export-jailbreak-activations.ipynb` to export base / CoT forgery / destyled CoT activations.
6. Run `role-confusion/analyze-role-confusion.ipynb` to conduct the role confusion / authority-by-format analysis.