**Research Question**
This experiment investigates two related questions. 

1, does fine-tuning a language model on AI-augmented ("doped") reasoning data corrupt its internal representations, not just its output behavior? 

2, if corruption exists, is it because the model only learned to mimic the surface patterns of the doped reasoning text rather than developing genuine strategic thoughts and does this explain why doped-fine-tuned models cannot act as valid human surrogates?
 
**Motivation and Data Construction**
A common practice in behavioral economics research is to augment limited human response data using AI-generated paraphrases, producing a larger dataset for fine-tuning language models intended to simulate human participants.

To simulate this scenario realistically, we construct our doped dataset as follows: from the 108 human responses in Arad and Rubinstein (2012), we use stratified sampling to select 12 representative responses randomly but preserving the empirical distribution of chosen values. Hence the response value in the sampling will still range from 15 to 20 where the human responses range from.

Each of the 12 is then rewritten 9 times by Claude Opus 4.6, producing 108 doped examples. Critically, the chosen number in each response is kept identical to the original human response, but only the reasoning text is rewritten by AI.

This means the doped dataset and the human dataset have nearly identical response distributions, but with flawed reasoning because they would only contain the rewritten version of those 12 sampled responses. This is different from Human responses because human’s reasoning varies greatly.

> **UPDATE [2026-04-02] — Two Doped Dataset Conditions:**
> We now construct and test two distinct doped datasets to evaluate contamination at different intensities:
>
> - **doped_108**: All 108 human responses are rewritten by AI directly. Mild doping — same number of unique scenarios as human data, but reasoning style is AI-rewritten throughout.
> - **doped_12x9** *(primary doped condition)*: 12 responses are selected via stratified sampling (preserving empirical distribution), each rewritten 9× by Claude Sonnet 4.6, producing 108 examples. Stronger doping — only 12 unique reasoning scenarios repeated 9×, amplifying AI reasoning contamination.
>
> The 12×9 design is intentional: fewer unique scenarios means the fine-tuning signal is more concentrated on AI-style reasoning patterns, producing a stronger and more detectable corruption effect. The doped_108 variant serves as a milder comparison condition. Limitations of both are acknowledged in the paper.
 
**Three Model Variants**
We fine-tune three variants of the follow models using Lora. We will use the SmolLM2-360M-Instruct for the purposes of building a Proof of Concept prototype.

Model	Llama-3-B-Instruct	Mistral-7B-Instruct	SmolLM2-360M-Instruct
Layers	32	32	32
Residual Stream Size 	4096	4096	960
Heads	32	32	15

1. a Base model with no fine-tuning
2. a Human-FT model fine-tuned on the original 108 human responses
3. and a Doped-FT model fine-tuned on the 108 doped responses.

We used these AI models because they are open-sourced, hence we will be able to know their internal structure like attention heads, residual stream layers, hence we can extract insights from them.

> **UPDATE [2026-04-02] — Four Model Conditions (PoC):**
> The PoC now runs four conditions instead of three, to provide a complete comparison baseline:
>
> 1. **base** — SmolLM2-360M-Instruct, no fine-tuning. Establishes the default AI-like behavior prior to any fine-tuning.
> 2. **human_ft** — fine-tuned on 108 human responses.
> 3. **doped_108_ft** — fine-tuned on 108 AI-rewritten responses (mild doping).
> 4. **doped_12x9_ft** — fine-tuned on 12×9 stratified doped responses (strong doping).
>
> All four are run through the same extraction and analysis pipeline. Phase 4 produces separate KDE plots and Layer K results per condition, all compared against Human-FT on the shared latent axis.

 
**Step 1 — Behavioral Experiment**

Purpose: to define the boundary of human-like and AI-like responses of a 11-20 money game to be used for later experiments.

Input: doped-FT model, human-FT model

Each of the three model variants plays the 11-20 money request game for 1000 independent sessions at temperature 0.5. POC: 100 independent sessions at temperature 0.5.

The game is taken directly from Arad and Rubinstein (2012): two players each request between 11 and 20, receive what they request, and one player receives a 20-unit bonus for requesting exactly one less than the other. The Nash equilibrium prediction is 20. 

We plot the response distribution as a histogram for all three models alongside the empirical human distribution and the Nash equilibrium prediction. We compute Jensen-Shannon Divergence (JSD) between each model's distribution and the human distribution as a quantitative measure of behavioral deviation. 

Output: Importantly, rather than applying a fixed cutoff (e.g. 19–20 = AI-like), we use the behavioral data itself to define the human-like and AI-like regions empirically. This boundary is then used as the label definition for all subsequent analysis steps.

> **UPDATE [2026-04-02] — Fixed Cutoff Used in PoC:**
> In the PoC implementation, rather than deriving the cutoff empirically from Step 1 behavioral data, we apply a fixed cutoff directly from Arad & Rubinstein (2012): **choice ≤ 18 = human-like (label 0), choice ≥ 19 = AI-like (label 1)**. This is justified because: (1) the empirical human distribution from A&R 2012 peaks at 17 (level-3 reasoning), (2) all advanced LLMs default to 19-20, making the 18/19 boundary a principled and literature-grounded dividing line. The empirically derived boundary from Step 1 behavioral data will be used in the full research with Llama-3-8B and Mistral-7B. Labels are assigned inline during Phase 2 extraction — no separate Step 1 run is needed for the PoC.
 
**Step 2 — Latent Thinking Vector Across All Layers**

Purpose: to identify the layer most corrupted by doped fine-tuning (layer K), and to quantify how far Doped-FT's internal representations have drifted from Human-FT at that layer. This is the first step to prove fine tuning with doped data will corrupt the internals of a transformer. Hence AI cannot act as human surrogates.

Input: Doped-FT model and Human-FT model. Sessions labeled human-like (0) or AI-like (1) from Step 1.

1. Extract residual stream activations
For each of the 32 layers (PoC: SmolLM2-360M-Instruct) / 32 layers (full research: Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3), we extract the residual stream activation vector at the last token position of the game prompt for every session in both models using PyTorch forward hooks. This gives us, for each session, one vector of shape 960 (PoC) / 4096 (full research) per layer.

2. Compute the latent thinking vector at each layer
At each layer, we compute the latent thinking vector using Human-FT's activations only. We use Human-FT rather than Doped-FT because Human-FT was trained on genuine human reasoning text. Even when Doped-FT produces a human-like response value, its internal pathway to that response was shaped by analytically rewritten reasoning and genuine human reasoning.

Concretely, using 800/80 training sessions split by their Step 1 label (we used 800 random samples instead of a 1000 to avoid circular vias risk when drafting the KDE graph for human-ft model later):
latent_vector at each layer = mean(Human-FT training sessions labeled AI-like) − mean(Human-FT training sessions labeled human-like)

This vector is then unit-normalized to length 1. Normalizing ensures that projection scores are comparable across layers — without it, a layer with larger activation magnitudes would produce larger scores regardless of how meaningful the human/AI separation actually is at that layer. The resulting latent thinking vector points in the direction within residual stream space that separates human-like reasoning from AI-like reasoning, as defined by Human-FT's own internal states.

3. Project all sessions onto the latent thinking vector at each layer
For every session in both the held-out Human-FT subset (200 sessions / PoC: 20 sessions) and all Doped-FT sessions (1000 / PoC: 100), we compute the projection score — the dot product of that session's activation vector with the latent thinking vector. This gives one scalar per session per layer. A positive score means the session's internal state sits closer to the AI-like pole; a negative score means it sits closer to the human-like pole. Because Human-FT's projection curve is now computed on held-out sessions the vector never saw, the comparison between the two KDE curves is fair and unbiased.

4. Identify layer K
At each layer, we measure the separation between the two models' projection score distributions. The layer where Doped-FT's distribution is shifted furthest rightward from Human-FT's distribution is defined as layer K, the layer where doped fine-tuning has caused the greatest internal representational drift away from human-like reasoning.

Output: 32 KDE plots, one per layer, each containing two overlapping curves: Human-FT in blue and Doped-FT in red. The x-axis is labeled "projection score (human pole ← → AI pole)" and the y-axis is labeled "session density." The layer with the greatest rightward shift of the Doped-FT curve relative to Human-FT is selected as layer K, which is then used as the focal point for Steps 3 and 4.

> **UPDATE [2026-04-02] — Layer K Selection Metric and Multiple Conditions:**
> Layer K is now selected by **Cohen's d** (not raw mean_diff). Cohen's d normalizes by pooled standard deviation, making the metric scale-invariant across layers — raw mean_diff is biased by activation magnitude differences between layers, which would unfairly favor layers with large activation scales. Cohen's d directly measures effect size and is the more scientifically defensible choice.
>
> Phase 4 now produces a **separate set of 32 KDE plots per condition** (base, doped_108, doped_12x9), each compared against Human-FT on the same latent axis. A final summary table reports Layer K for every condition side by side. Plots are saved to `results/figures/<condition>/`.

 
**Step 3 — Attention Patching at Layer K**

Purpose: to identify which specific attention head within layer K is responsible for the representational shift found in Steps 1 and 2. Finding this attention head H will further prove fine-tuning with doped data will corrupt the internals for transformers. Hence AI cannot act as human surrogates.

Input: Doped-FT model, Human-FT model, layer K identified in Step 2.

Each transformer layer contains 15 attention heads (PoC: SmolLM2-360M) / 32 attention heads (full research: Llama-3-8B, Mistral-7B). For each head H at layer K, we perform the following swap using a PyTorch forward hook: we run the Doped-FT model on the game prompt, but intercept the output of head H mid-computation and replace it with the output that Human-FT's head H would have produced on the same prompt. We then record the resulting response distribution across 100 sessions (PoC) / 1000 sessions (full research).

We measure the shift magnitude as the JSD between the patched Doped-FT distribution and the original Human-FT distribution — a smaller JSD means the swap brought Doped-FT closer to human-like behavior. We repeat this for all heads at layer K: 15 heads (PoC) / 32 heads (full research).

Output: a bar chart with head index on the x-axis and JSD-to-Human-FT on the y-axis. The head H with the lowest JSD after patching — meaning its replacement most recovered human-like behavior — is identified as the primary responsible head.

 
**Step 4 — Attention Heatmap for Head H at Layer K**

Purpose: to provide mechanistic evidence for why the model behaves differently is to show whether Head H has stopped attending to the strategically meaningful tokens in the game prompt after doped fine-tuning. This is the step to prove models only learned to mimic the surface patterns of the doped reasoning text rather than developing genuine strategic thoughts.

Input: Doped-FT model, Human-FT model, layer K and head H identified in Steps 2 and 3.

During a forward pass, after the attention patterns are computed and softmax is applied, each attention head produces a weight matrix and each Entry in this matrix represents how strongly the query token attends to the other key token. We extract this matrix for head H at layer K using a PyTorch forward hook, for a representative set of 1000 sessions per model.

We tokenize the full game prompt and use the resulting token strings as axis labels on both axes of the heatmap. We then average the attention weight matrices across the 1000 sessions to produce one stable heatmap per model. Both heatmaps use an identical color scale — lighter cells indicate low attention weight, darker cells indicate high attention weight.

Output: two side-by-side heatmaps, Human-FT on the left and Doped-FT on the right, for head H at layer K. A model performing genuine strategic reasoning should show head H attending strongly to the tokens "one less," "other player," and "additional 20" — the tokens that define the game's bonus rule. A model that has only mimicked the surface structure of analytical reasoning text will show those same cells dimmed or diffused, indicating that head H has stopped reading the strategic rule. 

This is the direct mechanistic evidence that doped fine-tuning causes the model to pattern-match rather than reason strategically, and therefore cannot serve as a valid human surrogate.


