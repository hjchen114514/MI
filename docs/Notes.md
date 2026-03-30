  PyTorch
  The deep learning framework that actually runs the neural networks. For your experiment it does three things:                                                                                      
  - Loads models via HuggingFace Transformers (which sits on top of PyTorch)                                                                                                                         
  - Forward hooks (model.register_forward_hook): lets you intercept and read the activations at any layer during a forward pass without modifying the model — this is how you extract residual stream
   vectors in Step 2 and attention weights in Step 4                                                                                                                                                 
  - Activation patching: hooks also let you replace a layer's output mid-pass — this is how you swap one model's attention head output into another model in Step 3                                  
                                   
  LoRA (Low-Rank Adaptation)                                                                                                                                                                         
  Fine-tuning a full 7B model updates billions of parameters — infeasible on a MacBook. LoRA instead freezes all original weights and injects small trainable "adapter" matrices into specific
  projection layers (q_proj, v_proj). The key insight: weight updates during fine-tuning are empirically low-rank, so you can approximate them as $\Delta W = BA$ where $B$ and $A$ are tiny matrices
   (rank 8 in your config). This means:
  - Training fits on your M4 with MPS                                                                                                                                                                
  - Human-FT and Doped-FT are just two different adapter files sitting on top of the same frozen base model                                                                                          
  - Easy to swap adapters to compare models                                                                
                                                                                                                                                                                                     
  HuggingFace PEFT is the library that implements LoRA. HuggingFace Transformers provides the model loading, tokenizer, and generation APIs.                                                                                                                                                                                                                                             
  scipy/numpy: Array math, KDE fitting (scipy.stats.gaussian_kde), JSD computation.                                                                                                                  
                                   
  matplotlib/seaborn: All the output visualizations — KDE plots, heatmaps, bar charts. 


  How to Improve Step 1 — Quantitative Boundary Method                                                                                                                                               
                                                                                                                                                                                                     
  You're right — "bare eyes" on a histogram is not defensible in a research paper. The current plan says the boundary is defined "empirically" but doesn't specify how. Here are three rigorous      
  options, from simplest to most sophisticated:                                                                                                                                                      
                                                                                                                                                                                                     
  Option A — Optimal threshold via ROC + Youden's J (recommended for PoC)                                                                                                                            
  You already have ground truth: the 108 Arad & Rubinstein human responses (known human-like) and the base model's responses (known AI-like tendency). Sweep every integer cutoff $c \in {11, ...,
  20}$. For each $c$, compute TPR (fraction of human responses classified as human-like, i.e., $< c$) and FPR (fraction of base model responses misclassified as human-like). Pick the $c$ that      
  maximizes Youden's J = TPR − FPR. This is ~10 lines of sklearn/numpy and gives a statistically grounded, single defensible cutoff.
                                                                                                                                                                                                     
  Option B — Gaussian Mixture Model (GMM)
  Fit a 2-component GMM to the pooled response distribution across all sessions. The boundary is where the posterior probability of the two components is equal (i.e., $P(\text{human-like} | x) =
  0.5$). Fully data-driven, no ground truth labels needed, but slightly harder to interpret.                                                                                                         
  
  Option C — KDE likelihood ratio                                                                                                                                                                    
  Fit KDEs to both the human empirical distribution (from A&R 2012) and the Doped-FT/base model distribution. The boundary is where the KDEs intersect — this is the Bayes-optimal decision boundary
  between the two distributions.                                                                                                                                                                     
  
  My recommendation for your plan: Use Option A (ROC/Youden) for the PoC. It's principled, fast, directly uses the existing A&R 2012 data as ground truth, and gives you something clean to report.  
  You can replace it with GMM or KDE for the full paper.


  ┌────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┐   
  │   Parameter    │                                                    What it does                                                    │                      Your setting                      │
  ├────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤   
  │ temperature    │ Randomness of sampling. Lower = more deterministic.                                                                │ 0.5 (per plan)                                         │
  ├────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤   
  │ do_sample      │ Must be True to use temperature sampling. False = greedy (always picks highest probability token, ignores temp).   │ True                                                   │  
  ├────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤   
  │ top_p          │ Nucleus sampling — only sample from tokens comprising top p% of probability mass.                                  │ Recommend 1.0 (disabled) to isolate temperature's      │   
  │                │                                                                                                                    │ effect                                                 │
  ├────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤   
  │ top_k          │ Only sample from top k tokens.                                                                                     │ Recommend 0 (disabled) for same reason                 │
  ├────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤   
  │ max_new_tokens │ Hard cap on response length.                                                                                       │ ~`300` — enough for a number + reasoning               │
  ├────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤   
  │ seed           │ Critical for reproducibility. Each of 100 sessions needs a different but fixed seed so results are reproducible    │ e.g. seeds 0–99 per session                            │
  │                │ later.                                                                                                             │                                                        │   
  └────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┘   
                                                                                                                                                                                                     