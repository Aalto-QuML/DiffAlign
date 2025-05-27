Excited to present our recent work "Equivariant ..." at #ICLR2025

@....[add all authors]

[calendar emoji]: 25.04.2025, 3 pm in Hall 3 + Hall 2B, poster #194
[paper emoji] proceedings: https://openreview.net/forum?id=onIro14tHv

Are you interested in #diffusion for graphs? Do you want to know more about the limitations of #equivariant models? Curious about one of the latest models in #retrosynthesis? Checkout this [thread] and come chat with 
@severi_rissanen or me anytime at #ICLR2025

[upload catchy rainbow photo]

1. we focus on diffusion for graph translation: i.e. we want to generate a graph conditioned on another graph 
with all the benefits of diffusion (diversity, inference time guidance, etc). A clear application for this setup is #retrosynthesis, where the goal is to predict reactants given a target product. 
[insert video]

2. diffusion for graphs often uses #equivariant denoisers in order to ensure the model can handle the input in any order. Equivariant denoisers struggle to map a self-symmetrical input into a less self-symmetrical output, which you can see for yourself in this toy [notebook](https://github.com/Aalto-QuML/DiffAlign/blob/gh-pages/notebooks/translate_selfsymmetrical_molecules.ipynb)

This is an issue for the denoising process in diffusion: the process starts with a noisy graph (e.g. an empty graph with dummy nodes and no edges), which is highly self-symmetrical, and is expected to output graphs that resemble more and more the target graph in the translation process.

3. why do #equivariant denoisers struggle with self-symmetrical input, exactly? 

The conflicting instructions of #breaking self-symmetry (i.e. assigning labels differentiating the components of the graph) while #maintaining equivariance force an optimally trained denoiser to output the marginal distribution of the node and edge labels in the training dataset! Since the denoising process involves sampling, we do eventually get less and less self-symmetrical input in each iteration, thus breaking out of the self-symmetry bottleneck, but very ineffectively. 
[upload denoising process figure]

4. To solve this, we need to help the model differentiate between the components of the self-symmetrical input while maintaining equivariance. We achieve this through *aligned equivariance*. The idea is simple: assign unique identifiers to paired graph components (nodes or edges) across the translation task. For instance, in chemical reactions, we know where the atoms of the molecular graph end up through atom-mapping information 


5. We match state-of-the-art models in top-k accuracy in #retrosynthesis on #uspto-50k while unlocking diffusion features like inpainting and inference-time guidance.


6. Checkout the project website: and our poster tomorrow at 10 am. Code coming soon (rainbow)