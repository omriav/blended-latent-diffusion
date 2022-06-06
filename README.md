# Blended Latent Diffusion
<a href="https://arxiv.org/abs/2206.02779"><img src="https://img.shields.io/badge/arXiv-2206.02779-b31b1b.svg" height=22.5></a>

<img src="docs/teaser.png" />

> **Blended Latent Diffusion**
>
> Omri Avrahami, Ohad Fried, Dani Lischinski
>
> Abstract: The tremendous progress in neural image generation, coupled with the emergence of seemingly omnipotent vision-language models has finally enabled text-based interfaces for creating and editing images. Handling *generic* images requires a diverse underlying generative model, hence the latest works utilize diffusion models, which were shown to surpass GANs in terms of diversity. One major drawback of diffusion models, however, is their relatively slow inference time. In this paper, we present an accelerated solution to the task of *local* text-driven editing of generic images, where the desired edits are confined to a user-provided mask. Our solution leverages a recent text-to-image Latent Diffusion Model (LDM), which speeds up diffusion by operating in a lower-dimensional latent space. We first convert the LDM into a local image editor by incorporating Blended Diffusion into it. Next we propose an optimization-based solution for the inherent inability of this LDM to accurately reconstruct images. Finally, we address the scenario of performing local edits using thin masks. We evaluate our method against the available baselines both qualitatively and quantitatively and demonstrate that in addition to being faster, our method achieves better precision than the baselines while mitigating some of their artifacts

<div>
  <img src="docs/object_editing.gif" width="200px"/>
  <img src="docs/new_object.gif" width="200px"/>
  <img src="docs/graffiti.gif" width="200px"/>
</div>

# Applications

### Background Editing
<img src="docs/applications/background_edit.png" />

### Text Generation
<img src="docs/applications/text.png" />

### Multiple Predictions
<img src="docs/applications/multiple_predictions.png" />

### Alter an Existing Object
<img src="docs/applications/object_edit.png" />

### Add a New Object
<img src="docs/applications/new_object.png" />

### Scribble Editing
<img src="docs/applications/scribble_edit.png" />

# Code
The code will be released in this repository soon.