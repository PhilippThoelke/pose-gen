# pose-gen
Generating images from human poses (and sound).

Maps pose estimation or audio input to the latent space of an image generation model.

# How to run
The best working model currently is PCA for image generation. To create the PCA model, run the `eigenface.py` script. This will fit PCA to an image dataset and stores the model as `models/eigenface.pt`.

To run everything, execute `run.py`. You can adjust the settings in the bottom section of the script (e.g. which model to use to generate images, which pose estimation or sound model should be used).

You can also train a convolutional VAE model using `vae.py` and use a model checkpoint in `run.py`, however, the current implementation of the VAE is rather limited and doesn't produce great images.