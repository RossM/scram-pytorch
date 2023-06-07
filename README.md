This repo contains various experiments I'm doing in optimizers and learning rate schedulers.

# scram_pytorch.scram
SCRAM (**Sc**ale and **R**otation Inv**a**riant **M**omentum)

This is similar to the [LION optimizer](https://github.com/lucidrains/lion-pytorch), but normalizes each parameter's updates using the root mean square (RMS) rather than the sign. This makes the optimizer invariant
to orthonormal transformations that rotate channels into each other.

Recommended hyperparameters for a model where AdamW is best at lr=1e-4:

|eps|learning rate|beta1|beta2|
|---|---|---|---|
|1e-15|1e-6|0.98|0.99|

For best results, gradient clipping should be disabled.

# scram_pytorch.simon
SIMON (**Si**gma **Mo**me**n**tum)
 
An AdaBelief derivative that incorporates the momentum modifications from Lion, and uses a slightly different way
of calculating the standard deviation. The best optimizer I've found for many problems in my tests.

Recommended hyperparameters for a model where AdamW is best at lr=1e-4:

|eps|learning rate|beta1|beta2|rmsclip|layerwise|normalize|
|---|---|---|---|---|---|---|
|1e-15|1e-4|0.98|0.99|False|False|False|

For best results, gradient clipping should be disabled.

# scram_pytorch.esgd
ESGD (**E**nsemble **S**tochastic **G**radient **D**descent)

A version of stochastic gradient descent (with momentum) that simulates a very large ensemble of
models by maintaining two copies of each weight, and randomly selecting one copy to use for each
weight independently at each optimization step. Also includes filterwise normalization.

ESGD seems to be particularly good at adversarial training.

Recommended hyperparameters for a model where AdamW is best at lr=1e-4:

|eps|learning rate|beta1|beta2|p|swap_ratio|
|---|---|---|---|---|---|
|1e-15|1e-4|0.99|0.99|0.5|0.99|

For best results, gradient clipping should be disabled.
