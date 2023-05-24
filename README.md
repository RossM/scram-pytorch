# scram-pytorch
 SCRAM (**Sc**ale and **R**otation Inv**a**riant **M**omentum) optimizer for PyTorch

This is similar to the [LION optimizer](https://github.com/lucidrains/lion-pytorch), but normalizes each parameter's updates using the root mean square (RMS) rather than the sign. This makes the optimizer invariant
to orthonormal transformations that rotate channels into each other.

I recommend using the same optimizer parameters you would use for LION. Epsilon is just to prevent divide-by-zero errors and can safely be set to a very small value, e.g. 1e-15.

Also contains SIMON (**Si**gma **Mo**me**n**tum) optimizer. This one is very experimental.
