# scram-pytorch
 SCRAM (**Sc**ale and **R**otation Inv**a**riant **M**omentum) optimizer for PyTorch

This is similar to the LION optimizer, but normalizes each parameters's gradients using the root mean square (RMS) rather than the sign. This makes the optimizer invariant
to orthonormal transformations that rotate channels into each other.

For parameters with a single value the result is identical to LION.
