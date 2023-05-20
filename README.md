# scram-pytorch
 SCRAM (**Sc**ale and **R**otation Inv**a**riant **M**omentum) optimizer for PyTorch

This is similar to the LION optimizer, but normalizes each parameter's updates using the root mean square (RMS) rather than the sign. This makes the optimizer invariant
to orthonormal transformations that rotate channels into each other.
