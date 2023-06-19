# The BLMP: Best Linear Monotonic Predictor

This repo contains various iterative attempts to move towards a BLMP -- or more specifically, a BLCP -- the Best Linear **Constrained** Predictor.

The BLCP allows one to use the standard machinery of the BLP, but it allows one to impose firm conditions on the nature of the predictions, such as monotonicity, bounds, integral norming and so on. 

In addition, we have formalised and justified a means of imposing a `Prior' onto the predictor using transforms on the data and on the kernel.