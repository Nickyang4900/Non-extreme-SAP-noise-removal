# Non-extreme-SAP-noise-removal
## Abstract
There are several previous methods based on neural
network can have great performance in denoising salt and pepper
noise. However, those methods work based on a hypothesis that
the value of salt and pepper noise is exactly 0 and 255. It is
not true in the real world. The result of those method deviation
sharply when the value changes form 0 and 255. To overcome
this weakness, our method aims at design a convolutional neural
network to detect the noise pixels in a wider range of value and
then a filter is used to modify it to 0 which is benefit for further
work. And another convolutional neural network is used followed
the filter to do the denoising and restoration work.
