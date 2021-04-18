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

## Results
Corrupted, 20% SAP noise at non-extreme value
![1](https://user-images.githubusercontent.com/48509610/115139100-acb95b80-a062-11eb-9f44-89fee856803d.png)

Stage1-Output, noise detection
![Blacked-1](https://user-images.githubusercontent.com/48509610/115139089-a0350300-a062-11eb-858e-28537c9cae33.png)

Stage2-Output, PSNR = 39.07dB
![Output_9](https://user-images.githubusercontent.com/48509610/115139095-a62ae400-a062-11eb-9658-eb31f8c7c3c0.png)

CounterPart, denoise directly PSNR = 35.52dB
![Counterpart-Output](https://user-images.githubusercontent.com/48509610/115139366-10905400-a064-11eb-96f8-40976e6d731e.PNG)

GT
![GT](https://user-images.githubusercontent.com/48509610/115139370-19812580-a064-11eb-8ed9-37b880ead77a.png)
