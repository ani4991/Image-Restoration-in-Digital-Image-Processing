# Image-Restoration
Grad project for Digital Image Processing course (Spring 19)

Basic Idea: To restore an Image back to original as close as possible through filtering techniques.

1. The project's motivation was to apply different types of static noise(Gamma, Uniform, Speckle,etc) to a Digital Image.
2. Analyzing the changes to histogram of the image after noise addition and experimenting the rate of noise addition by modifying the        probabaility.
3. we have utilized various filtering techniques to understand about the parameters involved and their optimal values for various            scenarios.
4. Implementation details:

Noises:
	- Gaussian
	- Gamma
	- Exponential
	- Rayleigh
	- Uniform
	- Salt and pepper
	- Salt
	- Pepper

5. We read an input image in grayscale, initially assumed the values of mean and variance(mean = 0, variance = 1) . We have implemented      individual functions for different kinds of noises. We found the values of a and b from mean and variance where a and b are height of      the curve's peak and position of the center of the peak respectively.
6. To sample points for the noise matrix, we have formed a range based on the variance and found the probability distribution curve for      this range. We also kept a track of values of the probabilities in an array(noise_array).

7. Filters:
     a. Mean filters:
           Arithmetic mean
           Geometric mean
           Harmonic mean
           Contraharmonic mean
           
     b. Order-statistics filters:
           Median
           Max
           Min
           Midpoint
           Alpha-trimmed mean
           
     c. Adaptive filters:
           Adaptive local noise reduction
           Adaptive median
           
 8. Once we had our new padded image we: (1) Created a mask with a default window size of 3x3 unless the user specifies a different size. (2) Looped through the image pixel by pixel and applied the selected filter calculation to obtain the new pixel value. Since we have eleven different filters we have eleven different implementations. (3) After performing the calculations for each pixel of the image the functions return the new restored image and the user is able to begin the analysis and comparison evaluation process.


