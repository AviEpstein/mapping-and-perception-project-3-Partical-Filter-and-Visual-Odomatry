# mapping-and-perception-project-3-Partical-Filter-and-Visual-Odomatry

Abstract
In this project we will be implementing and analyzing the following algorithms:

● Particle filter

● Visual Odometry (mono)

In the first section we will implement the Particle filter algorithm, we will load a pre defined data 
set containing landmarks and odamatry data. using this we will plot the ground truth trajectory 
starting at point (𝑥0
,𝑦0
, 𝜃0
) based on the following motion model:
We then added guassian noise to the motion model and to ground truth sensor measurements, 
we then initialized N particle's with an initial Gaussian distribution around (𝑥0
,𝑦0
, 𝜃0
) generating 
our hypothesis pose for each particle. In addition, we give each particle an initial weight. We 
then run our motion model on all particles (with a small amount of Gaussian noise added 
before) and apply sensor measurement (sensor measurement is assumed as a spin LiDAR 2D 
sensor (1 layer, 360 degrees) which in each iteration calculates the range and azimuth (𝑟, 𝜑) 
only from the closet landmarks).
We then apply sensor correction by recalculating the weight of each particle by using "normal 
Mahalanobis distance" and hence measuring how far is the hypothesis particle measurement 
from the ground truth and giving it a new weight based on this distance. We then resample the 
particles using low variance resampling from (Thrun, Burgard, and Fox's "Probabilistic 
Robotics") making our particle's with higher weight be resampled with high probability. We 
repeat this process of noised motion, measurement, weight correction and resampling until 
convergence to the true trajectory. We then calculate the MSE from the 50th frame using the 
pose of the particle with the highest weight and see how it performs. In addition, will then find 
the minimal amount of particle's that still give us good performance based on the MSE criteria.
In the second section we will implement a simple monocular Visual odometry algorithm.
We will extract from each frame its features using sift/orb/klt, each one of these feature 
extractions work a little different and will be explained in this report we then match the found 
features in both frame using brute force matching or FLANN, to get corresponding points 
between both frames. We then compute the essential matrix and remove outliers based on 
RANSAC, and decompose the essential matrix to recover the translation and rotation between 
the 2 camera positions the pose recovery verifies possible pose hypotheses by doing cheirality 
check. (which means that the triangulated 3D points should have positive depth)
Because we are using a mono camera it is hard to estimate the depth between frames and 
hence the translation is known up to scale. To correct this as best as we can we refine the 
translation vale by scaling each translation by a factor ratio between a known measurement 
such as IMU info or lidar (in our case we used the ground truth trajectory).
We then concatenate all these translations to get the full trajectory estimated from our mono 
camera rotation and translation we will see how this scale improves the result and why we 
receive drift in our trajectory.
