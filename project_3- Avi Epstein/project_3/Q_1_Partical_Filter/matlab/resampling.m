% resample the set of particles.
% A particle has a probability proportional to its weight to get
% selected. A good option for such a resampling method is the so-called low
% variance sampling, Probabilistic Robotics pg. 109
function newParticles = resampling(particles)

numParticles = length(particles);

w = [particles.weight];

% normalize the weight
w = w / sum(w);


% walk along the wheel to select the particles


