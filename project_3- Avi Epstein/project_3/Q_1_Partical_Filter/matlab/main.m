clear all;
close all;

% Make tools available
addpath('tools');

% Read sensor readings, i.e. odometry
data = read_data('odometry.dat');
landmarks=csvread('landmarks_Q1.csv');

% Settings
Q=%TODO% % initialize measurement noise
Odometer_noise = %TODO%; % initialize odometer noise
numParticles = 1000;

% Ground truth
figure
X=[0 ;0 ;0];
figure;plot(landmarks(:,1),landmarks(:,2),'ob')
hold off
Xr=landmarks(:,1);
Yr=landmarks(:,2);

for i=1:length(data.timestep)
    u.t=data.timestep(i).odometry.t;
    u.r1=data.timestep(i).odometry.r1;
    u.r2=data.timestep(i).odometry.r2;
    if i==1
        X(1,1)=u.t * cos(u.r1);
        X(2,1)=u.t * sin(u.r1);
        X(3,1)=normalize_angle(u.r1+u.r2);
    else 
        X(:,i)=X(:,i-1)+[u.t * cos(X(3,i-1) + u.r1); u.t * sin(X(3,i-1) + u.r1); normalize_angle(u.r1 + u.r2)];
    end
end
hold on
plot(X(1,:),X(2,:),'.k')
grid on
xlim([-10, 20])
ylim([-10, 20])
axis image
legend('Landmarks','GT')


% initialize the particles array
particles = struct;
for i = 1:numParticles
  particles(i).weight = %TODO%
  particles(i).pose = %TODO%
  particles(i).history = {};
end


figure
BestPart(:,1)=particles(1).pose;
aviobj = VideoWriter('AnimationParticles.avi');
aviobj.FrameRate=10;
open(aviobj)
for t = 1:size(data.timestep, 2)
    % Perform the prediction step of the particle filter
    particles = prediction_step(particles, data.timestep(t).odometry, Odometer_noise);
  
    % Perform the measurment step 
    obeservation_robot=measurement_step(X(:,t),landmarks);
    
    % Perform the corrections step 
    particles=corretion_step(particles,obeservation_robot,landmarks,Q);
    
    % resampling!
    particles = resampling(particles);
    
    % find the best particle
    [~,indx]=max([particles.weight]);
    BestPart(:,t)=particles(indx).pose;
    
    % Generate visualization plots of the current state of the filter
    plot_state(particles, t,X,Xr,Yr,BestPart);
    
    title(sprintf('ParticleFilter, Frame #%d',t))
    frame = getframe(gcf);
    writeVideo(aviobj,frame);
    
end
close(aviobj)
