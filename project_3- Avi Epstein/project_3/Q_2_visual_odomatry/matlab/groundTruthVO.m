%% Read in pose ground truth from Kitti odometry
posesname = '00.txt';
T = readtable(posesname,'Delimiter','space','ReadRowNames',false,'ReadVariableNames',false);
A = table2array(T);
M = zeros(3,4,length(A));

for i = 1: length(A)
    M(1:3,1:4,i) = [A(i,1:4);A(i,5:8);A(i,9:12)];
end
% The folder 'poses' contains the ground truth poses (trajectory) for the
% first 11 sequences. This information can be used for training/tuning your
% method. Each file xx.txt contains a N x 12 table, where N is the number of
% frames of this sequence. Row i represents the i'th pose of the left camera
% coordinate system (i.e., z pointing forwards) via a 3x4 transformation
% matrix. The matrices are stored in row aligned order (the first entries
% correspond to the first row), and take a point in the i'th coordinate
% system and project it into the first (=0th) coordinate system. Hence, the
% translational part (3x1 vector of column 4) corresponds to the pose of the
% left camera coordinate system in the i'th frame with respect to the first
% (=0th) frame. 


%length(M)
len = length(M);
Po = zeros(3,len);
for j = 1 : len
    Po(:,j) = M(:,:,j)*[0;0;0;1];
end
pos = [0;0;0];
Rpos = eye(3);
for k = 1:len
    pos_temp = Po(:,k);
    Rpos_temp = M(:,1:3,k);
    tr(:,k) = inv(Rpos)*(pos_temp - pos);
    R(:,:,k) = inv(Rpos)*Rpos_temp;
    pos = pos_temp;
    Rpos = Rpos_temp;
end

