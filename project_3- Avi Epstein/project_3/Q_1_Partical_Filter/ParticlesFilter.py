import numpy as np
import random
import copy
np.random.seed(333)
def normalize_angle(angle):
    if -np.pi < angle <= np.pi:
        return angle
    if angle > np.pi:
        angle = angle - 2 * np.pi
    if angle <= -np.pi:
        angle = angle + 2 * np.pi
    return normalize_angle(angle)

def calc_RMSE_maxE(X_Y_GT, X_Y_est):
    """
    That function calculates RMSE and maxE

    Args:
        X_Y_GT (np.ndarray): ground truth values of x and y
        X_Y_est (np.ndarray): estimated values of x and y

    Returns:
        (float, float): RMSE, maxE
    """
    maxE = 0
    e_x = X_Y_GT[50:,0] - X_Y_est[50:,0]
    e_y = X_Y_GT[50:,1] - X_Y_est[50:,1]
    maxE = max(abs(e_x)+abs(e_y))
    RMSE = np.sqrt(sum(np.power(e_x,2)+np.power(e_y,2))/(X_Y_GT.shape[0]-50))
    return RMSE, maxE

class ParticlesFilter:
    def __init__(self, numberOfPaticles, worldLandmarks, sigma_r1, sigma_t, sigma_r2):

        # Initialize parameters
        self.numberOfParticles = numberOfPaticles
        self.worldLandmarks = worldLandmarks

        self.sigma_r1 = sigma_r1
        self.sigma_t = sigma_t
        self.sigma_r2 = sigma_r2
        self.Q = np.array([[np.power(1,2), 0],[0, np.power(0.1,2)]]) # maybe should be intlized outside of class sigma_r and sigma_phi of lidar measserment
        self.total_weight = 0
        self.z_max = 4
        # Initialize particles
        self.particles = np.array([[np.random.normal(0,2),np.random.normal(0,2),np.random.normal(0.1,0.1),1/numberOfPaticles,0,0] for i in range(numberOfPaticles) ])

        ## TODO ## done!

    def apply(self, Zt, Ut):

        # Motion model based on odometry
        self.motionModel(Ut)
        # Observation model
        self.Observation()

        # Observation model
        self.weightParticles(Zt)

        # Resample particles
        self.resampleParticles()
        return self.particles

    def motionModel(self, odometry):

      for i in range(self.numberOfParticles):
        #adding noise ODAMITRY:
        dr1 = odometry['r1'] + np.random.normal(0,self.sigma_r1)  # np.random.normal(0,1.1*np.power(self.sigma_r1,2))
        dt = odometry['t'] + np.random.normal(0,self.sigma_t) # np.random.normal(0,1.5*np.power(self.sigma_t,2))
        dr2 = odometry['r2'] + np.random.normal(0,self.sigma_r2) #+ np.random.normal(0,np.power(self.sigma_r2,2))
        theta = self.particles[i][2]
        dMotion = np.array([dt * np.cos(theta + dr1), dt * np.sin(theta + dr1), dr1 + dr2])
        self.particles[i][0] = self.particles[i][0] + dMotion[0]
        self.particles[i][1] = self.particles[i][1] + dMotion[1]
        self.particles[i][2] = self.particles[i][2] + dMotion[2]
        self.particles[i][2] = normalize_angle(self.particles[i][2])
        #print("i", i ,"self.particles[i] ",self.particles[i])
      


    ## TODO ## done!

    def Observation(self):

      for i in range(self.numberOfParticles):
        ditance_x_y = np.subtract(self.worldLandmarks , [self.particles[i][:2]])
        closeset_landmark_idx  = np.argmin(np.sqrt(np.power(ditance_x_y[:,0],2) + np.power(ditance_x_y[:,1],2)))
        closeset_landmark = self.worldLandmarks[closeset_landmark_idx]
        #print("i", i, "closeset_landmark", closeset_landmark)
        #print("i", i ,"observation self.particles[i] ",self.particles[i])
        r_partical = np.sqrt(np.power(closeset_landmark[0] - self.particles[i][0],2) + np.power(closeset_landmark[1] - self.particles[i][1],2)) #+ np.random.normal(0,self.Q[0][0]) #possibly add noise ask roy
        phi_partical = np.arctan2(closeset_landmark[1] - self.particles[i][1],closeset_landmark[0] - self.particles[i][0]) #+ np.random.normal(0,0.5*self.Q[1][1]) #possibly add noise ask roy
        obeservation_partical = [r_partical,phi_partical]
        #print("i", i,"obeservation_robot" ,obeservation_robot)
        #if obeservation_partical[0] < self.z_max:
        self.particles[i][4:] = obeservation_partical
        #else:
          #self.particles[i][4:] = [0,0]

    ## TODO ## done!


    def weightParticles(self, worldMeasurment):

      ''' worldMeasurment is Zt = [r,phi] real messurment of closeset land mark having sensor noise Q. 
          giving probabilitys of each partical based on distance from real meassurment using Mahalanobis_distance as meassure of probaility as part of correction step
      '''
      self.total_weight = 0
      for i in range(self.numberOfParticles):
        self.particles[i][3] = 0
        diff = np.subtract(self.particles[i][4:] , worldMeasurment).T
        diff[1] = normalize_angle(diff[1])
        d = np.sqrt(np.matmul(diff.T,np.matmul(np.linalg.inv(self.Q),diff)))
        #print("i",i,"d weight" , d)
        self.particles[i][3] = np.exp(-np.power(d,2)/2)*(1.0/np.sqrt(np.linalg.det(2*np.pi*self.Q)))       
        self.total_weight += self.particles[i][3]

          
       
    ## TODO ## done!

    def resampleParticles(self):
      sum1 = 0
      # normalizing the weights
      for i in range(self.numberOfParticles):
        w = self.particles[i][3]
        self.particles[i][3] = w/self.total_weight
        sum1 +=self.particles[i][3]

      

      # resample using Low Variance resampling
      new_particles = []
      r = random.uniform(0,1.0)/self.numberOfParticles
      c  = copy.copy(self.particles[0][3])
      i = 0
      for m in range(0,self.numberOfParticles):
        U = r + m*(1.0/self.numberOfParticles)
        while U > c :
          i = i+1
          c += self.particles[i][3]
        new_particles.append(self.particles[i]) 


      for i in range(0,self.numberOfParticles):
        self.particles[i] = new_particles[i]

        '''
        #this scatters particals that have very low weight
        if self.particles[i][3]< 1/(1000*self.numberOfParticles):
          self.particles[i][0:2] = copy.copy(self.particles[i][0:2] +np.array([np.random.normal(0,0.5),np.random.normal(0,0.5)]))
        '''

      #return new_particles 
       
        ## TODO ## done!

    def bestKParticles(self, K):
 
      indexes = np.argsort(-self.particles[:, 3])
      bestK = indexes[:K]
      return self.particles[bestK,:]
      