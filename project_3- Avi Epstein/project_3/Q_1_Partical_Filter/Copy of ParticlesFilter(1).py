import numpy as np
import random

def normalize_angle(angle):
    if -np.pi < angle <= np.pi:
        return angle
    if angle > np.pi:
        angle = angle - 2 * np.pi
    if angle <= -np.pi:
        angle = angle + 2 * np.pi
    return normalize_angle(angle)

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
        
        # Initialize particles
        self.particles = dict()
        for i in range(numberOfPaticles):
          '''
          self.particles[i] = {'pose':np.expand_dims(np.array([np.random.normal(0,2),np.random.normal(0,2),np.random.normal(0.1,0.1)]),0),
                                'weight': 1/numberOfPaticles, 'history' : [],
                                'closeset_landmark' : np.zeros((1,2))} 
          '''
          self.particles[i] = {'pose':np.array([np.random.normal(0,2),np.random.normal(0,2),np.random.normal(0.1,0.1)]),
                      'weight': 1/numberOfPaticles, 'history' : [],
                      'closeset_landmark' : np.empty((1,2))}  
        ## TODO ## done!

    def apply(self, Zt, Ut):

        # Motion model based on odometry
        self.motionModel(Ut)
        print("after motion",self.particles)
        # Observation model
        self.Observation()

        # Observation model
        self.weightParticles(Zt)

        # Resample particles
        self.resampleParticles()

    def motionModel(self, odometry):

      for i in range(self.numberOfParticles):
        #self.particles[i]['history'].append(self.particles[i]['pose']) 
        #adding noise ODAMITRY:
        dr1 = odometry['r1'] + np.random.normal(0,np.power(self.sigma_r1,2))
        dt = odometry['t'] + np.random.normal(0,np.power(self.sigma_t,2))
        dr2 = odometry['r2'] + np.random.normal(0,np.power(self.sigma_r2,2))
        theta = self.particles[i]['pose'][2]
        dMotion = np.array([dt * np.cos(theta + dr1), dt * np.sin(theta + dr1), dr1 + dr2])
        past_position = self.particles[i]['pose'] 
        self.particles[i]['pose'] = past_position + dMotion
        self.particles[i]['pose'][2] = normalize_angle(self.particles[i]['pose'][2])
        print("i", i ,"self.particles[i]['pose'] ",self.particles[i]['pose'])
      


    ## TODO ## done!

    def Observation(self):

      for i in range(self.numberOfParticles):
        ditance_x_y = np.subtract(self.worldLandmarks , [self.particles[i]['pose'][0],self.particles[i]['pose'][1]])
        closeset_landmark_idx  = np.argmin(np.sqrt(np.power(ditance_x_y[:,0],2) + np.power(ditance_x_y[:,1],2)))
        closeset_landmark = self.worldLandmarks[closeset_landmark_idx]
        #print("i", i, "closeset_landmark", closeset_landmark)
        print("i", i ,"observation self.particles[i]['pose'] ",self.particles[i]['pose'])
        r_partical = np.sqrt(np.power(closeset_landmark[0] - self.particles[i]['pose'][0],2) + np.power(closeset_landmark[1] - self.particles[i]['pose'][1],2)) #possibly add noise ask roy
        phi_partical = np.arctan2(closeset_landmark[1] - self.particles[i]['pose'][1],closeset_landmark[0] - self.particles[i]['pose'][0]) #possibly add noise ask roy
        obeservation_robot = [r_partical,phi_partical]
        #print("i", i,"obeservation_robot" ,obeservation_robot)
        self.particles[i]['closeset_landmark'] = obeservation_robot

    ## TODO ## done!


    def weightParticles(self, worldMeasurment):

      ''' worldMeasurment is Zt = [r,phi] real messurment of closeset land mark having sensor noise Q. 
          giving probabilitys of each partical based on distance from real meassurment using Mahalanobis_distance as meassure of probaility as part of correction step
      '''
      self.total_weight = 0
      for i in range(self.numberOfParticles):
        self.particles[i]['weight'] = 0
        diff = np.subtract(self.particles[i]['closeset_landmark'] , worldMeasurment).T
        diff[1] = normalize_angle(diff[1])
        d = np.sqrt(np.matmul(diff.T,np.matmul(np.linalg.inv(self.Q),diff)))
        print("i",i,"d weight" , d)
        self.particles[i]['weight'] = np.exp(-np.power(d,2)/2)*(1.0/np.sqrt(np.linalg.det(2*np.pi*self.Q)))       
        self.total_weight += self.particles[i]['weight']

          
       
    ## TODO ## done!

    def resampleParticles(self):
      sum1 = 0
      # normalizing the weights
      for i in range(self.numberOfParticles):
        w = self.particles[i]['weight']
        print ("i", i, "w ", self.particles[i]['weight'])
        self.particles[i]['weight'] = w/self.total_weight
        sum1 +=self.particles[i]['weight']
        print ("i", i, "w normlized", self.particles[i]['weight'])
      #print("sum1:", sum1)
      print("total weight: ", self.total_weight)
      

      # resample using Low Variance resampling
      new_particles = []
      r = random.uniform(0,1.0)/self.numberOfParticles
      c  = self.particles[0]['weight']
      i = 0
      for m in range(0,self.numberOfParticles):
        U = r + m*(1.0/self.numberOfParticles)
        while U > c :
          i = i+1
          #print(self.particles[i]['weight'])
          c += self.particles[i]['weight']
        new_particles.append(self.particles[i]) 

      self.particles.clear()
      for i in range(0,self.numberOfParticles):
        self.particles[i] = new_particles[i]
      print(self.particles)
      #return new_particles 
       
        ## TODO ## done!

    def bestKParticles(self, K):
      weight_list = []
      for i in range(self.numberOfParticles):
        weight_list.append(self.particles[i]['weight'])
      indexes = np.argsort(-np.array(weight_list)) #indexes = np.argsort(-self.particles[:, 3]) # indexes = np.argsort(-self.particles[:]['weight'])
      bestK = indexes[:K]
      best_locations_list = []
      for i in bestK:
        best_locations_list.append(self.particles[i]['pose'])

      return best_locations_list  # return self.particles[bestK]['pose']

