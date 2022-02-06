import numpy as np
import cv2
from data_loader import DataLoader
from camera import Camera
import matplotlib.pyplot as plt
import graphs 
import matplotlib.animation as animation
import os
import copy

class VisualOdometry:

# write your code here 
  def __init__(self, vo_data):
    self.frame_itr = vo_data.images
    self.intrinsics_k = vo_data.cam.intrinsics
    self.extrinsics = vo_data.cam.extrinsics
    self.gt_poses = vo_data.gt_poses
    self.number_of_frames = vo_data.N
    #plt.imshow(next(self.frame_itr),cmap = 'gray')
    #plt.show()
    #plt.imshow(next(self.frame_itr),cmap = 'gray')
  
  # generate 2 images (Image sequence):
  # Feature detection:
  # find key points dicripturs or by orb(Fast+brief) or by sift() or by  cv.goodFeaturesToTrack()(bassed on harris or shi tomasi corner detector) and then runing sdiscritors on that area
  # do this fo every 2 sequential images:
  def extract_2_frames(self,cur_frame):
    frame_i_img = cur_frame
    frame_i_plus_1_img = next(self.frame_itr)
    return frame_i_img, frame_i_plus_1_img


  def match_with_optical_flow(self,frame1, frame2):

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 300,
                          qualityLevel = 0.2,
                          minDistance = 10,
                          blockSize = 1 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (7, 7),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    

    frame1_points = cv2.goodFeaturesToTrack(frame1, mask = None, **feature_params)

    # calculate optical flow
    frame2_points, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, frame1_points, None, **lk_params)
    if frame2_points is not None:
      good_new_points = frame2_points[st==1]
      good_old_points = frame1_points[st==1]
    print("good_new_points", len(good_new_points))

    return good_old_points, good_new_points #,frame1_points
  
  def extract_key_points_and_decripters_between_two_frames(self,frame1, frame2, detector_type = 'sift'):
    
    if detector_type == 'sift':
      #detector = cv2.xfeatures2d.SIFT_create()
      detector = cv2.SIFT_create()
    elif detector_type == 'orb':
      detector = cv2.ORB_create()
      
    frame1_keypoints, frame1_descriptor = detector.detectAndCompute(frame1, None)
    frame2_keypoints, frame2_descriptor = detector.detectAndCompute(frame2, None)
    return frame1_keypoints, frame1_descriptor, frame2_keypoints, frame2_descriptor


  def match_features_between_two_frames(self,frame1_descriptor,frame2_descriptor,detector_type = 'sift', matcher = 'BF'):
        # Create a Brute Force Matcher object.
    if matcher == 'BF':
      if detector_type == 'orb':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        match_list = bf.match(frame1_descriptor, frame2_descriptor)
        # The matches with shorter distance are the ones we want.
        matches = sorted(match_list, key = lambda x : x.distance)

       
 
 
      elif detector_type == 'sift':
        bf = cv2.BFMatcher()
        match_list = bf.knnMatch(frame1_descriptor,frame2_descriptor,k=2)
        # Apply ratio test
        matches = []
        for m,n in match_list:
          #print("m.distance", m.distance, "n.distance",n.distance)
          if m.distance < 0.6*n.distance:
              matches.append(m)
        
    elif matcher == 'FLANN': #fix this from this site !!!! Feature Matching !!!! in open cv
        #using KD/LSH 

        if detector_type == 'orb':
          FLANN_INDEX_LSH = 6
          index_params= dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6, # 12
                            key_size = 12,     # 20
                            multi_probe_level = 1) #2
          
        elif detector_type == 'sift':
          FLANN_INDEX_KDTREE = 1
          index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
          

        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Need to draw only good matches, so create a mask
        #matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        if detector_type == 'sift':
          match_list = matcher.knnMatch(frame1_descriptor, frame2_descriptor, k=2)
          matches = []
          #print("match_list", match_list)
          for m,n in  match_list:
              if m.distance < 0.6*n.distance:
                  matches.append(m)
         
        elif detector_type == 'orb':
          match_list = matcher.match(frame1_descriptor, frame2_descriptor)
          matches = sorted(match_list, key = lambda x : x.distance)
          
    return matches




    




  def esimate_motion_between_2_frames(self,matches,frame_i_kp,frame_i_plus_1_kp,detector_type = 'sift'):
    

    #essential_mat = np.zeros((3,4))
    ##print("self.intrinsics_k",self.intrinsics_k)
    
    frame_i_points = np.float32([frame_i_kp[m.queryIdx].pt for m in matches])
    frame_i_plus_1_points =  np.float32([frame_i_plus_1_kp[m.trainIdx].pt for m in matches])


    essential_mat = cv2.findEssentialMat( frame_i_points,
                          frame_i_plus_1_points,
                          self.intrinsics_k)[0]                      	
    ##print("essential_mat" , essential_mat)

    #This function decomposes an essential matrix using decomposeEssentialMat and then verifies possible pose hypotheses by doing cheirality check. 
    #The cheirality check means that the triangulated 3D points should have positive depth:
    _ , rotation_mat , translation_vector, _ = cv2.recoverPose(	essential_mat,
                                                              frame_i_points,
                                                              frame_i_plus_1_points,
                                                              self.intrinsics_k)	                                              

    return rotation_mat, translation_vector , frame_i_points, frame_i_plus_1_points

  def esimate_motion_between_2_frames_optical_flow(self,frame1,frame2):
    frame_i_points, frame_i_plus_1_points = self.match_with_optical_flow(frame1, frame2)
    essential_mat = cv2.findEssentialMat( frame_i_points,
                      frame_i_plus_1_points,
                      self.intrinsics_k)[0] 
    _ , rotation_mat , translation_vector, _ = cv2.recoverPose(	essential_mat,
                                                              frame_i_points,
                                                              frame_i_plus_1_points,
                                                              self.intrinsics_k)
    return rotation_mat, translation_vector , frame_i_points, frame_i_plus_1_points

  def find_rotation_and_translation_between_frames_no_scale(self,cur_frame,detector_type = 'sift' ,matcher = 'BF'):

      
      frame_i_img, frame_i_plus_1_img = self.extract_2_frames(cur_frame)
      if detector_type != 'optical flow':
        frame_i_keypoints, frame_i_descriptor, frame_i_plus_1_keypoints, frame_i_plus_1_descriptor = self.extract_key_points_and_decripters_between_two_frames(frame_i_img, frame_i_plus_1_img, detector_type)
        matches = self.match_features_between_two_frames(frame_i_descriptor, frame_i_plus_1_descriptor, detector_type, matcher)
        rotation_mat, translation_vector , frame_i_points, frame_i_plus_1_points = self.esimate_motion_between_2_frames(matches, frame_i_keypoints, frame_i_plus_1_keypoints)
      elif detector_type == 'optical flow':
        rotation_mat, translation_vector , frame_i_keypoints, frame_i_plus_1_keypoints = self.esimate_motion_between_2_frames_optical_flow(frame_i_img, frame_i_plus_1_img)
      
      #transform to homogines transformation:
      homogines_trans = np.eye(4)
      homogines_trans[:3,:3] = rotation_mat
      homogines_trans[:3,3] = translation_vector.T
      return frame_i_plus_1_img , homogines_trans, frame_i_keypoints, frame_i_plus_1_keypoints ,matches


  def apply_vo_on_all_frames(self,detector_type = 'sift',matcher = 'BF'):

    #get first frame:
    self.detector_type = detector_type
    self.matcher = matcher
    self.frame_i_points_list = []
    self.frame_i_plus_1_points = []
    self.matches_list = []
    frame_i_img = next(self.frame_itr)
    T_list = [] 
    for i in range(self.number_of_frames-1):
      frame_i_plus_1_img , homogines_trans, frame_i_points, frame_i_plus_1_points , matches = self.find_rotation_and_translation_between_frames_no_scale(frame_i_img,detector_type,matcher)
      frame_i_img = frame_i_plus_1_img
      T_list.append(homogines_trans) 
      self.frame_i_points_list.append(frame_i_points)
      self.frame_i_plus_1_points.append(frame_i_plus_1_points)
      self.matches_list.append(matches)
    return T_list







  def display_mateched_key_point(self,img1,img2,img1_keypoints, img1_descriptor,img2_keypoints, img2_descriptor,matches):
    '''
    image1 = np.copy(img1)
    image2 = np.copy(img2)

    #cv2.drawKeypoints(img1, img1_keypoints, keypoints_without_size, color = (0, 255, 0))

    image1 = cv2.drawKeypoints(img1, img1_keypoints, image1, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2 = cv2.drawKeypoints(img2, img2_keypoints, image2, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    
    # Display image with and without keypoints size
    fx, plots = plt.subplots(1, 2, figsize=(40,20))

    plots[0].set_title("image1 keypoints With Size")
    plots[0].imshow(image1, cmap='gray')

    plots[1].set_title("Train keypoints With Size")
    plots[1].imshow(image2, cmap='gray')
    plt.show()
    '''
    # Print the number of keypoints detected in the training image
    #print("Number of Keypoints Detected In The Training Image1: ", len(img1_keypoints))

    # Print the number of keypoints detected in the query image
    #print("Number of Keypoints Detected In The Query Image2: ", len(img2_keypoints))
  
    result = cv2.drawMatches(img1,img1_keypoints,img2,img2_keypoints,matches[:],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    '''
    # Display the best matching points
    plt.rcParams['figure.figsize'] = [16.0, 7.0]
    plt.title('Best Matching Points')
    plt.imshow(result)
    plt.show(result)
    '''
    # Print total number of matching points between the training and query images
    #print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
    return result


  def display_gt_trajectory(self):
    
    x_gt = [0]
    y_gt = [0]
    for gt_pose in self.gt_poses :
      x_gt.append(gt_pose[0][3])
      y_gt.append(gt_pose[2][3])
    
    xy_gt = np.vstack((x_gt,y_gt)).T
    graphs.plot_single_graph(xy_gt,'Trajectory:', 'x [m]','y [m]','Ground Truth')
    return xy_gt

# Feature matching (tracking) (outlier removal):
  # match them using bf matcher and remove outliears using ransac , if using orb use FlannBasedMatcher 
    
  
    

# Motion estimation 2D-2D:
  # compute essntial matrix from image pair l_k , l_k-1
  #decompose essential matrix into R_k and t_k to form T_k
  #compute relitive scale and rescale t_k acourdingly
  # concatinate transformation by computing C_k = c_k-1 @ T_k

# Local optimization:

  def diff(self, diff_arry_till_i_minus_1, xy_i):
    diff_arry_till_i_minus_1.append(xy_i - diff_arry_till_i_minus_1[-1] ) 
    return diff_arry_till_i_minus_1

  #make more efficaint! by adding elemnts to diff array each time
  def scale_correction(self,diff_gt_array_till_i_minus_1,diff_vo_array_till_i_minus_1,gt_xy_i,vo_xy_i):
    diff_gt_0_to_i = self.diff(diff_gt_array_till_i_minus_1, gt_xy_i)
    #print("diff_gt_0_to_i", diff_gt_0_to_i)
    diff_vo_0_to_i = self.diff(diff_vo_array_till_i_minus_1, vo_xy_i)
    #print("diff_vo_0_to_i", diff_vo_0_to_i)
    scale_i = np.median(np.linalg.norm(diff_gt_0_to_i,axis = 1))/np.median(np.linalg.norm(diff_vo_0_to_i, axis = 1))
    #print("scale_i", scale_i)
    return scale_i , diff_gt_0_to_i , diff_vo_0_to_i

  def calc_cur_rot_and_trans_with_scale(self,accumulated_t ,cur_t, accumulated_R, cur_R, scale):
    t = accumulated_t + scale*np.matmul(accumulated_R,cur_t)
    R = np.matmul(accumulated_R,cur_R)
    return R, t

  def calc_trajectory(self,T_list,xy_gt = None, scale = False):

    x_est = [0]
    y_est = [0]
    C_acumilated = np.eye(4)
    C_list = []
    # intlizing xy est 
    for T in T_list:
      C_acumilated = np.matmul(C_acumilated,T)
      C_list.append(C_acumilated)
      x_translation = C_acumilated[0,3] # x_est[-1] + 
      y_translation = C_acumilated[2,3] #y_est[-1] + C_acumilated[1,3]
      x_est.append(x_translation)
      y_est.append(-y_translation)
    xy_est = np.vstack((np.array(x_est),np.array(y_est))).T
    #print ("C_list",C_list)
    

    if(scale == False):
      graphs.plot_single_graph(xy_est,'Trajectory:', 'x [m]','y [m]','estimated trajectory without scale')

    elif (scale == True):
      #get scale_i
      diff_gt_array_till_i_minus_1 = [np.array([0.,0.])]
      diff_vo_array_till_i_minus_1 = [np.array([0.,0.])]
      scale_array = []
      #print(xy_gt)
      for i in range(xy_gt.shape[0]-1):
        scale_i , diff_gt_array_till_i_minus_1 , diff_vo_array_till_i_minus_1 = self.scale_correction(diff_gt_array_till_i_minus_1,diff_vo_array_till_i_minus_1,xy_gt[i],xy_est[i])
        scale_array.append(scale_i)

      #print("scale_array", scale_array)
      scale_array = scale_array[2:]
      #print("scale_array", scale_array)
  
      print("self.number_of_frames", self.number_of_frames)
      x_est_scaled_t_minus_1 = [0]
      y_est_scaled_t_minus_1 = [0]
      x_est_scaled = []
      accumulated_R = np.eye(3)
      accumulated_t = np.array([0,0,0])
      #cheek again :
      print("len(scale_array)",len(scale_array))
      print("len(T_list)",len(T_list))
      for i, T in enumerate(T_list): 
        if(i<len(scale_array)):
          accumulated_R, accumulated_t = self.calc_cur_rot_and_trans_with_scale(accumulated_t, T[:3,3], accumulated_R,T[:3,:3] ,scale_array[i])
          x_translation = accumulated_t[0]
          y_translation = accumulated_t[2]
          x_est_scaled_t_minus_1.append(x_translation)
          y_est_scaled_t_minus_1.append(-y_translation)


      xy_est_scaled = np.vstack((np.array(x_est_scaled_t_minus_1),np.array(y_est_scaled_t_minus_1))).T
      graphs.plot_single_graph(xy_est_scaled,'Trajectory:', 'x [m]','y [m]','estimated trajectory with scale')


      graphs.plot_three_graphs(xy_gt, xy_est, xy_est_scaled,'Ground truth trajectory vs estimated trajectory using {} detector and dicripter {} matcher and outlier removal with ratio test '.format(self.detector_type,self.matcher) ,'x [m]', 'y [m]' , 'ground truth trajectory', 'estimated trajectory no scale','estimated trajectory with scale') 

      return xy_est, xy_est_scaled



  def build_animation(self,X_Y0, X_Y1, X_Y2,frame_itr1,frame_itr2, title, xlabel, ylabel, label0, label1, label2):
    
    frames = []
    fig = plt.figure(figsize=(16, 8))
    ax_img = fig.add_subplot(2,1,1)
    ax = fig.add_subplot(2,1,2)
    self.i = 0
    
    print("Creating animation")
    
    x0, y0, x1, y1, x2, y2 = [], [], [], [], [],[] 
    val0, = plt.plot([], [], 'b-', animated=True, label=label0)
    val1, = plt.plot([], [], 'k-', animated=True, label=label1)
    val2, = plt.plot([], [], 'g-', animated=True, label=label2)
    frame_i_plus_1_img = next(frame_itr2)
    plt.legend()
    values = np.hstack((X_Y0, X_Y1, X_Y2))
    

    def init():
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        val0.set_data([],[])
        val1.set_data([],[])
        val2.set_data([],[])  
        return val0, val1, val2, 

    
    def update(frame):
        
        frame_i_img = next(frame_itr1)
        result = copy.copy(frame_i_img)
        #print(self.frame_i_points_list[self.i])
        result = cv2.drawKeypoints(frame_i_img,self.frame_i_points_list[self.i],result) #,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.i = self.i+1
        '''
        #to diplay matched points between images (runs slowly)
        frame_i_plus_1_img = next(frame_itr2)
        frame_i_keypoints, frame_i_descriptor, frame_i_plus_1_keypoints, frame_i_plus_1_descriptor = self.extract_key_points_and_decripters_between_two_frames(frame_i_img, frame_i_plus_1_img, detector_type = self.detector_type)
        matches = self.match_features_between_two_frames(frame_i_descriptor, frame_i_plus_1_descriptor, detector_type = self.detector_type, matcher = self.matcher)
        result = self.display_mateched_key_point(frame_i_img,frame_i_plus_1_img,frame_i_keypoints, frame_i_descriptor, frame_i_plus_1_keypoints, frame_i_plus_1_descriptor,matches)
        '''
        
        #print("result", result)
        # Display the best matching points
        #ax_img.rcParams['figure.figsize'] = [16.0, 7.0]
        ax_img.set_title('Best Matching Points between two frames')
        ax_img.imshow(result)
        ax_img.set_axis_off()
        
               
        ax.set_xlim(-10, 50 + frame[0])
        ax.set_ylim(-50 + frame[1], 60)
        x0.append(frame[0])
        y0.append(frame[1])
        x1.append(frame[2])
        y1.append(frame[3])
        x2.append(frame[4])
        y2.append(frame[5])
        val0.set_data(x0, y0)
        val1.set_data(x1, y1)
        val2.set_data(x2, y2)


        
        return val0, val1, val2,# frame_i_img
    
    anim = animation.FuncAnimation(fig, update, frames=values, init_func=init, interval=1, blit=True)
    return anim


  def save_animation(self,ani, basedir, file_name):
      print("Saving animation")
      Writer = animation.writers['ffmpeg']
      writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
      ani.save(os.path.join(basedir, f'{file_name}.mp4'), writer=writer)
      print("Animation saved")
 



      
    