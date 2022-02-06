import os
from visual_odometry import VisualOdometry
from data_loader import DataLoader


class ProjectQuestions:
    def __init__(self,vo_data):
        assert type(vo_data) is dict, "vo_data should be a dictionary"
        assert all([val in list(vo_data.keys()) for val in ['sequence', 'dir']]), "vo_data must contain keys: ['sequence', 'dir']"
        assert type(vo_data['sequence']) is int and (0 <= vo_data['sequence'] <= 10), "sequence must be an integer value between 0-10"
        assert type(vo_data['dir']) is str and os.path.isdir(vo_data['dir']), "dir should be a directory"
        self.vo_data = vo_data
    
    
    def Q2(self):
        vo_data = DataLoader(self.vo_data)
        vo = VisualOdometry(vo_data)
        xy_gt = vo.display_gt_trajectory()
        #detector_type = 'optical flow' 'sift', 'orb'    matcher = 'BF','FLANN'
        detctor = 'sift'
        match = 'FLANN'
        T_list = vo.apply_vo_on_all_frames(detector_type = detctor,matcher = match)
        vo.calc_trajectory(T_list)
        xy_est, xy_est_scaled = vo.calc_trajectory(T_list,xy_gt,scale=True)
        vo_data_for_animation1 = DataLoader(self.vo_data)
        vo_data_for_animation2 = DataLoader(self.vo_data)
        ani = vo.build_animation(xy_gt[:xy_est_scaled.shape[0]],xy_est[:xy_est_scaled.shape[0]], xy_est_scaled, vo_data_for_animation1.images,vo_data_for_animation2.images,'VO using {}, {} matching with threshhold = 0.6 and ransac'.format(detctor,match),'x [m]', 'y [m]' , 'gt' ,'est no scale', 'est with scale')
        vo.save_animation(ani, '/content/drive/MyDrive/mapping_and_perception/project_3/Q_2_visual_odomatry', "VO animation {} {} with key points" .format(detctor,match))
        
        
        
        #vo.find_rotation_and_translation_between_frames_no_scale()

        
        
    
    def run(self):
        self.Q2()
    