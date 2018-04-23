# python3 and python2
import numpy as np
import cv2
import glob, os
import tqdm
from itertools import repeat
from multiprocessing import Pool
from functools import partial

def align_2p(img, left_eye, right_eye):
    width = 256
    eye_width = 70

    transform = np.matrix([
            [1, 0, left_eye[0]],
            [0, 1, left_eye[1]],
            [0, 0, 1]
    ], dtype='float')

    th = np.pi + -np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])
    transform *= np.matrix([
            [np.cos(th), np.sin(th), 0],
            [-np.sin(th), np.cos(th), 0],
            [0, 0, 1]
    ], dtype='float')

    scale = np.sqrt((left_eye[1] - right_eye[1]) ** 2 + (left_eye[0] - right_eye[0]) ** 2) / eye_width
    transform *= np.matrix([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
    ], dtype='float')

    transform *= np.matrix([
            [1, 0, -(width - eye_width) / 2],
            [0, 1, -width / 2.42],
            [0, 0, 1]
    ], dtype='float')

    transform = np.linalg.inv(transform)
    jmg = cv2.warpAffine(img, transform[:2], (width, width))
    return jmg                                                                                                                                                                                                                                      

def align_face_2p(img, landmarks):
    left_eye = (landmarks[0], landmarks[1])
    right_eye = (landmarks[2], landmarks[3])
    aligned_img = align_2p(img, left_eye, right_eye)
    return aligned_img
                                                                                                                                                                                                                                         
# average landmarks                                                                                                                                                                                                                         
mean_face_lm5p = np.array([                                                                                                                                                                                                                 
    [-0.17607, -0.172844],  # left eye pupil                                                                                                                                                                                                
    [0.1736, -0.17356],  # right eye pupil                                                                                                                                                                                                  
    [-0.00182, 0.0357164],  # nose tip                                                                                                                                                                                                      
    [-0.14617, 0.20185],  # left mouth corner                                                                                                                                                                                               
    [0.14496, 0.19943],  # right mouth corner                                                                                                                                                                                               
]) 

def _get_align_5p_mat23_size_256(lm):                                                                                                                                                                                                       
    # legacy code                                                                                                                                                                                                                           
    width = 256                                                                                                                                                                                                                             
    mf = mean_face_lm5p.copy()                                                                                                                                                                                                              
                                                                                                                                                                                                                                            
    # Assumptions:                                                                                                                                                                                                                          
    # 1. The output image size is 256x256 pixels                                                                                                                                                                                            
    # 2. The distance between two eye pupils is 70 pixels                                                                                                                                                                                   
    ratio = 70.0 / (                                                                                                                                                                                                                        
       256.0 * 0.34967                                                                                                                                                                                                                     
    )  # magic number 0.34967 to compensate scaling from average landmarks                                                                                                                                                                  
                                                                                                                                                                                                                                            
    left_eye_pupil_y = mf[0][1]                                                                                                                                                                                                             
    # In an aligned face image, the ratio between the vertical distances from eye to the top and bottom is 1:1.42                                                                                                                           
    ratioy = (left_eye_pupil_y * ratio + 0.5) * (1 + 1.42)                                                                                                                                                                                  
    mf[:, 0] = (mf[:, 0] * ratio + 0.5) * width                                                                                                                                                                                             
    mf[:, 1] = (mf[:, 1] * ratio + 0.5) * width / ratioy                                                                                                                                                                                    
    mx = mf[:, 0].mean()                                                                                                                                                                                                                    
    my = mf[:, 1].mean()                                                                                                                                                                                                                    
    dmx = lm[:, 0].mean()                                                                                                                                                                                                                   
    dmy = lm[:, 1].mean()                                                                                                                                                                                                                   
    mat = np.zeros((3, 3), dtype=float)                                                                                                                                                                                                     
    ux = mf[:, 0] - mx                                                                                                                                                                                                                      
    uy = mf[:, 1] - my                                                                                                                                                                                                                      
    dux = lm[:, 0] - dmx                                                                                                                                                                                                                    
    duy = lm[:, 1] - dmy                                                                                                                                                                                                                    
    c1 = (ux * dux + uy * duy).sum()                                                                                                                                                                                                        
    c2 = (ux * duy - uy * dux).sum()                                                                                                                                                                                                        
    c3 = (dux**2 + duy**2).sum()                                                                                                                                                                                                            
    a = c1 / c3                                                                                                                                                                                                                             
    b = c2 / c3                                                                                                                                                                                                                             
                                                                                                                                                                                                                                            
    kx = 1                                                                                                                                                                                                                                  
    ky = 1                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                            
    s = c3 / (c1**2 + c2**2)                                                                                                                                                                                                                
    ka = c1 * s                                                                                                                                                                                                                             
    kb = c2 * s                                                                                                                                                                                                                             
                                                                                                                                                                                                                                            
    transform = np.zeros((2, 3))                                                                                                                                                                                                            
    transform[0][0] = kx * a                                                                                                                                                                                                                
    transform[0][1] = kx * b                                                                                                                                                                                                                
    transform[0][2] = mx - kx * a * dmx - kx * b * dmy                                                                                                                                                                                      
    transform[1][0] = -ky * b                                                                                                                                                                                                               
    transform[1][1] = ky * a                                                                                                                                                                                                                
    transform[1][2] = my - ky * a * dmy + ky * b * dmx                                                                                                                                                                                      
    return transform     

def get_align_5p_mat23(lm5p, size):                                                                                                                                                                                                         
    """Align a face given 5 facial landmarks of                                                                                                                                                                                             
    left_eye_pupil, right_eye_pupil, nose_tip, left_mouth_corner, right_mouth_corner                                                                                                                                                        
                                                                                                                                                                                                                                            
    :param lm5p: nparray of (5, 2), 5 facial landmarks,                                                                                                                                                                                     
                                                                                                                                                                                                                                            
    :param size: an integer, the output image size. The face is aligned to the mean face                                                                                                                                                    
                                                                                                                                                                                                                                            
    :return: a affine transformation matrix of shape (2, 3)                                                                                                                                                                                 
    """                                                                                                                                                                                                                                     
    mat23 = _get_align_5p_mat23_size_256(lm5p.copy())                                                                                                                                                                                       
    mat23 *= size / 256                                                                                                                                                                                                                     
    return mat23                                                                                                                                                                                                                            
                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                            
def align_given_lm5p(img, lm5p, size):                                                                                                                                                                                                      
    mat23 = get_align_5p_mat23(lm5p, size)                                                                                                                                                                                                  
    return cv2.warpAffine(img, mat23, (size, size))  


def align_face_5p(img, landmarks):
    aligned_img = align_given_lm5p(img, np.array(landmarks).reshape((5, 2)), 256)  
    return aligned_img

def work(data_dir, out_dir, landmarks, i):
    src_imname = os.path.join(data_dir, 'data', '{:06d}.jpg'.format(i+1))
    des_imname = os.path.join(out_dir, '{:06d}.jpg'.format(i+1))
    img = cv2.imread(src_imname)
    aligned_img = align_face_5p(img, landmarks[i])
    cv2.imwrite(des_imname, aligned_img)
    return 0

def main(data_dir, out_dir, thread_num):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(data_dir, 'list_landmarks_celeba.txt'), 'r') as f:
        landmarks = [list(map(int, x.split()[1:11])) for x in f.read().strip().split('\n')[2:]]

    im_list = glob.glob(os.path.join(data_dir, 'data/*.jpg'))

    pool = Pool(thread_num)
    # pool.starmap(work, zip(range(len(im_list)), repeat(data_dir), repeat(out_dir), repeat(landmarks)))
    partial_work = partial(work, data_dir, out_dir, landmarks)
    pool.map(partial_work, range(len(im_list)))
    pool.close()
    pool.join()


if __name__ == '__main__':  
    os.environ["CUDA_VISIBLE_DEVICES"] = '' 
    main('./datasets/celebA/', './datasets/celebA/align_5p/', 30)
