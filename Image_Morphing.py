import cv2
import numpy as np
import matplotlib.pyplot as plt
import ctypes
from scipy.spatial import Delaunay


def Bilinear_Interpolation(valid_pts, M, img):
    backward_pts = valid_pts@M.T    
    backward_pts[:,0][backward_pts[:,0] > img.shape[0]-1] = img.shape[0]-1
    backward_pts[:,0][backward_pts[:,0] < 0] = 0        
    backward_pts[:,1][backward_pts[:,1] > img.shape[1]-1] = img.shape[1]-1
    backward_pts[:,1][backward_pts[:,1] < 0] = 0    
    logic1 = np.logical_and(backward_pts[:,0] > 0, backward_pts[:,0] < img.shape[0]-1)
    logic2 = np.logical_and(backward_pts[:,1] > 0, backward_pts[:,1] < img.shape[1]-1)
    logic = np.logical_and(logic1, logic2)
    inregion_pts = backward_pts[np.where(logic)[0],:]    
    # corner1 ........ corner3
    #   .                .
    #   .                .
    #   .                .
    # corner2 ........ corner4
    interpolated_pixels = np.zeros((backward_pts.shape[0], 3))
    interpolated_pixels[np.where(~logic)[0],:] = img[np.int64(backward_pts[np.where(~logic)[0],:])[:,0],
                                                     np.int64(backward_pts[np.where(~logic)[0],:])[:,1],:].copy()
    temp = np.where(logic)[0]
    corner1 = np.int64(inregion_pts)
    corner2 = np.int64(inregion_pts) + np.array([1, 0])
    corner3 = np.int64(inregion_pts) + np.array([0, 1])
    corner4 = np.int64(inregion_pts) + 1
    area = (inregion_pts - corner1)[:,0] * (inregion_pts - corner1)[:,1]
    for i in range(3):
        interpolated_pixels[temp,i] += img[corner4[:,0], corner4[:,1], i]*area
    
    area = (corner4 - inregion_pts)[:,0] * (corner4 - inregion_pts)[:,1]
    for i in range(3):
        interpolated_pixels[temp,i] += img[corner1[:,0], corner1[:,1], i]*area
    
    area = abs((inregion_pts - corner2)[:,0] * (inregion_pts - corner2)[:,1])
    for i in range(3):
        interpolated_pixels[temp,i] += img[corner3[:,0], corner3[:,1], i]*area
    
    area = abs((inregion_pts - corner3)[:,0] * (inregion_pts - corner3)[:,1])
    for i in range(3):
        interpolated_pixels[temp,i] += img[corner2[:,0], corner2[:,1], i]*area
        
    return np.int64(interpolated_pixels)

def Counter_Clock_Checker(v_x, v_y):
    
    if np.sum((np.roll(v_x, -1) - v_x)*(np.roll(v_y,-1)+v_y)) < 0:
        print('The points are counter-clockwise')
        return (v_x, v_y)
    else:
        print('The points are clockwise')
        print('direction changed to counter-clockwise')
        return (v_x[::-1], v_y[::-1])
    
    
def affine(x, y):
    b = np.reshape(y, len(y)*2)
    A = np.zeros((len(y)*2, 6))
    for i in range(A.shape[0]):
        if i % 2 == 0:
            t = i//2
            A[i][0] = x[t][0]
            A[i][1] = x[t][1]
            A[i][2] = 1
        
        else:
            t = (i-1)//2
            A[i][3] = x[t][0]
            A[i][4] = x[t][1]
            A[i][5] = 1
            
    coef = np.linalg.inv(A.T @ A) @ A.T @ b

    M = np.array([[coef[0], coef[1], coef[2]],[coef[3], coef[4], coef[5]]], dtype=np.float64)
    return M

def Triangle_Interior(Triangle_Vertices):
    # Triangle Vertices in a 3*2 array
    # The vertices should be in counterclockwise form
    # THe output is the corrdinates of interior points of the triangle
    
    pts_min = np.min(Triangle_Vertices, axis = 0)
    pts_max = np.max(Triangle_Vertices, axis = 0)
    
    rectangle_pts = np.zeros(((pts_max[0] - pts_min[0] + 1)*(pts_max[1]-pts_min[1] + 1),
                              2), dtype = np.float64)
    
    rectangle_pts[:,0] = np.tile(np.arange(pts_min[0], pts_max[0]+1, 1),
                                 pts_max[1]-pts_min[1] + 1).astype(np.float64)
    
    
    temp = np.zeros((pts_max[1]-pts_min[1] + 1, 1))
    temp[:,0] = np.arange(pts_min[1], pts_max[1] + 1, 1)
    
    rectangle_pts[:,1] =  np.reshape(np.tile(temp, pts_max[0] - pts_min[0] + 1),
                                     (rectangle_pts.shape[0], 1), order = 'C').T[0]
    
    Pa, Pb, Pc = Triangle_Vertices[:].copy().astype(np.float64)
    Mab, Mbc, Mca = (Pa + Pb)/2, (Pb + Pc)/2, (Pc + Pa)/2 # Medians of trangle

    Vab, Vbc, Vca = Pb - Pa, Pc - Pb, Pa - Pc
    Nab, Nbc, Nca = np.zeros(2), np.zeros(2), np.zeros(2)
    Nab[0], Nab[1] = -Vab[1].copy(), Vab[0].copy()
    Nbc[0], Nbc[1] = -Vbc[1].copy(), Vbc[0].copy()
    Nca[0], Nca[1] = -Vca[1].copy(), Vca[0].copy()

    dot1 = (rectangle_pts - Mab)@Nab
    dot2 = (rectangle_pts - Mbc)@Nbc
    dot3 = (rectangle_pts - Mca)@Nca
    
    logic_temp = np.logical_and(dot1 >= 0, dot2 >= 0)*1.0
    logic = np.logical_and(logic_temp != 0, dot3 >= 0)*1.0
    valid_pts = rectangle_pts[np.where(logic != 0)[0]]
    return valid_pts


def Video_Maker(path, frames, fps):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), fps,
                          frames[0].shape[:2][::-1])
 
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

def Image_Morphing(img1, img2,
                   points1, points2,
                   triangle,
                   frame_num,
                   path1, path2):
    video_frames = []
    frame = 1
    for t in np.linspace(0, 1, frame_num):    
        img_warp1 = np.zeros(img1.shape, dtype = np.uint8)
        img_warp2 = np.zeros(img2.shape, dtype = np.uint8)
        for i in range(triangle.nsimplex):
            tri_vertices1 = points1[triangle.simplices[i]]
            tri_vertices2 = points2[triangle.simplices[i]]
            tri_vertices_middle = tri_vertices1 + t*(tri_vertices2 - tri_vertices1)
            M = affine(tri_vertices_middle, tri_vertices1)
            
            valid_pts = Triangle_Interior(np.int64(tri_vertices_middle))
            valid_pts = np.concatenate((valid_pts, np.ones((1, valid_pts.shape[0])).T), axis = 1)   
            img_warp1[np.int64(valid_pts[:,0]),
                      np.int64(valid_pts[:,1]),:] = Bilinear_Interpolation(valid_pts,
                                                                           M, img1)
            
            
            M = affine(tri_vertices_middle, tri_vertices2)
            img_warp2[np.int64(valid_pts[:,0]),
                      np.int64(valid_pts[:,1]),:] = Bilinear_Interpolation(valid_pts,
                                                                           M, img2)
                                                                           
        img_morph = np.uint8(img_warp1*(1-t) + img_warp2*t)
        
        if frame == np.int64(1/3*frame_num):
            cv2.imwrite(path1, img_morph)
        if frame == np.int64(2/3*frame_num):
            cv2.imwrite(path2, img_morph)
            
        video_frames.append(img_morph)  
        frame += 1
    video_frames = video_frames + video_frames[::-1]    
    return video_frames
        
def Read_Points(path):
    with open(path) as f:
        lines = f.readlines()
    points = np.zeros((len(lines), 2))
    for i in range(len(lines)):
        temp = lines[i][0:-1].split(',')
        points[i, 0] = int(temp[0])
        points[i, 1] = int(temp[1])
    return points


def nothing(x):
    # Do nothing!!
    pass


# Functions used for finding corresponding points between two images with the
# help of user. (Corresponding points should be chosen in order!)
def Click_Point1(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN:
        img_copy1[y-2:y+2,x-2:x+2,:] = (0,255,0)
        cv2.imshow(title1, img_copy1)
        X1.append(y)
        Y1.append(x)

def Click_Point2(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN:
        img_copy2[y-2:y+2,x-2:x+2,:] = (0,255,0)
        cv2.imshow(title2, img_copy2)
        X2.append(y)
        Y2.append(x)
        
def Choosing_Points(img1, img2, scale1=1, scale2=1):
    global X1, X2
    global Y1, Y2
    global img_copy1, img_copy2
    global title1, title2
    title1 = 'img1'
    title2 = 'img2'
    
    ratio1 = img1.shape[1]/img1.shape[0]
    ratio2 = img2.shape[1]/img2.shape[0]
    
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    X1 = []
    Y1 = []
    X2 = []
    Y2 = []

    img_copy1 = cv2.resize(img1.copy(), (np.int64(screensize[1]//2*ratio1), screensize[1]//2))
    img_copy2 = cv2.resize(img2.copy(), (np.int64(screensize[1]//2*ratio2), screensize[1]//2))
    
    cv2.imshow(title1, img_copy1)
    cv2.imshow(title2, img_copy2)
    cv2.setMouseCallback(title1, Click_Point1)
    cv2.setMouseCallback(title2, Click_Point2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (np.array(X1)*img1.shape[0]/img_copy1.shape[0],
            np.array(Y1)*img1.shape[1]/img_copy1.shape[1],
            np.array(X2)*img2.shape[0]/img_copy2.shape[0],
            np.array(Y2)*img2.shape[1]/img_copy2.shape[1])






if __name__ == '__main__':
    img1 = cv2.imread("Img1.jpg")
    img2 = cv2.imread("Img2.jpg")
    x1,y1,x2,y2 = Choosing_Points(img1, img2)
    
    points1 = np.zeros((x1.shape[0]+4, 2))
    points2 = np.zeros((x2.shape[0]+4, 2))
    points1[:-4, 0] = x1
    points1[:-4, 1] = y1
    points1[-4, 0] = img1.shape[0]-1
    points1[-3, 1] = img1.shape[1]-1
    points1[-2, :] = (img1.shape[0]-1, img1.shape[1]-1)
    
    points2[:-4, 0] = x2
    points2[:-4, 1] = y2
    points2[-4, 0] = img2.shape[0]-1
    points2[-3, 1] = img2.shape[1]-1
    points2[-2, :] = (img2.shape[0]-1, img2.shape[1]-1)
    
    tri1 = Delaunay(points1)
    
    video_frames = Image_Morphing(img1, img2,
                                  points1, points2,
                                  tri1, 45,
                                  'Morph1.jpg', 'Morph2.jpg')
    Video_Maker('Video_Result.mp4', video_frames, 30.0)
    