# Summary - This code performs calculates the intrinsic matrix and uses the distortion settings to produce an undistorted image for each dataset
# Author - Kushal, based on Ajay's work
# Created: 06/22/2023
# Last updated: 8/01/2024
import csv
import os

# import statements
# import os
import cv2
import numpy as np


folderDay = ['Maca20110111', '20110110-tm1-ub-maca', '20110121-tm3-uy-macaNoSynchro','Maca20101120', 'Maca20101122','Maca20101121']

cams = [0,1]
DataNames = [folder + "_cam" + str(cam) for folder in folderDay for cam in cams]

folders = ['Datasets2']
rows = [[], []]
# useFlag=[ True, False]
i = -1
for folder in folders:
    i += 1
    for currFolderDay in folderDay:
        for camNum in cams:
            base = '/Users/Kushal/Coding/PycharmProjects/UGSRPenguin/' + folder + '/'
            output = "Out/Output6/"
            if folder == 'Datasets':
                output = "Out/Output5/"
            if (camNum == 0):
                folder_name = base + currFolderDay + '/Calibration/intrinsic/cam0/saved_frames1'
                file_name = base + currFolderDay + '/Calibration/intrinsic/cam0/sampleFrame'
                # imageNameOG = '/Users/Kushal/Coding/PycharmProjects/UGSRPenguin/Output/' +currFolderDay + 'cam0OG.png'
                imageName = '/Users/Kushal/Coding/PycharmProjects/UGSRPenguin/' + output + currFolderDay + 'cam0.png'
            else:
                folder_name = base + currFolderDay + '/Calibration/intrinsic/cam1/saved_frames1'
                file_name = base + currFolderDay + '/Calibration/intrinsic/cam1/sampleFrame'
                # imageNameOG = '/Users/Kushal/Coding/PycharmProjects/UGSRPenguin/Output/' +currFolderDay + 'cam1OG.png'
                imageName = '/Users/Kushal/Coding/PycharmProjects/UGSRPenguin/' + output + currFolderDay + 'cam1.png'
            # custom class for camera calibration
            from camera_calibration2 import Camera_Calibration

            cam0_img_folder = folder_name
            cam1_img_folder = folder_name
            print(currFolderDay + " " + str(i))
            print('Camera:', camNum)
            # file_name = 'data/' + currFolderDay + '/Calibration/intrinsic/cam1/sampleFrame'
            imgList = os.listdir(file_name)
            # pick one of the image files to show distorted and undistrorted
            j=imgList[0]
            if(".DS" in j):
                j="4924_IMAQdxcam1_357.png"
            imageFile = file_name + '/' + j
            # imageFile= file_name + "/" + '4924_IMAQdxcam1_357.png'
            # print(imageFile)
            # initialize camera calibration
            calib_cam = Camera_Calibration(cam0_img_folder, cam1_img_folder)
            # calib_cam.stereo_calibrate(image_folder0='20110110-tm1-ub-maca/Calibration/Extrinseque/cam0', image_folder1='20110110-tm1-ub-maca/Calibration/Extrinseque/cam1', check_undistortion=False, custom_pts_filename= "manual_pts_data.pkl" )
            image_files = calib_cam.get_image_files(folder_name)  # extracting image files from the folder
            # image_files = calib_cam.get_image_files(folder_name='Maca20101120/Calibration/Intrinseque/cam0LotsofPhotos') #extracting image files from the folder
            threedpoints, twodpoints, grey = calib_cam.detect_checker_cornersv2(image_files, verify_corners=False,
                                                                                skip=1)  # get 2d/3d points correspondences
            print(str(len(threedpoints)) + " " + str(len(twodpoints)))
            ret, cam_matrix_0, dist_c0, r_vecs_0, t_vecs_0, repr_error_0 = calib_cam.calibrate_one_camera(threedpoints,
                                                                                                          twodpoints,
                                                                                                          grey,
                                                                                                          flag=True)  # caibrate cam
            # print("old distortion: ", dist_c0)

            # load pickled calibration result
            # with open('calibration_result.pickle', 'rb') as file:
            #     calib_result= pickle.load(file)
            # P0,P1= calib_cam.get_projection_mat(matrix_0=calib_result['camera_matrix_c0'], matrix_1=calib_result['camera_matrix_c1'], R=calib_result['c0_R_c1'], T=calib_result['c0_t_c1'])

            # cam_matrix_0 = calib_result['camera_matrix_c0']
            # cam_matrix_1 = calib_result['camera_matrix_c1']
            # dist_c0= calib_result['distortion_c0']
            # dist_c1= calib_result['distortion_c1']

            # rectify_scale = 0.5  # Free scaling parameter check this https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#fisheye-stereorectify
            # R0, R1, P0, P1, Q, roi1, roi2 = cv2.stereoRectify(cam_matrix_0, dist_c0, cam_matrix_1, dist_c1, (480, 480),
            #                                                   calib_result['c0_R_c1'], calib_result['c0_t_c1'],
            #                                                 #   alpha=rectify_scale,
            #                                                   )

            # distortion_0[0][-3:] = np.array([0.,0.,0.]) #set the last 3 distortion coeffs to 0
            #cam_matrix_0=np.array([[724.51171464 ,0,322.03569772],[0,719.13120469,203.1987187],[0,0,1]],dtype=np.float64)
            img_c0 = cv2.imread(imageFile)
            print(imageFile)
            # /Users/Kushal/Coding/PycharmProjects/UGSRPenguin/FixedUpData_July102024/Maca20101121v2/Calibration/intrinsic/cam0/sampleFrame/1708_IMAQdxcam0_406.png
            # img_c0 = cv2.imread('20110110-tm1-ub-maca/Calibration/Intrinseque/cam1/0746_IMAQdxcam1_100.png')
            h0, w0 = img_c0.shape[:2]
            # actual_img= img_c0.copy()

            # actual_img = cv2.line(img_c0,(23,312),(478,370),(0,0,255),1)
            # actual_img = cv2.line(actual_img,(130,228),(440,260),(0,0,255),1)

            # cv2.imshow("actual image",img_c0)
            # cv2.waitKey(0)

            # cv2.imwrite(imageNameOG, img_c0)

            # line points
            # (23,312),(478,370) - l1
            # (130,228), (440,260) -l2
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_matrix_0, dist_c0, (w0, h0), 1,
                                                              (w0, h0))  # refine the camera matrix with the image size

            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_matrix_1, dist_c1, (w0,h0), 1, (w0,h0)) #refine the camera matrix with the image size
            map0_x, map0_y = cv2.initUndistortRectifyMap(cam_matrix_0, dist_c0, None, newcameramtx, (h0, w0),
                                                         cv2.CV_32FC1)
            im_remapped_0 = cv2.remap(img_c0, map0_x, map0_y, cv2.INTER_LANCZOS4, cv2.INTER_AREA)
            # map1_x, map1_y = cv2.initUndistortRectifyMap(cam_matrix_1, dist_c1, None, newcameramtx, (h0,w0), cv2.CV_16SC2)
            # im_remapped_0 = cv2.remap(img_c0, map1_x, map1_y, cv2.INTER_LANCZOS4)

            # #im_remapped_0 = cv2.undistort(img_c0, matrix_0, distortion_0, None, newcameramtx)
            #
            # x, y, w, h = roi
            # im_remapped_0 = im_remapped_0[y:y+h, x:x+w]

            # after undistortion
            # (24,312), (474,365)

            # code to draw lines on the image

            # im_remapped_0 = cv2.line(im_remapped_0,(23,312),(478,370),(0,0,255),1)
            # im_remapped_0 = cv2.line(im_remapped_0,(130,228),(440,260),(0,0,255),1)

            # im_remapped_0 = cv2.line(im_remapped_0,(24,312), (474,365),(0,0,255),1)

            # cv2.imwrite("undistorted_image_5coeff.png",im_remapped_0)

            # cv2.imshow('undistorted image', im_remapped_0)
            cv2.imwrite(imageName, im_remapped_0)
            rows[i].append(repr_error_0)
with open('/Users/Kushal/Coding/PycharmProjects/UGSRPenguin/UndistortFixed/output2.csv', mode='w', newline='') as file:
    print('k')
    writer = csv.writer(file)

    # Write the header
    writer.writerow(DataNames)

    # Write the rows
    writer.writerows(rows)
