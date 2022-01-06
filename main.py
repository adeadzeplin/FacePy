import random
from math import sin

import mediapipe as mp
import cv2

import pyvirtualcam



def fake_cam():



    # with pyvirtualcam.Camera(width=1280, height=720, fps=20) as cam:
    #     print(f'Using virtual camera: {cam.device}')
    #     frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    #     while True:
    #         frame[:] = cam.frames_sent % 255  # grayscale animation
    #         cam.send(frame)
    #         cam.sleep_until_next_frame()

    height = 720
    width = 1280

    mp_drawing = mp.solutions.drawing_utils

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        scroller = 0
        with pyvirtualcam.Camera(width=width, height=height, fps=30) as cam:
            while cap.isOpened():
                scroller = (scroller + 2) % 256
                ret, frame = cap.read()

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Make Detections
                results = holistic.process(image)

                # print(type(results.face_landmarks))

                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hchannel = hsv[:, :, :]
                # hchannel = 0 * hchannel
                hsv[:, :, :] = hchannel
                image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # n = 256
                # TWO_PI = 3.14159 * 2
                # red = 128 + sin(scroller * TWO_PI / n + 0) * 127
                # grn = 128 + sin(scroller * TWO_PI / n + TWO_PI / 3) * 127
                # blu = 128 + sin(scroller * TWO_PI / n + 2 * TWO_PI / 3) * 127
                #
                # face_col = (red, grn, blu)
                face_col = (0,200,0)

                face_coords = mp_drawing.get_landmark_coords(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1),  # dots
                                          mp_drawing.DrawingSpec(color=face_col, thickness=1, circle_radius=1)  # Lines
                                          )
                # print(face_coords)
                min_c = (width,height)
                max_c = (0,0)
                if face_coords is not None:
                    for i in face_coords:
                        if min_c[0] > face_coords[i][0]:
                            min_c = (face_coords[i][0], min_c[1])
                        if min_c[1] > face_coords[i][1]:
                            min_c = (min_c[0], face_coords[i][1])

                        if max_c[0] < face_coords[i][0]:
                            max_c = (face_coords[i][0], max_c[1])
                        if max_c[1] < face_coords[i][1]:
                            max_c = (max_c[0], face_coords[i][1])

                    # cv2.rectangle(image,min_c,max_c,(100,100,100),3)
                cv2.rectangle(image, (0,0), (min_c[0],height), (0, 0, 0), -1)
                cv2.rectangle(image, (min_c[0],0), (max_c[0],min_c[1]), (0, 0, 0), -1)
                cv2.rectangle(image, (max_c[0],0), (width,height), (0, 0, 0), -1)
                cv2.rectangle(image, (min_c[0],max_c[1]), (max_c[0],height), (0, 0, 0), -1)




                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=face_col, thickness=1, circle_radius=1)
                                          )

                # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=face_col, thickness=1, circle_radius=1)
                                          )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=face_col, thickness=1, circle_radius=1)
                                          )
                image = cv2.resize(image, (width, height ))
                # cv2.imshow('Raw Webcam Feed', image)

                cam.send(image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

    pass



def main_tracking():
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(0)
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        scroller = 0

        while cap.isOpened():
            scroller = (scroller + 2) % 256
            ret, frame = cap.read()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)




            # Make Detections
            results = holistic.process(image)




            # print(type(results.face_landmarks))


            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks



            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hchannel = hsv[:, :, :]
            hchannel = 0 * hchannel
            hsv[:, :, :] = hchannel
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


            n = 256
            TWO_PI = 3.14159 * 2
            red = 128 + sin(scroller * TWO_PI / n + 0)*127
            grn = 128 + sin(scroller * TWO_PI / n + TWO_PI / 3)*127
            blu = 128 + sin(scroller * TWO_PI / n + 2 * TWO_PI / 3)*127

            face_col = (red, grn, blu)
            # face_col = (0,200,0)


            
            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION ,
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1), # dots
                                      mp_drawing.DrawingSpec(color=face_col, thickness=1, circle_radius=1)  # Lines
                                      )

            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=face_col, thickness=1, circle_radius=1)
                                      )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=face_col, thickness=1, circle_radius=1)
                                      )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=face_col, thickness=1, circle_radius=1)
                                      )

            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    # main_tracking()
    fake_cam()

    # # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    # import cv2
    # import mediapipe as mp
    # mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles
    # mp_face_mesh = mp.solutions.face_mesh
    #
    # # For static images:
    # IMAGE_FILES = []
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    # with mp_face_mesh.FaceMesh(
    #     static_image_mode=True,
    #     max_num_faces=1,
    #     min_detection_confidence=0.5) as face_mesh:
    #   for idx, file in enumerate(IMAGE_FILES):
    #     image = cv2.imread(file)
    #     # Convert the BGR image to RGB before processing.
    #     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #
    #     # Print and draw face mesh landmarks on the image.
    #     if not results.multi_face_landmarks:
    #       continue
    #     annotated_image = image.copy()
    #     for face_landmarks in results.multi_face_landmarks:
    #       print('face_landmarks:', face_landmarks)
    #       mp_drawing.draw_landmarks(
    #           image=annotated_image,
    #           landmark_list=face_landmarks,
    #           connections=mp_face_mesh.FACEMESH_TESSELATION,
    #           landmark_drawing_spec=None,
    #           connection_drawing_spec=mp_drawing_styles
    #           .get_default_face_mesh_tesselation_style())
    #       mp_drawing.draw_landmarks(
    #           image=annotated_image,
    #           landmark_list=face_landmarks,
    #           connections=mp_face_mesh.FACEMESH_CONTOURS,
    #           landmark_drawing_spec=None,
    #           connection_drawing_spec=mp_drawing_styles
    #           .get_default_face_mesh_contours_style())
    #     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    #
    # # For webcam input:
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    # cap = cv2.VideoCapture(0)
    # with mp_face_mesh.FaceMesh(
    #     min_detection_confidence=0.5,
    #     min_tracking_confidence=0.5) as face_mesh:
    #   while cap.isOpened():
    #     success, image = cap.read()
    #     if not success:
    #       print("Ignoring empty camera frame.")
    #       # If loading a video, use 'break' instead of 'continue'.
    #       continue
    #
    #     # Flip the image horizontally for a later selfie-view display, and convert
    #     # the BGR image to RGB.
    #     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    #     # To improve performance, optionally mark the image as not writeable to
    #     # pass by reference.
    #     image.flags.writeable = False
    #     results = face_mesh.process(image)
    #
    #     # Draw the face mesh annotations on the image.
    #     image.flags.writeable = True
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     if results.multi_face_landmarks:
    #       for face_landmarks in results.multi_face_landmarks:
    #         mp_drawing.draw_landmarks(
    #             image=image,
    #             landmark_list=face_landmarks,
    #             connections=mp_face_mesh.FACEMESH_TESSELATION,
    #             landmark_drawing_spec=None,
    #             connection_drawing_spec=mp_drawing_styles
    #             .get_default_face_mesh_tesselation_style())
    #         mp_drawing.draw_landmarks(
    #             image=image,
    #             landmark_list=face_landmarks,
    #             connections=mp_face_mesh.FACEMESH_CONTOURS,
    #             landmark_drawing_spec=None,
    #             connection_drawing_spec=mp_drawing_styles
    #             .get_default_face_mesh_contours_style())
    #     cv2.imshow('MediaPipe FaceMesh', image)
    #     if cv2.waitKey(5) & 0xFF == 27:
    #       break
    # cap.release()
