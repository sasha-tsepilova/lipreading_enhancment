import cv2
import mediapipe as mp
import numpy as np
import argparse

from mediapipeDemos.videosource import FileSource
from mediapipeDemos.custom.face_geometry import (
    PCF,
    get_metric_landmarks
)

lip_indeces = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
lip_indeces.sort()




def main(args):
    source = FileSource(args.input_path)
    video=cv2.VideoWriter(args.output_path.split('.')[0] + ".mp4", -1, source.get_fps(), (96,96))
    frame_width, frame_height = source.get_image_size()

    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeff = np.zeros((4, 1))
    refine_landmarks = True

    pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],
    )

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        # the model is getting 96 x 96 pixels, making all landmarks fit there and leaving some padding
        new_frame = np.zeros((96,96))
        for idx, (frame, frame_rgb) in enumerate(source):
        # Capture each frame 
            new_frame = np.zeros((96,96))
            results = face_mesh.process(frame_rgb)
            multi_face_landmarks = results.multi_face_landmarks

            if multi_face_landmarks:
                face_landmarks = multi_face_landmarks[0]
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
                landmarks = landmarks.T

                if refine_landmarks:
                    landmarks = landmarks[:, :468]

                metric_landmarks, pose_transform_mat = get_metric_landmarks(
                    landmarks.copy(), pcf
                )

                model_points = metric_landmarks[0:3, lip_indeces].T

                # see here:
                # https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
                pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]

            
                # we want to have canonical view on the lips landmarks
                identity_rotation = np.identity(3)
                identity_rotation[2][2] = -1
                identity_rotation[1][1] = -1
                identity_translation = np.zeros((3,1))
                identity_translation[2] = 50

                projected_model_points, jacobian_projected = cv2.projectPoints(
                    model_points,
                    identity_rotation,
                    identity_translation,
                    camera_matrix,
                    dist_coeff,
                )
                x_coords, y_coords = [point[0][0] for point in projected_model_points], [point[0][1] for point in projected_model_points]
                x_min, x_max, y_min, y_max = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
                x_c, y_c = (x_min+x_max) // 2, (y_min+y_max) // 2

                # fitting landmarks into 90x90 (drawing them with radius 1) and want to introduce a padding
                lip_reg_width = 90
                scaling_coef = min([lip_reg_width/(x_c-x_min), lip_reg_width/(x_max-x_c), lip_reg_width/(y_c-y_min), lip_reg_width/(y_max-y_c)])/2
                x_resized = (x_coords - x_c) * scaling_coef + lip_reg_width//2 + 3
                y_resized = (y_coords - y_c) * scaling_coef + lip_reg_width//2 + 3
                resized_projections = list(zip(x_resized,y_resized))

                for proj in resized_projections:
                    new_frame = cv2.circle(new_frame, (int(proj[0]),int(proj[1])), radius=1, color=255, thickness=-1)


            video.write(cv2.cvtColor(new_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
    video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('-o', '--output_path')
    args = parser.parse_args()
    if not args.input_path:
        raise Exception("Input video file should be specified.")
    if not args.output_path:
        args.output_path = args.input_path.split('.')[0] + "_out"
    main(args)