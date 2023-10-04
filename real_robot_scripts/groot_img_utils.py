import numpy as np
import cv2

class ImageProcessor():
    def __init__(self):
        pass

    def get_fx_fy_dict(self, img_size=224):
        if img_size == 224:
            fx_fy_dict = {
                "k4a": {
                    0: {"fx": 0.35, "fy": 0.35},
                    1: {"fx": 0.35, "fy": 0.35},
                    2: {"fx": 0.4, "fy": 0.6}
                },
                "rs": {
                    0: {"fx": 0.49, "fy": 0.49},
                    1: {"fx": 0.49, "fy": 0.49},
                }
            }
        # elif img_size == 128:
        #     fx_fy_dict = {0: {"fx": 0.2, "fy": 0.2}, 1: {"fx": 0.2, "fy": 0.2}, 2: {"fx": 0.2, "fy": 0.3}}
        # elif img_size == 84:
        #     fx_fy_dict = {0: {"fx": 0.13, "fy": 0.13}, 1: {"fx": 0.13, "fy": 0.13}, 2: {"fx": 0.15, "fy": 0.225}}
        return fx_fy_dict

    def resize_img(
                self,
                img: np.ndarray,
                camera_type: str, 
                img_w: int=224, 
                img_h: int=224, 
                offset_w: int=0, 
                offset_h: int=0,
                fx: float=None,
                fy: float=None) -> np.ndarray:
        if camera_type == "k4a":
            if fx is None:
                fx = 0.2
            if fy is None:
                fy = 0.2
            resized_img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation = cv2.INTER_NEAREST)
            w = resized_img.shape[0]
            h = resized_img.shape[1]

        if camera_type == "rs":
            if fx is None:
                fx = 0.2
            if fy is None:
                fy = 0.3
            resized_img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation = cv2.INTER_NEAREST)
            w = resized_img.shape[0]
            h = resized_img.shape[1]

        resized_img = resized_img[w//2-img_w//2:w//2+img_w//2, h//2-img_h//2:h//2+img_h//2, ...]
        return resized_img

    def resize_intrinsics(
            self,
            original_image_size: np.ndarray,
            intrinsic_matrix: np.ndarray,
            camera_type: str,
            img_w: int=224,
            img_h: int=224,
            fx: float=None,
            fy: float=None) -> np.ndarray:
        if camera_type == "k4a":
            if fx is None:
                fx = 0.2
            if fy is None:
                fy = 0.2
        elif camera_type == "rs":
            if fx is None:
                fx = 0.2
            if fy is None:
                fy = 0.3
            
        fake_image = np.zeros((original_image_size[0], original_image_size[1], 3))

        resized_img = cv2.resize(fake_image, (0, 0), fx=fx, fy=fy)
        new_intrinsic_matrix = intrinsic_matrix.copy()
        w, h = resized_img.shape[0], resized_img.shape[1]
        new_intrinsic_matrix[0, 0] = intrinsic_matrix[0, 0] * fx
        new_intrinsic_matrix[1, 1] = intrinsic_matrix[1, 1] * fy
        new_intrinsic_matrix[0, 2] = intrinsic_matrix[0, 2] * fx
        new_intrinsic_matrix[1, 2] = intrinsic_matrix[1, 2] * fy
        new_intrinsic_matrix[0, 2] = new_intrinsic_matrix[0, 2] - (w//2-img_w//2)
        new_intrinsic_matrix[1, 2] = new_intrinsic_matrix[1, 2] - (h//2-img_h//2)
        return new_intrinsic_matrix
