import numpy as np
import mediapipe as mp
import cv2
from typing import Any, List, Tuple

from tensorboard.summary.v1 import image


class MallaFacialMediapipe:
    def __init__(self):
        #mediapipe
        self.dibujo_mp = mp.solutions.drawing_utils
        self.config_dibujo = self.dibujo_mp.DrawingSpec(color=(255,127,0), thickness=1, circle_radius=1)

        self.objeto_malla_facial = mp.solutions.face_mesh
        self.malla_facial_mp = self.objeto_malla_facial.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                                 refine_landmarks=False, min_detection_confidence=0.6,
                                                                 min_tracking_confidence=0.6)
        self.puntos_malla = None
        #face points
        #right parietal
        self.rp_x: int = 0
        self.rp_y: int = 0
        #left parietal
        self.lp_x: int = 0
        self.lp_y: int = 0
        #right eyebrow
        self.re_x: int = 0
        self.re_y: int = 0
        #left eyebrow
        self.le_x: int = 0
        self.le_y: int = 0

    def malla_facial_mediapipe(self, imagen_facial:np.ndarray) ->Tuple[bool, Any]:
        imagen_rgb = imagen_facial.copy()
        imagen_rgb = cv2.cvtColor(imagen_rgb, cv2.COLOR_BGR2RGB)

        malla_facial = self.malla_facial_mp.process(imagen_rgb)
        if malla_facial.multi_face_landmarks is None:
            return False, malla_facial
        else:
            return True, malla_facial

    def extraer_puntos_malla_facial(self,imagen_facial: np.ndarray, malla_facial_info: Any, viz: bool) ->List[List[int]]:
        height, width, _ = imagen_facial.shape
        self.puntos_malla = []
        for malla_facial in malla_facial_info.multi_face_landmarks:
            for i, puntos in enumerate(malla_facial.landmark):
                x, y = int(puntos.x * width), int(puntos.y * height)
                self.puntos_malla.append([i, x, y])

            if viz:
                self.dibujo_mp.draw_landmarks(imagen_facial, malla_facial, self.objeto_malla_facial.FACEMESH_TESSELATION,
                                              self.config_dibujo, self.config_dibujo)
        return self.puntos_malla

    def checkeo_centro_rostro(self, puntos_faciales: List[List[int]]) -> bool:
        if len(puntos_faciales) == 468:
            self.rp_x, self.rp_y = puntos_faciales[139][1:]
            self.lp_x, self.lp_y = puntos_faciales[368][1:]
            self.re_x, self.re_y = puntos_faciales[70][1:]
            self.le_x, self.le_y = puntos_faciales[300][1:]

            if self.re_x > self.rp_x and self.le_x < self.lp_x:
                return True
            else:
                return False


    def config_color(self, color: Tuple[int, int, int]):
        self.config_dibujo = self.dibujo_mp.DrawingSpec(color = color, thickness=1, circle_radius=1)
