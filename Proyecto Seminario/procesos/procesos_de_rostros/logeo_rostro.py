import numpy as np
from typing import Tuple

from procesos.procesos_de_rostros.herramientos_rostros import HerramientasRostros
from procesos.basededatos.configuracion import RutaBaseDeDatos

class FacialLogin:
    def __init__(self):
        self.basededatos = RutaBaseDeDatos()
        self.herramientas_faciales = HerramientasRostros()

        self.comparador = None
        self.comparacion = False
        self.cont_frame = 0

    def procesos(self, imagen_facial: np.ndarray):
        #step 1: check face detection
        checkeo_deteccion_facial, info_facial, guardar_rostro = self.herramientas_faciales.checkeo_facial(imagen_facial)
        if checkeo_deteccion_facial is False:
            return imagen_facial, self.comparador, 'Rostro no detectado'

         # step 2: face mesh
        checheo_malla_facial, malla_facial_info = self.herramientas_faciales.malla_facial(imagen_facial)
        if checheo_malla_facial is False:
            return imagen_facial, self.comparador, 'Malla no detectada'

        # step 3: extract face mesh
        list_puntos_malla_facial = self.herramientas_faciales.extraer_malla_facial(imagen_facial, malla_facial_info)

        # step 4: check face center
        checkeo_centro_rostro = self.herramientas_faciales.checkeo_centro_rostro(list_puntos_malla_facial)

        # step 5: show state
        self.herramientas_faciales.mostrar_estado_login(imagen_facial, state=self.comparador)

        if checkeo_centro_rostro:
            # step 6: extract face info
            # bbox & key_points
            self.cont_frame = self.cont_frame + 1
            if self.cont_frame == 48:
                bbox_facial = self.herramientas_faciales.extraer_bbox_facial(imagen_facial, info_facial)
                puntos_faciales = self.herramientas_faciales.extraer_puntos_faciales(imagen_facial, info_facial)

                # step 7: face crop
                recortar_rostro = self.herramientas_faciales.recortar_rostro(guardar_rostro, bbox_facial)

                # step 8: read database
                basededatos_rostros, basededatos_nombres, info = self.herramientas_faciales.leer_rostros_basededatos(self.basededatos.rostros)

                if len(basededatos_rostros) != 0 and not self.comparacion and self.comparador is None:
                    self.comparacion = True
                    # step 9: compare faces
                    self.comparador, nombre_usuario = self.herramientas_faciales.comparador_rostros(recortar_rostro, basededatos_rostros, basededatos_nombres)

                    if self.comparador:
                        # step 10: save data & time
                        self.herramientas_faciales.usuario_check_in(nombre_usuario, self.basededatos.usuarios)
                        return imagen_facial, self.comparador, 'Acceso a usuario aprobado!'
                    else:
                        return imagen_facial, self.comparador, 'Usuario no aprobado'
                else:
                    return imagen_facial, self.comparador, 'Base de datos vacia'
            else:
                return imagen_facial, self.comparador, 'wait frames'
        else:
            return imagen_facial, self.comparador, 'Rostro no centrado'



