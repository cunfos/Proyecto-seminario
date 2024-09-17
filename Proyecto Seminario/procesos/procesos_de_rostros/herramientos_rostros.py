import os
import numpy as np
import cv2
from ultralytics import YOLO
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import List, Tuple, Any
from procesos.procesos_de_rostros.modelo_detector_rostros.detector_rostros import DetectorRostrosMediaPipe
from procesos.procesos_de_rostros.modelo_mallas_rostros.mallas_rostros import MallaFacialMediapipe
from procesos.procesos_de_rostros.modelo_comparacion_rostros.comparacion_rostros import ModeloComparacionRostros
from datetime import datetime  # Importar el módulo datetime
class HerramientasRostros:
    def __init__(self):
        #face detect
        self.detector_rostros = DetectorRostrosMediaPipe()
        #face mesh
        self.detector_mallas = MallaFacialMediapipe()

        #detector de objetos
        self.modelo_yolo = YOLO("yolov8n.pt")  # Puedes usar el modelo que prefieras
        self.clases_a_detectar = ["glasses", "hat", "cap"]  # Nombres de las clases que YOLO detecta (gafas, gorra, etc.)

        #face matcher
        self.comparacion_rostros = ModeloComparacionRostros()

        self.angulo = None

        #variables
        self.rostros_bd = []
        self.nombres_rostros = []
        self.comparador: bool = False
        self.distancia: float = 0.0
        self.usuario_registrado = False

        #face matcher
    def checkeo_facial(self, imagen_facial: np.ndarray)-> Tuple[bool, Any, np.ndarray]:
        guardar_rostro = imagen_facial.copy()
        checkeo_facial, info_facial = self.detector_rostros.deteccion_rostros_mediapipe(imagen_facial)
        return checkeo_facial, info_facial, guardar_rostro

    def extraer_bbox_facial(self, imagen_facial: np.ndarray, info_facial: Any):
        h_img, w_img, _ = imagen_facial.shape
        bbox = self.detector_rostros.extraer_face_bbox_mediapipe(w_img, h_img, info_facial)
        return bbox

    def extraer_puntos_faciales(self, imagen_facial: np.ndarray, info_facial: Any):
        h_img, w_img, _ = imagen_facial.shape
        puntos_faciales = self.detector_rostros.extraer_puntos_faciales_mediapipe(h_img, w_img, info_facial)
        return puntos_faciales

    def malla_facial(self, imagen_facial: np.ndarray) -> Tuple[bool, Any]:
        checkeo_malla_facial, info_malla_facial = self.detector_mallas.malla_facial_mediapipe(imagen_facial)
        return checkeo_malla_facial, info_malla_facial

    def extraer_malla_facial(self, imagen_facial: np.ndarray, malla_facial_info: Any)->List[List[int]]:
        list_puntos_malla_facial = self.detector_mallas.extraer_puntos_malla_facial(imagen_facial, malla_facial_info, viz=True)
        return list_puntos_malla_facial

    def checkeo_centro_rostro(self, puntos_faciales: List[List[int]]) -> bool:
        checkeo_centro_rostro = self.detector_mallas.checkeo_centro_rostro(puntos_faciales)
        return checkeo_centro_rostro

    def recortar_rostro(self, imagen_facial: np.ndarray, bbox_facial: List[int])->np.ndarray:
        h, w, _ = imagen_facial.shape
        offset_x, offset_y = int(w * 0.025), int(h * 0.025)
        xi, yi, xf, yf = bbox_facial
        xi, yi, xf, yf = xi - offset_x, yi - (offset_y * 4), xf + offset_x, yf
        return imagen_facial[yi:yf, xi:xf]

    def guardar_rostro(self, recortar_rostro: np.ndarray, dni: str, path: str):
        if len(recortar_rostro) != 0:
                recortar_rostro = cv2.cvtColor(recortar_rostro, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"{path}/{dni}.png", recortar_rostro)
                return True
        else:
            return False

    #draw
    def mostrar_estado_registro(self, imagen_facial: np.ndarray, state: bool):
        if state:
            texto = 'Proceso facial, mire a la camara!'
            tamaño_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = tamaño_texto[0], tamaño_texto[1]
            cv2.rectangle(imagen_facial, (370, 650 - dim[1]-baseline),(370 + dim[0], 650 + baseline), (0,0,0), cv2.FILLED)
            cv2.putText(imagen_facial, texto, (370, 650-5), cv2.FONT_HERSHEY_DUPLEX, 0.75,(0,255,0), 1)
            self.detector_mallas.config_color((0, 255, 0))

        else:
            texto = 'Guardando rostro, espere 3 segundos por favor'
            tamaño_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = tamaño_texto[0], tamaño_texto[1]
            cv2.rectangle(imagen_facial, (370, 650 - dim[1] - baseline), (370 + dim[0], 650 + baseline), (0, 0, 0),
                          cv2.FILLED)
            cv2.putText(imagen_facial, texto, (370, 650 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 0), 1)
            self.detector_mallas.config_color((255, 0, 0))

    def mostrar_estado_login(self, imagen_facial: np.ndarray, state: bool):
        # Guardar una copia de la imagen sin la malla facial
        imagen_sin_malla = imagen_facial.copy()

        if state:
            texto = 'Persona aprobada, bienvenido!'
            tamaño_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = tamaño_texto[0], tamaño_texto[1]
            cv2.rectangle(imagen_facial, (370, 650 - dim[1] - baseline), (370 + dim[0], 650 + baseline), (0, 0, 0),
                          cv2.FILLED)
            cv2.putText(imagen_facial, texto, (370, 650 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 1)
            self.detector_mallas.config_color((0, 255, 0))

        elif state is None:
            texto = 'Comparando rostros, mira a la camara y espera 3 segundos!'
            tamaño_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = tamaño_texto[0], tamaño_texto[1]
            cv2.rectangle(imagen_facial, (250, 650 - dim[1] - baseline), (250 + dim[0], 650 + baseline), (0, 0, 0),
                          cv2.FILLED)
            cv2.putText(imagen_facial, texto, (250, 650 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 0), 1)
            self.detector_mallas.config_color((255, 255, 0))

        elif state is False:
            texto = 'Rostro no aprobado, registrese por favor!'
            tamaño_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = tamaño_texto[0], tamaño_texto[1]
            cv2.rectangle(imagen_facial, (370, 650 - dim[1] - baseline), (370 + dim[0], 650 + baseline), (0, 0, 0),
                          cv2.FILLED)
            cv2.putText(imagen_facial, texto, (370, 650 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 1), 1)
            self.detector_mallas.config_color((255, 0, 1))

            # Convertir de BGR a RGB la imagen sin la malla
            imagen_rgb_sin_malla = cv2.cvtColor(imagen_sin_malla, cv2.COLOR_BGR2RGB)

            # Guardar la imagen del rostro no autorizado sin malla
            nombre_imagen = "rostro_no_autorizado_sin_malla.jpg"
            cv2.imwrite(nombre_imagen, imagen_rgb_sin_malla)  # Guarda la imagen facial convertida a RGB

            # Envía el correo con la foto sin la malla adjunta
            self.enviar_correo_alerta(nombre_imagen)

    def enviar_correo_alerta(self, imagen_path: str):
        # Configuración del servidor y credenciales
        smtp_server = "smtp.gmail.com"
        port = 587
        sender_email = "pignuoliluca@gmail.com"
        password = "tgmf socc aiqu kwyz"

        # Obtener la fecha y hora actual
        fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Crear el mensaje
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = "sgomezlr30@gmail.com"
        message["Subject"] = "Intento de acceso de usuario no registrado"

        # Cuerpo del correo con la fecha y hora actuales
        body = f"Un usuario intentó ingresar sin estar registrado. Se adjunta la imagen de la persona no autorizada.\n\nFecha y hora del intento: {fecha_hora_actual}"
        message.attach(MIMEText(body, "plain"))

        # Adjuntar la imagen sin malla facial
        try:
            with open(imagen_path, "rb") as img_file:
                img = MIMEImage(img_file.read())
                img.add_header("Content-Disposition", f"attachment; filename={os.path.basename(imagen_path)}")
                message.attach(img)
        except Exception as e:
            print(f"Error al adjuntar la imagen: {e}")

        try:
            # Conectar al servidor
            server = smtplib.SMTP(smtp_server, port)
            server.starttls()  # Iniciar la conexión segura
            server.login(sender_email, password)
            server.sendmail(sender_email, "sgomezlr30@gmail.com", message.as_string())
            print("Correo enviado exitosamente con la imagen adjunta")
        except Exception as e:
            print(f"Error al enviar el correo: {e}")
        finally:
            server.quit()

    def leer_rostros_basededatos(self, basededatos_path: str) -> Tuple[List[np.ndarray], List[str], str]:
        self.rostros_bd: List[np.ndarray] = []
        self.nombres_rostros: List[str] = []

        for file in os.listdir(basededatos_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(basededatos_path, file)
                img_read = cv2.imread(img_path)
                if img_read is not None:
                    self.rostros_bd.append(img_read)
                    self.nombres_rostros.append(os.path.splitext(file)[0])

        return self.rostros_bd, self.nombres_rostros, f'Comparando{len(self.rostros_bd)}rostros!'


    def comparador_rostros(self, imagen_actual: np.ndarray, rostros_bd: List[np.ndarray], nombre_bd: List[str])-> Tuple[bool, str]:
        nombre_usuario: str = ''
        imagen_actual = cv2.cvtColor(imagen_actual, cv2.COLOR_RGB2BGR)
        for idx, rostro_img in enumerate(rostros_bd):
            self.comparador, self.distancia = self.comparacion_rostros.face_matching_deepid_model(imagen_actual, rostro_img)
            print(f'Validando rostro con: {nombre_bd[idx]}')
            if self.comparador:
                nombre_usuario = nombre_bd[idx]
                return True, nombre_usuario

        return False, '¡Rostro desconocido!'

    def usuario_check_in(self, nombre_usuario: str, ruta_usuario: str):
        if not self.usuario_registrado:
            now = datetime.datetime.now()
            date_time = now.strftime("%y-%m-%d %H:%M:%S")
            usuario_archivo_path = os.path.join(ruta_usuario, f"{nombre_usuario}.txt")
            with open(usuario_archivo_path, "a") as usuario_archivo:
                usuario_archivo.write(f'\n Acceso garantizado at: {date_time}\n')

            self.usuario_registrado = True








