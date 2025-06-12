import cv2
import mediapipe as mp
import os
from math import dist as distancia_euclidiana

# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Cargar rutas de imágenes
letras = {
    chr(i): os.path.join(os.path.dirname(__file__), "images", f"{chr(i)}.png")
    for i in range(65, 91)
}


def draw_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        x_min, y_min = min(x, x_min), min(y, y_min)
        x_max, y_max = max(x, x_max), max(y, y_max)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


def detectar_letra_en_imagen(image):
    """
    Recibe una imagen, detecta la mano usando MediaPipe y aplica lógica para detectar la letra en ASL.
    Retorna una letra (A-Z) o None si no se detecta.
    """
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Convertir la imagen de BGR a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]

        # Obtener puntos normalizados y convertir a píxeles
        h, w, _ = image.shape
        puntos = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

        # Dibujar mano en la imagen (opcional)
        # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Lógica de detección
        return detectar_letra(puntos)


def detectar_letra(hand_points):
    
    index_finger_mcp = hand_points[5]
    
    index_finger_tip = hand_points[8]
    index_finger_mid = hand_points[7]
    index_finger_pip = hand_points[6]
    
    thumb_tip = hand_points[4]
    thumb_ip = hand_points[3]
    thumb_pip = hand_points[2]
    
    middle_finger_tip  = hand_points[12]
    middle_finger_mcp = hand_points[9]
    index_palm  = hand_points[11]
    middle_finger_mid = hand_points [11]
    middle_finger_pip = hand_points[10]
    
    ring_finger_tip = hand_points[16]
    ring_finger_mid = hand_points[15]
    ring_finger_pip = hand_points[14]
    
    pinky_tip = hand_points[20]
    pinky_mid = hand_points[19]
    pinky_pip = hand_points[18]
    
    wrist = hand_points[0]
    
    ring_finger_pip2  = hand_points[5]
    
    letra = None

    # Detectar la letra "A" en LSA
    # Pulgar cerca de todos los dedos (índice, medio, anular, meñique)
    # Las distancias entre el pulgar y cada dedo son menores a valores específicos
    if abs(thumb_tip[1] - index_finger_pip[1]) <50 and abs(thumb_ip[1]-index_finger_pip[1]) <10\
                    and (ring_finger_pip[1] - ring_finger_tip[1]) < 0\
                    and (middle_finger_pip[1] - ring_finger_tip[1] < 0) \
                    and (index_finger_pip[1] - index_finger_tip[1] < 0)\
                    and (pinky_pip[1] - pinky_tip[1]) < 0:
        letra = "A"

    # Detectar la letra "N" en LSA
    # El pulgar está a la izquierda de los dedos índice y medio
    # El índice está doblado hacia abajo (PIP > TIP)
    # El dedo medio está extendido hacia arriba, mientras que el anular y el meñique están doblados hacia abajo
    # La distancia entre el pulgar y el dedo medio está en un rango específico
    elif thumb_tip[0] < index_finger_pip[0] and \
                    thumb_tip[0] < middle_finger_pip[0] and \
                    index_finger_pip[1] - index_finger_tip[1] < 0 and \
                    middle_finger_tip[1] > middle_finger_pip[1] and \
                    ring_finger_tip[1] > ring_finger_pip[1] and \
                    pinky_tip[1] > pinky_pip[1] and \
                    abs(thumb_tip[1] - middle_finger_pip[1]) > 35 and abs(thumb_tip[1] - middle_finger_pip[1]) < 60:
            letra = "N"
    # Detectar la letra "B" en LSA
    # Los dedos índice, medio, anular y meñique están estirados (distancia positiva entre las falanges)
    # La distancia entre el dedo medio y el anular es negativa, indicando que están juntos
    # Pulgar cerca del anular
    elif index_finger_pip[1] - index_finger_tip[1]>0 and pinky_pip[1] - pinky_tip[1] > 0 and \
        middle_finger_pip[1] - middle_finger_tip[1] >0 and ring_finger_pip[1] - ring_finger_tip[1] >0 and \
            middle_finger_tip[1] - ring_finger_tip[1] <0 and abs(thumb_tip[1] - ring_finger_pip2[1])<40:
        letra ="B"
    
    # Detectar la letra "O" en LSA
    # Los dedos están doblados hacia asbajo (PIP > TIP para todos los dedos)
    # La distancia entre el pulgar y el dedo medio es corta
    # La distancia entre el pulgar y la palma es suficientemente grande
    elif pinky_pip[1] - pinky_tip[1] < 0 and middle_finger_pip[1] - middle_finger_tip[1] < 0 and index_finger_pip[1] - index_finger_tip[1] < 0 and ring_finger_pip[1]- ring_finger_tip[1] < 0 and\
        distancia_euclidiana(thumb_tip, middle_finger_tip) < 40 and\
        distancia_euclidiana(thumb_pip, index_palm) > 120 :
        letra ="O"
    
    # Detectar la letra "C" en LSA
    # El índice está cerca del pulgar (diferencia pequeña en ambas coordenadas)
    # El dedo índice está más arriba que el dedo medio
    # Los dedos anular y meñique también están más arriba que sus correspondientes bases 
    elif abs(index_finger_tip[0] - thumb_tip[0]) < 100 and \
                    abs(index_finger_tip[1] - thumb_tip[1]) < 100 and \
                    index_finger_tip[1] > middle_finger_pip[1] and \
                    ring_finger_tip[1] > ring_finger_pip[1] and \
                    pinky_tip[1] > pinky_pip[1]\
                    and thumb_tip[0]-index_finger_pip[0] < 0:
        letra ="C"
    
            
         
    
    # Detectar la letra "D" en LSA
    # La distancia entre el pulgar y el dedo medio, y entre el pulgar y el anular, es menor a 65
    # El dedo meñique está doblado hacia abajo
    # El dedo índice está más arriba que su base
    elif distancia_euclidiana(thumb_tip, middle_finger_tip) < 65 \
        and distancia_euclidiana(thumb_tip, ring_finger_tip) < 65 \
        and  pinky_pip[1] - pinky_tip[1]<0\
        and index_finger_pip[1] - index_finger_tip[1]>0:
        letra = "D"
                 
    # Detectar la letra "Z" en LSA
    # El dedo índice está doblado hacia abajo, mientras que los otros dedos (medio, anular, meñique) están doblados hacia arriba
    # El pulgar está por encima de la base del dedo medio
    elif index_finger_pip[1] - index_finger_tip[1] > 0 and \
         middle_finger_pip[1] - middle_finger_tip[1] < 0 and \
         ring_finger_pip[1] - ring_finger_tip[1] < 0 and \
         pinky_pip[1] - pinky_tip[1] < 0 and \
         thumb_tip[1] > middle_finger_mcp[1]:
        letra = "Z"
        
    # Detectar la letra "E" en LSA
    # Los dedos índice, meñique, medios y anular están doblados hacia abajo
    # El pulgar está por encima de todos los dedos (es más alto que los demás)
    # El pulgar está más cerca del índice, del medio, del anular y del meñique
    # La distancia entre el pulgar y el índice, el medio, el anular y el meñique es menor a 100
    elif index_finger_pip[1] - index_finger_tip[1] < 0 and pinky_pip[1] - pinky_tip[1] < 0 and \
        middle_finger_pip[1] - middle_finger_tip[1] < 0 and ring_finger_pip[1] - ring_finger_tip[1] < 0 \
            and abs(index_finger_tip[1] - thumb_tip[1]) < 100 and \
                thumb_tip[1] - index_finger_tip[1] > 0 \
                and thumb_tip[1] - middle_finger_tip[1] > 0 \
                and thumb_tip[1] - ring_finger_tip[1] > 0 \
                and thumb_tip[1] - pinky_tip[1] > 0:
            letra ="E"
            
        # Detectar la letra "Q" en LSA
    #Los dedos medio, anular y meñique deben de estar abajo
    #Los dedos indice y el pulgar tocan sus puntas
    #Para asegurar que es la q debe estar el indice hacia abajo
    elif pinky_pip[1] - pinky_mid[1] < 0 and ring_finger_pip[1] - ring_finger_mid[1] < 0 and middle_finger_pip[1] - middle_finger_mid[1] < 0 and\
        distancia_euclidiana(thumb_tip, index_finger_tip) < 30 and middle_finger_pip[1] < index_finger_tip[1]:
        letra ="Q"
                    

    # Detectar la letra "F" en LSA
    # Los dedos meñique, medio y anulasr están doblados hacia abajo
    # El dedo índice está estirado hacia arriba
    # El pulgar está ligeramente separado del dedo índice
    # La distancia entre el índice y el pulgar es menor que 65                    
    elif  pinky_pip[1] - pinky_tip[1] > 0 and middle_finger_pip[1] - middle_finger_tip[1] > 0 and \
        ring_finger_pip[1] - ring_finger_tip[1] > 0 and index_finger_pip[1] - index_finger_tip[1] < 0 \
            and abs(thumb_pip[1] - thumb_tip[1]) > 0 and distancia_euclidiana(index_finger_tip, thumb_tip) <65:
        letra = "F"


    # Detectar la letra "G" en LSA
    # El dedo índice está doblado hacisa abajo (menos de 30 unidades de diferencia entre el extremo y la articulación)
    # El pulgar está cerca de la base de la palma (menos de 25 unidades de diferencia entre el extremo y la articulación)
    # Los dedos medio, anular y meñique están estirados hacia arriba por encima del dedo índice
    elif abs(index_finger_tip[1] - index_finger_pip[1]) < 30 and \
            abs(thumb_tip[1] - thumb_pip[1]) < 25 and \
            middle_finger_tip[1] > index_finger_pip[1] and \
            ring_finger_tip[1] > index_finger_pip[1] and \
            pinky_tip[1] > index_finger_pip[1]:
         letra = "G"
    
        # Detectar la letra "R" en LSA
    # El dedo meñique y el anular están abajo,
    # El pulgar se encuentra dentro de la palma sosteniendo el dedo anular
    # El dedo índice y el medio están arriba y cruzados
    elif pinky_pip[1] - pinky_tip[1] < 0 and ring_finger_pip[1] - ring_finger_tip[1] < 0 and\
        distancia_euclidiana(thumb_tip, ring_finger_tip) < 70 and\
        distancia_euclidiana(index_finger_tip, middle_finger_tip ) < 30:
        letra = "R"
    
         
    # Detectar la letra "U" en LSA
    # El dedo meñique y el anular están abajo,
    # El dedo índice y el medio están arriba y juntos
    # El pulgar se encuentra dentro de la palma sosteniendo el anular
    #Evaluamos la separacion de los dedos indice y medio para que no sea una v
    elif pinky_pip[1] - pinky_tip[1] < 0 and ring_finger_pip[1] - ring_finger_tip[1] < 120 and\
        index_finger_pip[1] - index_finger_tip[1] > 0 and middle_finger_pip[1] - middle_finger_tip[1] > 0 and index_finger_tip[1] - middle_finger_tip[1] < 50 and\
        distancia_euclidiana(thumb_tip, ring_finger_tip) < 30 and\
        distancia_euclidiana(index_finger_tip, middle_finger_tip) > 35 and distancia_euclidiana(index_finger_tip, middle_finger_tip) < 62 :

        letra ="U"
    
    # Detectar la letra "K" en LSA
    # Los dedos índice y medio están extendidos hacia arriba (con separación de más de 20 unidades entre PIP y tip)
    # Los dedos anular y meñique están doblados hacia abajo (pinky_tip < pinky_pip y ring_finger_tip < ring_finger_pip)
    # El pulgar está cerca del PIP del dedo medio (distancia_euclidiana < 50)
    # La distancia entre las puntas del índice y el medio es menor que 90
    elif (index_finger_pip[1] - index_finger_tip[1] > 20) and \
         (middle_finger_pip[1] - middle_finger_tip[1] > 20) and \
         (ring_finger_pip[1] - ring_finger_tip[1] < 0) and \
         (pinky_pip[1] - pinky_tip[1] < 0) and \
         (distancia_euclidiana(thumb_tip, middle_finger_pip) < 50) and \
         (distancia_euclidiana(index_finger_tip, middle_finger_tip) < 90):
        letra = "K"
                
    
    # Detectar la letra "V" en LSA
    # Los dedos índice y medio están doblados hacia abajo
    # El anular y meñique están doblados hacia arriba
    # La distancia horizontal entre el dedo índice y el dedo medio es significativa
    elif index_finger_pip[1] - index_finger_tip[1] > 0 and \
         middle_finger_pip[1] - middle_finger_tip[1] > 0 and \
         ring_finger_pip[1] - ring_finger_tip[1] < 0 and \
         pinky_pip[1] - pinky_tip[1] < 0 and \
         abs(index_finger_tip[0] - middle_finger_tip[0]) > 50:
        letra = "V"
    
    
    # Detectar la letra "H" en LSA
    # Los dedos índice y medio están cserca uno del otro (menos de 50 unidades de diferencia)
    # Los dedos índice y medio están por debajo de la base del dedo anular (con tolerancia de 10 unidades)
    # El dedo anular está ligeramente doblado
    # El meñique está extendido hacia arriba, por encima de su base
    elif abs(index_finger_tip[1] - middle_finger_tip[1]) < 50 and \
                    index_finger_tip[1] < ring_finger_pip[1] + 10 and \
                    middle_finger_tip[1] < ring_finger_pip[1] + 10 and \
                    ring_finger_tip[1] > ring_finger_pip[1] - 10 and \
                    pinky_tip[1] > pinky_pip[1] - 10:  # Más tolerancia en el doblez del meñique
            letra = "H"
         
    # Detectar la letra "I" en LSA
    # El dedo meñique está doblado hacia abarriba (pinky_pip > pinky_tip)
    # El dedo índice, medio y anular están doblados hacia abajo (sus PIPs están más cerca de las puntas)
    # el pulgar esta junto a al indice parado
    elif pinky_pip[1] - pinky_tip[1] > 0 and \
        index_finger_pip[1] - index_finger_tip[1] < 0 and \
        middle_finger_pip[1] - middle_finger_tip[1] < 0 and \
        ring_finger_pip[1] - ring_finger_tip[1] < 0 and\
        thumb_ip[1] - index_finger_pip[1] <30 and\
        index_finger_pip[1] - thumb_tip[1] >20:
            
        letra ="I"
    
    #Vamos a detectar la Y 
    #Los dedos indice, medio y anular estan abajo
    #El dedio meñique esta extendido y el pulgar tambien
    #EL pulgar no esta junto al indice
    elif index_finger_pip[1] - index_finger_tip[1] < 0 and middle_finger_pip[1] - middle_finger_tip[1] < 0 and ring_finger_pip[1] - middle_finger_tip[1] < 0 and\
        pinky_pip[1] - pinky_tip[1] > 0 and thumb_pip[1] - thumb_tip[1] >0 and\
        thumb_ip[1] - index_finger_pip[1] > 35 :
        letra ="Y"
                    
    
    # Detectar la letra "J" en LSA
    # El dedo meñique está doblado hacia abajo (pinky_tip < pinky_pip)
    # El índice, medio y anular están extendidos (sus puntas están por encima de los PIPs)
    # El pulgar está hacia la izquierda del índice (thumb_tip < index_finger_pip)
    elif pinky_tip[1] < pinky_pip[1] and \
        index_finger_tip[1] > index_finger_pip[1] and \
        middle_finger_tip[1] > middle_finger_pip[1] and \
        ring_finger_tip[1] > ring_finger_pip[1] and \
        thumb_tip[0] < index_finger_pip[0]: 
        letra =  "J"
    
    # Detectar la letra "L" en LSA
    # El dedo índice está extendido hacia arriba (PIP > TIP)
    # Los dedos medio, anular y meñique están doblados hacia abajo (PIP < TIP)
    # El pulgar está ubicado a la izquierda del MCP del índice
    elif index_finger_pip[1] - index_finger_tip[1] > 0 and \
         middle_finger_pip[1] - middle_finger_tip[1] < 0 and \
         ring_finger_pip[1] - ring_finger_tip[1] < 0 and \
         pinky_pip[1] - pinky_tip[1] < 0 and \
         thumb_tip[0] < index_finger_mcp[0]:
        letra ="L"
        

    
    # Detectar la letra "M" en LSA
    # El pulgar está a la izquierda de los dedos índice, medio y anular
    # El índice está doblado hacia abajo (PIP > TIP)
    # Los dedos medio, anular y meñique están extendidos hacia arriba
    # La distancia entre el pulgar y el anular está en un rango específico
    elif thumb_tip[0] < index_finger_pip[0] and \
            thumb_tip[0] < middle_finger_pip[0] and \
            thumb_tip[0] < ring_finger_pip[0] and \
            index_finger_pip[1] - index_finger_tip[1] < 0 and \
            middle_finger_tip[1] > middle_finger_pip[1] and \
            ring_finger_tip[1] > ring_finger_pip[1] and \
            pinky_tip[1] > pinky_pip[1] and \
            abs(thumb_tip[1] - ring_finger_pip[1]) > 35 and abs(thumb_tip[1] - ring_finger_pip[1]) < 60:
         letra = "M"
         
    
    
    
    # Detectar la letra "P" en LSA
    # El índice está más abajo que el dedo anular, el medio está por encima del índice
    # La distancia entre el pulgar y el dedo medio es pequeña
    # El meñique está doblado hacia abajo
    elif  index_finger_tip[1] < ring_finger_pip[1] + 20 and \
                    middle_finger_tip[1] > index_finger_tip[1] + 20 and \
                    abs(thumb_tip[0] - middle_finger_tip[0]) < 50 and \
                    ring_finger_tip[1] > ring_finger_pip[1] - 20 and \
                    pinky_tip[1] > pinky_pip[1] - 20:  # Meñique doblado
        letra = "P"
         
    # Detectar la letra "Q" en LSA
    #Los dedos medio, anular y meñique deben de estar abajo
    #Los dedos indice y el pulgar tocan sus puntas
    #Para asegurar que es la q debe estar el indice hacia abajo
    elif pinky_pip[1] - pinky_mid[1] < 0 and ring_finger_pip[1] - ring_finger_mid[1] < 0 and middle_finger_pip[1] - middle_finger_mid[1] < 0 and\
        distancia_euclidiana(thumb_tip, index_finger_tip) < 30 and middle_finger_pip[1] < index_finger_tip[1]:
        letra ="Q"
                    

    # Detectar la letra "S" en LSA
    # Los cuatro dedos (índice, medio, anular y meñique) están doblados hacia abajo
    # El pulgar está arriba del índice y cerca de la posición del dedo índice
    elif index_finger_pip[1] - index_finger_tip[1] < 0 and \
         middle_finger_pip[1] - middle_finger_tip[1] < 0 and \
         ring_finger_pip[1] - ring_finger_tip[1] < 0 and \
         pinky_pip[1] - pinky_tip[1] < 0 and \
         thumb_tip[1] > index_finger_pip[1] and \
         abs(thumb_tip[0] - index_finger_pip[0]) < 40:
         letra = "S"

    # Detectar la letra "T" en LSA
    # El dedo índice está por encima del pulgar y doblado hacia abajo
    # El medio, anular y meñique están doblados hacia abajo
    # La distancia entre el pulgar y el índice es pequeña
    # El dedo medio está a la izquierda del pulgar, y el dedo índice está a la derecha del pulgar
    elif index_finger_tip[1] > thumb_tip[1]  and \
            index_finger_pip[1] - index_finger_tip[1] < 0 and \
            middle_finger_pip[0] < thumb_tip[0] < index_finger_tip[0] and \
            pinky_pip[1] - pinky_tip[1] < 0 and \
            ring_finger_pip[1] - ring_finger_tip[1] < 0 and \
            middle_finger_pip[1] - middle_finger_tip[1] < 0 and \
            distancia_euclidiana(thumb_ip, index_finger_tip) < 35:
         letra = "T"

    

    # Detectar la letra "W" en LSA
    # Los dedos índice, medio, anular sy meñique están doblados hacia abajo
    # La distancia entre el pulgar y el meñique es pequeña
    elif middle_finger_pip[1] - index_finger_tip[1] > 0 and \
            ring_finger_pip[1] - index_finger_tip[1] > 0 and \
            pinky_pip[1] - index_finger_tip[1] > 0 and \
            distancia_euclidiana(thumb_tip, pinky_tip) <50:
         letra = "W"
    
    
    # Detectar la letra "X" en LSA
    # La distancia entre el dedo índice y el dedo índice en la segunda articulación es grande
    # Las distancias entre el pulgar y otros dedos (medio, anular, meñique) son pequeñas
    elif distancia_euclidiana(index_finger_tip, index_finger_pip) > 40 and \
         abs(thumb_tip[1] - middle_finger_pip[1]) < 30 and \
         abs(thumb_tip[1] - ring_finger_pip[1]) < 30 and \
         abs(thumb_tip[1] - pinky_pip[1]) < 30:
        letra = "X"
    
    else:
        pass

    print(letra)
    return letra

def extraer_puntos_landmark(hand_landmarks, image_shape):
    h, w, _ = image_shape
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]


def procesar_frame(frame, hands_instance):
    letra_detectada = None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_instance.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            draw_bounding_box(frame, hand_landmarks)
            puntos = extraer_puntos_landmark(hand_landmarks, frame.shape)
            letra_detectada = detectar_letra(puntos)
            if letra_detectada:
                cv2.putText(frame, f"Letra: {letra_detectada}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame, letra_detectada
