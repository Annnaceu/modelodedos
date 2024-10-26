import numpy as np
import cv2
import mediapipe as mp
import joblib 

try:
    modelo = joblib.load('modelo_contagem_dedos.pkl')
except FileNotFoundError:
    print("Modelo não encontrado. Certifique-se de que 'modelo_contagem_dedos.pkl' está no diretório correto.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

def contar_dedos(hand_landmarks):
    dedos_levantados = 0
    landmarks = hand_landmarks.landmark

    if landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x:
        dedos_levantados += 1

    dedos = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
             mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    juntas = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
              mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]

    for dedo_tip, junta_pip in zip(dedos, juntas):
        if landmarks[dedo_tip].y < landmarks[junta_pip].y:
            dedos_levantados += 1

    return dedos_levantados

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            dedos_levantados = contar_dedos(hand_landmarks)

            cv2.putText(frame, f'Dedos levantados: {dedos_levantados}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks)

    cv2.imshow('Contagem de Dedos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
