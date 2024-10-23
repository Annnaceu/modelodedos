import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Adicionado para desenhar os marcos
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
data = []

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
            dedos = contar_dedos(hand_landmarks)

            # Desenhar os landmarks da mão na imagem
            mp_drawing.draw_landmarks(frame, hand_landmarks)

            # Exibir a contagem de dedos na tela
            cv2.putText(frame, f'Dedos levantados: {dedos}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Extrair características para dados
            features = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            data.append((features.tolist(), dedos))

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Salvar os dados coletados em um arquivo
data = np.array(data, dtype=object)
np.save('dados_dedos.npy', data)

cap.release()
cv2.destroyAllWindows()

