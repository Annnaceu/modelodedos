import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np

st.set_page_config(page_title="Contagem de Dedos com Webcam", layout="centered")

st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
            font-family: Arial, sans-serif;
        }
        
        /* Caixa principal */
        .stApp {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            margin: auto;
        }
        
        /* Cabeçalho */
        h1 {
            color: #ff4b4b;
            font-size: 2.4em;
            text-align: center;
            font-weight: bold;
            margin-bottom: 1em;
        }

        /* Botão */
        .stButton > button {
            background-color: #ff4b4b;
            color: white;
            font-size: 1.1em;
            padding: 0.6em 1.2em;
            border-radius: 8px;
            transition: 0.3s ease;
            border: none;
            box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.2);
        }
        .stButton > button:hover {
            background-color: #e53e3e;
            box-shadow: 0px 8px 12px rgba(0, 0, 0, 0.3);
        }

        /* Exibição de vídeo */
        .stVideo {
            border: 2px solid #e3e4e8;
            border-radius: 8px;
            margin: 1em 0;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }

        /* Histórico de dedos */
        .stText {
            background-color: #eef2f7;
            border-radius: 8px;
            padding: 1em;
            color: #333;
            font-size: 1.2em;
            margin-top: 20px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

modelo = joblib.load('modelo_contagem_dedos.pkl')
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.7
)

def prever_dedos(hand_landmarks):
    landmarks = hand_landmarks.landmark
    features = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).flatten()
    
    if features.size != 63:
        st.warning("Número de características não é o esperado. Características capturadas: " + str(features.size))
        return 0
    dedos_levantados = modelo.predict([features])[0]
    return dedos_levantados

def main():
    st.title("Contagem de Dedos com Webcam")

    if 'contagem_dedos' not in st.session_state:
        st.session_state.contagem_dedos = []

    stop_capture = st.button('Parar Captura')

    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    historico = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Erro ao capturar vídeo.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                dedos_levantados = prever_dedos(hand_landmarks)
                if not st.session_state.contagem_dedos or st.session_state.contagem_dedos[-1] != dedos_levantados:
                    st.session_state.contagem_dedos.append(dedos_levantados)
                cv2.putText(frame, f'Dedos levantados: {dedos_levantados}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        stframe.image(frame, channels="BGR", use_column_width=True)
        historico.markdown(f"<div class='stText'>Histórico de Dedos Levantados: {', '.join([str(d) for d in st.session_state.contagem_dedos])}</div>", unsafe_allow_html=True)

        if stop_capture:
            break

    cap.release()
    st.success("Captura de vídeo parada.")
    st.write("Histórico final de dedos levantados durante a gravação:")
    st.write(", ".join([f"{dedos} dedos" for dedos in st.session_state.contagem_dedos]))

if __name__ == "__main__":
    main()


