import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
from packaging import version
import pandas as pd
from datetime import date

st.set_page_config(
    page_title="Detector de Sementes",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌽 Detector de Sementes de Milho")
st.markdown("Faça o upload de uma imagem ou use sua câmera para detectar sementes.")
st.markdown("---")

# ----------------- ALTERAÇÃO 1: SELEÇÃO DA CÂMERA/MODELO YOLO -----------------
# Dicionário que mapeia o nome da câmera para o nome do arquivo do modelo
camera_options = {
    "Câmera Casual (RGB)": "best.pt",
    "Câmera RGN": "rgn.pt",
    "Câmera RE": "re.pt",
    "Câmera NIR": "nir.pt",
    "RGB": "rgb.pt"
}

# ----------------- Sidebar para as configurações de detecção -----------------
st.sidebar.header("Opções de Detecção")

# Caixa de seleção para escolher a câmera
selected_camera = st.sidebar.selectbox(
    "Qual câmera foi utilizada?",
    list(camera_options.keys()),
    help="Selecione o tipo de câmera para carregar o modelo de detecção correspondente."
)

# Carrega o modelo com base na seleção
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo '{model_path}': {e}")
        st.info(f"Verifique se o arquivo do modelo '{model_path}' existe no diretório.")
        return None

# Carrega o modelo dinamicamente
model_path_to_load = camera_options[selected_camera]
model = load_model(model_path_to_load)

# Slider de confiança
confidence_threshold = st.sidebar.slider(
    "Limiar de Confiança:",
    min_value=0.01,
    max_value=1.0,
    value=0.25,
    step=0.01,
    help="Ajuste para mostrar detecções mais (valor baixo) ou menos (valor alto) confiantes."
)

# ----------------- Estado da sessão -----------------
if "seed_count" not in st.session_state:
    st.session_state.seed_count = 0
if "uploaded_image_count" not in st.session_state:
    st.session_state.uploaded_image_count = None
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "processed_images_history" not in st.session_state:
    st.session_state.processed_images_history = []
if "camera_history" not in st.session_state:
    st.session_state.camera_history = []

# ----------------- Função de detecção -----------------
def predict_and_display(image, model, confidence, is_camera=False):
    seed_count = 0
    im_array = None

    if model:
        results = model(image, conf=confidence)
        if results and results[0].boxes:
            seed_count = len(results[0].boxes)
            try:
                im_array = results[0].plot()
                im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            except Exception as e:
                st.warning(f"Erro ao gerar imagem anotada: {e}")
                im_array = None

    if im_array is None:
        try:
            im_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            st.error("Erro ao processar imagem.")
            return None, 0 if is_camera else 0

    if is_camera:
        return im_array, seed_count
    else:
        if im_array is not None:
            st.image(im_array, caption="Imagem com Detecções", use_container_width=True)

    return seed_count

# ----------------- Interface principal com as abas -----------------
if version.parse(st.__version__) >= version.parse("1.18.0"):
    tab1, tab2, tab3 = st.tabs(["Upload de Imagem", "Câmera Ao Vivo", "Estatísticas"])

    with tab1:
        st.header("Upload de Imagem")
        uploaded_file = st.file_uploader(
            "Selecione uma imagem (.jpg, .jpeg, .png)",
            type=['jpg', 'jpeg', 'png']
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.subheader("Imagem Original:")
            st.image(image, caption="Imagem Carregada", use_container_width=True, channels="BGR")
            st.subheader("Resultado da Detecção:")
            with st.spinner("Processando..."):
                seed_count = predict_and_display(image, model, confidence_threshold)
                st.session_state.uploaded_image_count = seed_count

                st.session_state.processed_images_history.append({
                    "Data/Hora": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Fonte": "Upload",
                    "Modelo": selected_camera,
                    "Sementes": seed_count,
                    "Limiar": f"{confidence_threshold:.2f}"
                })

    with tab2:
        st.header("Câmera Ao Vivo")
        st.warning("⚠️ Pode não funcionar no navegador mobile ou em ambientes como o Streamlit Cloud.")
        col_button1, col_button2 = st.columns(2)
        with col_button1:
            start_button = st.button("📷 Iniciar Detecção Ao Vivo", key="live_detection_button")
        with col_button2:
            stop_button = st.button("🛑 Parar Detecção", key="stop_detection_button")
        if start_button:
            st.session_state.run_camera = True
        if stop_button:
            st.session_state.run_camera = False

        frame_placeholder = st.empty()

        if st.session_state.run_camera:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Não foi possível acessar a câmera.")
                st.session_state.run_camera = False
            else:
                while st.session_state.run_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Falha ao capturar imagem da câmera.")
                        st.session_state.run_camera = False
                        break

                    annotated_frame, seed_count = predict_and_display(frame, model, confidence_threshold, is_camera=True)
                    st.session_state.seed_count = seed_count

                    if annotated_frame is not None:
                        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                    else:
                        st.warning("⚠️ Imagem da câmera não pôde ser processada.")

                    st.session_state.camera_history.append({
                        "Data/Hora": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Fonte": "Câmera",
                        "Modelo": selected_camera,
                        "Sementes": seed_count,
                        "Limiar": f"{confidence_threshold:.2f}"
                    })

                cap.release()
                cv2.destroyAllWindows()
                frame_placeholder.empty()

    with tab3:
        st.header("Estatísticas")
        
        # ----------------- ALTERAÇÃO 2: FILTRO POR DATA -----------------
        st.subheader("Filtrar por Período")
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Data de Início", value=pd.to_datetime("today").date())
        with col_end:
            end_date = st.date_input("Data de Fim", value=pd.to_datetime("today").date())
        
        # Histórico de Upload
        st.markdown("### Histórico de Análises de Imagens (Upload)")
        if st.session_state.processed_images_history:
            df_history_upload = pd.DataFrame(st.session_state.processed_images_history)
            df_history_upload['Data/Hora'] = pd.to_datetime(df_history_upload['Data/Hora'])
            
            filtered_df_upload = df_history_upload[
                (df_history_upload['Data/Hora'].dt.date >= start_date) &
                (df_history_upload['Data/Hora'].dt.date <= end_date)
            ]
            
            st.dataframe(filtered_df_upload, use_container_width=True)
            if st.button("Limpar Histórico de Upload", key="clear_upload"):
                st.session_state.processed_images_history = []
                st.rerun()
        else:
            st.info("Nenhuma imagem foi processada ainda.")

        st.markdown("---")

        # Histórico da Câmera
        st.markdown("### Histórico de Análises da Câmera (Ao Vivo)")
        if st.session_state.camera_history:
            df_history_camera = pd.DataFrame(st.session_state.camera_history)
            df_history_camera['Data/Hora'] = pd.to_datetime(df_history_camera['Data/Hora'])
            
            filtered_df_camera = df_history_camera[
                (df_history_camera['Data/Hora'].dt.date >= start_date) &
                (df_history_camera['Data/Hora'].dt.date <= end_date)
            ]

            st.dataframe(filtered_df_camera, use_container_width=True)
            if st.button("Limpar Histórico da Câmera", key="clear_camera"):
                st.session_state.camera_history = []
                st.rerun()
        else:
            st.info("A contagem da câmera será mostrada aqui.")

else:
    st.warning("Sua versão do Streamlit não suporta abas. Atualize para >= 1.18.0.")