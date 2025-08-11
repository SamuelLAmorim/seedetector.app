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
st.markdown("Faça o upload de uma ou várias imagens, ou use sua câmera para detectar sementes.")
st.markdown("---")

# ----------------- MODELOS DISPONÍVEIS -----------------
camera_options = {
    "Câmera Casual (RGB)": "best.pt",
    "Câmera RGN": "rgn.pt",
    "Câmera RE": "re.pt",
    "Câmera NIR": "nir.pt",
    "RGB": "rgb.pt"
}

# ----------------- SIDEBAR -----------------
st.sidebar.header("Opções de Detecção")

selected_camera = st.sidebar.selectbox(
    "Qual câmera foi utilizada?",
    list(camera_options.keys()),
    help="Selecione o tipo de câmera para carregar o modelo de detecção correspondente."
)

@st.cache_resource
def load_model(model_path):
    import torch
    try:
        torch.serialization.add_safe_globals([YOLO])
    except:
        pass
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo '{model_path}': {e}")
        return None

model_path = camera_options[selected_camera]
model = load_model(model_path)

confidence_threshold = st.sidebar.slider(
    "Limiar de Confiança:",
    min_value=0.01,
    max_value=1.0,
    value=0.25,
    step=0.01
)

# ----------------- ESTADO DA SESSÃO -----------------
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "processed_images_history" not in st.session_state:
    st.session_state.processed_images_history = []
if "camera_history" not in st.session_state:
    st.session_state.camera_history = []
if "uploaded_files_processed" not in st.session_state:
    st.session_state.uploaded_files_processed = False

# ----------------- FUNÇÃO DE DETECÇÃO -----------------
def predict_and_display(image, model, confidence, is_camera=False):
    seed_count = 0
    inteiras_count = 0
    pedradas_count = 0
    im_array = None

    if model:
        # Adicionado verbose=False para suprimir a saída do terminal
        results = model(image, conf=confidence, verbose=False)
        if results and results[0].boxes:
            seed_count = len(results[0].boxes)
            
            class_names = model.names
            
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = class_names[class_id]
                
                if class_name == "inteira": 
                    inteiras_count += 1
                elif class_name == "pedrada": 
                    pedradas_count += 1
            
            try:
                im_array = results[0].plot()
                im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            except Exception as e:
                # Se a plotagem falhar, ainda retorna as contagens
                st.warning(f"Erro ao gerar imagem anotada: {e}")
                im_array = None

    if im_array is None:
        try:
            im_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            st.error("Erro ao processar imagem.")
            if is_camera:
                return None, (0, 0, 0)
            else:
                return (0, 0, 0)

    if is_camera:
        return im_array, (seed_count, inteiras_count, pedradas_count)
    else:
        # Apenas retorna as contagens para o loop de upload
        return im_array, (seed_count, inteiras_count, pedradas_count)

# ----------------- INTERFACE COM ABAS -----------------
if version.parse(st.__version__) >= version.parse("1.18.0"):
    tab1, tab2, tab3 = st.tabs(["Upload de Imagem/Pasta", "Câmera Ao Vivo", "Estatísticas"])

    with tab1:
        st.header("Upload de Imagem ou Pasta")
        uploaded_files = st.file_uploader(
            "Selecione uma ou mais imagens (.jpg, .jpeg, .png)",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )

        if uploaded_files:
            # Processa as imagens apenas se houver novos uploads e não foram processadas ainda
            if not st.session_state.uploaded_files_processed:
                
                # Barra de progresso para melhor feedback visual
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)

                    # Obtém a imagem anotada e as contagens
                    annotated_image, (seed_count, inteiras_count, pedradas_count) = predict_and_display(image, model, confidence_threshold, is_camera=False)
                    
                    # Salva os valores em colunas separadas no histórico
                    st.session_state.processed_images_history.append({
                        "Data/Hora": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Fonte": "Upload",
                        "Arquivo": uploaded_file.name,
                        "Modelo": selected_camera,
                        "Total": seed_count,
                        "Inteiras": inteiras_count,
                        "Pedras": pedradas_count,
                        "Limiar": f"{confidence_threshold:.2f}",
                        "Imagem_Processada": annotated_image # Salva a imagem processada
                    })
                    
                    # Atualiza a barra de progresso
                    progress_value = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress_value)
                    status_text.text(f"Processando imagem {i+1} de {len(uploaded_files)}: {uploaded_file.name}")
                
                st.session_state.uploaded_files_processed = True
                progress_bar.empty()
                status_text.empty()
                st.rerun()

            # Exibe os resultados após o processamento
            if st.session_state.uploaded_files_processed and st.session_state.processed_images_history:
                st.subheader("Resultados do Upload")
                num_columns = 3 # Número de colunas para exibir as imagens
                cols = st.columns(num_columns)

                for i, record in enumerate(st.session_state.processed_images_history):
                    with cols[i % num_columns]:
                        st.image(record["Imagem_Processada"], caption=f"Arquivo: {record['Arquivo']}\nTotal: {record['Total']}\nInteiras: {record['Inteiras']}\nPedras: {record['Pedras']}", use_container_width=True)
                
                if st.button("Limpar Imagens do Upload"):
                    st.session_state.processed_images_history = []
                    st.session_state.uploaded_files_processed = False
                    st.rerun()

    with tab2:
        st.header("Câmera Ao Vivo")
        st.warning("⚠️ Pode não funcionar no navegador mobile ou no Streamlit Cloud.")
        col_button1, col_button2 = st.columns(2)
        with col_button1:
            start_button = st.button("📷 Iniciar Detecção Ao Vivo")
        with col_button2:
            stop_button = st.button("🛑 Parar Detecção")
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

                    # Captura os 3 valores retornados pela função
                    annotated_frame, (seed_count, inteiras_count, pedradas_count) = predict_and_display(frame, model, confidence_threshold, is_camera=True)
                    
                    # Salva os valores no histórico (pode ser otimizado para salvar a cada X frames)
                    st.session_state.camera_history.append({
                        "Data/Hora": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Fonte": "Câmera",
                        "Modelo": selected_camera,
                        "Total": seed_count,
                        "Inteiras": inteiras_count,
                        "Pedras": pedradas_count,
                        "Limiar": f"{confidence_threshold:.2f}"
                    })

                    if annotated_frame is not None:
                        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                    else:
                        st.warning("⚠️ Imagem da câmera não pôde ser processada.")
            
            cap.release()
            cv2.destroyAllWindows()
            frame_placeholder.empty()

    with tab3:
        st.header("Estatísticas")
        st.subheader("Filtrar por Período")
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Data de Início", value=pd.to_datetime("today").date())
        with col_end:
            end_date = st.date_input("Data de Fim", value=pd.to_datetime("today").date())
        
        st.markdown("---")
        st.subheader("Resumo do Período Selecionado")
        total_inteiras = 0
        total_pedras = 0

        # Calcula o resumo do período para os dados de upload
        if st.session_state.processed_images_history:
            df_upload = pd.DataFrame(st.session_state.processed_images_history).drop(columns=['Imagem_Processada'])
            df_upload['Data/Hora'] = pd.to_datetime(df_upload['Data/Hora'])
            filtered_upload = df_upload[
                (df_upload['Data/Hora'].dt.date >= start_date) &
                (df_upload['Data/Hora'].dt.date <= end_date)
            ]
            total_inteiras += filtered_upload['Inteiras'].sum()
            total_pedras += filtered_upload['Pedras'].sum()

        # Calcula o resumo do período para os dados da câmera
        if st.session_state.camera_history:
            df_camera = pd.DataFrame(st.session_state.camera_history)
            df_camera['Data/Hora'] = pd.to_datetime(df_camera['Data/Hora'])
            filtered_camera = df_camera[
                (df_camera['Data/Hora'].dt.date >= start_date) &
                (df_camera['Data/Hora'].dt.date <= end_date)
            ]
            total_inteiras += filtered_camera['Inteiras'].sum()
            total_pedras += filtered_camera['Pedras'].sum()

        col_inteiras, col_pedras = st.columns(2)
        with col_inteiras:
            st.metric(label="Sementes Inteiras (Total)", value=int(total_inteiras))
        with col_pedras:
            st.metric(label="Pedras (Total)", value=int(total_pedras))

        st.markdown("---")

        # Histórico de Upload
        st.markdown("### Histórico de Análises de Imagens (Upload)")
        if st.session_state.processed_images_history:
            df_history_upload = pd.DataFrame(st.session_state.processed_images_history).drop(columns=['Imagem_Processada'])
            df_history_upload['Data/Hora'] = pd.to_datetime(df_history_upload['Data/Hora'])
            filtered_df_upload = df_history_upload[
                (df_history_upload['Data/Hora'].dt.date >= start_date) &
                (df_history_upload['Data/Hora'].dt.date <= end_date)
            ]
            st.dataframe(filtered_df_upload, use_container_width=True)
            if st.button("Limpar Histórico de Upload"):
                st.session_state.processed_images_history = []
                st.session_state.uploaded_files_processed = False
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
            if st.button("Limpar Histórico da Câmera"):
                st.session_state.camera_history = []
                st.rerun()
        else:
            st.info("A contagem da câmera será mostrada aqui.")

else:
    st.warning("Sua versão do Streamlit não suporta abas. Atualize para >= 1.18.0.")
