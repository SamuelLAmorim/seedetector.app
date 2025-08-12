import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
from packaging import version
import pandas as pd

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
    except Exception:
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
    st.session_state.processed_images_history = []  # histórico completo para estatísticas
if "processed_images_display" not in st.session_state:
    st.session_state.processed_images_display = []  # imagens exibidas na aba Upload
if "camera_history" not in st.session_state:
    st.session_state.camera_history = []
if "uploaded_files_processed" not in st.session_state:
    st.session_state.uploaded_files_processed = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "last_uploaded_file_names" not in st.session_state:
    st.session_state.last_uploaded_file_names = None

# ----------------- FUNÇÃO DE DETECÇÃO -----------------
def predict_and_display(image, model, confidence, is_camera=False):
    seed_count = 0
    inteiras_count = 0
    predadas_count = 0
    im_array = None

    if model is not None:
        try:
            results = model(image, conf=confidence, verbose=False)
        except Exception as e:
            st.warning(f"Erro ao inferir com o modelo: {e}")
            results = None

        if results and len(results) > 0 and getattr(results[0], 'boxes', None):
            seed_count = len(results[0].boxes)
            class_names = model.names if hasattr(model, 'names') else {}

            for box in results[0].boxes:
                try:
                    class_id = int(box.cls)
                    class_name = class_names.get(class_id, str(class_id))
                except Exception:
                    class_name = str(getattr(box, 'cls', ''))

                if class_name == "inteira":
                    inteiras_count += 1
                elif class_name == "predada":
                    predadas_count += 1

            try:
                im_array = results[0].plot()
                im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            except Exception as e:
                st.warning(f"Erro ao gerar imagem anotada: {e}")
                im_array = None

    if im_array is None:
        try:
            im_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            st.error("Erro ao processar imagem.")
            return None, (0, 0, 0)

    return im_array, (seed_count, inteiras_count, predadas_count)

# ----------------- INTERFACE COM ABAS -----------------
if version.parse(st.__version__) >= version.parse("1.18.0"):
    tab1, tab2, tab3 = st.tabs(["Upload de Imagem/Pasta", "Câmera Ao Vivo", "Estatísticas"]) 

    # ----------------- ABA 1 - UPLOAD -----------------
    with tab1:
        st.header("Upload de Imagem ou Pasta")
        show_saved_history = st.checkbox("Mostrar histórico salvo de uploads processados", value=False)

        uploaded_files = st.file_uploader(
            "Selecione uma ou mais imagens (.jpg, .jpeg, .png)",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}"
        )

        uploaded_files_list = list(uploaded_files) if uploaded_files else []

        current_names = [f.name for f in uploaded_files_list]
        if current_names != (st.session_state.last_uploaded_file_names or []):
            st.session_state.processed_images_display = []
            st.session_state.uploaded_files_processed = False
            st.session_state.last_uploaded_file_names = current_names

        if uploaded_files_list and not st.session_state.uploaded_files_processed:
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files_list):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)

                annotated_image, (seed_count, inteiras_count, predadas_count) = predict_and_display(
                    image, model, confidence_threshold, is_camera=False
                )

                new_record = {
                    "Data/Hora": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Fonte": "Upload",
                    "Arquivo": uploaded_file.name,
                    "Modelo": selected_camera,
                    "Total": seed_count,
                    "Inteiras": inteiras_count,
                    "Predadas": predadas_count,
                    "Limiar": f"{confidence_threshold:.2f}",
                    "Imagem_Processada": annotated_image,
                    "Imagem_Original": image
                }

                st.session_state.processed_images_history.append(new_record)
                st.session_state.processed_images_display.append(new_record)

                progress_bar.progress((i + 1) / len(uploaded_files_list))
                status_text.text(f"Processando imagem {i+1} de {len(uploaded_files_list)}: {uploaded_file.name}")

            st.session_state.uploaded_files_processed = True
            progress_bar.empty()
            status_text.empty()

        if st.session_state.processed_images_display:
            st.subheader("Resultados do Upload")
            num_columns = 3
            cols = st.columns(num_columns)

            for i, record in enumerate(st.session_state.processed_images_display):
                with cols[i % num_columns]:
                    if record["Imagem_Processada"] is not None:
                        st.image(record["Imagem_Processada"],
                                 caption=(f"Arquivo: {record['Arquivo']}\n"
                                          f"Total: {record['Total']}\n"
                                          f"Inteiras: {record['Inteiras']}\n"
                                          f"Predadas: {record['Predadas']}"),
                                 use_container_width=True)

                    if st.button(f"🔄 Refazer Análise - {record['Arquivo']}", key=f"refazer_{i}"):
                        image_to_reprocess = record["Imagem_Original"]
                        annotated_image, (seed_count, inteiras_count, predadas_count) = predict_and_display(
                            image_to_reprocess, model, confidence_threshold, is_camera=False
                        )
                        updated_record = record.copy()
                        updated_record.update({
                            "Data/Hora": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Total": seed_count,
                            "Inteiras": inteiras_count,
                            "Predadas": predadas_count,
                            "Limiar": f"{confidence_threshold:.2f}",
                            "Imagem_Processada": annotated_image
                        })
                        st.session_state.processed_images_display[i] = updated_record
                        st.session_state.processed_images_history.append(updated_record)
                        st.rerun()

            if st.button("Limpar Imagens do Upload", key="limpar_upload"):
                st.session_state.processed_images_display.clear()
                st.session_state.uploaded_files_processed = False
                st.session_state.last_uploaded_file_names = []
                st.session_state.uploader_key += 1
                st.rerun()
        else:
            if show_saved_history and st.session_state.processed_images_history:
                st.subheader("Resultados Salvos (Histórico)")
                num_columns = 3
                cols = st.columns(num_columns)
                for i, record in enumerate(st.session_state.processed_images_history):
                    with cols[i % num_columns]:
                        if record["Imagem_Processada"] is not None:
                            st.image(record["Imagem_Processada"],
                                     caption=(f"Arquivo: {record['Arquivo']}\n"
                                              f"Total: {record['Total']}\n"
                                              f"Inteiras: {record['Inteiras']}\n"
                                              f"Predadas: {record['Predadas']}"),
                                     use_container_width=True)
            else:
                st.info("Nenhum arquivo carregado. Faça upload para analisar ou marque 'Mostrar histórico salvo'.")

    # ----------------- ABA 2 - CÂMERA -----------------
    with tab2:
        st.header("Câmera Ao Vivo")
        col_button1, col_button2 = st.columns(2)
        with col_button1:
            start_button = st.button("📷 Iniciar Detecção Ao Vivo", key="start_camera")
        with col_button2:
            stop_button = st.button("🛑 Parar Detecção", key="stop_camera")
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
                        break
                    annotated_frame, (seed_count, inteiras_count, predadas_count) = predict_and_display(
                        frame, model, confidence_threshold, is_camera=True
                    )
                    st.session_state.camera_history.append({
                        "Data/Hora": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Fonte": "Câmera",
                        "Modelo": selected_camera,
                        "Total": seed_count,
                        "Inteiras": inteiras_count,
                        "Predadas": predadas_count,
                        "Limiar": f"{confidence_threshold:.2f}"
                    })
                    if annotated_frame is not None:
                        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
            cap.release()
            cv2.destroyAllWindows()

    # ----------------- ABA 3 - ESTATÍSTICAS -----------------
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
        total_predadas = 0

        if st.session_state.processed_images_history:
            df_upload = pd.DataFrame(st.session_state.processed_images_history).drop(columns=['Imagem_Processada', 'Imagem_Original'], errors='ignore')
            df_upload['Data/Hora'] = pd.to_datetime(df_upload['Data/Hora'])
            filtered_upload = df_upload[
                (df_upload['Data/Hora'].dt.date >= start_date) &
                (df_upload['Data/Hora'].dt.date <= end_date)
            ]
            total_inteiras += filtered_upload['Inteiras'].sum()
            total_predadas += filtered_upload['Predadas'].sum()

        if st.session_state.camera_history:
            df_camera = pd.DataFrame(st.session_state.camera_history)
            df_camera['Data/Hora'] = pd.to_datetime(df_camera['Data/Hora'])
            filtered_camera = df_camera[
                (df_camera['Data/Hora'].dt.date >= start_date) &
                (df_camera['Data/Hora'].dt.date <= end_date)
            ]
            total_inteiras += filtered_camera['Inteiras'].sum()
            total_predadas += filtered_camera['Predadas'].sum()

        col_inteiras, col_predadas = st.columns(2)
        with col_inteiras:
            st.metric(label="Sementes Inteiras (Total)", value=int(total_inteiras))
        with col_predadas:
            st.metric(label="Predadas (Total)", value=int(total_predadas))

        st.markdown("---")

        st.markdown("### Histórico de Análises de Imagens (Upload)")
        if st.session_state.processed_images_history:
            st.dataframe(filtered_upload, use_container_width=True)
            if st.button("🗑️ Limpar Histórico de Uploads", key="limpar_hist_upload"):
                st.session_state.processed_images_history.clear()
                st.rerun()
        else:
            st.info("Nenhuma imagem foi processada ainda.")

        st.markdown("---")
        st.markdown("### Histórico de Análises da Câmera (Ao Vivo)")
        if st.session_state.camera_history:
            st.dataframe(filtered_camera, use_container_width=True)
            if st.button("🗑️ Limpar Histórico da Câmera", key="limpar_hist_camera"):
                st.session_state.camera_history.clear()
                st.rerun()
        else:
            st.info("A contagem da câmera será mostrada aqui.")
else:
    st.warning("Atualize o Streamlit para >= 1.18.0 para usar abas.")
