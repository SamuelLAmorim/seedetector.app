import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
from packaging import version
import pandas as pd
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn import Sequential
from ultralytics.nn.modules.conv import Conv

st.set_page_config(
    page_title="Detector de Sementes",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    img {
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

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

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    # Adiciona classe customizada para deserialização segura do PyTorch 2.6+
    try:
        torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv])
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
for key, default_value in {
    "run_camera": False,
    "processed_images_history": [],
    "processed_images_display": [],
    "camera_history": [],
    "uploaded_files_processed": False,
    "uploader_key": 0,
    "last_uploaded_file_names": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ----------------- FUNÇÃO DE DETECÇÃO -----------------
def predict_and_display(image, model, confidence, is_camera=False):
    seed_count = 0
    inteiras_count = 0
    predadas_count = 0
    quebradas_count = 0
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
                elif class_name == "pedrada": # Corrigido de "predada" para "pedrada"
                    predadas_count += 1
                elif class_name == "quebrada":
                    quebradas_count += 1
                                    

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
            return None, (0, 0, 0, 0)

    return im_array, (seed_count, inteiras_count, predadas_count, quebradas_count)

# --- NEW: Função para gerar instruções da API ---
def generate_api_instructions(inteiras_count, predadas_count, quebradas_count):
    total_detected = inteiras_count + predadas_count + quebradas_count
    
    # DEBUG: Para verificar os valores de entrada na função
    print(f"DEBUG_FUNC_INPUT: Inteiras={inteiras_count}, Predadas={predadas_count}, Quebradas={quebradas_count}, Total={total_detected}")

    instructions = []

    if total_detected == 0:
        instructions.append("Não foram detectadas sementes na imagem. Por favor, verifique a imagem e o limiar de confiança.")
    else:
        # Calculate percentages only if total_detected > 0 to avoid ZeroDivisionError
        perc_inteiras = (inteiras_count / total_detected) * 100
        perc_predadas = (predadas_count / total_detected) * 100
        perc_quebradas = (quebradas_count / total_detected) * 100

        # Priority for predadas
        if predadas_count > 0 and perc_predadas >= 20: # Threshold for "high quantity"
            instructions.append(f"🔍 **Observação: Alta quantidade de sementes predadas ({predadas_count} sementes, {perc_predadas:.1f}%)**")
            instructions.append("Isso pode indicar um **alto índice de pragas** no seu plantio.")
            instructions.append("### Perguntas para Reflexão:")
            instructions.append("- Você notou sinais de insetos, roedores ou pássaros na área de plantio?")
            instructions.append("- Qual o método de controle de pragas atualmente utilizado?")
            instructions.append("- Há histórico de pragas nesta área em safras anteriores?")
            instructions.append("### Sugestões:")
            instructions.append("- Considere a aplicação de **inseticidas específicos** para as pragas identificadas. Consulte um agrônomo para a escolha do produto adequado.")
            instructions.append("- Avalie a utilização de **fertilizantes** que promovam a saúde da planta, tornando-a mais resistente a ataques de pragas.")
            instructions.append("- Implemente **monitoramento regular** das pragas para identificar e agir precocemente.")
        
        # Priority for quebradas (if predadas is not the main issue or if quebradas are also significant)
        if quebradas_count > 0 and perc_quebradas >= 20: # Threshold for "high quantity"
            if len(instructions) > 0: instructions.append("---") # Separator if there are prior instructions
            instructions.append(f"💔 **Observação: Alta quantidade de sementes quebradas ({quebradas_count} sementes, {perc_quebradas:.1f}%)**")
            instructions.append("Isso sugere que a **colheita pode estar sendo feita de forma muito bruta**, resultando em **dano mecânico** nas sementes.")
            instructions.append("### Perguntas para Reflexão:")
            instructions.append("- Qual equipamento de colheita está sendo utilizado e há quanto tempo foi revisado?")
            instructions.append("- A velocidade da colheitadeira está adequada para as condições da lavoura?")
            instructions.append("- Há pontos de impacto excessivo ou ajustes inadequados na máquina?")
            instructions.append("### Sugestões:")
            instructions.append("- **Revise e ajuste a colheitadeira**: Verifique configurações como a abertura do côncavo, velocidade do cilindro trilhador e o espaçamento dos peneiras.")
            instructions.append("- **Reduza a velocidade de colheita**: Uma velocidade excessiva pode aumentar significativamente o dano mecânico.")
            instructions.append("- Treine a equipe de operação para manusear o equipamento com maior cuidado.")

        # General message for inteiras
        # Only show "ótima" if there are NO predadas or quebradas, AND it's mostly inteiras.
        if inteiras_count > 0 and perc_inteiras >= 80 and predadas_count == 0 and quebradas_count == 0:
            if len(instructions) > 0: instructions.append("---") # Separator if there are prior instructions
            instructions.append(f"✅ **Parabéns! Sua colheita está ótima! ({inteiras_count} sementes inteiras, {perc_inteiras:.1f}%)**")
            instructions.append("A alta proporção de sementes inteiras indica excelente qualidade no seu manejo e colheita.")
            instructions.append("Continue com as boas práticas!")
        
        # Fallback for general analysis if no specific high-priority issues were flagged
        if not instructions and total_detected > 0: # If instructions list is still empty, and total_detected > 0
            instructions.append("📈 **Análise Geral**: A proporção de sementes parece equilibrada ou não há um problema predominante. Continue monitorando!")

    final_instructions = "\n".join(instructions)
    # DEBUG: Para verificar a string de saída da função
    print(f"DEBUG_FUNC_OUTPUT: '{final_instructions}'")
    return final_instructions


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

                annotated_image, (seed_count, inteiras_count, predadas_count, quebradas_count) = predict_and_display(
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
                    "Quebradas": quebradas_count,
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
                    
                    # --- Display API Instructions for uploaded images ---
                    st.markdown("---")
                    st.markdown("### Recomendações da Análise:")
                    instructions = generate_api_instructions(
                        record['Inteiras'], record['Predadas'], record['Quebradas']
                    )
                    st.markdown(instructions) # Using st.markdown to render the formatted string
                    st.markdown("---") # Add a separator after instructions for clarity

                    if st.button(f"🔄 Refazer Análise - {record['Arquivo']}", key=f"refazer_{i}"):
                        image_to_reprocess = record["Imagem_Original"]
                        annotated_image, (seed_count, inteiras_count, predadas_count, quebradas_count) = predict_and_display(
                            image_to_reprocess, model, confidence_threshold, is_camera=False
                        )
                        updated_record = record.copy()
                        updated_record.update({
                            "Data/Hora": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Total": seed_count,
                            "Inteiras": inteiras_count,
                            "Predadas": predadas_count,
                            "Quebradas": quebradas_count,
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
                            # --- Display API Instructions for saved history images ---
                            st.markdown("---")
                            st.markdown("### Recomendações da Análise (Histórico):")
                            instructions = generate_api_instructions(
                                record['Inteiras'], record['Predadas'], record['Quebradas']
                            )
                            st.markdown(instructions)
                            st.markdown("---")
            else:
                st.info("Nenhum arquivo carregado. Faça upload para analisar ou marque 'Mostrar histórico salvo'.")

    # ----------------- ABA 2 - CÂMERA -----------------
    with tab2: # Conteúdo da ABA 2 - Câmera
        st.header("Câmera Ao Vivo - Captura de Fotos")

        camera_input = st.camera_input("Abra a câmera e tire fotos para detectar sementes")

        if camera_input is not None:
            file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated_frame, (seed_count, inteiras_count, predadas_count, quebradas_count) = predict_and_display(
                frame, model, confidence_threshold, is_camera=True
            )
            st.image(annotated_frame, channels="RGB", use_container_width=True)

            # --- Display API Instructions for camera input ---
            st.markdown("---")
            st.markdown("### Recomendações da Análise:")
            instructions = generate_api_instructions(inteiras_count, predadas_count, quebradas_count)
            st.markdown(instructions)
            st.markdown("---")

            # Salva no histórico
            st.session_state.camera_history.append({
                "Data/Hora": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Fonte": "Câmera Foto",
                "Modelo": selected_camera,
                "Total": seed_count,
                "Inteiras": inteiras_count,
                "Predadas": predadas_count,
                "Quebradas": quebradas_count,
                "Limiar": f"{confidence_threshold:.2f}"
            })
        else:
            st.info("Use o botão acima para capturar uma foto da câmera.")

# ---------------- ESTATÍSTICAS MELHORADAS ----------------

    with tab3: # Conteúdo da ABA 3 - Estatísticas
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
        total_quebradas = 0

        def process_dataframe(history):
            if not history:
                return pd.DataFrame()
            df = pd.DataFrame(history)
            if 'Data/Hora' in df.columns:
                df['Data/Hora'] = pd.to_datetime(df['Data/Hora'], errors='coerce')
            # Converter contagens para int, preencher nulos com 0
            for col in ['Inteiras', 'Predadas', 'Quebradas']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            return df

        df_upload = process_dataframe(st.session_state.processed_images_history)
        df_camera = process_dataframe(st.session_state.camera_history)

        filtered_upload = df_upload[
            (df_upload['Data/Hora'].dt.date >= start_date) &
            (df_upload['Data/Hora'].dt.date <= end_date)
        ] if not df_upload.empty else pd.DataFrame()

        filtered_camera = df_camera[
            (df_camera['Data/Hora'].dt.date >= start_date) &
            (df_camera['Data/Hora'].dt.date <= end_date)
        ] if not df_camera.empty else pd.DataFrame()

        total_inteiras += filtered_upload['Inteiras'].sum() if not filtered_upload.empty else 0
        total_inteiras += filtered_camera['Inteiras'].sum() if not filtered_camera.empty else 0

        total_predadas += filtered_upload['Predadas'].sum() if not filtered_upload.empty else 0
        total_predadas += filtered_camera['Predadas'].sum() if not filtered_camera.empty else 0

        total_quebradas += filtered_upload['Quebradas'].sum() if not filtered_upload.empty else 0
        total_quebradas += filtered_camera['Quebradas'].sum() if not filtered_camera.empty else 0

        col_inteiras, col_predadas, col_quebradas = st.columns(3)
        with col_inteiras:
            st.metric(label="Sementes Inteiras (Total)", value=int(total_inteiras))
        with col_predadas:
            st.metric(label="Predadas (Total)", value=int(total_predadas))
        with col_quebradas:
            st.metric(label="Quebradas (Total)", value=int(total_quebradas))

        st.markdown("---")
        # --- NEW: Display API Instructions for filtered statistics ---
        st.subheader("Recomendações da Análise para o Período Selecionado:")
        stats_instructions = generate_api_instructions(total_inteiras, total_predadas, total_quebradas)
        
        # --- DEBUGGING LINES ---
        st.write(f"Totais passados para a função: Inteiras={total_inteiras}, Predadas={total_predadas}, Quebradas={total_quebradas}")
        st.write("Instruções:")
        st.code(stats_instructions)
        # --- END DEBUGGING LINES ---

        st.markdown(stats_instructions)
        st.markdown("---")


        st.markdown("### Histórico de Análises de Imagens (Upload)")
        if not filtered_upload.empty:
            st.dataframe(filtered_upload.drop(columns=['Imagem_Processada', 'Imagem_Original'], errors='ignore'), use_container_width=True)
            if st.button("🗑️ Limpar Histórico de Uploads", key="limpar_hist_upload"):
                st.session_state.processed_images_history.clear()
                st.rerun()
        else:
                st.info("Nenhuma imagem foi processada no período selecionado.")

        st.markdown("---")
        st.markdown("### Histórico de Análises da Câmera (Ao Vivo)")
        if not filtered_camera.empty:
            st.dataframe(filtered_camera, use_container_width=True)
            if st.button("🗑️ Limpar Histórico da Câmera", key="limpar_hist_camera"):
                st.session_state.camera_history.clear()
                st.rerun()
        else:
                st.info("A contagem da câmera será mostrada aqui.")
    
else:
    st.warning("Atualize o Streamlit para >= 1.18.0 para usar abas.")


