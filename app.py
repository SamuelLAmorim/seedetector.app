# app.py
import streamlit as st
import sqlite3
import bcrypt
import re
import time
import pandas as pd
import numpy as np
import cv2
from packaging import version

# ===== Imports do detector (só serão usados após login) =====
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn import Sequential
from ultralytics.nn.modules.conv import Conv

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
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

# ============================================================
# BANCO DE DADOS (SQLite) - Usuários
# ============================================================
def init_db():
    conn = sqlite3.connect("usuarios.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        nome TEXT,
        senha_hash TEXT,
        cpf TEXT,
        tamanho_plantacao REAL,
        local TEXT
    )
    """)
    conn.commit()
    conn.close()

def add_user(username: str, nome: str, senha: str, cpf: str, tamanho_plantacao: float, local: str) -> bool:
    conn = sqlite3.connect("usuarios.db")
    c = conn.cursor()
    senha_hash = bcrypt.hashpw(senha.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        c.execute(
            """
            INSERT INTO usuarios (username, nome, senha_hash, cpf, tamanho_plantacao, local)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (username.strip().lower(), nome.strip(), senha_hash, cpf.strip(), tamanho_plantacao, local.strip())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def check_user(username: str, senha: str):
    conn = sqlite3.connect("usuarios.db")
    c = conn.cursor()
    c.execute("SELECT senha_hash, nome FROM usuarios WHERE username=?", (username.strip().lower(),))
    row = c.fetchone()
    conn.close()
    if row:
        senha_hash, nome = row
        try:
            if bcrypt.checkpw(senha.encode('utf-8'), senha_hash.encode('utf-8')):
                return True, nome
        except Exception:
            return False, None
    return False, None

# ============================================================
# FUNÇÃO PARA VALIDAR CPF
# ============================================================
def validar_cpf(cpf: str) -> bool:
    cpf = re.sub(r'[^0-9]', '', cpf)
    if len(cpf) != 11 or cpf == cpf[0] * 11:
        return False
    soma1 = sum(int(cpf[i]) * (10 - i) for i in range(9))
    dig1 = (soma1 * 10 % 11) % 10
    soma2 = sum(int(cpf[i]) * (11 - i) for i in range(10))
    dig2 = (soma2 * 10 % 11) % 10
    return dig1 == int(cpf[9]) and dig2 == int(cpf[10])

# ============================================================
# TELAS DE AUTENTICAÇÃO
# ============================================================
def login_screen():
    st.subheader("🔐 Login")
    with st.form("form_login", clear_on_submit=False):
        username = st.text_input("Usuário")
        senha = st.text_input("Senha", type="password")
        submit = st.form_submit_button("Entrar")
    if submit:
        ok, nome = check_user(username, senha)
        if ok:
            st.session_state["logado"] = True
            st.session_state["usuario"] = username.strip().lower()
            st.session_state["nome"] = nome
            st.success(f"Bem-vindo, {nome}!")
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos.")

def signup_screen():
    st.subheader("📝 Cadastro")
    with st.form("form_signup", clear_on_submit=False):
        nome = st.text_input("Nome completo")
        username = st.text_input("Usuário (login, min. 3 caracteres)")
        cpf = st.text_input("CPF (somente números)")
        senha = st.text_input("Senha (min. 6 caracteres)", type="password")
        senha2 = st.text_input("Confirmar senha", type="password")
        tamanho_plantacao = st.number_input("Tamanho da plantação (ha)", min_value=0.01, step=0.01)
        local = st.text_input("Local da fazenda/espaço de cultivo")
        submit = st.form_submit_button("Cadastrar")

    if submit:
        # Valida campos obrigatórios
        if not nome.strip() or not username.strip() or not senha or not cpf.strip() or not local.strip():
            st.error("Preencha todos os campos obrigatórios.")
            return
        # Nome completo
        if len(nome.strip().split()) < 2:
            st.warning("Digite seu nome completo (pelo menos 2 palavras).")
            return
        # Usuário e senha
        if len(username.strip()) < 3 or len(senha) < 6:
            st.warning("Usuário deve ter no mínimo 3 caracteres e senha no mínimo 6 caracteres.")
            return
        # Senhas conferem?
        if senha != senha2:
            st.error("As senhas não coincidem.")
            return
        # CPF válido?
        if not validar_cpf(cpf):
            st.error("CPF inválido. Digite um CPF válido.")
            return
        # Tamanho da plantação positivo
        if tamanho_plantacao <= 0:
            st.error("Informe um tamanho de plantação válido (>0).")
            return
        # Tentar adicionar usuário
        if add_user(username, nome, senha, cpf, tamanho_plantacao, local):
            st.success("Usuário cadastrado com sucesso! Agora faça login.")
        else:
            st.error("Usuário já existe. Tente outro login.")

# ============================================================
# APLICAÇÃO PRINCIPAL (seu detector) - Só aparece após login
# ============================================================
def main_app():
    # ----------------- CABEÇALHO E TOPO -----------------
    st.sidebar.success(f"👤 Logado como: {st.session_state.get('nome', '')}")
    if st.sidebar.button("Sair"):
        st.session_state.clear()
        st.rerun()

    st.title("🌽 Detector de Sementes de Milho")
    st.markdown("Faça o upload de uma ou várias imagens, ou use sua câmera para detectar sementes.")
    st.markdown("---")

    # ----------------- MODELOS DISPONÍVEIS -----------------
    camera_options = {
        "Câmera Casual (RGB)": "models/best.pt",
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
                    elif class_name == "pedrada":  # importante: sua classe está assim
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

    # --- Gera recomendações (API instructions) ---
    def generate_api_instructions(inteiras_count, predadas_count, quebradas_count):
        total_detected = inteiras_count + predadas_count + quebradas_count
        instructions = []

        if total_detected == 0:
            instructions.append("Não foram detectadas sementes na imagem. Verifique a imagem e o limiar de confiança.")
        else:
            perc_inteiras = (inteiras_count / total_detected) * 100 if total_detected else 0
            perc_predadas = (predadas_count / total_detected) * 100 if total_detected else 0
            perc_quebradas = (quebradas_count / total_detected) * 100 if total_detected else 0

            if predadas_count > 0 and perc_predadas >= 20:
                instructions += [
                    f"🔍 **Observação: Alta quantidade de sementes predadas ({predadas_count} sementes, {perc_predadas:.1f}%)**",
                    "Isso pode indicar um **alto índice de pragas** no seu plantio.",
                    "### Perguntas para Reflexão:",
                    "- Você notou sinais de insetos, roedores ou pássaros na área de plantio?",
                    "- Qual o método de controle de pragas atualmente utilizado?",
                    "- Há histórico de pragas nesta área em safras anteriores?",
                    "### Sugestões:",
                    "- Considere a aplicação de **inseticidas específicos** (consulte um agrônomo).",
                    "- Avalie **fertilizantes** que melhorem a saúde da planta.",
                    "- Implemente **monitoramento regular** das pragas."
                ]

            if quebradas_count > 0 and perc_quebradas >= 20:
                if instructions: instructions.append("---")
                instructions += [
                    f"💔 **Observação: Alta quantidade de sementes quebradas ({quebradas_count} sementes, {perc_quebradas:.1f}%)**",
                    "Isso sugere **dano mecânico** na colheita.",
                    "### Perguntas para Reflexão:",
                    "- Qual equipamento de colheita está sendo utilizado? Quando foi revisado?",
                    "- A velocidade da colheitadeira está adequada?",
                    "- Há ajustes inadequados ou pontos de impacto excessivo?",
                    "### Sugestões:",
                    "- **Revisar e ajustar colheitadeira** (côncavo, cilindro, peneiras).",
                    "- **Reduzir velocidade de colheita**.",
                    "- Treinar a equipe para manusear o equipamento com cuidado."
                ]

            if inteiras_count > 0 and perc_inteiras >= 80 and predadas_count == 0 and quebradas_count == 0:
                if instructions: instructions.append("---")
                instructions += [
                    f"✅ **Parabéns! Colheita ótima! ({inteiras_count} sementes inteiras, {perc_inteiras:.1f}%)**",
                    "A alta proporção de inteiras indica excelente manejo e colheita. Continue assim!"
                ]

            if not instructions and total_detected > 0:
                instructions.append("📈 **Análise Geral**: Proporções equilibradas. Continue monitorando!")

        return "\n".join(instructions)

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
                            st.image(
                                record["Imagem_Processada"],
                                caption=(f"Arquivo: {record['Arquivo']}\n"
                                         f"Total: {record['Total']}\n"
                                         f"Inteiras: {record['Inteiras']}\n"
                                         f"Predadas: {record['Predadas']}"),
                                use_container_width=True
                            )

                        st.markdown("---")
                        st.markdown("### Recomendações da Análise:")
                        instructions = generate_api_instructions(
                            record['Inteiras'], record['Predadas'], record['Quebradas']
                        )
                        st.markdown(instructions)
                        st.markdown("---")

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
                                st.image(
                                    record["Imagem_Processada"],
                                    caption=(f"Arquivo: {record['Arquivo']}\n"
                                             f"Total: {record['Total']}\n"
                                             f"Inteiras: {record['Inteiras']}\n"
                                             f"Predadas: {record['Predadas']}"),
                                    use_container_width=True
                                )
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
        with tab2:
            st.header("Câmera Ao Vivo - Captura de Fotos")
            camera_input = st.camera_input("Abra a câmera e tire fotos para detectar sementes")

            if camera_input is not None:
                file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                annotated_frame, (seed_count, inteiras_count, predadas_count, quebradas_count) = predict_and_display(
                    frame, model, confidence_threshold, is_camera=True
                )
                st.image(annotated_frame, channels="RGB", use_container_width=True)

                st.markdown("---")
                st.markdown("### Recomendações da Análise:")
                instructions = generate_api_instructions(inteiras_count, predadas_count, quebradas_count)
                st.markdown(instructions)
                st.markdown("---")

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
            total_quebradas = 0

            def process_dataframe(history):
                if not history:
                    return pd.DataFrame()
                df = pd.DataFrame(history)
                if 'Data/Hora' in df.columns:
                    df['Data/Hora'] = pd.to_datetime(df['Data/Hora'], errors='coerce')
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
            st.subheader("Recomendações da Análise para o Período Selecionado:")
            stats_instructions = generate_api_instructions(total_inteiras, total_predadas, total_quebradas)

            # (debug opcional) — comente se não quiser ver
            st.write(f"Totais passados: Inteiras={total_inteiras}, Predadas={total_predadas}, Quebradas={total_quebradas}")
            st.code(stats_instructions)

            st.markdown(stats_instructions)
            st.markdown("---")

            st.markdown("### Histórico de Análises de Imagens (Upload)")
            if not filtered_upload.empty:
                st.dataframe(filtered_upload.drop(columns=['Imagem_Processada', 'Imagem_Original'], errors='ignore'), use_container_width=True)
                if st.button("🗑️ Limpar Histórico de Uploads", key="limpar_hist_upload"):
                    st.session_state.processed_images_history.clear()
                    st.rerun()
            else:
                st.info("Nenhuma imagem processada no período selecionado.")

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

# ============================================================
# EXECUÇÃO
# ============================================================
init_db()

if "logado" not in st.session_state:
    st.session_state["logado"] = False

if not st.session_state["logado"]:
    tab_login, tab_signup = st.tabs(["Login", "Cadastro"])
    with tab_login:
        login_screen()
    with tab_signup:
        signup_screen()
else:
    main_app()
