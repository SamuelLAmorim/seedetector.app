import streamlit as st
import sqlite3
import bcrypt
import re
import time
import pandas as pd
import numpy as np
import cv2
from packaging import version
from PIL import Image
import altair as alt

# ===== Imports do detector (só serão usados após login) - Mantidos =====
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn import Sequential
from ultralytics.nn.modules.conv import Conv

# ============================================================
# CONFIGURAÇÃO DA PÁGINA E TEMA
# ============================================================
st.set_page_config(
    page_title="🌽 Detector de Sementes",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema Customizado (Ajuste as cores para sua marca)
# st.markdown("""
# <style>
# [data-testid="stSidebar"] {
#     background-color: #f0f2f6; /* Um cinza claro */
# }
# [data-testid="stHeader"] {
#     background-color: rgba(0,0,0,0); /* Transparente */
# }
# </style>
# """, unsafe_allow_html=True)


# CSS para melhorar a estética
st.markdown("""
    <style>
    /* Estilo para imagens e boxes (cards) */
    img {
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    }
    
    /* Melhora a legibilidade dos subcabeçalhos */
    h2 {
        color: #2e8b57; /* Verde esmeralda */
        border-bottom: 2px solid #3cb371; /* Fina linha verde */
        padding-bottom: 5px;
    }
    
    /* Estilo para as métricas (para dar um 'card' effect) */
    [data-testid="stMetricValue"] {
        color: #2e8b57; /* Cor do valor da métrica */
        font-size: 2em;
    }
    
    /* Customiza botões primários para serem mais verdes */
    .stButton>button {
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        opacity: 0.9;
    }
    
    /* Container para resultados (expander) */
    .stExpander {
        border-radius: 10px;
        border: 1px solid #dcdcdc;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Estilo para as mensagens de Recomendações */
    .recommendation-box {
    /* Define o fundo da caixa ligeiramente mais escuro/claro que o fundo principal */
    background-color: var(--st-secondary-background); 
    
    /* Linha de destaque verde - mantém para consistência */
    border-left: 5px solid #3cb371; 
    
    padding: 10px;
    margin-top: 10px;
    margin-bottom: 10px;
    border-radius: 5px;
}
    </style>
""", unsafe_allow_html=True)


# ============================================================
# BANCO DE DADOS (SQLite) - Usuários (Mantido)
# ============================================================

# Função para conectar/inicializar o DB (mantida, mas o uso será com 'with')
def get_db_connection():
    return sqlite3.connect("usuarios.db")

def init_db():
    # Usando o 'with' para garantir que a conexão feche
    with get_db_connection() as conn:
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

def add_user(username: str, nome: str, senha: str, cpf: str, tamanho_plantacao: float, local: str) -> bool:
    senha_hash = bcrypt.hashpw(senha.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        # Usando o 'with' para garantir que a conexão feche e o commit/rollback seja tratado
        with get_db_connection() as conn:
            c = conn.cursor()
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
        # Integridade violada (usuário já existe)
        return False
    except Exception as e:
        # Outros erros de DB
        # print(f"Erro ao adicionar usuário: {e}")
        return False

# Retorna um dicionário com os dados do usuário, ou None se falhar
def check_user(username: str, senha: str) -> dict | None:
    username = username.strip().lower()
    
    # Adicionando tratamento de exceção para garantir que o 'with' feche
    try:
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row # Permite acessar colunas por nome
            c = conn.cursor()
            # Seleciona todos os campos necessários
            c.execute(
                "SELECT senha_hash, nome, cpf, tamanho_plantacao, local FROM usuarios WHERE username=?",
                (username,)
            )
            row = c.fetchone()
            
            if row:
                user_data = dict(row)
                senha_hash = user_data.pop('senha_hash') # Remove o hash para não expor

                # Verifica a senha
                if bcrypt.checkpw(senha.encode('utf-8'), senha_hash.encode('utf-8')):
                    # Adiciona o username de volta antes de retornar
                    user_data['username'] = username
                    return user_data
    except Exception:
        # Erro de DB ou de criptografia
        return None
        
    return None

# ============================================================
# FUNÇÃO PARA VALIDAR CPF (Mantida)
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
# TELAS DE AUTENTICAÇÃO (Melhorado o visual)
# ============================================================
def login_screen():
    st.header("🔐 Login") # Usando st.header com emoji
    with st.form("form_login", clear_on_submit=False):
        username = st.text_input("Usuário", placeholder="Seu nome de usuário")
        senha = st.text_input("Senha", type="password", placeholder="******")
        submit = st.form_submit_button("Entrar", type="primary")
    
    if submit:
        user_data = check_user(username, senha)
        
        if user_data:
            st.session_state["logado"] = True
            
            # Armazena todos os dados importantes na sessão
            st.session_state["usuario"] = user_data['username']
            st.session_state["nome"] = user_data['nome']
            st.session_state["cpf"] = user_data['cpf']
            st.session_state["tamanho_plantacao"] = user_data['tamanho_plantacao']
            st.session_state["local"] = user_data['local']
            
            st.success(f"🎉 **Bem-vindo, {user_data['nome']}!** Acesso concedido.")
            time.sleep(1) # Pequena pausa para o usuário ver a mensagem
            st.rerun()
        else:
            st.error("❌ Usuário ou senha inválidos. Tente novamente.")

def signup_screen():
    st.header("📝 Cadastro") # Usando st.header com emoji
    with st.form("form_signup", clear_on_submit=False):
        # Layout em colunas para os campos de entrada, melhora a organização
        col1, col2 = st.columns(2)
        with col1:
            nome = st.text_input("Nome completo", placeholder="João da Silva")
            username = st.text_input("Usuário (login, min. 3 caracteres)", placeholder="joao.silva")
            cpf = st.text_input("CPF (somente números)", placeholder="12345678900")
        with col2:
            senha = st.text_input("Senha (min. 6 caracteres)", type="password", placeholder="******")
            senha2 = st.text_input("Confirmar senha", type="password", placeholder="******")
            local = st.text_input("Local da fazenda/espaço de cultivo", placeholder="Fazenda Esperança, MG")
            tamanho_plantacao = st.number_input("Tamanho da plantação (ha)", min_value=0.01, step=0.01, format="%.2f")
            
        st.markdown("---")
        submit = st.form_submit_button("✅ Cadastrar", type="primary")

    if submit:
        # Valida campos obrigatórios
        if not nome.strip() or not username.strip() or not senha or not cpf.strip() or not local.strip():
            st.error("🛑 **Preencha todos os campos obrigatórios.**")
            return
        # Nome completo
        if len(nome.strip().split()) < 2:
            st.warning("⚠️ Digite seu nome completo (pelo menos 2 palavras).")
            return
        # Usuário e senha
        if len(username.strip()) < 3 or len(senha) < 6:
            st.warning("⚠️ Usuário deve ter no mínimo **3 caracteres** e senha no mínimo **6 caracteres**.")
            return
        # Senhas conferem?
        if senha != senha2:
            st.error("❌ As senhas não coincidem.")
            return
        # CPF válido?
        if not validar_cpf(cpf):
            st.error("❌ CPF inválido. Digite um CPF válido.")
            return
        # Tamanho da plantação positivo
        if tamanho_plantacao <= 0:
            st.error("❌ Informe um tamanho de plantação válido (>0).")
            return
        # Tentar adicionar usuário
        if add_user(username, nome, senha, cpf, tamanho_plantacao, local):
            st.success("🥳 **Usuário cadastrado com sucesso!** Agora faça login na aba ao lado.")
        else:
            st.error("🛑 Usuário já existe. Tente outro login.")

# ============================================================
# APLICAÇÃO PRINCIPAL (Seu detector) - Melhorias Visuais
# ============================================================
def main_app():
    # ----------------- SIDEBAR - Dados do Usuário Aprimorados -----------------
    st.sidebar.markdown(f"## 🌾 **{st.session_state.get('nome', 'Usuário')}**")
    st.sidebar.markdown(f"**Fazenda:** {st.session_state.get('local', 'Não Informado')}")
    tamanho_display = st.session_state.get('tamanho_plantacao', 0.0)
    # Formatação mais clara para área
    st.sidebar.markdown(f"**Área:** **{tamanho_display:,.2f} ha**".replace(",", "X").replace(".", ",").replace("X", ".")) 
    st.sidebar.markdown("---")
    
    # Botão Sair - mais visual
    if st.sidebar.button("🚪 Sair da Aplicação", type="secondary"):
        st.session_state.clear()
        st.info("Desconectando...")
        time.sleep(0.5)
        st.rerun()

    st.title("🌽 Detector de Sementes de Milho - Análise Pós-Colheita")
    st.markdown("Utilize a tecnologia de Visão Computacional para **avaliar a qualidade** da sua colheita de milho.")
    st.markdown("---")

    # ----------------- MODELOS DISPONÍVEIS -----------------
    camera_options = {
        "Câmera Casual (RGB)": "models/best.pt",
        "Câmera RGN": "rgn.pt",
        "Câmera RE": "re.pt",
        "Câmera NIR": "nir.pt",
        "RGB": "rgb.pt"
    }

    # ----------------- SIDEBAR - Opções de Detecção -----------------
    st.sidebar.header("⚙️ Opções de Detecção")
    selected_camera = st.sidebar.selectbox(
        "📸Qual câmera foi utilizada?",
        list(camera_options.keys()),
        help="Selecione o tipo de câmera para carregar o modelo de detecção correspondente.📸",
        key="camera_select" # Adicionei key para evitar warning
    )

    @st.cache_resource(show_spinner="Carregando Modelo...")
    def load_model(model_path):
        # Adiciona classe customizada para deserialização segura do PyTorch 2.6+
        try:
            torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv])
        except Exception:
            pass
        try:
            # st.toast(f"Carregando {model_path}...")
            model = YOLO(model_path)
            return model
        except Exception as e:
            st.error(f"❌ Erro ao carregar o modelo '{model_path}': {e}")
            return None

    model_path = camera_options[selected_camera]
    model = load_model(model_path)

    confidence_threshold = st.sidebar.slider(
        "Limiar de Confiança:",
        min_value=0.01,
        max_value=1.0,
        value=0.25,
        step=0.01,
        format="%.2f",
        help="Aumente para reduzir falsos positivos (detectar menos, mas com mais certeza)."
    )

    # ----------------- ESTADO DA SESSÃO (Mantido) -----------------
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

    # ----------------- FUNÇÃO DE DETECÇÃO (Mantida) -----------------
    def predict_and_display(image, model, confidence, is_camera=False):
        seed_count = 0
        inteiras_count = 0
        predadas_count = 0
        quebradas_count = 0
        im_array = None

        if model is not None:
            try:
                # Usando .copy() para garantir que a imagem não seja modificada in-place por outra thread/função
                results = model(image.copy(), conf=confidence, verbose=False) 
            except Exception as e:
                # st.warning(f"Erro ao inferir com o modelo: {e}")
                results = None

            if results and len(results) > 0 and getattr(results[0], 'boxes', None):
                seed_count = len(results[0].boxes)
                class_names = model.names if hasattr(model, 'names') else {}

                for box in results[0].boxes:
                    try:
                        # Convertendo box.cls para garantir o tipo
                        class_id = int(box.cls[0].item()) # Usando item() para obter o valor numérico puro
                        class_name = class_names.get(class_id, str(class_id))
                    except Exception:
                        class_name = str(getattr(box, 'cls', ''))

                    # Assegurando que os nomes das classes estejam corretos
                    if class_name.lower() == "inteira":
                        inteiras_count += 1
                    elif class_name.lower() == "predada" or class_name.lower() == "pedrada": 
                        predadas_count += 1
                    elif class_name.lower() == "quebrada":
                        quebradas_count += 1

                try:
                    im_array = results[0].plot()
                    # A plotagem do YOLO retorna BGR, Streamlit espera RGB
                    im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB) 
                except Exception as e:
                    # st.warning(f"Erro ao gerar imagem anotada: {e}")
                    im_array = None

        if im_array is None:
            # Se a plotagem falhar, retorna a imagem original em RGB
            try:
                im_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception:
                #st.error("Erro ao processar imagem.")
                # Se não conseguir converter nem mesmo a original
                return None, (0, 0, 0, 0) 

        return im_array, (seed_count, inteiras_count, predadas_count, quebradas_count)

    # --- Gera recomendações (API instructions) --- (Aprimorada com mais formatação)
    def generate_api_instructions(inteiras_count, predadas_count, quebradas_count):
        total_detected = inteiras_count + predadas_count + quebradas_count
        instructions = []

        if total_detected == 0:
            instructions.append("⚠️ Não foram detectadas sementes na imagem. **Verifique a imagem e o Limiar de Confiança.**")
        else:
            perc_inteiras = (inteiras_count / total_detected) * 100 if total_detected else 0
            perc_predadas = (predadas_count / total_detected) * 100 if total_detected else 0
            perc_quebradas = (quebradas_count / total_detected) * 100 if total_detected else 0

            instructions.append("### 📊 Resumo da Análise")
            instructions.append(f"- **Inteiras:** {inteiras_count} ({perc_inteiras:.1f}%)")
            instructions.append(f"- **Predadas:** {predadas_count} ({perc_predadas:.1f}%)")
            instructions.append(f"- **Quebradas:** {quebradas_count} ({perc_quebradas:.1f}%)")
            instructions.append("---")

            if predadas_count > 0 and perc_predadas >= 15: # Reduzi o limiar para 15% para ser mais sensível
                instructions += [
                    f"🐛 **Alto Dano por Pragas ({perc_predadas:.1f}%)**",
                    "Isso pode indicar um **alto índice de pragas** (insetos, roedores) no seu plantio, ou sementes de baixa qualidade.",
                    "#### Ações Recomendadas:",
                    "* Considere a aplicação de **inseticidas específicos** (consulte um agrônomo).",
                    "* Implemente **monitoramento regular** das pragas, com armadilhas ou inspeções visuais.",
                    "* Avalie a origem e o tratamento das sementes."
                ]

            if quebradas_count > 0 and perc_quebradas >= 10: # Reduzi o limiar para 10%
                if instructions and instructions[-1] != "---": instructions.append("---")
                instructions += [
                    f"🔨 **Alto Dano Mecânico ({perc_quebradas:.1f}%)**",
                    "Isso sugere **dano mecânico** na colheita ou no manuseio pós-colheita.",
                    "#### Ações Recomendadas:",
                    "* **Revisar e ajustar a colheitadeira** (velocidade do cilindro, abertura do côncavo).",
                    "* **Reduzir a velocidade de colheita** para diminuir o impacto nas sementes.",
                    "* Verificar se o transporte e o ensacamento não estão causando danos adicionais."
                ]

            if not instructions or (inteiras_count > 0 and perc_inteiras >= 80):
                if instructions and instructions[-1] != "---": instructions.append("---")
                instructions += [
                    f"⭐ **Excelente Qualidade ({perc_inteiras:.1f}%)**",
                    "A alta proporção de sementes inteiras indica **excelente manejo** de pragas e **colheita otimizada**. Mantenha o padrão de qualidade!"
                ]

        # Envolve a lista de instruções em um box customizado
        return f'<div class="recommendation-box">{"<br>".join(instructions)}</div>'

    # ----------------- INTERFACE COM ABAS -----------------
    if version.parse(st.__version__) >= version.parse("1.18.0"):
        tab1, tab2, tab3 = st.tabs(["📤 Upload e Análise", "📸 Câmera Ao Vivo", "📈 Estatísticas e Histórico"])

        # ----------------- ABA 1 - UPLOAD -----------------
        with tab1:
            st.header("Upload de Imagens para Análise")
            
            # Novo layout para histórico e uploader
            col_upload_left, col_upload_right = st.columns([3, 1])
            
            with col_upload_right:
                show_saved_history = st.checkbox("Mostrar histórico salvo?", value=False)
                
            uploaded_files = col_upload_left.file_uploader(
                "Selecione uma ou mais imagens (.jpg, .jpeg, .png)",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key=f"uploader_{st.session_state.uploader_key}"
            )
            
            # Lógica de processamento mantida
            uploaded_files_list = list(uploaded_files) if uploaded_files else []
            current_names = [f.name for f in uploaded_files_list]
            
            # Recarrega se novos arquivos forem detectados
            if current_names != (st.session_state.last_uploaded_file_names or []):
                st.session_state.processed_images_display = []
                st.session_state.uploaded_files_processed = False
                st.session_state.last_uploaded_file_names = current_names

            if uploaded_files_list and not st.session_state.uploaded_files_processed:
                
                with st.spinner("Processando imagens..."): # Adiciona um spinner de progresso
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, uploaded_file in enumerate(uploaded_files_list):
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        image = cv2.imdecode(file_bytes, 1) # Imagem em BGR

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
                            "Imagem_Original": image # Mantemos a original para reprocessamento
                        }

                        st.session_state.processed_images_history.append(new_record)
                        st.session_state.processed_images_display.append(new_record)

                        progress_bar.progress((i + 1) / len(uploaded_files_list))
                        status_text.text(f"Processando imagem {i+1} de {len(uploaded_files_list)}: {uploaded_file.name}")

                    st.session_state.uploaded_files_processed = True
                    progress_bar.empty()
                    status_text.empty()
                    st.toast("✅ Todas as imagens processadas com sucesso!", icon='🎉')


            if st.session_state.processed_images_display:
                st.subheader("🖼️ Resultados do Upload Recente")
                
                # Exibição dos resultados em cards/expanders
                for i, record in enumerate(st.session_state.processed_images_display):
                    
                    # Usando um Expander para organizar melhor o conteúdo de cada imagem
                    header_text = (f"**{record['Arquivo']}** | Total: **{record['Total']}** "
                                   f"({record['Inteiras']} Inteiras, {record['Predadas']} Predadas, {record['Quebradas']} Quebradas)")
                    
                    with st.expander(header_text, expanded=True):
                        
                        col_img, col_stats = st.columns([2, 1])
                        
                        with col_img:
                            if record["Imagem_Processada"] is not None:
                                st.image(
                                    record["Imagem_Processada"],
                                    caption=f"Análise: Modelo {record['Modelo']} @ Limiar {record['Limiar']}",
                                    use_container_width=True
                                )
                            else:
                                st.warning("Não foi possível renderizar a imagem processada.")
                                
                        with col_stats:
                            # Tabela de Resultados mais clara
                            st.markdown("#### Detalhes da Contagem")
                            st.table(pd.DataFrame({
                                'Tipo': ['Inteiras', 'Predadas', 'Quebradas'],
                                'Contagem': [record['Inteiras'], record['Predadas'], record['Quebradas']]
                            }).set_index('Tipo'))
                            
                            st.markdown("---")
                            
                            # Botão de reanálise dentro da coluna de estatísticas
                            if st.button("🔄 Refazer Análise com Novo Limiar", key=f"refazer_{i}", use_container_width=True):
                                with st.spinner(f"Re-analisando {record['Arquivo']}..."):
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
                                    st.toast("✅ Análise refeita!", icon='👍')
                                    st.rerun()
                                    
                        st.markdown("---")
                        st.markdown("### 💡 Recomendações da Análise:")
                        instructions = generate_api_instructions(
                            record['Inteiras'], record['Predadas'], record['Quebradas']
                        )
                        st.markdown(instructions, unsafe_allow_html=True) # Usando unsafe_allow_html para o box customizado
                        st.markdown("---")


                st.markdown("---")
                # Botão de limpar o display
                if st.button("🗑️ Limpar Imagens Exibidas (Mantém Histórico)", key="limpar_upload"):
                    st.session_state.processed_images_display.clear()
                    st.session_state.uploaded_files_processed = False
                    st.session_state.last_uploaded_file_names = []
                    st.session_state.uploader_key += 1 # Aumenta a key para forçar o uploader a resetar
                    st.toast("Imagens do display limpas.")
                    st.rerun()

            else: # Se não há arquivos processados recentemente
                if show_saved_history and st.session_state.processed_images_history:
                    st.subheader("💾 Histórico de Análises Salvas")
                    # Exibir apenas os dados sumarizados, sem as imagens, para não sobrecarregar
                    df_history = pd.DataFrame(st.session_state.processed_images_history).drop(
                        columns=['Imagem_Processada', 'Imagem_Original'], errors='ignore')
                    st.dataframe(df_history, use_container_width=True)
                else:
                    st.info("⬆️ Faça o upload de uma ou mais imagens para iniciar a análise.")

        # ----------------- ABA 2 - CÂMERA -----------------
        with tab2:
            st.header("📸 Captura de Fotos - Câmera Ao Vivo")
            st.warning("Para um bom resultado, utilize a câmera em ambiente com boa iluminação e foco fixo.")
            
            camera_input = st.camera_input("Tire uma foto para detectar as sementes")

            if camera_input is not None:
                with st.spinner("Analisando foto da câmera..."):
                    file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # Imagem em BGR
                    
                    annotated_frame, (seed_count, inteiras_count, predadas_count, quebradas_count) = predict_and_display(
                        frame, model, confidence_threshold, is_camera=True
                    )
                    
                st.image(annotated_frame, caption="Resultado da Detecção", channels="RGB", use_container_width=True)

                st.markdown("---")
                st.subheader("💡 Recomendações da Análise Instantânea:")
                instructions = generate_api_instructions(inteiras_count, predadas_count, quebradas_count)
                st.markdown(instructions, unsafe_allow_html=True)
                st.markdown("---")
                st.toast("✅ Foto analisada!", icon='📷')

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
                st.info("Clique em 'Tirar foto' acima para começar a análise com a sua câmera.")

        # ----------------- ABA 3 - ESTATÍSTICAS -----------------
        with tab3:
            st.header("📈 Estatísticas Agregadas")
            st.markdown("Visualize o desempenho de sua colheita e tendências.")

            st.subheader("Filtro por Período de Análise")
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input("Data de Início", value=pd.to_datetime("today").date() - pd.Timedelta(days=7)) # Exibe 7 dias por padrão
            with col_end:
                end_date = st.date_input("Data de Fim", value=pd.to_datetime("today").date())

            st.markdown("---")
            st.subheader("Resumo dos Danos no Período")
            
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
            
            # Combina e filtra os dados
            df_combined = pd.concat([df_upload, df_camera], ignore_index=True)
            
            if not df_combined.empty:
                filtered_df = df_combined[
                    (df_combined['Data/Hora'].dt.date >= start_date) &
                    (df_combined['Data/Hora'].dt.date <= end_date)
                ]
            else:
                filtered_df = pd.DataFrame()


            total_inteiras = filtered_df['Inteiras'].sum() if not filtered_df.empty else 0
            total_predadas = filtered_df['Predadas'].sum() if not filtered_df.empty else 0
            total_quebradas = filtered_df['Quebradas'].sum() if not filtered_df.empty else 0
            total_global = total_inteiras + total_predadas + total_quebradas
            
            # Display das métricas com cores e ícones
            col_inteiras, col_predadas, col_quebradas, col_total = st.columns(4)
            with col_inteiras:
                st.metric(label="💚 Sementes Inteiras", value=f"{int(total_inteiras):,}".replace(",", "."))
            with col_predadas:
                st.metric(label="🐛 Sementes Predadas", value=f"{int(total_predadas):,}".replace(",", "."), 
                          delta=f"{((total_predadas/total_global)*100):.1f}% do Total" if total_global else None, delta_color="inverse")
            with col_quebradas:
                st.metric(label="💔 Sementes Quebradas", value=f"{int(total_quebradas):,}".replace(",", "."),
                          delta=f"{((total_quebradas/total_global)*100):.1f}% do Total" if total_global else None, delta_color="inverse")
            with col_total:
                 st.metric(label="∑ Total de Sementes", value=f"{int(total_global):,}".replace(",", "."))

            st.markdown("---")
            st.subheader("Recomendações Agregadas para o Período")
            stats_instructions = generate_api_instructions(total_inteiras, total_predadas, total_quebradas)
            st.markdown(stats_instructions, unsafe_allow_html=True)
            st.markdown("---")
            
            # Gráfico de barras simples
        if not filtered_df.empty and total_global > 0:
            st.subheader("Distribuição dos Tipos de Sementes")
            
            data_chart = pd.DataFrame({
                'Tipo': ['Inteiras', 'Predadas', 'Quebradas'],
                'Contagem': [total_inteiras, total_predadas, total_quebradas]
            })
            
            # 1. Defina o esquema de cores usando alt.Scale
            color_scale = alt.Scale(
                domain=['Inteiras', 'Predadas', 'Quebradas'],
                range=['#3cb371', '#ff4b4b', '#ffaa00']
            )
        
            # 2. Crie o gráfico Altair
            chart = alt.Chart(data_chart).mark_bar().encode(
                # Eixo X usa a coluna 'Tipo'
                x=alt.X('Tipo:N', axis=alt.Axis(title='Tipo de Semente')),
                # Eixo Y usa a coluna 'Contagem'
                y=alt.Y('Contagem:Q', axis=alt.Axis(title='Número de Sementes')),
                # Cor é mapeada pela coluna 'Tipo', usando a escala definida acima
                color=alt.Color('Tipo:N', scale=color_scale, legend=None),
                # Adiciona tooltip para exibir os dados ao passar o mouse
                tooltip=['Tipo', 'Contagem']
            ).properties(
                # Define o título e a largura do gráfico
                title="Contagem de Sementes por Tipo"
            ).interactive() # Permite zoom e pan
        
            # 3. Renderize o gráfico no Streamlit
            st.altair_chart(chart, use_container_width=True)
        
    
                
                
            st.markdown("### 📜 Detalhes do Histórico de Análises")
            if not filtered_df.empty:
                st.dataframe(filtered_df.drop(columns=['Imagem_Processada', 'Imagem_Original'], errors='ignore'), use_container_width=True)
                
                col_clear_upload, col_clear_camera = st.columns(2)
                with col_clear_upload:
                    if st.button("🗑️ Limpar Histórico de Uploads", key="limpar_hist_upload", type="secondary", use_container_width=True):
                        st.session_state.processed_images_history.clear()
                        st.toast("Histórico de uploads limpo!")
                        st.rerun()
                with col_clear_camera:
                    if st.button("🗑️ Limpar Histórico da Câmera", key="limpar_hist_camera", type="secondary", use_container_width=True):
                        st.session_state.camera_history.clear()
                        st.toast("Histórico da câmera limpo!")
                        st.rerun()
            else:
                st.info("Nenhuma análise realizada no período selecionado.")

    else:
        st.error("🚨 **Atualize o Streamlit!** Versão `>= 1.18.0` é necessária para usar as abas e ter a melhor experiência.")

# ============================================================
# EXECUÇÃO (Mantida)
# ============================================================
init_db()

if "logado" not in st.session_state:
    st.session_state["logado"] = False

if not st.session_state["logado"]:
    # Usando st.container para dar mais margem aos formulários de login/cadastro
    with st.container(border=True):
        tab_login, tab_signup = st.tabs(["Login", "Cadastro"])
        with tab_login:
            login_screen()
        with tab_signup:
            signup_screen()
else:
    main_app() 
