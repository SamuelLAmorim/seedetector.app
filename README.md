🌱 Aplicativo de Seleção de Sementes no Agronegócio

Um aplicativo/site desenvolvido para auxiliar na seleção e análise de sementes agrícolas, começando pelas sementes de milho, utilizando a tecnologia YOLO (You Only Look Once) para detecção de objetos com alta precisão.

O sistema é compatível com alguns tipos de câmeras, possibilitando diferentes cenários de análise:

📱 Casual (Celular, GoPro)
🌈 RGN(Em desenvolvimento)
🎨 RGB(Em desenvolvimento)
🌌 NIR (Infravermelho Próximo)(Em desenvolvimento)
🔴 RE (Red Edge)(Em desenvolvimento)

🚀 Funcionalidades

Detecção baseada em YOLOv8 para classificação precisa de sementes.

Menu de seleção de câmera na tela inicial.

Análise em tempo real de vídeo para monitoramento dinâmico.

Suporte a diferentes tipos de câmeras utilizadas no agronegócio.

Compatibilidade com desktop e dispositivos móveis.

🛠️ Tecnologias Utilizadas

Python (Backend & Processamento)

YOLOv8 (Detecção de objetos)

OpenCV (Processamento de imagens e vídeos)

Streamlit (Interface web)

Android Studio (opcional – versão mobile)

📸 Tipos de Câmeras
Tipo de Câmera	Uso Principal	Vantagem
Casual	Celular ou GoPro	Detecção rápida e acessível
RGN	Imagem agrícola especializada	Análise de saúde da planta
RGB	Câmera de cor padrão	Detecção de uso geral
NIR	Infravermelho próximo	Monitoramento de estresse e hidratação
RE	Red Edge	Monitoramento da saúde da vegetação
🎯 Objetivo

Simplificar e agilizar a seleção de sementes no agronegócio, melhorando a precisão da análise e permitindo que agricultores, pesquisadores e agrônomos tomem decisões mais assertivas com base em dados em tempo real.

🖥️ Fluxo de Uso (Exemplo)

Selecionar o tipo de câmera na página inicial.

Fazer upload de uma imagem ou habilitar o vídeo ao vivo.

O modelo YOLO detecta e classifica as sementes.

Exibir resultados detalhados com níveis de confiança.

📦 Passo a Passo de Instalação e Uso
1. Baixar o repositório

2. Criar ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Instalar dependências(IMPORTANTE)
pip install -r requirements.txt

4. Executar o sistema no navegador (via Streamlit)
streamlit run app.py

5. Usar o aplicativo

Escolha o tipo de câmera na tela inicial.

Faça upload de imagens ou habilite a câmera para análise em tempo real.

Receba os resultados da classificação com base no modelo YOLO.

🌱 Agribusiness Seed Selection App

An application/website designed to assist in the selection and analysis of agricultural seeds, starting with corn seeds, using YOLO (You Only Look Once) object detection technology for high accuracy and precision.

The system is compatible with several types of camera, enable different analysis scenarios:

📱 Casual (Cell phone, GoPro)
🌈 RGN(developing)
🎨 RGB(developing)
🌌 NIR (Near-Infrared)(developing)
🔴 RE (Red Edge)(developing)

🚀 Features

YOLOv8-based detection for precise seed classification.

Camera selection menu on the initial page.

Real-time video analysis for dynamic monitoring.

Support for different camera types used in agribusiness.

Compatibility with desktop and mobile devices.

🛠️ Tech Stack

Python (Backend & Processing)

YOLOv8 (Object detection)

OpenCV (Image & video processing)

Streamlit (Web interface)

Android Studio (optional – mobile version)

📸 Camera Options
Camera Type	Use Case	Advantage
Casual	Cell phone or GoPro	Quick and accessible detection
RGN	Specialized agricultural imaging	Enhanced plant health analysis
RGB	Standard color camera	General-purpose detection
NIR	Near-Infrared	Plant stress and hydration monitoring
RE	Red Edge	Vegetation health monitoring
🎯 Objective

To simplify and speed up seed selection in agribusiness, improving accuracy of analysis and enabling farmers, researchers, and agronomists to make better decisions based on real-time data.

🖥️ Example Flow

Select the camera type on the home page.

Upload an image or enable live video feed.

YOLO model detects and classifies the seeds.

View detailed results with detection confidence levels.

📦 Installation & Usage Guide
1. Clone the repository
git clone https://github.com/your-repo/agribusiness-seed-selection.git
cd agribusiness-seed-selection

2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Run the system in the browser (via Streamlit)
streamlit run app.py

5. Use the application

Select the camera type on the start screen.

Upload images or enable camera for real-time analysis.

Get classification results based on YOLO detection.
