🌱 Agribusiness Seed Selection App

Um aplicativo/site desenvolvido para auxiliar na seleção e análise de sementes agrícolas, começando pelas sementes de milho, utilizando tecnologia YOLO (You Only Look Once) para detecção de objetos com alta precisão.

O sistema é compatível com vários tipos de câmeras, permitindo diferentes cenários de análise no agronegócio.

📸 Tipos de Câmeras Compatíveis
Tipo de Câmera	Uso Principal	Vantagem
📱 Casual	Celular, GoPro	Detecção rápida e acessível
🌈 RGN	Imagem agrícola especializada	Análise avançada de saúde da planta
🎨 RGB	Câmera de cor padrão	Detecção de uso geral
🌌 NIR	Infravermelho Próximo	Monitoramento de estresse e hidratação
🔴 RE	Red Edge	Análise de saúde da vegetação
📊 Status de Desenvolvimento

Casual: ✔️ Implementado

RGN, RGB, NIR, RE: 🔧 Em Desenvolvimento

🚀 Funcionalidades

✔️ Detecção baseada em YOLOv8

✔️ Interface amigável em Streamlit

✔️ Menu para seleção de tipo de câmera

✔️ Análise em tempo real via webcam

✔️ Suporte a imagens estáticas e vídeo

✔️ Compatível com desktop e dispositivos móveis

🔧 Versão mobile (Android) — em desenvolvimento

🛠️ Tecnologias Utilizadas

Python — Backend e processamento

YOLOv8 (Ultralytics) — Detecção de objetos

OpenCV — Processamento de imagens

Streamlit — Interface Web

Android Studio — Versão mobile (opcional)

🎯 Objetivo

O objetivo é simplificar e agilizar a seleção de sementes no agronegócio, oferecendo:

Maior precisão nas análises

Detecção rápida e acessível

Suporte a múltiplos tipos de câmera

Informações em tempo real para agricultores, agrônomos e pesquisadores

🖥️ Fluxo de Uso (Exemplo)

Selecionar o tipo de câmera

Escolher entre:

Upload de imagem

Uso da câmera ao vivo

O modelo YOLO faz a detecção

O sistema exibe:

Caixa delimitadora

Classe da semente

Porcentagem de confiança

Visualização em tempo real

📦 Passo a Passo de Instalação
🔽 1. Baixar o repositório
git clone https://github.com/your-repo/agribusiness-seed-selection.git
cd agribusiness-seed-selection

🧪 2. Criar ambiente virtual (opcional, recomendado)
Windows
python -m venv venv
venv\Scripts\activate

Linux/Mac
python3 -m venv venv
source venv/bin/activate

📦 3. Instalar dependências
pip install -r requirements.txt

▶️ 4. Executar o sistema (via Streamlit)
streamlit run app.py


Acessar no navegador:
http://localhost:8501

🌱 Uso da Aplicação

Escolha o tipo de câmera

Faça upload de imagem ou ative a webcam

Aguarde a detecção YOLO

Veja os resultados com porcentagem de confiança

🤝 Contribuições

Contribuições são bem-vindas!
Sinta-se à vontade para abrir issues, enviar pull requests ou sugerir melhorias.
