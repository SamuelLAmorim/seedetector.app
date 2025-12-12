# 🌱 Agribusiness Seed Selection App

Um aplicativo/site desenvolvido para auxiliar na seleção e análise de sementes agrícolas, começando pelas sementes de milho, utilizando tecnologia **YOLO (You Only Look Once)** para detecção de objetos com alta precisão.

O sistema é compatível com vários tipos de câmeras, permitindo diferentes cenários de análise no agronegócio.

---

## 📸 Tipos de Câmeras Compatíveis

| Tipo de Câmera | Uso Principal | Vantagem |
|----------------|---------------|----------|
| 📱 **Casual** | Celular, GoPro | Detecção rápida e acessível |
| 🌈 **RGN** | Imagem agrícola especializada | Análise avançada de saúde da planta |
| 🎨 **RGB** | Câmera de cor padrão | Detecção de uso geral |
| 🌌 **NIR** | Infravermelho Próximo | Monitoramento de estresse e hidratação |
| 🔴 **RE** | Red Edge | Análise de saúde da vegetação |

### ✔️ Status de Desenvolvimento

- **Casual:** ✔️ Implementado  
- **RGN, RGB, NIR, RE:** 🔧 Em Desenvolvimento  

---

## 🚀 Funcionalidades

- ✔️ Detecção baseada em **YOLOv8**  
- ✔️ Interface amigável em **Streamlit**  
- ✔️ Menu de seleção do tipo de câmera  
- ✔️ Análise em tempo real via webcam  
- ✔️ Suporte a imagens estáticas e vídeo  
- ✔️ Compatível com desktop e mobile  
- 🔧 Versão Android — Em desenvolvimento  

---

## 🛠️ Tecnologias Utilizadas

- **Python** — Backend e processamento  
- **YOLOv8 (Ultralytics)** — Detecção de objetos  
- **OpenCV** — Processamento de imagens e vídeo  
- **Streamlit** — Interface web  
- **Android Studio** — Versão mobile (opcional)  

---

## 🎯 Objetivo

O objetivo deste sistema é:

- Melhorar a análise e seleção de sementes  
- Entregar precisão, rapidez e acessibilidade  
- Fornecer dados em tempo real  
- Auxiliar agricultores, agrônomos e pesquisadores  

---

## 🖥️ Fluxo de Uso

1. Selecionar o tipo de câmera  
2. Escolher:
   - Upload de imagem  
   - Webcam ao vivo  
3. O modelo YOLO analisa a imagem  
4. O sistema exibe:
   - Classes  
   - Confiança  
   - Caixas delimitadoras  
   - Resultados em tempo real  

---

## 📦 Passo a Passo de Instalação

### 🔽 1. Baixar o repositório
```bash 
git clone https://github.com/your-repo/agribusiness-seed-selection.git
cd agribusiness-seed-selection
```

### 🧪 2. Criar ambiente virtual (opcional, recomendado)
```bash
# 🪟 Windows
python -m venv venv
venv\Scripts\activate

# 🐧 Linux
python3 -m venv venv
source venv/bin/activate

# 🍎 Mac
python3 -m venv venv
source venv/bin/activate
```

### 📦 3. Instalar dependências
```bash
pip install -r requirements.txt
```

### ▶️ 4. Executar o sistema
```bash
python -m streamlit run app.py
```

### 🌱 Uso da Aplicação

-Escolha o tipo de câmera

-Faça upload da imagem ou ative a webcam

-Aguarde a análise YOLO

-Verifique classes, caixas delimitadoras e níveis de confiança

### 🤝 Contribuições

Contribuições são bem-vindas!
Sinta-se à vontade para abrir issues, enviar pull requests ou sugerir melhorias.

### Feito por: 
## Samuel Leal Amorim
