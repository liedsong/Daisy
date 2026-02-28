# Daisy (恋爱僚机) - 智能聊天分析与回复助手

Daisy 是一个基于 **Streamlit** 和 **LLM (大语言模型)** 构建的智能恋爱助手。它可以分析你与 Crush (心动对象) 的聊天记录截图，通过 OCR 提取文字，并结合心理学知识为你生成高情商的回复建议。

## 🌟 核心功能

*   **📷 聊天截图分析**: 上传聊天记录截图，自动识别双方对话内容。
*   **🧩 长图智能拼接**: 支持一次性上传多张连续截图，系统自动去重并拼接成长图 (V1.1 新增)。
*   **🧠 高情商回复生成**: 基于深度定制的 Prompt，提供“幽默推拉”、“情绪价值”、“高冷神秘”等多种风格的回复。
*   **💬 可编辑对话**: OCR 识别有误？别担心，你可以手动修正每一条消息的内容和角色。
*   **🌍 多语言支持**: 界面支持中英文切换，AI 会根据你的界面语言设定自动调整回复语言。
*   **🔍 深度思考模式**: 集成 DeepSeek-R1 模型，展示 AI 完整的推理思考过程。

## 🛠️ 技术栈

*   **Frontend**: Streamlit
*   **OCR**: PaddleOCR / EasyOCR (支持 GPU 加速)
*   **LLM**: OpenAI API (兼容 DeepSeek, GPT-4)
*   **Image Processing**: OpenCV (去噪、锐化、拼接)

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/Daisy.git
cd Daisy
```

### 2. 安装依赖

建议使用 Python 3.10+ 环境：

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 API Key：

```bash
cp .env.example .env
```

在 `.env` 文件中：

```ini
OPENAI_API_KEY=sk-xxxxxx
OPENAI_BASE_URL=https://api.deepseek.com  # 如果使用 DeepSeek
```

### 4. 运行应用

```bash
streamlit run app.py
```

## 📅 开发进度

- [x] 核心项目结构搭建
- [x] OCR 引擎集成 (支持 GPU)
- [x] 聊天记录解析逻辑 (基于坐标的角色判定)
- [x] LLM 客户端与“反舔狗”提示词设计
- [x] Streamlit 基础界面
- [x] 高级图像预处理 (去噪) - V1.1
- [x] 多图垂直自动拼接 - V1.1
- [ ] 针对不同聊天 App 的 UI 适配 (微信/Telegram/WhatsApp) - 规划中 V1.2

## 🤝 贡献

欢迎提交 Issue 和 PR！让我们一起把 Daisy 变得更聪明。

## 📄 许可证

MIT License
