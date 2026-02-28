# Daisy (Love Assistant) - Intelligent Chat Analysis & Reply System

> **Project Goal**: A desktop application that analyzes chat screenshots using OCR and generates high-EQ, "pursuit-style" replies using Large Language Models (LLMs).
> **Core Philosophy**: "Attraction is about attitude, not pleasing."

---

## 1. Installation & Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (Recommended for faster OCR)

### Step 1: Install Dependencies
The system requires PyTorch with GPU support. The installation script will handle this automatically if you run the setup command, but due to large file sizes, it's recommended to install manually:

1.  **Install PyTorch (GPU Version)**:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```
    *Note: Adjust `cu124` to your CUDA version if needed.*

2.  **Install Other Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Configure Environment
1.  Copy `.env.example` to `.env`.
2.  Add your OpenAI or DeepSeek API Key.
    *   **OpenAI**: Use `https://api.openai.com/v1` (default).
    *   **DeepSeek**: Use `https://api.deepseek.com`.

### Step 3: Run the Application
```bash
streamlit run app.py
```

---

## 2. System Architecture & Flow

### 2.1 Core Workflow (Mermaid)

```mermaid
graph TD
    A[User Uploads Screenshot] --> B{Image Pre-processing}
    B -->|Grayscale/Binarization| C[OCR Engine (EasyOCR)]
    C --> D[Raw Text Data]
    
    subgraph "Structural Analysis Layer"
    D --> E[Identify Text Bubbles]
    E --> F{Role Classification}
    F -->|Right Side/Green Bubble| G[Me - User]
    F -->|Left Side/White Bubble| H[Target - Crush]
    end
    
    subgraph "Cognitive Layer (LLM)"
    H --> I[Extract Last 3-5 Messages]
    I --> J[Inject 'No-Simp' Persona Prompts]
    J --> K[Call LLM API (GPT/Yi/Wenxin)]
    end
    
    K --> L[Generate 3 Reply Options]
    L --> M[User Selection & Feedback]
```

---

## 3. "No Simp" Reply Rules (The Soul of Daisy)

1.  **Rule of Investment Match**: Never write a paragraph in response to a one-word reply. Match the target's energy and length.
2.  **Rule of Emotional Independence**: If the target is venting/complaining, offer *perspective* or *playful teasing*, not just blind agreement or "poor you."
3.  **Rule of Mystery**: Do not answer every question directly. Use humor, deflection, or counter-questions to keep the conversation dynamic.
4.  **Rule of Self-Respect**: If the target is cold or rude, call it out playfully or withdraw attention. Do not apologize unless you actually made a mistake.
5.  **Rule of Value Display**: Subtly imply you have a life outside the chat. Do not be "always available."

---

## 4. Development Status
- [x] Core Project Structure
- [x] OCR Engine Implementation (GPU-enabled)
- [x] Chat Parsing Logic (Coordinate-based Role Distinction)
- [x] LLM Client with "No-Simp" Prompts
- [x] Streamlit UI (Basic)
- [ ] Advanced Image Preprocessing (Noise Removal) - Planned V1.1
- [ ] Multi-image stitching - Planned V1.2
