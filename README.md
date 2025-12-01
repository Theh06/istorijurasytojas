# Istorijų rašytojas – N-gram + LLM (Ollama)

Projektas yra paprastas **istorijų rašytojas**, kuris sujungia klasikinį n-gram kalbos modelį ir lokaliai veikiančią LLM per **Ollama**.

Pagrindinė schema:

> **Įvestis → N-gram juodraštis → LLM patobulinimas → Galutinis tęsinys**

Aplikacija paleidžiama: `python main.py`

---

## Reikalavimai

- **Python** 3.10+
- **requirements.txt** `pip install -r requirements.txt`
- **Ollama**, įdiegta lokaliai ir paleista
  (turi veikti `http://localhost:11434`)
- Ollama modelis: `ollama pull gemma3:4b`

