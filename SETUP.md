# SoloBro Flask App Setup

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the application:
```bash
python app.py
```

4. Open browser to: http://localhost:5000

## API Configuration

### OpenAI Setup
- Get API key from: https://platform.openai.com/api-keys
- Add to .env file: `OPENAI_API_KEY=your_key_here`

### Ollama Setup
- Install Ollama: https://ollama.ai/
- Pull Mistral model: `ollama pull mistral`
- Ensure Ollama is running on port 11434

## Features

- **Podcast Search**: Search Apple Podcasts (iTunes API)
- **Content Generation**: Generate custom podcast content using OpenAI or Ollama
- **Tone Control**: Interactive triangle slider for content tone (Analytical/Professor/Debate)
- **Voice Generation**: Text-to-speech using OpenAI TTS or Ollama
- **Multi-language Support**: Generate content in different languages
- **Duration Control**: Specify content length (5-45 minutes)

## API Endpoints

- `POST /search_podcasts` - Search for podcasts
- `POST /contentGenerate` - Generate podcast content
- `POST /voiceGenerate` - Generate audio from text