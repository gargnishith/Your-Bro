from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
import os
import uuid
# from pydub import AudioSegment
import subprocess

app = Flask(__name__)
# Enable CORS for all routes to prevent cross-origin errors
CORS(app)

# API configuration - Set your API keys as environment variables
#OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# OPENAI_API_KEY="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"
# OPENAI_API_KEY="your-api-key-here"  # Alternative: hardcode for testing (not recommended for production)
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# HUGGING_FACE_BARK_KEY = "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"

# Popular podcast platforms and their API endpoints
PODCAST_PLATFORMS = {
    'spotify': {
        'name': 'Spotify',
        'base_url': 'https://open.spotify.com/show/',
        'search_endpoint': 'https://api.spotify.com/v1/search'
    },
    'apple': {
        'name': 'Apple Podcasts',
        'base_url': 'https://podcasts.apple.com/podcast/',
        'search_endpoint': 'https://itunes.apple.com/search'
    },
    'google': {
        'name': 'Google Podcasts',
        'base_url': 'https://podcasts.google.com/feed/',
        'search_endpoint': 'https://www.googleapis.com/customsearch/v1'
    }
}

@app.route('/')
def index():
    """Main page with podcast generator interface"""
    return render_template('index.html')

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    """Serve generated audio files"""
    return send_from_directory('static/audio', filename)

@app.route('/search_podcasts', methods=['POST'])
def search_podcasts():
    """API endpoint to search for podcasts across platforms"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        # Search across different platforms
        results = {}
        
        # Apple Podcasts search (iTunes API - free to use)
        apple_results = search_apple_podcasts(query)
        results['apple'] = apple_results
        
        # Note: Spotify API requires authentication and app registration
        # Google Podcasts API requires API key
        
        return jsonify({
            'success': True,
            'results': results,
            'platforms': PODCAST_PLATFORMS
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def search_apple_podcasts(query: str, limit: int = 10):
    """Search Apple Podcasts using iTunes API"""
    try:
        url = "https://itunes.apple.com/search"
        params = {
            'term': query,
            'media': 'podcast',
            'limit': limit
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        podcasts = []
        
        for item in data.get('results', []):
            podcasts.append({
                'name': item.get('trackName', ''),
                'artist': item.get('artistName', ''),
                'description': item.get('description', ''),
                'artwork': item.get('artworkUrl100', ''),
                'url': item.get('trackViewUrl', ''),
                'genre': item.get('primaryGenreName', '')
            })
        
        return podcasts
        
    except Exception as e:
        print(f"Error searching Apple Podcasts: {e}")
        return []

@app.route('/contentGenerate', methods=['POST'])
def content_generate():
    """API endpoint for content generation using OpenAI or Ollama"""
    try:
        data = request.get_json()
        
        # Extract parameters
        topic = data.get('topic', '')
        duration = data.get('duration', 30)  # minutes
        language = data.get('language', 'English')
        tone_ratios = data.get('tone_ratios', {'analytical': 33, 'professor': 33, 'debate': 34})
        provider = data.get('provider', 'openai')  # 'openai' or 'ollama'
        toneJourney = data.get('journey_type', 'lesson')
        
        wordsCountInContent = 160*duration

        # Create prompt based on tone ratios
        #prompt = create_content_prompt(topic, duration, language, tone_ratios)
        SystemPrompt = generate_content_openai_prompt(topic, duration, language, toneJourney)
        print(SystemPrompt)
        # Generate content based on provider
        if provider == 'openai':
            content = generate_content_openai(SystemPrompt, topic, wordsCountInContent)
        else:
            content = generate_content_ollama(SystemPrompt)
        
        return jsonify({
            'success': True,
            'content': content,
            'provider_used': provider
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/voiceGenerate', methods=['POST'])
def voice_generate():
    """API endpoint for voice generation using OpenAI or Ollama"""
    try:
        data = request.get_json()
        
        text = data.get('text', '')
        voice_style = data.get('voice_style', 'alloy')
        provider = data.get('provider', 'openai')  # 'openai' or 'ollama'
        toneJourney = data.get('journey_type', 'lesson')

        # voiceStyle: str

        if toneJourney == "podcast": voice_style = "onyx"
        elif toneJourney == "lesson": voice_style = "fable"
        else: voice_style = "nova"

        # Generate voice based on provider
        if provider == 'openai':
            audio_url = generate_voice_openai_full(text, voice_style)
        else:
            audio_url = generate_voice_bark(text, voice_style)
            # audio_url = generate_voice_ollama(text, voice_style)
        
        return jsonify({
            'success': True,
            'audio_url': audio_url,
            'provider_used': provider
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_content_openai_prompt(topic: str, duration: int, language: str, toneOfSpeech: str) -> str:
    """
    Generate podcast-style script content using OpenAI Chat API.
    
    Parameters:
    - topic (str): The topic of the episode
    - duration (int): Target duration in minutes
    - language (str): Desired language for the output
    - toneOfSpeech (dict): Dict with `label` key: "Podcast", "Story Reading", "Lesson in Class"
    
    Returns:
    - str: Generated script content
    """
    # Word count calculation
    words_per_minute = 160  # Slightly generous for natural pacing
    target_word_count = duration * words_per_minute

    # Select tone instructions
    tone_label = toneOfSpeech.lower()
    if tone_label == "podcast":
        tone_instruction = (
            "Write in a conversational, analytical, and debateful tone. "
            "Cover different perspectives—highlighting flaws, comebacks, and redemptions. "
            "Engage the listener like a podcast with natural pauses and emotional commentary."
        )
    elif tone_label == "story":
        tone_instruction = (
            "Write like a captivating story by a master author. "
            "Use vivid storytelling, character-driven flow, metaphors, and deep immersion. "
            "Let the script feel like reading a fiction or memoir book—rich in detail, suspense, and pacing."
        )
    elif tone_label == "lesson":
        tone_instruction = (
            "Write in the style of a world-class professor from MIT or Harvard. "
            "Use structured explanations, historical or scientific facts, and clear, engaging delivery. "
            "The goal is to educate and enlighten, not entertain—purely factual but approachable."
        )
    else:
        tone_instruction = (
            "Write clearly, informatively, and engagingly—use appropriate tone for the subject."
        )

    # System Prompt
    system_prompt = f"""
                You are an expert podcast scriptwriter who blends journalism, narration, and factual storytelling. 
                You will write a {duration}-minute script (approx. {target_word_count} words) in {language} 
                on the topic: "{topic}".

                {tone_instruction}

                Do NOT include sound effects, background music, or listener instructions like “Welcome to...” or “Let’s begin.” 
                Write the core narrative content only.

                Maintain a flowing structure with a compelling opening, a coherent middle, and a thought-provoking end.
                Keep the voice natural and suited for text-to-speech narration.

                if require put your name as "your bro" and can introduce yourself as "hey i am your bro" 
                """
    return system_prompt


def create_content_prompt(topic: str, duration: int, language: str, tone_ratios: dict) -> str:
    """Create a detailed prompt based on user preferences and tone ratios"""
    
    # Calculate tone percentages
    total = sum(tone_ratios.values())
    analytical_pct = (tone_ratios.get('analytical', 0) / total) * 100
    professor_pct = (tone_ratios.get('professor', 0) / total) * 100
    debate_pct = (tone_ratios.get('debate', 0) / total) * 100
    
    # prompt = f"""
    # Create a {duration}-minute podcast script about "{topic}" in {language} considering that is should be approx 155 words per minute 
    # if converts the content to speech.
    
    # Tone composition:
    # - Analytical approach: {analytical_pct:.1f}% (data-driven, research-focused)
    # - Professorial style: {professor_pct:.1f}% (educational, explanatory)
    # - Debate format: {debate_pct:.1f}% (contrasting viewpoints, critical analysis)
    
    # Structure the content to fit exactly {duration} minutes of speaking time.
    # Include timestamps, speaking cues, and natural transitions.
    # Make it engaging for solo listeners during gym sessions or travel. dont include the music in your contetn as that can't be converted to
    # into audio so include content only according to described tone and content requested.
    # """




    prompt  = f"""
    Create a podcast script on the topic: "{topic}"

    Tone composition:
    - Analytical approach: {analytical_pct:.1f}% (data-driven, research-focused)
    - Professorial style: {professor_pct:.1f}% (educational, explanatory)
    - Debate format: {debate_pct:.1f}% (contrasting viewpoints, critical analysis)
     
Preferred language: "{language}"  
Duration: "{duration}" minutes

The audience is primarily gym-goers who are listening while working out, so keep the energy engaging, clear, and focused. The speech should be continuous and natural for narration, not like a blog post.

Structure the script like a flowing monologue or narrated audio episode. Feel free to include analogies, surprising facts, expert opinions, and a storytelling arc.

Respond only with the script content—no additional instructions or summaries.
    """
    
    return prompt

def generate_content_openai(systemPrompt: str, topic: str, wordsCountInContent: int) -> str:
    """Generate content using OpenAI API via HTTP requests"""
    try:
        if not OPENAI_API_KEY:
            return "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        
        # OpenAI Chat Completions API endpoint
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        # payload = {
        #     "model": "gpt-4o",
        #     "messages": [
        #         {
        #             "role": "system", 
        #             "content": "You are an expert podcast content creator with experience as a professional storyteller, journalist, anchor, and researcher. Your role is to generate high-quality, engaging, and factually sound podcast scripts tailored for audio narration."
        #             "Your outputs should:"
        #             "- Flow like a compelling story"
        #             "- Include vivid narration, anecdotes, metaphors, and emotional hooks- Be structured with a beginning, middle, and end- Include data or examples where relevant (with simple, conversational explanations)"
        #             "- Match the requested tone (e.g., debate, professor, motivational)"
        #             "- Respect the maximum time limit (based on average 150 words per minute)"
        #             "- Be in the specified language (use translation if needed)"
        #             "Avoid technical jargon unless explicitly asked. Always aim for clarity, impact, and listener engagement."
        #         },
        #         {"role": "user", "content": prompt}
        #     ],
        #     "max_tokens": 2500,
        #     "temperature": 0.7,
        #     "stream": True
        # }



        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": f"Generate the full script for the topic: {topic} in {wordsCountInContent}" 
                 "tone should match a to the prompt given to you for system role. but be sure to be in "
                 "approx {wordsCountInContent} not less than"}
            ],
            "temperature": 0.7,
            "max_tokens": 3000,
            "stream": True
        }

        print(payload)

        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        print(response)

        full_response = ""

        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                line_content = line[len("data: "):]
                if line_content == "[DONE]":
                    break
                try:
                    json_data = json.loads(line_content)
                    delta = json_data["choices"][0]["delta"]
                    content = delta.get("content", "")
                    full_response += content
                except Exception as e:
                    print("Error parsing chunk:", e)

        return full_response
        
    except Exception as e:
        print(e)
        return f"Error generating content with OpenAI: {str(e)}"

def generate_content_ollama(prompt: str) -> str:
    """Generate content using Ollama with Mistral model"""
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        
        payload = {
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', 'No content generated')
        
    except Exception as e:
        return f"Error generating content with Ollama: {str(e)}"

def generate_voice_openai(text: str, voice_style: str) -> str:
    """Generate voice using OpenAI Text-to-Speech API via HTTP requests"""
    try:
        if not OPENAI_API_KEY:
            raise Exception("OpenAI API key not configured")
        
        # OpenAI TTS API endpoint
        url = "https://api.openai.com/v1/audio/speech"

        model = "tts-1-hd" 
        # model = "tts-1"
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "voice": voice_style,
            "input": text[:3500]  # Limit text length for TTS
        }
        
        print(payload)

        response = requests.post(url, headers=headers, json=payload)

        print(response)
        
        # Check if the response indicates an API key issue
        if response.status_code == 403:
            raise Exception("OpenAI API key is invalid or doesn't have access to audio/speech endpoint. Please check your API key permissions.")
        elif response.status_code == 401:
            raise Exception("OpenAI API key is unauthorized. Please verify your API key.")
        elif response.status_code == 429:
            raise Exception("OpenAI API rate limit exceeded. Please try again later.")
        
        response.raise_for_status()
        
        # Save audio file
        filename = f"audio_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = f"static/audio/{filename}"
        
        # Create directory if it doesn't exist
        os.makedirs("static/audio", exist_ok=True)
        
        # Write audio content to file
        with open(audio_path, "wb") as f:
            f.write(response.content)
        
        return f"/{audio_path}"
        
    except Exception as e:
        raise Exception(f"Error generating voice with OpenAI: {str(e)}")

def generate_voice_ollama(text: str, voice_style: str) -> str:
    """Generate voice using Ollama (with appropriate TTS model)"""
    try:
        # Ollama TTS implementation would depend on available models
        # This is a placeholder for TTS integration
        return f"Voice generation with Ollama TTS (style: {voice_style}) - Implementation needed"
        
    except Exception as e:
        return f"Error generating voice with Ollama: {str(e)}"


def generate_voice_bark(text: str, voice_style: str) -> str:
    """Generate voice using bark Text-to-Speech API via HTTP requests"""
    try:
        if not HUGGING_FACE_BARK_KEY:
            raise Exception("hugging face API key not configured")
        
        # OpenAI TTS API endpoint
        url = "https://api-inference.huggingface.co/models/suno/bark"
        
        headers = {
            "Authorization": f"Bearer {HUGGING_FACE_BARK_KEY}"
        }

        response = requests.post(
            url,
            headers=headers,
            json={"inputs": text},
        )

        # Check if response is successful
        if response.status_code == 200:
            with open("bark_output.wav", "wb") as f:
                f.write(response.content)
            print("✅ Audio saved as 'bark_output.wav'")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

        # Check if the response indicates an API key issue
        if response.status_code == 403:
            raise Exception("OpenAI API key is invalid or doesn't have access to audio/speech endpoint. Please check your API key permissions.")
        elif response.status_code == 401:
            raise Exception("OpenAI API key is unauthorized. Please verify your API key.")
        elif response.status_code == 429:
            raise Exception("OpenAI API rate limit exceeded. Please try again later.")
        
        response.raise_for_status()
        
        # Save audio file
        filename = f"audio_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = f"static/audio/{filename}"
        
        # Create directory if it doesn't exist
        os.makedirs("static/audio", exist_ok=True)
        
        # Write audio content to file
        with open(audio_path, "wb") as f:
            f.write(response.content)
        
        return f"/{audio_path}"
        
    except Exception as e:
        raise Exception(f"Error generating voice with OpenAI: {str(e)}")
 

CHUNK_LIMIT = 3500  # Safe chunk size for TTS input


def chunk_text(text, max_chars=CHUNK_LIMIT):
    """Split text into safe-size chunks for OpenAI TTS API."""
    sentences = text.split(". ")
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 < max_chars:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks




def generate_voice_openai_full(text: str, voice_style: str, output_name: str = None) -> str:
    """Generate full-length voice using OpenAI TTS with chunking and ffmpeg merging."""
    try:
        if not OPENAI_API_KEY:
            raise Exception("OpenAI API key not configured")

        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        model = "tts-1-hd"
        chunks = chunk_text(text)
        print(f"🔹 Total chunks: {len(chunks)}")

        temp_files = []

        # Generate audio chunks
        for i, chunk in enumerate(chunks):
            payload = {
                "model": model,
                "voice": voice_style,
                "input": chunk
            }
            print(f"🔹 Processing chunk {i+1}")
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            temp_filename = f"temp_{uuid.uuid4().hex[:8]}.mp3"
            with open(temp_filename, "wb") as f:
                f.write(response.content)

            temp_files.append(temp_filename)

        # Create silence.mp3 if not exists (0.3s silence)
        silence_path = "silence_300ms.mp3"
        if not os.path.exists(silence_path):
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                "-t", "0.3", "-q:a", "9", "-acodec", "libmp3lame", silence_path
            ], check=True)

        # Create a concat list for ffmpeg
        concat_list_path = "concat_list.txt"
        with open(concat_list_path, "w") as f:
            for file in temp_files:
                f.write(f"file '{file}'\n")
                f.write(f"file '{silence_path}'\n")

        # Merge using ffmpeg
        final_name = output_name or f"audio_{uuid.uuid4().hex[:8]}.mp3"
        output_path = f"static/audio/{final_name}"
        os.makedirs("static/audio", exist_ok=True)

        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-c", "copy", output_path
        ], check=True)

        # Cleanup
        for file in temp_files:
            os.remove(file)
        os.remove(concat_list_path)

        print(f"✅ Audio file created: {output_path}")
        return f"/{output_path}"

    except Exception as e:
        raise Exception(f"Error generating full voice: {str(e)}")




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
