# whisper-tube
A Python tool that downloads YouTube videos and transcribes them using OpenAI's Whisper. Extracts audio and generates accurate transcriptions with timestamps in multiple languages and formats (TXT, JSON, SRT). Perfect for creating subtitles and making content accessible.


```shell
➜  whisper-tube git:(main) uv run main.py  
Enter the YouTube video URL: https://www.youtube.com/watch?v=LPZh9BOjkQs
Choose output format:
1. Text file (.txt)
2. JSON with timestamps (.json)
3. Subtitle file (.srt)
Enter your choice (1-3) [default: 1]: 2
Choose compute device (mps, cuda, cpu) [default: mps]: mps

Language options:
- Leave empty for automatic language detection
- Enter 'en' for English
- Enter other language code (e.g., 'fr', 'es', 'de', etc.)
Select language [default: auto-detect]: en
Downloading audio...
Audio saved as audio.mp4
Transcribing audio...
Device set to use mps
Device set to use mps

--- Transcription Preview ---
 Imagine you happen across a short movie script that describes a scene between a person and their AI assistant. The script has what the person asks the AI, but the AI's response has been torn off. Suppose you also have this powerful magical machine that can take any text and provide a sensible prediction of what word comes next. You could then finish the script by feeding in what you have to the machine, seeing what it would predict to start the AI's answer, and then repeating this over and over...
Transcription saved as transcript_LPZh9BOjkQs.json
```