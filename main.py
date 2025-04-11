import json
import ssl

import certifi
import torch
from pytubefix import YouTube
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

# Set the default HTTPS context to use certifi's certificate bundle
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


def download_audio(youtube_url, filename="audio.mp4"):
    """
    Download the audio stream from a YouTube video and save it as an MP4 file.
    """
    try:
        yt = YouTube(youtube_url)
        # Get the highest quality audio stream available
        stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        if stream is None:
            raise ValueError("No audio stream found for the provided URL.")
        # Download the audio and save it with the specified filename
        stream.download(output_path=".", filename=filename)
        return filename
    except Exception as e:
        raise Exception(f"Error downloading from YouTube: {str(e)}")


def transcribe_audio(audio_file, device=None, language=None):
    """
    Transcribe the given audio file using the Whisper ASR pipeline.

    Args:
        audio_file: Path to the audio file
        device: Compute device ("mps" for Mac, "cuda" for NVIDIA GPU, "cpu" for CPU)
        language: Language code (e.g., "en" for English, None for auto-detection)

    Returns:
        Dictionary containing the transcription and timestamps
    """
    # Auto-select the best available device if none specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Validate the requested device is available
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("Warning: MPS device requested but not available. Falling back to CPU.")
        device = "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA device requested but not available. Falling back to CPU.")
        device = "cpu"

    print(f"Device set to use {device}")

    # Configure pipeline with proper parameters
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device=device,
        model_kwargs={
            "attn_implementation": "flash_attention_2" if is_flash_attn_2_available() else "sdpa",
            "pad_token_id": 50256,
            "eos_token_id": 50257,
        },
    )

    # Set up generation parameters
    generate_kwargs = {}

    # Handle language selection to fix the language detection warning
    if language:
        # In Whisper pipeline, we need to set language in generate_kwargs
        generate_kwargs["language"] = language
        generate_kwargs["task"] = "transcribe"  # Explicitly set to transcribe when language is provided

    # Run the transcription
    outputs = pipe(
        audio_file,
        chunk_length_s=30,
        batch_size=16 if device != "cpu" else 4,  # Lower batch size for CPU
        return_timestamps=True,
        generate_kwargs=generate_kwargs
    )

    return outputs


def save_transcription(transcription, output_format="txt", output_file="transcription"):
    """
    Save the transcription to a file in the specified format.

    Args:
        transcription: The transcription dictionary from the Whisper model
        output_format: The output format ("txt", "json", or "srt")
        output_file: The base name for the output file (without extension)
    """
    if output_format == "txt":
        with open(f"{output_file}.txt", "w", encoding="utf-8") as f:
            f.write(transcription["text"])

    elif output_format == "json":
        with open(f"{output_file}.json", "w", encoding="utf-8") as f:
            json.dump(transcription, f, ensure_ascii=False, indent=2)

    elif output_format == "srt":
        with open(f"{output_file}.srt", "w", encoding="utf-8") as f:
            for i, chunk in enumerate(transcription["chunks"]):
                start_time, end_time = chunk["timestamp"]
                # Convert seconds to SRT format (HH:MM:SS,mmm)
                start_formatted = format_timestamp_srt(start_time)
                end_formatted = format_timestamp_srt(end_time)

                f.write(f"{i + 1}\n")
                f.write(f"{start_formatted} --> {end_formatted}\n")
                f.write(f"{chunk['text']}\n\n")

    print(f"Transcription saved as {output_file}.{output_format}")


def format_timestamp_srt(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"


def main():
    youtube_url = input("Enter the YouTube video URL: ").strip()
    if not youtube_url:
        print("No URL provided. Exiting.")
        return

    # Get video ID for naming the output file
    try:
        video_id = youtube_url.split("v=")[1].split("&")[0]
        output_base = f"transcript_{video_id}"
    except:
        output_base = "transcription"

    # Ask for output format
    print("Choose output format:")
    print("1. Text file (.txt)")
    print("2. JSON with timestamps (.json)")
    print("3. Subtitle file (.srt)")

    format_choice = input("Enter your choice (1-3) [default: 1]: ").strip() or "1"
    format_map = {"1": "txt", "2": "json", "3": "srt"}
    output_format = format_map.get(format_choice, "txt")

    # Ask for device
    device_choice = input("Choose compute device (mps, cuda, cpu) [default: mps]: ").strip() or "mps"

    # Ask for language preference
    print("\nLanguage options:")
    print("- Leave empty for automatic language detection")
    print("- Enter 'en' for English")
    print("- Enter other language code (e.g., 'fr', 'es', 'de', etc.)")
    language = input("Select language [default: auto-detect]: ").strip() or None

    print("Downloading audio...")
    try:
        audio_file = download_audio(youtube_url)
        print(f"Audio saved as {audio_file}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Transcribing audio...")
    try:
        transcript = transcribe_audio(audio_file, device=device_choice, language=language)

        print("\n--- Transcription Preview ---")
        # Print first 500 characters as preview
        preview_text = transcript["text"][:500] + "..." if len(transcript["text"]) > 500 else transcript["text"]
        print(preview_text)

        # Save the transcription
        save_transcription(transcript, output_format=output_format, output_file=output_base)

    except Exception as e:
        print(f"Error transcribing audio: {e}")


if __name__ == "__main__":
    main()