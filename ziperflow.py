from __future__ import annotations

import io
import json
import os
import sys
import wave
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pyperclip
import sounddevice as sd


def record_audio() -> bytes:
    sample_rate = int(sd.query_devices(kind="input")["default_samplerate"])
    audio = bytearray()
    warnings: list[str] = []

    def capture(data: np.ndarray, _frames: int, _time: object, status: object) -> None:
        audio.extend(data.tobytes())
        if status:
            warnings.append(str(status))

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            callback=capture,
        ):
            input("Recording... press Enter to stop.\n")
    except sd.PortAudioError as error:
        raise RuntimeError(f"Could not record from the microphone: {error}") from error

    if warnings:
        print(f"Audio warning: {'; '.join(warnings)}", file=sys.stderr)
    if not audio:
        raise RuntimeError("No audio was captured. Check the microphone and try again.")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio)
    return buffer.getvalue()


def transcribe(audio_wav: bytes) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "Transcription dependencies are missing. Run "
            "`python -m pip install -e .`."
        ) from error

    model_name = os.environ.get("ZIPERFLOW_MODEL", "base")
    print(f"Loading speech model '{model_name}'...")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(io.BytesIO(audio_wav), vad_filter=True)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    if not transcript:
        raise RuntimeError("No speech was recognized. Try recording again.")
    return transcript


def rewrite_prompt(transcript: str) -> str:
    model_name = os.environ.get("ZIPERFLOW_PROMPT_MODEL", "qwen3:4b")
    request_data = {
        "model": model_name,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 512},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Turn the transcript into a concise, actionable prompt for an "
                    "AI agent. Remove speech fillers and repetitions, fix grammar, "
                    "and preserve the exact intent, facts, technical details, and "
                    "constraints. For a multi-part or rambling request, organize "
                    "only the information actually stated into short sections or "
                    "bullets. Never infer causes or implementation details, broaden "
                    "scope, strengthen requirements, or add acceptance criteria. "
                    "Do not solve or answer the request. Keep uncertainty when "
                    "the transcript is uncertain. Return a JSON object with exactly "
                    "one string field named 'prompt' containing only the rewritten "
                    "prompt."
                ),
            },
            {"role": "user", "content": f"Rewrite this transcript:\n\n{transcript}"},
        ],
    }
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(request_data, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}: {detail or error.reason}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            "Could not reach local Ollama. Start Ollama and make sure the "
            f"'{model_name}' model is installed (`ollama pull {model_name}`)."
        ) from error
    except TimeoutError as error:
        raise RuntimeError("Ollama took too long to rewrite the prompt.") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Ollama returned an invalid response.") from error

    try:
        prompt = json.loads(result["message"]["content"])["prompt"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Ollama returned an unexpected prompt format.") from error
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Ollama returned an empty prompt.")
    return prompt.strip()


def main() -> int:
    print("ZiperFlow ✦\n")

    try:
        input("Press Enter to start recording.")
        audio = record_audio()
        print("❊ Transcribing and writing your agent prompt...")
        prompt = rewrite_prompt(transcribe(audio))
        print(f"\nYour agent-ready prompt:\n\n{prompt}")

        try:
            pyperclip.copy(prompt)
        except pyperclip.PyperclipException as error:
            print(f"\nCould not copy to clipboard: {error}", file=sys.stderr)
            print("You can still select and copy the prompt above.")
            return 1

        print("\nCopied to clipboard. Switch apps and press Ctrl+V.")
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except EOFError:
        print("\nInput closed before recording completed.", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
