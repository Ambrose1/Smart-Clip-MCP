"""Subtitle extraction module using Whisper."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile

from smart_clip.models.subtitle import Segment, SubtitleResult

logger = logging.getLogger(__name__)


class SubtitleExtractor:
    """Extract timestamped subtitles from video using Whisper."""

    def __init__(self, mode: str = "local", language: str = "zh"):
        self.mode = mode
        self.language = language

    async def extract(self, video_path: str, language: str | None = None) -> SubtitleResult:
        """
        Extract subtitles from video.

        Args:
            video_path: Path to the input video file.
            language: Override language code (default: instance language).

        Returns:
            SubtitleResult with segments, language, and full text.
        """
        lang = language or self.language

        if self.mode == "local":
            return await self._via_local_whisper(video_path, lang)
        else:
            return await self._via_api(video_path, lang)

    async def _via_local_whisper(self, video_path: str, language: str) -> SubtitleResult:
        """Use local Whisper model for transcription."""
        try:
            import whisper
        except ImportError:
            logger.warning("whisper not installed, falling back to API mode")
            return await self._via_api(video_path, language)

        logger.info(f"Loading Whisper model for {video_path}")
        model = whisper.load_model("large-v3")

        result = model.transcribe(
            video_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )

        segments = []
        full_text_parts = []
        for i, seg in enumerate(result["segments"]):
            segments.append(
                Segment(
                    index=i,
                    start=round(seg["start"], 2),
                    end=round(seg["end"], 2),
                    text=seg["text"].strip(),
                    confidence=seg.get("avg_logprob", 0.0),
                )
            )
            full_text_parts.append(seg["text"].strip())

        return SubtitleResult(
            segments=segments,
            language=language,
            full_text=" ".join(full_text_parts),
        )

    async def _via_api(self, video_path: str, language: str) -> SubtitleResult:
        """Use OpenAI Whisper API for transcription."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        # Extract audio first (API has size limits)
        audio_path = await self._extract_audio(video_path)

        logger.info(f"Calling Whisper API for {video_path}")
        with open(audio_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = []
        full_text_parts = []
        for i, seg in enumerate(transcript.segments):
            segments.append(
                Segment(
                    index=i,
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    text=seg.text.strip(),
                )
            )
            full_text_parts.append(seg.text.strip())

        # Clean up temp audio
        if audio_path != video_path and os.path.exists(audio_path):
            os.remove(audio_path)

        return SubtitleResult(
            segments=segments,
            language=language,
            full_text=" ".join(full_text_parts),
        )

    async def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video as WAV for API upload."""
        suffix = os.path.splitext(video_path)[1]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", tmp.name,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return tmp.name
