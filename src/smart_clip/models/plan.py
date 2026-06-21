"""Data models for edit plans and clip candidates."""

from __future__ import annotations

from pydantic import BaseModel


class ClipScores(BaseModel):
    """Scoring breakdown for a clip candidate."""

    information_density: float = 0.0
    emotional_tension: float = 0.0
    completeness: float = 0.0
    virality: float = 0.0
    rhythm_fit: float = 0.0


class ClipCandidate(BaseModel):
    """A candidate highlight clip identified by the LLM."""

    segment_index: int = 0
    start: float  # seconds
    end: float  # seconds
    title: str = ""
    reason: str = ""
    scores: ClipScores = ClipScores()
    weighted_score: float = 0.0
    suggested_hook: str = ""
    quote_text: str | None = None
    hook_type: str | None = None
    clip_type: str | None = None
    energy_level: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class EditPlan(BaseModel):
    """A structured editing plan ready for execution."""

    clips: list[ClipCandidate] = []
    total_selected_duration: float = 0.0
    platform: str = "original"
    with_subtitles: bool = True
    with_bgm: bool = False
    content_type: str = ""
    tone: str = ""
    summary: str = ""


class ExecuteConfig(BaseModel):
    """Configuration for clip execution."""

    video_path: str
    output_dir: str = "./smart-clip-output"
    format: str = "mp4"
    quality: str = "high"
    codec: str = "libx264"
