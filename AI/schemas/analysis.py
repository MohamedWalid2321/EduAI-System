"""Pydantic models for the proctoring analysis endpoint."""

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ModuleResult(BaseModel):
    """Result from a single AI proctoring module."""

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(..., description="Module identifier (1=Gaze, 2=YOLO, 3=Face)")
    timestamp: str = Field(..., description="ISO-8601 timestamp of the analysis")
    flag: bool = Field(..., description="Whether suspicious activity was detected")
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0–1)",
    )
    evidence: str = Field(..., description="Human-readable evidence description")


class AnalysisResponse(BaseModel):
    """Aggregated response from all proctoring modules."""

    gaze_detection: ModuleResult = Field(
        ..., description="Eye-gaze analysis result"
    )
    object_detection: ModuleResult = Field(
        ..., description="Prohibited-object detection result"
    )
    face_recognition: ModuleResult = Field(
        ..., description="Face recognition/verification result"
    )
