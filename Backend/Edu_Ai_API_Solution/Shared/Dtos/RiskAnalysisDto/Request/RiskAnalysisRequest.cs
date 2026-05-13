using System.Text.Json.Serialization;

namespace Shared.Dtos.RiskAnalysisDto.Request
{
    // ── Top-level payload ───────────────────────────────────────────────────────

    public class RiskAnalysisRequest
    {
        [JsonPropertyName("report_metadata")]
        public ReportMetadata ReportMetadata { get; set; } = new();

        [JsonPropertyName("session_summary")]
        public SessionSummary SessionSummary { get; set; } = new();

        [JsonPropertyName("questions")]
        public List<QuestionViolation> Questions { get; set; } = [];
    }

    // ── report_metadata ─────────────────────────────────────────────────────────

    public class ReportMetadata
    {
        [JsonPropertyName("generated_at")]
        public DateTimeOffset GeneratedAt { get; set; }

        [JsonPropertyName("student_id")]
        public string StudentId { get; set; } = string.Empty;

        [JsonPropertyName("Attempt_Id")]
        public string AttemptId { get; set; } = string.Empty;

        [JsonPropertyName("mode")]
        public string Mode { get; set; } = string.Empty;

        [JsonPropertyName("normalisation")]
        public string Normalisation { get; set; } = string.Empty;
    }

    // ── session_summary ─────────────────────────────────────────────────────────

    public class SessionSummary
    {
        [JsonPropertyName("total_questions")]
        public int TotalQuestions { get; set; }

        [JsonPropertyName("questions_violated")]
        public int QuestionsViolated { get; set; }

        [JsonPropertyName("questions_clean")]
        public int QuestionsClean { get; set; }

        [JsonPropertyName("violation_rate")]
        public double ViolationRate { get; set; }

        [JsonPropertyName("weights_used")]
        public WeightsUsed WeightsUsed { get; set; } = new();
    }

    public class WeightsUsed
    {
        [JsonPropertyName("face_absence_mismatch")]
        public double FaceAbsenceMismatch { get; set; } = 1.0;

        [JsonPropertyName("suspicious_movement")]
        public double SuspiciousMovement { get; set; } = 1.0;

        [JsonPropertyName("conversation_noise")]
        public double ConversationNoise { get; set; } = 1.0;

        [JsonPropertyName("forbidden_objects")]
        public double ForbiddenObjects { get; set; } = 1.0;
    }

    // ── questions[] ─────────────────────────────────────────────────────────────

    public class QuestionViolation
    {
        [JsonPropertyName("question_id")]
        public int QuestionId { get; set; }

        [JsonPropertyName("face_detection")]
        public int FaceDetection { get; set; }

        [JsonPropertyName("face_recognition")]
        public int FaceRecognition { get; set; }

        [JsonPropertyName("eye_gaze")]
        public int EyeGaze { get; set; }

        [JsonPropertyName("speech_detection")]
        public int SpeechDetection { get; set; }

        [JsonPropertyName("object_detection")]
        public int ObjectDetection { get; set; }

        [JsonPropertyName("violation_total")]
        public int ViolationTotal { get; set; }
    }
}
