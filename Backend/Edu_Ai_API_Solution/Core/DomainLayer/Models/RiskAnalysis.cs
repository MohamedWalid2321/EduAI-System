namespace DomainLayer.Models
{
    /// <summary>
    /// Persists the raw per-question violation counts sent by the AI Proctoring
    /// desktop app for a single quiz attempt.  One row per (AttemptId, QuestionId).
    /// Having every student's counts in the table lets us run cohort
    /// MIN/MAX queries directly in the database.
    /// </summary>
    public class RiskAnalysis : BaseEntity<int>
    {
        // ── Identity ────────────────────────────────────────────────────────────
        public int    AttemptId  { get; set; }
        public string StudentId  { get; set; } = string.Empty;
        public int    QuestionId { get; set; }

        // ── Session-level metadata (duplicated per row for easy filtering) ──────
        public double ViolationRate { get; set; }

        // ── Per-question violation counts ────────────────────────────────────────
        public int FaceDetection   { get; set; }
        public int FaceRecognition { get; set; }
        public int EyeGaze         { get; set; }
        public int SpeechDetection { get; set; }
        public int ObjectDetection { get; set; }

        // ── Weights (stored so we can re-calculate without re-parsing the payload) ─
        public double WeightFaceAbsenceMismatch { get; set; }
        public double WeightSuspiciousMovement  { get; set; }
        public double WeightConversationNoise   { get; set; }
        public double WeightForbiddenObjects    { get; set; }
    }
}
