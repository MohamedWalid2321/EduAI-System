using DomainLayer.Contracts;
using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Shared.Dtos.RiskAnalysisDto.Request;
using Shared.Dtos.RiskAnalysisDto.Response;

namespace ServiceLayer.Jobs
{
    /// <summary>
    /// Hangfire fire-and-forget job that:
    ///   1. Persists raw RiskAnalysis rows (one per question) for the current attempt.
    ///   2. Performs cohort-level Min-Max normalization per question.
    ///   3. Applies per-category weights.
    ///   4. Writes the final RiskScore back to the CheatingReport.
    /// </summary>
    public class RiskScoreCalculationJob(
        IUnitOfWork unitOfWork,
        ILogger<RiskScoreCalculationJob> logger)
    {
        private readonly IUnitOfWork _unitOfWork = unitOfWork;
        private readonly ILogger<RiskScoreCalculationJob> _logger = logger;

        // ────────────────────────────────────────────────────────────────────────
        //  Public entry-point (called by Hangfire)
        // ────────────────────────────────────────────────────────────────────────

        public async Task ExecuteAsync(RiskAnalysisRequest request, int cheatingReportId)
        {
            _logger.LogInformation(
                "RiskScoreCalculationJob started – AttemptId={AttemptId}, ReportId={ReportId}",
                request.ReportMetadata.AttemptId, cheatingReportId);

            try
            {
                // 1. Persist / upsert raw violation rows for this attempt (cohort input data).
                await PersistRawRowsAsync(request);

                // 2. Calculate per-question scores and the overall session score.
                var result = await CalculateScoresAsync(request);

                // 3. Persist the full structured result (session + per-question breakdown).
                await PersistResultsAsync(result, cheatingReportId);

                // 4. Update the convenience RiskScore column on CheatingReport.
                await UpdateCheatingReportAsync(cheatingReportId, result.OverallSessionRiskScore);

                _logger.LogInformation(
                    "RiskScoreCalculationJob finished – AttemptId={AttemptId}, OverallScore={Score}",
                    request.ReportMetadata.AttemptId, result.OverallSessionRiskScore);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex,
                    "RiskScoreCalculationJob failed – AttemptId={AttemptId}",
                    request.ReportMetadata.AttemptId);
                throw;
            }
        }

        // ────────────────────────────────────────────────────────────────────────
        //  Step 1 – Persist raw rows
        // ────────────────────────────────────────────────────────────────────────

        private async Task PersistRawRowsAsync(RiskAnalysisRequest request)
        {
            if (!int.TryParse(request.ReportMetadata.AttemptId, out var attemptId))
                throw new InvalidOperationException(
                    $"Attempt_Id '{request.ReportMetadata.AttemptId}' is not a valid integer.");

            var repo = _unitOfWork.GetRepository<RiskAnalysis, int>();
            var weights = request.SessionSummary.WeightsUsed;

            foreach (var q in request.Questions)
            {
                // Upsert: remove the old row for this (attempt, question) pair if it exists.
                var allRows = await repo.GetAllAsync();
                var existing = allRows.FirstOrDefault(
                    r => r.AttemptId == attemptId && r.QuestionId == q.QuestionId);

                if (existing is not null)
                    repo.HardDelete(existing);

                var row = new RiskAnalysis
                {
                    AttemptId  = attemptId,
                    StudentId  = request.ReportMetadata.StudentId,
                    QuestionId = q.QuestionId,

                    ViolationRate  = request.SessionSummary.ViolationRate,

                    FaceDetection   = q.FaceDetection,
                    FaceRecognition = q.FaceRecognition,
                    EyeGaze         = q.EyeGaze,
                    SpeechDetection = q.SpeechDetection,
                    ObjectDetection = q.ObjectDetection,

                    WeightFaceAbsenceMismatch = weights.FaceAbsenceMismatch,
                    WeightSuspiciousMovement  = weights.SuspiciousMovement,
                    WeightConversationNoise   = weights.ConversationNoise,
                    WeightForbiddenObjects    = weights.ForbiddenObjects,
                };

                await repo.AddAsync(row);
            }

            await _unitOfWork.SaveChangesAsync();
        }

        // ────────────────────────────────────────────────────────────────────────
        //  Step 2 – Calculate per-question & overall scores
        // ────────────────────────────────────────────────────────────────────────

        public async Task<RiskAnalysisResponse> CalculateScoresAsync(RiskAnalysisRequest request)
        {
            if (!int.TryParse(request.ReportMetadata.AttemptId, out var attemptId))
                throw new InvalidOperationException(
                    $"Attempt_Id '{request.ReportMetadata.AttemptId}' is not a valid integer.");

            var repo = _unitOfWork.GetRepository<RiskAnalysis, int>();
            var weights = request.SessionSummary.WeightsUsed;

            // Load all rows from the DB once and group by QuestionId.
            var allRows = (await repo.GetAllAsync()).ToList();

            var questionResults = new List<QuestionRiskResult>();

            foreach (var q in request.Questions)
            {
                // All rows across the cohort for this specific question.
                var cohortRows = allRows.Where(r => r.QuestionId == q.QuestionId).ToList();

                // Cohort min/max per violation column.
                var minFD  = cohortRows.Min(r => r.FaceDetection);
                var maxFD  = cohortRows.Max(r => r.FaceDetection);
                var minFR  = cohortRows.Min(r => r.FaceRecognition);
                var maxFR  = cohortRows.Max(r => r.FaceRecognition);
                var minEG  = cohortRows.Min(r => r.EyeGaze);
                var maxEG  = cohortRows.Max(r => r.EyeGaze);
                var minSD  = cohortRows.Min(r => r.SpeechDetection);
                var maxSD  = cohortRows.Max(r => r.SpeechDetection);
                var minOD  = cohortRows.Min(r => r.ObjectDetection);
                var maxOD  = cohortRows.Max(r => r.ObjectDetection);

                // ── Student's normalized score for this question ────────────────
                var studentScore = ComputeScore(
                    fd:  q.FaceDetection,   minFD, maxFD,
                    fr:  q.FaceRecognition, minFR, maxFR,
                    eg:  q.EyeGaze,         minEG, maxEG,
                    sd:  q.SpeechDetection, minSD, maxSD,
                    od:  q.ObjectDetection, minOD, maxOD,
                    weights);

                // ── Cohort average: re-score every student's row ────────────────
                double cohortSum = 0;
                foreach (var row in cohortRows)
                {
                    cohortSum += ComputeScore(
                        fd:  row.FaceDetection,   minFD, maxFD,
                        fr:  row.FaceRecognition, minFR, maxFR,
                        eg:  row.EyeGaze,         minEG, maxEG,
                        sd:  row.SpeechDetection, minSD, maxSD,
                        od:  row.ObjectDetection, minOD, maxOD,
                        weights);
                }

                var cohortAvg = cohortRows.Count > 0
                    ? cohortSum / cohortRows.Count
                    : 0.0;

                questionResults.Add(new QuestionRiskResult
                {
                    QuestionId         = q.QuestionId,
                    StudentRiskScore   = Math.Round((decimal)studentScore, 2),
                    CohortAvgRiskScore = Math.Round((decimal)cohortAvg,   2),
                });
            }

            var overallScore = questionResults.Count > 0
                ? questionResults.Average(r => r.StudentRiskScore)
                : 0m;

            return new RiskAnalysisResponse
            {
                StudentId             = request.ReportMetadata.StudentId,
                AttemptId             = request.ReportMetadata.AttemptId,
                OriginalViolationRate = request.SessionSummary.ViolationRate,
                QuestionsRisk         = questionResults,
                OverallSessionRiskScore = Math.Round(overallScore, 2),
            };
        }

        // ────────────────────────────────────────────────────────────────────────
        //  Step 3 – Update CheatingReport.RiskScore
        // ────────────────────────────────────────────────────────────────────────

        private async Task UpdateCheatingReportAsync(int reportId, decimal overallScore)
        {
            var reportRepo = _unitOfWork.GetRepository<CheatingReport, int>();
            var report = await reportRepo.GetByIdAsync(reportId)
                ?? throw new KeyNotFoundException(
                    $"CheatingReport {reportId} not found when trying to update RiskScore.");

            report.RiskScore = overallScore;
            reportRepo.Update(report);
            await _unitOfWork.SaveChangesAsync();
        }

        // ────────────────────────────────────────────────────────────────────────
        //  Step 3 – Persist full result (session + per-question breakdown)
        // ────────────────────────────────────────────────────────────────────────

        private async Task PersistResultsAsync(RiskAnalysisResponse result, int cheatingReportId)
        {
            if (!int.TryParse(result.AttemptId, out var attemptId))
                throw new InvalidOperationException(
                    $"Attempt_Id '{result.AttemptId}' is not a valid integer.");

            var repo = _unitOfWork.GetRepository<RiskAssessmentResult, int>();

            // Upsert: hard-delete any previous result for this attempt (re-submission).
            var allExisting = await repo.GetAllAsync();
            var existing = allExisting.FirstOrDefault(r => r.AttemptId == attemptId);
            if (existing is not null)
                repo.HardDelete(existing);

            // Build the session-level record with its question children.
            var assessment = new RiskAssessmentResult
            {
                StudentId               = result.StudentId,
                AttemptId               = attemptId,
                SessionViolationRate    = result.OriginalViolationRate,
                OverallSessionRiskScore = result.OverallSessionRiskScore,
                CheatingReportId        = cheatingReportId,

                Questions = result.QuestionsRisk.Select(q => new RiskQuestionResult
                {
                    QuestionId         = q.QuestionId,
                    StudentRiskScore   = q.StudentRiskScore,
                    CohortAvgRiskScore = q.CohortAvgRiskScore,
                }).ToList()
            };

            await repo.AddAsync(assessment);
            await _unitOfWork.SaveChangesAsync();
        }

        // ────────────────────────────────────────────────────────────────────────
        //  Core math helpers

        // ────────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Normalizes a single violation count using min-max, then multiplies by weight.
        /// Returns 0 when max == min (avoids division by zero).
        /// </summary>
        private static double Normalize(int value, int min, int max, double weight)
        {
            if (max == min) return 0.0;
            return ((double)(value - min) / (max - min)) * weight;
        }

        /// <summary>
        /// Applies weight mapping and sums all normalized violation scores for one question row.
        ///   face_detection + face_recognition  → face_absence_mismatch weight
        ///   eye_gaze                            → suspicious_movement weight
        ///   speech_detection                    → conversation_noise weight
        ///   object_detection                    → forbidden_objects weight
        /// </summary>
        private static double ComputeScore(
            int fd, int minFD, int maxFD,
            int fr, int minFR, int maxFR,
            int eg, int minEG, int maxEG,
            int sd, int minSD, int maxSD,
            int od, int minOD, int maxOD,
            WeightsUsed w)
        {
            return Normalize(fd, minFD, maxFD, w.FaceAbsenceMismatch)
                 + Normalize(fr, minFR, maxFR, w.FaceAbsenceMismatch)
                 + Normalize(eg, minEG, maxEG, w.SuspiciousMovement)
                 + Normalize(sd, minSD, maxSD, w.ConversationNoise)
                 + Normalize(od, minOD, maxOD, w.ForbiddenObjects);
        }
    }
}
