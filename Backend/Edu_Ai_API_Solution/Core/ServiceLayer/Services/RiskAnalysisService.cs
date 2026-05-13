using DomainLayer.Contracts;
using DomainLayer.Models;
using Hangfire;
using Microsoft.Extensions.Logging;
using ServiceAbstractionLayer;
using ServiceLayer.Jobs;
using ServiceLayer.Specifications.CheatingReportSpecifications;
using Shared.Dtos.RiskAnalysisDto.Request;
using Shared.Dtos.RiskAnalysisDto.Response;

namespace ServiceLayer.Services
{
    public class RiskAnalysisService(
        IUnitOfWork unitOfWork,
        IBackgroundJobClient backgroundJobClient,
        ILogger<RiskAnalysisService> logger) : IRiskAnalysisService
    {
        private readonly IUnitOfWork _unitOfWork = unitOfWork;
        private readonly IBackgroundJobClient _backgroundJobClient = backgroundJobClient;
        private readonly ILogger<RiskAnalysisService> _logger = logger;

        /// <inheritdoc />
        public async Task<RiskAnalysisResponse> SubmitAsync(
            RiskAnalysisRequest request,
            CancellationToken cancellationToken = default)
        {
            // ── Validate Attempt_Id ─────────────────────────────────────────────
            if (!int.TryParse(request.ReportMetadata.AttemptId, out var attemptId))
                throw new ArgumentException(
                    $"Attempt_Id '{request.ReportMetadata.AttemptId}' must be a valid integer.");

            // ── Locate the corresponding CheatingReport ─────────────────────────
            var reportRepo = _unitOfWork.GetRepository<CheatingReport, int>();
            var spec = new CheatingReportByAttemptSpecification(attemptId);
            var report = await reportRepo.GetFirstOrDefaultAsync(spec, cancellationToken)
                ?? throw new KeyNotFoundException(
                    $"No CheatingReport found for QuizAttempt {attemptId}. " +
                    "Create one first via POST /api/cheating-reports.");

            // ── Enqueue the background job (fire-and-forget) ────────────────────
            var jobId = _backgroundJobClient.Enqueue<RiskScoreCalculationJob>(
                job => job.ExecuteAsync(request, report.Id));

            _logger.LogInformation(
                "Risk analysis job enqueued – JobId={JobId}, AttemptId={AttemptId}, ReportId={ReportId}",
                jobId, attemptId, report.Id);

            // ── Return an immediate lightweight acknowledgement ──────────────────
            // The actual scores will be null until the job finishes;
            // here we return the unprocessed metadata so the caller gets a 202.
            return new RiskAnalysisResponse
            {
                StudentId             = request.ReportMetadata.StudentId,
                AttemptId             = request.ReportMetadata.AttemptId,
                OriginalViolationRate = request.SessionSummary.ViolationRate,
                QuestionsRisk         = [],          // populated later by the job
                OverallSessionRiskScore = 0m,        // updated on CheatingReport by job
            };
        }
    }
}
