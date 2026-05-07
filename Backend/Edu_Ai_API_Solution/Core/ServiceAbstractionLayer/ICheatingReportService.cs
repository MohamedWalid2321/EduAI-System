using Shared.Dtos.CheatingReportDto.Request;
using Shared.Dtos.CheatingReportDto.Response;

namespace ServiceAbstractionLayer
{
    public interface ICheatingReportService
    {
        // Get the cheating report for a specific quiz attempt
        Task<CheatingReportResponse> GetByAttemptIdAsync(int attemptId, CancellationToken cancellationToken = default);

        // Get all cheating reports for a quiz
        Task<IEnumerable<CheatingReportResponse>> GetByQuizIdAsync(int quizId, CancellationToken cancellationToken = default);

        // Create an empty report for a quiz attempt (called by proctoring desktop app)
        Task<CheatingReportResponse> CreateAsync(int attemptId, CancellationToken cancellationToken = default);

        // Add a new violation to an existing report
        Task<CheatingViolationResponse> AddViolationAsync(int reportId, AddViolationRequest request, CancellationToken cancellationToken = default);

        // Delete a single violation
        Task DeleteViolationAsync(int violationId, CancellationToken cancellationToken = default);

        // Delete the full report
        Task DeleteReportAsync(int reportId, CancellationToken cancellationToken = default);
    }
}