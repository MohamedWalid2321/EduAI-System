using Shared.Dtos.RiskAnalysisDto.Request;
using Shared.Dtos.RiskAnalysisDto.Response;

namespace ServiceAbstractionLayer
{
    public interface IRiskAnalysisService
    {
        /// <summary>
        /// Persists raw violation rows and enqueues a Hangfire background job
        /// that will calculate the normalized risk score and update the cheating report.
        /// </summary>
        /// <returns>A lightweight acknowledgement response (job enqueued).</returns>
        Task<RiskAnalysisResponse> SubmitAsync(
            RiskAnalysisRequest request,
            CancellationToken cancellationToken = default);
    }
}
