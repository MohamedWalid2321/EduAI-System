using Shared.Dtos.RiskAnalysisDto.Request;
using Shared.Dtos.RiskAnalysisDto.Response;

namespace PresentationLayer.Controllers
{
    /// <summary>
    /// Receives AI-Proctoring violation reports and kicks off
    /// the background risk-score calculation pipeline.
    /// </summary>
    [Route("api/risk-analysis")]
    public class RiskAnalysisController(IRiskAnalysisService riskAnalysisService) : ApiControllerBase
    {
        private readonly IRiskAnalysisService _riskAnalysisService = riskAnalysisService;

        /// <summary>
        /// Submit a proctoring violation report.
        /// Returns 202 Accepted immediately; the risk score is calculated
        /// asynchronously and stored on the matching CheatingReport.
        /// </summary>
        /// <response code="202">Report accepted; background job enqueued.</response>
        /// <response code="400">Attempt_Id is not a valid integer.</response>
        /// <response code="404">No CheatingReport exists for the supplied Attempt_Id.</response>
        [HttpPost]
        [ProducesResponseType(typeof(RiskAnalysisResponse), StatusCodes.Status202Accepted)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> SubmitReport(
            [FromBody] RiskAnalysisRequest request,
            CancellationToken cancellationToken)
        {
            var result = await _riskAnalysisService.SubmitAsync(request, cancellationToken);
            return Accepted(result);
        }
    }
}
