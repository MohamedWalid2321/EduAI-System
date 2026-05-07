using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using ServiceAbstractionLayer;
using Shared.Constants;
using Shared.Dtos.CheatingReportDto.Request;

namespace PresentationLayer.Controllers
{
    public class CheatingReportController(ICheatingReportService _cheatingReportService) : ApiControllerBase
    {
        [HttpGet("attempt/{attemptId:int}")]
        [Authorize(Policy = Permissions.GetCheatingReport)]
        public async Task<IActionResult> GetByAttempt(int attemptId, CancellationToken cancellationToken)
        {
            var result = await _cheatingReportService.GetByAttemptIdAsync(attemptId, cancellationToken);
            return Ok(result);
        }

        [HttpGet("quiz/{quizId:int}")]
        [Authorize(Policy = Permissions.GetCheatingReport)]
        public async Task<IActionResult> GetByQuiz(int quizId, CancellationToken cancellationToken)
        {
            var result = await _cheatingReportService.GetByQuizIdAsync(quizId, cancellationToken);
            return Ok(result);
        }

        [HttpPost("attempt/{attemptId:int}")]
        [Authorize(Policy = Permissions.AddCheatingReport)]
        public async Task<IActionResult> Create(int attemptId, CancellationToken cancellationToken)
        {
            var result = await _cheatingReportService.CreateAsync(attemptId, cancellationToken);
            return CreatedAtAction(nameof(GetByAttempt), new { attemptId }, result);
        }

        [HttpPost("{reportId:int}/violations")]
        [Authorize(Policy = Permissions.AddCheatingReport)]
        public async Task<IActionResult> AddViolation(int reportId, [FromBody] AddViolationRequest request, CancellationToken cancellationToken)
        {
            var result = await _cheatingReportService.AddViolationAsync(reportId, request, cancellationToken);
            return Ok(result);
        }

        [HttpDelete("{reportId:int}")]
        [Authorize(Policy = Permissions.DeleteCheatingReport)]
        public async Task<IActionResult> DeleteReport(int reportId, CancellationToken cancellationToken)
        {
            await _cheatingReportService.DeleteReportAsync(reportId, cancellationToken);
            return NoContent();
        }

        [HttpDelete("violations/{violationId:int}")]
        [Authorize(Policy = Permissions.DeleteCheatingReport)]
        public async Task<IActionResult> DeleteViolation(int violationId, CancellationToken cancellationToken)
        {
            await _cheatingReportService.DeleteViolationAsync(violationId, cancellationToken);
            return NoContent();
        }
    }
}