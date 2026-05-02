using Shared.Dtos.AssignmentSubmissionDto.Request;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PresentationLayer.Controllers
{
    public class AssignmentSubmissionController(IServiceManager _serviceManger) : ApiControllerBase
    {
        [HttpPost("Assignment/Submit")]
        public async Task<IActionResult> SubmitAssignment([FromForm] AssignmentSubmissionRequestDto request, List<IFormFile?> attachmentFiles, CancellationToken cancellationToken)
        {
            var studentId = User.GetUserId();

            var result = await _serviceManger.AssignmentSubmissionService.SubmitAssignmentAsync(studentId, request, attachmentFiles, cancellationToken);
            return Ok(result);
        }

        [HttpGet("AssignmentSubmission/{submissionId}")]
        public async Task<IActionResult> GetAssignmentSubmission(int submissionId, CancellationToken cancellationToken)
        {
            var result = await _serviceManger.AssignmentSubmissionService.GetSubmissionByIdAsync(submissionId, cancellationToken);
            return Ok(result);
        }

        [HttpGet("Student/Assignment/Submissions")]
        public async Task<IActionResult> GetStudentSubmissions(CancellationToken cancellationToken)
        {
            var studentId = User.GetUserId();
            var result = await _serviceManger.AssignmentSubmissionService.GetSubmissionsByStudentIdAsync(studentId, cancellationToken);
            return Ok(result);
        }
        [HttpGet("Assignment/{assignmentId}/Students")]
        public async Task<IActionResult> GetSubmissionsToAssignment(int assignmentId, CancellationToken cancellationToken)
        {
            var result = await _serviceManger.AssignmentSubmissionService.GetSubmissionsByAssignmentIdAsync(assignmentId, cancellationToken);
            return Ok(result);
        }
        [HttpDelete("Assignment/Submission/{submissionId}")]
        public async Task<IActionResult> DeleteAssignment(int submissionId, CancellationToken cancellationToken)
        {
            await _serviceManger.AssignmentSubmissionService.DeleteSubmissionAsync(submissionId, cancellationToken);
            return Ok();
        }
        [HttpPut("Assignment/Submission/{submissionId}/Grade")]
        public async Task<IActionResult> GradeAssignment(int submissionId, [FromBody] GradeAssignmentSubmissionRequestDto request, CancellationToken cancellationToken)
        {
            var result = await _serviceManger.AssignmentSubmissionService.GradeSubmissionAsync(submissionId, request, cancellationToken);
            return Ok(result);
        }
    }
}
