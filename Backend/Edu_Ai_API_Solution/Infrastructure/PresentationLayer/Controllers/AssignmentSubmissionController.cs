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
        public async Task<IActionResult> SubmitAssignment([FromForm] AssignmentSubmissionRequestDto request, List<IFormFile?> attachmentFiles)
        {
            var studentId = User.GetUserId();

            var result = await _serviceManger.AssignmentSubmissionService.SubmitAssignmentAsync(studentId, request, attachmentFiles);
            return Ok(result);
        }

        [HttpGet("AssignmentSubmission/{submissionId}")]
        public async Task<IActionResult> GetAssignmentSubmission(int submissionId)
        {
            var result = await _serviceManger.AssignmentSubmissionService.GetSubmissionByIdAsync(submissionId);
            return Ok(result);
        }

        [HttpGet("Student/Assignment/Submissions")]
        public async Task<IActionResult> GetStudentSubmissions()
        {
            var studentId = User.GetUserId();
            var result = await _serviceManger.AssignmentSubmissionService.GetSubmissionsByStudentIdAsync(studentId);
            return Ok(result);
        }
        [HttpGet("Assignment/{assignmentId}/Students")]
        public async Task<IActionResult> GetSubmissionsToAssignment(int assignmentId)
        {
            var result = await _serviceManger.AssignmentSubmissionService.GetSubmissionsByAssignmentIdAsync(assignmentId);
            return Ok(result);
        }
        [HttpDelete("Assignment/Submission/{submissionId}")]
        public  async Task<IActionResult> DeleteAssignment(int submissionId)
        {
            await _serviceManger.AssignmentSubmissionService.DeleteSubmissionAsync(submissionId);
            return Ok();
        }
        [HttpPut("Assignment/Submission/{submissionId}/Grade")]
        public async Task<IActionResult> GradeAssignment(int submissionId, [FromBody] GradeAssignmentSubmissionRequestDto request)
        {
            var result = await _serviceManger.AssignmentSubmissionService.GradeSubmissionAsync(submissionId, request);
            return Ok(result);
        }
    }
}
