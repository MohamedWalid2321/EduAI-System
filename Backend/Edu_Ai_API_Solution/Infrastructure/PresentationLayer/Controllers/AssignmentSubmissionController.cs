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
        /// <summary>Student submits an assignment. (Student only)</summary>
        [HasPermission(Permissions.SolveAss)]
        [HttpPost("Assignment/Submit")]
        public async Task<IActionResult> SubmitAssignment(
            [FromForm] AssignmentSubmissionRequestDto request,
            List<IFormFile?> attachmentFiles,
            CancellationToken cancellationToken)
        {
            var studentId = User.GetUserId();
            var result = await _serviceManger.AssignmentSubmissionService
                .SubmitAssignmentAsync(studentId, request, attachmentFiles, cancellationToken);
            return Ok(result);
        }

        /// <summary>Get a submission by ID. (Student + Instructor)</summary>
        [HasPermission(Permissions.GetAssSubmission)]
        [HttpGet("AssignmentSubmission/{submissionId}")]
        public async Task<IActionResult> GetAssignmentSubmission(int submissionId, CancellationToken cancellationToken)
        {
            var result = await _serviceManger.AssignmentSubmissionService
                .GetSubmissionByIdAsync(submissionId, cancellationToken);
            return Ok(result);
        }

        /// <summary>Get the current student's own submissions. (Student only)</summary>
        [HasPermission(Permissions.SolveAss)]
        [HttpGet("Student/Assignment/Submissions")]
        public async Task<IActionResult> GetStudentSubmissions(CancellationToken cancellationToken)
        {
            var studentId = User.GetUserId();
            var result = await _serviceManger.AssignmentSubmissionService
                .GetSubmissionsByStudentIdAsync(studentId, cancellationToken);
            return Ok(result);
        }

        /// <summary>Get all student submissions for a specific assignment. (Instructor only)</summary>
        [HasPermission(Permissions.GetAllAssSubmissions)]
        [HttpGet("Assignment/{assignmentId}/Students")]
        public async Task<IActionResult> GetSubmissionsToAssignment(int assignmentId, CancellationToken cancellationToken)
        {
            var result = await _serviceManger.AssignmentSubmissionService
                .GetSubmissionsByAssignmentIdAsync(assignmentId, cancellationToken);
            return Ok(result);
        }

        /// <summary>Delete a submission. (Instructor / Admin only)</summary>
        [HasPermission(Permissions.DeleteAssSubmission)]
        [HttpDelete("Assignment/Submission/{submissionId}")]
        public async Task<IActionResult> DeleteAssignment(int submissionId, CancellationToken cancellationToken)
        {
            await _serviceManger.AssignmentSubmissionService
                .DeleteSubmissionAsync(submissionId, cancellationToken);
            return Ok();
        }

        /// <summary>Grade a student's submission. (Instructor only)</summary>
        [HasPermission(Permissions.GradeAss)]
        [HttpPut("Assignment/Submission/{submissionId}/Grade")]
        public async Task<IActionResult> GradeAssignment(
            int submissionId,
            [FromBody] GradeAssignmentSubmissionRequestDto request,
            CancellationToken cancellationToken)
        {
            var result = await _serviceManger.AssignmentSubmissionService
                .GradeSubmissionAsync(submissionId, request, cancellationToken);
            return Ok(result);
        }
    }
}
