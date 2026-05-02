namespace PresentationLayer.Controllers
{
	public class AssignmentController(IServiceManager serviceManager) : ApiControllerBase
	{
		[HttpGet("course/{courseId}")]
		public async Task<IActionResult> GetAllAssignmentsByCourseId(int courseId, CancellationToken cancellationToken)
		{
			var assignments = await serviceManager.AssignmentService.GetAllAssigmentsByCourseIdAsync(courseId, cancellationToken);
			return Ok(assignments);
		}

		[HttpPost("course/{courseId}")]
		public async Task<IActionResult> CreateOrUpdateAssignmentForCourse(
			int courseId,
			[FromBody] AssigmentRequestDto assignmentDto, CancellationToken cancellationToken)
		{
			var createdOrUpdatedAssignment = await serviceManager.AssignmentService
				.CreateOrUpdateAssigmentForCourse(courseId, assignmentDto, cancellationToken);
			return Ok(createdOrUpdatedAssignment);
		}

		[HttpDelete("{id}")]
		public async Task<IActionResult> DeleteAssignment(int id, CancellationToken cancellationToken)
		{
			await serviceManager.AssignmentService.DeleteAssigmentAsync(id, cancellationToken);
			return Ok();
		}

		[HttpGet("{id}")]
		public async Task<IActionResult> GetAssignmentById(int id, CancellationToken cancellationToken)
		{
			var assignment = await serviceManager.AssignmentService.GetAssigmentByIdAsync(id, cancellationToken);
			return Ok(assignment);
		}

		[HttpPost("{assignmentId}/attachments")]
		public async Task<IActionResult> AddAttachmentToAssignment(
			int assignmentId,
			[FromForm] List<IFormFile?> attachmentFiles, CancellationToken cancellationToken)
		{
			var updatedAssignment = await serviceManager.AssignmentService
				.AddAttachmentToAssigment(assignmentId, attachmentFiles, cancellationToken);
			return Ok(updatedAssignment);
		}

		[HttpDelete("attachments/{attachmentId}")]
		public async Task<IActionResult> RemoveAttachment(Guid attachmentId, CancellationToken cancellationToken)
		{
			await serviceManager.AssignmentService.RemoveAttachment(attachmentId, cancellationToken);
			return Ok();
		}
	}
}