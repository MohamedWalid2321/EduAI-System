namespace PresentationLayer.Controllers
{
	public class AssignmentController(IServiceManager serviceManager, ICacheService cacheService) : ApiControllerBase
	{
		private const string AssignmentsPattern = "/api/assignment*";

		[HasPermission(Permissions.GetAss)]
		[HttpGet("course/{courseId}")]
		[Cache(300)]
		public async Task<IActionResult> GetAllAssignmentsByCourseId(int courseId, CancellationToken cancellationToken)
		{
			var assignments = await serviceManager.AssignmentService.GetAllAssigmentsByCourseIdAsync(courseId, cancellationToken);
			return Ok(assignments);
		}

		[HasPermission(Permissions.GetAss)]
		[HttpGet("{id}")]
		[Cache(300)]
		public async Task<IActionResult> GetAssignmentById(int id, CancellationToken cancellationToken)
		{
			var assignment = await serviceManager.AssignmentService.GetAssigmentByIdAsync(id, cancellationToken);
			return Ok(assignment);
		}

		[HasPermission(Permissions.AddOrUpdateAss)]
		[HttpPost("course/{courseId}")]
		public async Task<IActionResult> CreateOrUpdateAssignmentForCourse(
			int courseId,
			[FromBody] AssigmentRequestDto assignmentDto, CancellationToken cancellationToken)
		{
			var createdOrUpdatedAssignment = await serviceManager.AssignmentService
				.CreateOrUpdateAssigmentForCourse(courseId, assignmentDto, cancellationToken);
			await cacheService.RemoveByPatternAsync(AssignmentsPattern);
			return Ok(createdOrUpdatedAssignment);
		}
		[HasPermission(Permissions.DeleteAss)]
		[HttpDelete("{id}")]
		public async Task<IActionResult> DeleteAssignment(int id, CancellationToken cancellationToken)
		{
			await serviceManager.AssignmentService.DeleteAssigmentAsync(id, cancellationToken);
			await cacheService.RemoveByPatternAsync(AssignmentsPattern);
			return Ok();
		}
		[HasPermission(Permissions.AddOrUpdateAss)]
		[HttpPost("{assignmentId}/attachments")]
		public async Task<IActionResult> AddAttachmentToAssignment(
			int assignmentId,
			[FromForm] List<IFormFile?> attachmentFiles, CancellationToken cancellationToken)
		{
			var updatedAssignment = await serviceManager.AssignmentService
				.AddAttachmentToAssigment(assignmentId, attachmentFiles, cancellationToken);
			await cacheService.RemoveByPatternAsync(AssignmentsPattern);
			return Ok(updatedAssignment);
		}
		[HasPermission(Permissions.DeleteAss)]
		[HttpDelete("attachments/{attachmentId}")]
		public async Task<IActionResult> RemoveAttachment(Guid attachmentId, CancellationToken cancellationToken)
		{
			await serviceManager.AssignmentService.RemoveAttachment(attachmentId, cancellationToken);
			await cacheService.RemoveByPatternAsync(AssignmentsPattern);
			return Ok();
		}
	}
}