<<<<<<< HEAD
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using ServiceAbstractionLayer;
using Shared.Dtos.AssigmentDto.Request;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

=======
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
namespace PresentationLayer.Controllers
{
	public class AssignmentController(IServiceManager serviceManager) : ApiControllerBase
	{
		[HttpGet("course/{courseId}")]
		public async Task<IActionResult> GetAllAssignmentsByCourseId(int courseId)
		{
			var assignments = await serviceManager.AssignmentService.GetAllAssigmentsByCourseIdAsync(courseId);
			return Ok(assignments);
		}

		[HttpPost("course/{courseId}")]
		public async Task<IActionResult> CreateOrUpdateAssignmentForCourse(
			int courseId,
			[FromBody] AssigmentRequestDto assignmentDto)
		{
			var createdOrUpdatedAssignment = await serviceManager.AssignmentService
				.CreateOrUpdateAssigmentForCourse(courseId, assignmentDto);
			return Ok(createdOrUpdatedAssignment);
		}

		[HttpDelete("{id}")]
		public async Task<IActionResult> DeleteAssignment(int id)
		{
			await serviceManager.AssignmentService.DeleteAssigmentAsync(id);
			return Ok();
		}

		[HttpGet("{id}")]
		public async Task<IActionResult> GetAssignmentById(int id)
		{
			var assignment = await serviceManager.AssignmentService.GetAssigmentByIdAsync(id);
			return Ok(assignment);
		}

		[HttpPost("{assignmentId}/attachments")]
		public async Task<IActionResult> AddAttachmentToAssignment(
			int assignmentId,
			[FromForm] List<IFormFile?> attachmentFiles)
		{
			var updatedAssignment = await serviceManager.AssignmentService
				.AddAttachmentToAssigment(assignmentId, attachmentFiles);
			return Ok(updatedAssignment);
		}

		[HttpDelete("attachments/{attachmentId}")]
		public async Task<IActionResult> RemoveAttachment(Guid attachmentId)
		{
			await serviceManager.AssignmentService.RemoveAttachment(attachmentId);
			return Ok();
		}
	}
}