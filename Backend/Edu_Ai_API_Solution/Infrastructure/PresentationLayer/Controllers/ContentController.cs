<<<<<<< HEAD
﻿using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using ServiceAbstractionLayer;
using Shared.Dtos.ContentDto.ContentRequest;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

=======
﻿
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
namespace PresentationLayer.Controllers
{
	public class ContentController(IServiceManager serviceManager): ApiControllerBase
	{
		[HttpGet("course/{courseId}")]
		public async Task<IActionResult> GetAllContentByCourseId(int courseId)
		{
			var contents = await serviceManager.ContentService.GetAllContentsByCourseIdAsync(courseId);
			return Ok(contents);
		}
		
		[HttpPost("course/{courseId}")]
		public async Task<IActionResult> CreateOrUpdateContentForCourse(
			int courseId,
			[FromBody] ContentRequestDto contentDto)
		{
			var createdOrUpdatedContent = await serviceManager.ContentService
				.CreateOrUpdateContentForCourse(courseId, contentDto);
			return Ok(createdOrUpdatedContent);
		}
		
		[HttpDelete("{id}")]
		public async Task<IActionResult> DeleteContent(int id)
		{
			await serviceManager.ContentService.DeleteContentAsync(id);
			return Ok();
		}
		
		[HttpGet("{id}")]
		public async Task<IActionResult> GetContentById(int id)
		{
			var content = await serviceManager.ContentService.GetContentByIdAsync(id);
			return Ok(content);
		}
		
		[HttpPost("{contentId}/attachments")]
		public async Task<IActionResult> AddAttachmentToContent(
			int contentId, 
			[FromForm] List<IFormFile?> attachmentFiles)
		{
			var updatedContent = await serviceManager.ContentService
				.AddAttachmentToContent(contentId, attachmentFiles);
			return Ok(updatedContent);
		}
		
		[HttpDelete("attachments/{attachmentId}")]
		public async Task<IActionResult> RemoveAttachmentFromContent(Guid attachmentId)
		{
			await serviceManager.ContentService.RemoveAttachment(attachmentId);
			return Ok();
		}
	}
}
