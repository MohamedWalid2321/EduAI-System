
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
				.AddContentForCourse(courseId, contentDto);
			return Ok(createdOrUpdatedContent);
		}
		[HttpPut("{ContentId}")]
		public async Task<IActionResult> UpdateContent(int ContentId, [FromBody] ContentRequestDto contentDto)
		{
			await serviceManager.ContentService.UpdateContentForCourse(ContentId, contentDto);
			return Ok();
		}

		[HttpDelete("{ContentId}")]
		public async Task<IActionResult> DeleteContent(int ContentId)
		{
			await serviceManager.ContentService.DeleteContentAsync(ContentId);
			return Ok();
		}
		
		[HttpGet("{ContentId}")]
		public async Task<IActionResult> GetContentById(int ContentId)
		{
			var content = await serviceManager.ContentService.GetContentByIdAsync(ContentId);
			return Ok(content);
		}
		
		[HttpPost("{contentId}/attachments")]
		[RequestSizeLimit(524288000)] // 500 MB
		[RequestFormLimits(MultipartBodyLengthLimit = 524288000)]
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
