namespace PresentationLayer.Controllers
{
	public class ContentController(IServiceManager serviceManager): ApiControllerBase
	{
		[HttpGet("course/{courseId}")]
		public async Task<IActionResult> GetAllContentByCourseId(int courseId, CancellationToken cancellationToken)
		{
			var contents = await serviceManager.ContentService.GetAllContentsByCourseIdAsync(courseId, cancellationToken);
			return Ok(contents);
		}
		
		[HttpPost("course/{courseId}")]
		public async Task<IActionResult> CreateOrUpdateContentForCourse(
			int courseId,
			[FromBody] ContentRequestDto contentDto, CancellationToken cancellationToken)
		{
			var createdOrUpdatedContent = await serviceManager.ContentService
				.AddContentForCourse(courseId, contentDto, cancellationToken);
			return Ok(createdOrUpdatedContent);
		}
		[HttpPut("{ContentId}")]
		public async Task<IActionResult> UpdateContent(int ContentId, [FromBody] ContentRequestDto contentDto, CancellationToken cancellationToken)
		{
			await serviceManager.ContentService.UpdateContentForCourse(ContentId, contentDto, cancellationToken);
			return Ok();
		}

		[HttpDelete("{ContentId}")]
		public async Task<IActionResult> DeleteContent(int ContentId, CancellationToken cancellationToken)
		{
			await serviceManager.ContentService.DeleteContentAsync(ContentId, cancellationToken);
			return Ok();
		}
		
		[HttpGet("{ContentId}")]
		public async Task<IActionResult> GetContentById(int ContentId, CancellationToken cancellationToken)
		{
			var content = await serviceManager.ContentService.GetContentByIdAsync(ContentId, cancellationToken);
			return Ok(content);
		}
		
		[HttpPost("{contentId}/attachments")]
		[RequestSizeLimit(524288000)] // 500 MB
		[RequestFormLimits(MultipartBodyLengthLimit = 524288000)]
		public async Task<IActionResult> AddAttachmentToContent(
			int contentId, 
			[FromForm] List<IFormFile?> attachmentFiles, CancellationToken cancellationToken)
		{
			var updatedContent = await serviceManager.ContentService
				.AddAttachmentToContent(contentId, attachmentFiles, cancellationToken);
			return Ok(updatedContent);
		}
		
		[HttpDelete("attachments/{attachmentId}")]
		public async Task<IActionResult> RemoveAttachmentFromContent(Guid attachmentId, CancellationToken cancellationToken)
		{
			await serviceManager.ContentService.RemoveAttachment(attachmentId, cancellationToken);
			return Ok();
		}
	}
}
