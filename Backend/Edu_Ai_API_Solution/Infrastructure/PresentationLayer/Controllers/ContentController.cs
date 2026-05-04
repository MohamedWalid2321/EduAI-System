namespace PresentationLayer.Controllers
{
	public class ContentController(IServiceManager serviceManager, ICacheService cacheService) : ApiControllerBase
	{
		private const string ContentsPattern = "/api/content*";
		[HasPermission(Permissions.GetContent)]
		[HttpGet("course/{courseId}")]
		[Cache(300)]
		public async Task<IActionResult> GetAllContentByCourseId(int courseId, CancellationToken cancellationToken)
		{
			var contents = await serviceManager.ContentService.GetAllContentsByCourseIdAsync(courseId, cancellationToken);
			return Ok(contents);
		}
		
		[HasPermission(Permissions.GetContent)]
		[HttpGet("{ContentId}")]
		[Cache(300)]
		public async Task<IActionResult> GetContentById(int ContentId, CancellationToken cancellationToken)
		{
			var content = await serviceManager.ContentService.GetContentByIdAsync(ContentId, cancellationToken);
			return Ok(content);
		}
		[HasPermission(Permissions.AddContent)]
		[HttpPost("course/{courseId}")]
		public async Task<IActionResult> CreateOrUpdateContentForCourse(
			int courseId,
			[FromBody] ContentRequestDto contentDto, CancellationToken cancellationToken)
		{
			var createdOrUpdatedContent = await serviceManager.ContentService
				.AddContentForCourse(courseId, contentDto, cancellationToken);
			await cacheService.RemoveByPatternAsync(ContentsPattern);
			return Ok(createdOrUpdatedContent);
		}
		[HasPermission(Permissions.UpdateContent)]
		[HttpPut("{ContentId}")]
		public async Task<IActionResult> UpdateContent(int ContentId, [FromBody] ContentRequestDto contentDto, CancellationToken cancellationToken)
		{
			await serviceManager.ContentService.UpdateContentForCourse(ContentId, contentDto, cancellationToken);
			await cacheService.RemoveByPatternAsync(ContentsPattern);
			return Ok();
		}
		[HasPermission(Permissions.DeleteContent)]
		[HttpDelete("{ContentId}")]
		public async Task<IActionResult> DeleteContent(int ContentId, CancellationToken cancellationToken)
		{
			await serviceManager.ContentService.DeleteContentAsync(ContentId, cancellationToken);
			await cacheService.RemoveByPatternAsync(ContentsPattern);
			return Ok();
		}
		[HasPermission(Permissions.AddContent)]
		[HttpPost("{contentId}/attachments")]
		[RequestSizeLimit(524288000)] // 500 MB
		[RequestFormLimits(MultipartBodyLengthLimit = 524288000)]
		public async Task<IActionResult> AddAttachmentToContent(
			int contentId, 
			[FromForm] List<IFormFile?> attachmentFiles, CancellationToken cancellationToken)
		{
			var updatedContent = await serviceManager.ContentService
				.AddAttachmentToContent(contentId, attachmentFiles, cancellationToken);
			await cacheService.RemoveByPatternAsync(ContentsPattern);
			return Ok(updatedContent);
		}
		[HasPermission(Permissions.DeleteContent)]
		[HttpDelete("attachments/{attachmentId}")]
		public async Task<IActionResult> RemoveAttachmentFromContent(Guid attachmentId, CancellationToken cancellationToken)
		{
			await serviceManager.ContentService.RemoveAttachment(attachmentId, cancellationToken);
			await cacheService.RemoveByPatternAsync(ContentsPattern);
			return Ok();
		}
	}
}
