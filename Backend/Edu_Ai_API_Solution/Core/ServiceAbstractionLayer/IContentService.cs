namespace ServiceAbstractionLayer
{
	public interface IContentService
	{
		Task<ContentResponseDto> AddContentForCourse(int courseId, ContentRequestDto contentRequest, CancellationToken cancellationToken = default);
		Task UpdateContentForCourse(int contentId, ContentRequestDto contentRequest, CancellationToken cancellationToken = default);
		Task<IEnumerable<ContentResponseDto>> GetAllContentsByCourseIdAsync(int courseId, CancellationToken cancellationToken = default);
		Task<ContentResponseDto> GetContentByIdAsync(int contentId, CancellationToken cancellationToken = default);
		Task DeleteContentAsync(int contentId, CancellationToken cancellationToken = default);
		Task RemoveAttachment(Guid AttachmentId, CancellationToken cancellationToken = default);
		Task<ContentResponseDto> AddAttachmentToContent(int ContentId, List<IFormFile?> Files, CancellationToken cancellationToken = default);
	}
}
