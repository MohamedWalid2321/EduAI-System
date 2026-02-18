


namespace ServiceAbstractionLayer
{
	public interface IContentService
	{
		Task<ContentResponseDto> CreateOrUpdateContentForCourse(int courseId, ContentRequestDto contentRequest);
		Task<IEnumerable<ContentResponseDto>> GetAllContentsByCourseIdAsync(int courseId);
		Task<ContentResponseDto> GetContentByIdAsync(int contentId);
		Task DeleteContentAsync(int contentId);
		Task RemoveAttachment(Guid AttachmentId);
		Task<ContentResponseDto> AddAttachmentToContent(int ContentId, List<IFormFile?> Files);
	}
}
