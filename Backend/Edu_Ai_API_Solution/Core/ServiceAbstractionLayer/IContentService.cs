using Microsoft.AspNetCore.Http;
using Shared.Dtos.ContentDto.ContentRequest;
using Shared.Dtos.ContentDto.ContentResponse;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
	public interface IContentService
	{
		Task<ContentResponseDto> CreateOrUpdateContentForCourse(int courseId, ContentRequestDto contentRequest,List<IFormFile?> Files);
		Task<IEnumerable<ContentResponseDto>> GetAllContentsByCourseIdAsync(int courseId);
		Task<ContentResponseDto> GetContentByIdAsync(int contentId);
		Task DeleteContentAsync(int contentId);
		Task RemoveAttachment(Guid AttachmentId);
		Task<ContentResponseDto> AddAttachmentToContent(int ContentId, List<IFormFile?> Files);
	}
}
