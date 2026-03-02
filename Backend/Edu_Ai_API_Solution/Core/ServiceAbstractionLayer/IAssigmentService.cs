

namespace ServiceAbstractionLayer
{
	public interface IAssigmentService
	{
		Task<AssigmentResponseDto> CreateOrUpdateAssigmentForCourse(int courseId, AssigmentRequestDto assigmentRequest);
		Task<IEnumerable<AssigmentResponseDto>> GetAllAssigmentsByCourseIdAsync(int courseId);
		Task<AssigmentResponseDto> GetAssigmentByIdAsync(int AssigmentId);
		Task DeleteAssigmentAsync(int AssigmentId);
		Task RemoveAttachment(Guid AttachmentId);
		Task<AssigmentResponseDto> AddAttachmentToAssigment(int AssigmentId, List<IFormFile?> Files);
	}
}
