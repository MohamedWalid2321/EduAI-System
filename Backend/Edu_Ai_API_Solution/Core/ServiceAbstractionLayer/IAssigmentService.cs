namespace ServiceAbstractionLayer
{
	public interface IAssigmentService
	{
		Task<AssigmentResponseDto> CreateOrUpdateAssigmentForCourse(int courseId, AssigmentRequestDto assigmentRequest, CancellationToken cancellationToken = default);
		Task<IEnumerable<AssigmentResponseDto>> GetAllAssigmentsByCourseIdAsync(int courseId, CancellationToken cancellationToken = default);
		Task<AssigmentResponseDto> GetAssigmentByIdAsync(int AssigmentId, CancellationToken cancellationToken = default);
		Task DeleteAssigmentAsync(int AssigmentId, CancellationToken cancellationToken = default);
		Task RemoveAttachment(Guid AttachmentId, CancellationToken cancellationToken = default);
		Task<AssigmentResponseDto> AddAttachmentToAssigment(int AssigmentId, List<IFormFile?> Files, CancellationToken cancellationToken = default);
	}
}
