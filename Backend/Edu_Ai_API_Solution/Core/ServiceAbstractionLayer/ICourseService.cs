using Shared.Dtos.AssesmentDto;
using Shared.Dtos.AssesmentDto.AssesmentDto;

namespace ServiceAbstractionLayer
{
	public interface ICourseService
	{
		Task<IEnumerable<CourseResponseDto>> GetAllCourseforDepartmentAsync(int departmentId, CancellationToken cancellationToken = default);
		Task<IEnumerable<CourseResponseDto>> GetAllCourseAsync(CancellationToken cancellationToken = default);
		Task<IEnumerable<CourseResponseDto>> GetCoursesAsync(string userId, int? departmentId, CancellationToken cancellationToken = default);
		Task<IEnumerable<CourseResponseDto>> GetUserEnrolledCoursesAsync(string userId, CancellationToken cancellationToken = default);
		Task<FullCourseResponse> GetCourseByIdAsync(int departmentId, int courseId, CancellationToken cancellationToken = default);
		Task<CourseResponseDto> AddCourseAsync(int departmentId, CourseRequestDto request, IFormFile? ImageFile, CancellationToken cancellationToken = default);
		Task UpdateCourseAsync(int courseId, CourseRequestDto request, IFormFile? ImageFile, CancellationToken cancellationToken = default);
		Task ToggleCouresStatus(int CourseId, CancellationToken cancellationToken = default);
		Task<FullCourseResponse> AddAssesment(int CourseId, List<AssesmentRequest> assesments, CancellationToken cancellationToken = default);
		Task UpdateAssesment(int CourseId, List<AssesmentRequest> assesments, CancellationToken cancellationToken = default);
		Task DeleteCourseAsync(int courseId, CancellationToken cancellationToken = default);
		Task<UserCourseResponse> ManualEnrollUserAsync(int courseId, string userId, string enrolledBy, CancellationToken cancellationToken = default);
		Task ManualUnenrollUserAsync(int courseId, string userId, CancellationToken cancellationToken = default);
		Task<IEnumerable<UserCourseResponse>> GetCourseEnrolledUsersAsync(int courseId, CancellationToken cancellationToken = default);
		Task<IEnumerable<AssessmentResponseDto>> GetAssessmentsByCourseIdAsync(int courseId, CancellationToken cancellationToken = default);
	}
}
