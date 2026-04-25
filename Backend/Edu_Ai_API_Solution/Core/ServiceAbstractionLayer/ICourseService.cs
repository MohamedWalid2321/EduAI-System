namespace ServiceAbstractionLayer
{
	public interface ICourseService
	{
		Task<IEnumerable<CourseResponseDto>> GetAllCourseforDepartmentAsync(int departmentId);
		Task<IEnumerable<CourseResponseDto>> GetAllCourseAsync();
		Task<IEnumerable<CourseResponseDto>> GetCoursesAsync(string userId, int? departmentId);
		Task<IEnumerable<CourseResponseDto>> GetUserEnrolledCoursesAsync(string userId);
		Task<FullCourseResponse> GetCourseByIdAsync(int departmentId, int courseId);
		Task<CourseResponseDto> AddCourseAsync(int departmentId, CourseRequestDto request, IFormFile? ImageFile);
		Task UpdateCourseAsync(int departmentId, int courseId, CourseRequestDto request, IFormFile? ImageFile);
		Task ToggleCouresStatus(int CourseId);
		Task<FullCourseResponse> AddAssesment(int CourseId, List<AssesmentDto> assesments);
		Task UpdateAssesment(int CourseId, List<AssesmentDto> assesments);
		Task DeleteCourseAsync(int courseId);

		// User enrollment (unified for all IsEnrolled roles)
		Task<UserCourseResponse> ManualEnrollUserAsync(int courseId, string userId, string enrolledBy);
		Task ManualUnenrollUserAsync(int courseId, string userId);
		Task<IEnumerable<UserCourseResponse>> GetCourseEnrolledUsersAsync(int courseId);
	}
}
