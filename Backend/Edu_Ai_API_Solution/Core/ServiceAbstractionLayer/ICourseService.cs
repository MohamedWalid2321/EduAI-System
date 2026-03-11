namespace ServiceAbstractionLayer
{
	public interface ICourseService
	{
		Task<IEnumerable<CourseResponseDto>> GetAllCourseforDepartmentAsync(int departmentId);
		Task<IEnumerable<CourseResponseDto>> GetAllStudentCourse(string UserId);
		Task<IEnumerable<CourseResponseDto>> GetAllCourseAsync();
		Task<FullCourseResponse> GetCourseByIdAsync(int departmentId, int courseId);
		Task<CourseResponseDto> AddCourseAsync(int departmentId, CourseRequestDto request , IFormFile? ImageFile);
		Task UpdateCourseAsync(int departmentId, int courseId,CourseRequestDto request, IFormFile? ImageFile);
		Task ToggleCouresStatus(int CourseId);
		Task<FullCourseResponse> AddAssesment(int CourseId, List<AssesmentDto> assesments);
		Task UpdateAssesment(int CourseId, List<AssesmentDto> assesments);
		Task DeleteCourseAsync(int courseId);

		// Instructor enrollment methods
		Task<InstructorCourseResponse> EnrollInstructorAsync(int courseId, string instructorId, string assignedBy);
		Task UnenrollInstructorAsync(int courseId, string instructorId);
		Task<IEnumerable<InstructorCourseResponse>> GetCourseInstructorsAsync(int courseId);
		Task<IEnumerable<CourseResponseDto>> GetInstructorCoursesAsync(string instructorId);
	}
}
