namespace ServiceAbstractionLayer
{
	public interface ICourseService
	{

		Task<IEnumerable<FullCourseResponse>> GetAllCourseAsync(int departmentId);
		Task<CourseResponseDto> GetCourseByIdAsync(int departmentId, int courseId);
		Task<CourseResponseDto> AddCourseAsync(int departmentId, CourseRequestDto request , IFormFile? ImageFile);
		Task UpdateCourseAsync(int departmentId, int courseId,CourseRequestDto request, IFormFile? ImageFile);
		Task<FullCourseResponse> AddAssesment(int CourseId, List<AssesmentDto> assesments);
		Task UpdateAssesment(int CourseId, List<AssesmentDto> assesments);
		Task DeleteCourseAsync(int courseId);


	}
}
