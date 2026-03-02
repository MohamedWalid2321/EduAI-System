namespace ServiceAbstractionLayer
{
	public interface ICourseService
	{
		Task<CourseResponseDto> CreateOrUpdateCourseAsync(CourseRequestDto courseDto , IFormFile? ImageFile);
		Task<IEnumerable<CourseResponseDto>> GetAllCourseAsync();
		Task<CourseResponseDto> GetCourseByIdAsync(int courseId);
		Task DeleteCourseAsync(int courseId);


	}
}
