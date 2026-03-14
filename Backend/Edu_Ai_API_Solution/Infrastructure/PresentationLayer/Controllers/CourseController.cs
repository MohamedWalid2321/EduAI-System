using DomainLayer.Models;

namespace PresentationLayer.Controllers
{
	public class CourseController(IServiceManager serviceManager , ICacheService _cacheService) : ApiControllerBase
	{
		private const string AllCoursesCacheKey = "/api/course/All";
		private const string CoursesPatternCacheKey = "/api/course|user:*";
		[HttpGet("")]
		[Cache(300)]
		[HasPermission(Permissions.GetCourse)]
		public async Task<IActionResult> GetCourses()
		{
			var courses = await serviceManager.CourseService.GetCoursesAsync(User.GetUserId()!, User.GetDepartmentIdOrThrow());
			return Ok(courses);
		}

		[HttpGet("All")]
		[Cache(300)]
		[HasPermission(Permissions.GetAllCourses)]
		public async Task<IActionResult> GetAllCourse()
		{
			var courses = await serviceManager.CourseService.GetAllCourseAsync();
			return Ok(courses);
		}

		[HttpGet("{Courseid}/FromDepartment/{departmentId}")]
		[HasPermission(Permissions.GetCourse)]
		public async Task<IActionResult> GetCourseById(int departmentId, int Courseid)
		{
			var course = await serviceManager.CourseService.GetCourseByIdAsync(departmentId, Courseid);
			return Ok(course);
		}

		[HttpPost("{departmentId}")]
		[HasPermission(Permissions.AddCourse)]
		public async Task<IActionResult> AddCourse(int departmentId, [FromForm] CourseRequestDto courseDto, IFormFile ImageFile)
		{
			var createdCourse = await serviceManager.CourseService.AddCourseAsync(departmentId, courseDto, ImageFile);
			await _cacheService.RemoveByPatternAsync(CoursesPatternCacheKey);
			await _cacheService.RemoveAsync(AllCoursesCacheKey);
			return CreatedAtAction(nameof(GetCourseById), new { departmentId, Courseid = createdCourse.Id }, createdCourse);
		}

		[HttpPut("{departmentId}/{id}")]
		[HasPermission(Permissions.UpdateCourse)]
		public async Task<IActionResult> UpdateCourse(int departmentId, int id, [FromForm] CourseRequestDto courseDto, IFormFile? ImageFile)
		{
			await serviceManager.CourseService.UpdateCourseAsync(departmentId, id, courseDto, ImageFile);
			await _cacheService.RemoveByPatternAsync(CoursesPatternCacheKey);
			await _cacheService.RemoveAsync(AllCoursesCacheKey);
			return Ok();
		}

		[HttpPut("{CourseId}/Toggle_Status")]
		[HasPermission(Permissions.UpdateCourse)]
		public async Task<IActionResult> ToggleCourseStatus(int CourseId)
		{
			await serviceManager.CourseService.ToggleCouresStatus(CourseId);
			await _cacheService.RemoveByPatternAsync(CoursesPatternCacheKey);
			await _cacheService.RemoveAsync(AllCoursesCacheKey);
			return Ok();
		}

		[HttpPost("{CourseId}/AddAssesment")]
		[HasPermission(Permissions.AddCourse)]
		public async Task<IActionResult> AddAssesment(int CourseId, List<AssesmentDto> assesments)
		{
			var updatedCourse = await serviceManager.CourseService.AddAssesment(CourseId, assesments);
			return Ok(updatedCourse);
		}

		[HttpPut("{CourseId}/UpdateAssesment")]
		[HasPermission(Permissions.UpdateCourse)]
		public async Task<IActionResult> UpdateAssesment(int CourseId, List<AssesmentDto> assesments)
		{
			await serviceManager.CourseService.UpdateAssesment(CourseId, assesments);
			return Ok();
		}

		[HttpDelete("{id}")]
		[HasPermission(Permissions.DeleteCourse)]
		public async Task<IActionResult> DeleteCourse(int id)
		{
			await serviceManager.CourseService.DeleteCourseAsync(id);
			await _cacheService.RemoveByPatternAsync(CoursesPatternCacheKey);
			await _cacheService.RemoveAsync(AllCoursesCacheKey);
			return Ok();
		}

		// User enrollment endpoints (unified for all IsEnrolled roles)

		[HttpPost("{courseId}/users")]
		[HasPermission(Permissions.EnrollInstructor)]
		public async Task<IActionResult> ManualEnrollUser(int courseId, [FromBody] EnrollUserRequest request)
		{
			var enrolledBy = User.GetUserId()!;
			var result = await serviceManager.CourseService.ManualEnrollUserAsync(courseId, request.UserId, enrolledBy);
			return CreatedAtAction(nameof(GetCourseEnrolledUsers), new { courseId }, result);
		}

		[HttpDelete("{courseId}/users/{userId}")]
		[HasPermission(Permissions.UnenrollInstructor)]
		public async Task<IActionResult> ManualUnenrollUser(int courseId, string userId)
		{
			await serviceManager.CourseService.ManualUnenrollUserAsync(courseId, userId);
			return NoContent();
		}

		[HttpGet("{courseId}/users")]
		[HasPermission(Permissions.GetCourse)]
		public async Task<IActionResult> GetCourseEnrolledUsers(int courseId)
		{
			var users = await serviceManager.CourseService.GetCourseEnrolledUsersAsync(courseId);
			return Ok(users);
		}

		[HttpGet("users/{userId}/courses")]
		[HasPermission(Permissions.GetCourse)]
		public async Task<IActionResult> GetUserEnrolledCourses(string userId)
		{
			var courses = await serviceManager.CourseService.GetUserEnrolledCoursesAsync(userId);
			return Ok(courses);
		}
	}
}
