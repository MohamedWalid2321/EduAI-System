using DomainLayer.Models;
using Shared.Dtos.AssesmentDto;

namespace PresentationLayer.Controllers
{
    public class CourseController(IServiceManager serviceManager, ICacheService _cacheService) : ApiControllerBase
    {
        private const string CoursesPatternCacheKey = "/api/course*|user:*";

        [HttpGet("")]
        [Cache(300)]
        [HasPermission(Permissions.GetCourse)]
        public async Task<IActionResult> GetCourses(CancellationToken cancellationToken)
        {
            var courses = await serviceManager.CourseService.GetCoursesAsync(User.GetUserId()!, User.GetDepartmentIdOrThrow(), cancellationToken);
            return Ok(courses);
        }

        [HttpGet("All")]
        [Cache(300)]
        [HasPermission(Permissions.GetAllCourses)]
        public async Task<IActionResult> GetAllCourse(CancellationToken cancellationToken)
        {
            var courses = await serviceManager.CourseService.GetAllCourseAsync(cancellationToken);
            return Ok(courses);
        }

        [HttpGet("{Courseid}/FromDepartment/{departmentId}")]
        [HasPermission(Permissions.GetCourse)]
        public async Task<IActionResult> GetCourseById(int departmentId, int Courseid, CancellationToken cancellationToken)
        {
            var course = await serviceManager.CourseService.GetCourseByIdAsync(departmentId, Courseid, cancellationToken);
            return Ok(course);
        }

        [HttpPost("{departmentId}")]
        [HasPermission(Permissions.AddCourse)]
        public async Task<IActionResult> AddCourse(int departmentId, [FromForm] CourseRequestDto courseDto, IFormFile ImageFile, CancellationToken cancellationToken)
        {
            var createdCourse = await serviceManager.CourseService.AddCourseAsync(departmentId, courseDto, ImageFile, cancellationToken);
            await _cacheService.RemoveByPatternAsync(CoursesPatternCacheKey);
            return CreatedAtAction(nameof(GetCourseById), new { departmentId, Courseid = createdCourse.Id }, createdCourse);
        }

        [HttpPut("{id}")]
        [HasPermission(Permissions.UpdateCourse)]
        public async Task<IActionResult> UpdateCourse(int id, [FromForm] CourseRequestDto courseDto, IFormFile? ImageFile, CancellationToken cancellationToken)
        {
            await serviceManager.CourseService.UpdateCourseAsync(id, courseDto, ImageFile, cancellationToken);
            await _cacheService.RemoveByPatternAsync(CoursesPatternCacheKey);
            return Ok();
        }

        [HttpPut("{CourseId}/Toggle_Status")]
        [HasPermission(Permissions.UpdateCourse)]
        public async Task<IActionResult> ToggleCourseStatus(int CourseId, CancellationToken cancellationToken)
        {
            await serviceManager.CourseService.ToggleCouresStatus(CourseId, cancellationToken);
            await _cacheService.RemoveByPatternAsync(CoursesPatternCacheKey);
            return Ok();
        }

        [HttpPost("{CourseId}/AddAssesment")]
        [HasPermission(Permissions.AddCourse)]
        public async Task<IActionResult> AddAssesment(int CourseId, List<AssesmentRequest> assesments, CancellationToken cancellationToken)
        {
            var updatedCourse = await serviceManager.CourseService.AddAssesment(CourseId, assesments, cancellationToken);
            return Ok(updatedCourse);
        }

        [HttpPut("{CourseId}/UpdateAssesment")]
        [HasPermission(Permissions.UpdateCourse)]
        public async Task<IActionResult> UpdateAssesment(int CourseId, List<AssesmentRequest> assesments, CancellationToken cancellationToken)
        {
            await serviceManager.CourseService.UpdateAssesment(CourseId, assesments, cancellationToken);
            return Ok();
        }

        [HttpDelete("{id}")]
        [HasPermission(Permissions.DeleteCourse)]
        public async Task<IActionResult> DeleteCourse(int id, CancellationToken cancellationToken)
        {
            await serviceManager.CourseService.DeleteCourseAsync(id, cancellationToken);
            await _cacheService.RemoveByPatternAsync(CoursesPatternCacheKey);
            return Ok();
        }

        // User enrollment endpoints (unified for all IsEnrolled roles)

        [HttpPost("{courseId}/users")]
        [HasPermission(Permissions.EnrollInstructor)]
        public async Task<IActionResult> ManualEnrollUser(int courseId, [FromBody] EnrollUserRequest request, CancellationToken cancellationToken)
        {
            var enrolledBy = User.GetUserId()!;
            var result = await serviceManager.CourseService.ManualEnrollUserAsync(courseId, request.UserId, enrolledBy, cancellationToken);
            return CreatedAtAction(nameof(GetCourseEnrolledUsers), new { courseId }, result);
        }

        [HttpDelete("{courseId}/users/{userId}")]
        [HasPermission(Permissions.UnenrollInstructor)]
        public async Task<IActionResult> ManualUnenrollUser(int courseId, string userId, CancellationToken cancellationToken)
        {
            await serviceManager.CourseService.ManualUnenrollUserAsync(courseId, userId, cancellationToken);
            return NoContent();
        }

        [HttpGet("{courseId}/users")]
        [HasPermission(Permissions.GetCourse)]
        public async Task<IActionResult> GetCourseEnrolledUsers(int courseId, CancellationToken cancellationToken)
        {
            var users = await serviceManager.CourseService.GetCourseEnrolledUsersAsync(courseId, cancellationToken);
            return Ok(users);
        }

        [HttpGet("users/{userId}/courses")]
        [HasPermission(Permissions.GetCourse)]
        public async Task<IActionResult> GetUserEnrolledCourses(string userId, CancellationToken cancellationToken)
        {
            var courses = await serviceManager.CourseService.GetUserEnrolledCoursesAsync(userId, cancellationToken);
            return Ok(courses);
        }

        [HttpGet("{courseId}/assessments")]
        [Cache(300)]
        [HasPermission(Permissions.GetAssesment)]
        public async Task<IActionResult> GetAssessmentsByCourseId(int courseId, CancellationToken cancellationToken)
        {
            var assessments = await serviceManager.CourseService.GetAssessmentsByCourseIdAsync(courseId, cancellationToken);
            return Ok(assessments);
        }
    }
}
