
using DomainLayer.Models;

namespace PresentationLayer.Controllers
{
	public class CourseController(IServiceManager serviceManager): ApiControllerBase
	{
		//[HttpGet("{departmentId}")]
		////[Cache(300)]
		//public async Task<IActionResult> GetAllCourse(int departmentId)
		//{
		//	var courses = await serviceManager.CourseService.GetAllCourseforDepartmentAsync(departmentId);
		//	return Ok(courses);

		//}
		[HttpGet("")]
		[Authorize]
		//[Cache(300)]
		public async Task<IActionResult> GetAllCourseByDepartmentId()
		{
			var courses = await serviceManager.CourseService.GetAllCourseforDepartmentAsync(User.GetDepartmentIdOrThrow());
			return Ok(courses);

		}
		[HttpGet("StudenCourse")]
		[Authorize]
		public async Task<IActionResult> GetAllCourseForStudent()
		{
			var courses = await serviceManager.CourseService.GetAllStudentCourse(User.GetUserId()!);
			return Ok(courses);
		}

		[HttpGet("All")]
		public async Task<IActionResult> GetAllCourse()
		{
			var courses = await serviceManager.CourseService.GetAllCourseAsync();
			return Ok(courses);
		}
		[HttpGet("{Courseid}/FromDepartment/{departmentId}")]
		public async Task<IActionResult> GetCourseById(int departmentId, int Courseid)
		{
			var course = await serviceManager.CourseService.GetCourseByIdAsync(departmentId, Courseid);
			return Ok(course);
		}
		[HttpPost("{departmentId}")]
		public async Task<IActionResult> AddCourse(int departmentId, [FromForm] CourseRequestDto courseDto,IFormFile ImageFile)
		{
			var createdCourse = await serviceManager.CourseService.AddCourseAsync(departmentId, courseDto,ImageFile);
			return CreatedAtAction(nameof(GetCourseById), new { departmentId = departmentId, Courseid = createdCourse.Id }, createdCourse);
		}
		
		[HttpPut("{departmentId}/{id}")]
		public async Task<IActionResult> UpdateCourse(
			int departmentId,
			int id,
			[FromForm] CourseRequestDto courseDto,
			IFormFile? ImageFile)
		{
			await serviceManager.CourseService.UpdateCourseAsync(departmentId,id, courseDto, ImageFile);
			return Ok();
		}
		[HttpPut("{CourseId}/Toggle_Status")]
		public async Task<IActionResult> ToggleCourseStatus(int CourseId)
		{
			await serviceManager.CourseService.ToggleCouresStatus(CourseId);
			return Ok();
		}
		[HttpPost("{CourseId}/AddAssesment")]
		public async Task<IActionResult> AddAssesment(int CourseId, List<AssesmentDto> assesments)
		{
			var updatedCourse = await serviceManager.CourseService.AddAssesment(CourseId, assesments);
			return Ok(updatedCourse);
		}
		[HttpPut("{CourseId}/UpdateAssesment")]
		public async Task<IActionResult> UpdateAssesment(int CourseId, List<AssesmentDto> assesments)
		{
			await serviceManager.CourseService.UpdateAssesment(CourseId, assesments);
			return Ok();
		}

		[HttpDelete("{id}")]
		public async Task<IActionResult> DeleteCourse(int id)
		{
			await serviceManager.CourseService.DeleteCourseAsync(id);
			return Ok();
		}


	}
}
