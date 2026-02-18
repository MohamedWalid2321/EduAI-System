
namespace PresentationLayer.Controllers
{
	public class CourseController(IServiceManager serviceManager): ApiControllerBase
	{
		[HttpPost]
		public async Task<IActionResult> CreateOrUpdateCourse([FromForm] CourseRequestDto courseDto,IFormFile ImageFile)
		{
			var createdOrUpdatedCourse = await serviceManager.CourseService.CreateOrUpdateCourseAsync(courseDto,ImageFile);
			return Ok(createdOrUpdatedCourse);
		}
		[HttpGet]
		[Cache(300)]
		public async Task<IActionResult> GetAllCourse()
		{
			var courses = await serviceManager.CourseService.GetAllCourseAsync();
			return Ok(courses);
		}
		[HttpGet("{id}")]
		public async Task<IActionResult> GetCourseById(int id)
		{
			var course = await serviceManager.CourseService.GetCourseByIdAsync(id);
			return Ok(course);
		}
		[HttpDelete("{id}")]
		public async Task<IActionResult> DeleteCourse(int id)
		{
			await serviceManager.CourseService.DeleteCourseAsync(id);
			return Ok();
		}


	}
}
