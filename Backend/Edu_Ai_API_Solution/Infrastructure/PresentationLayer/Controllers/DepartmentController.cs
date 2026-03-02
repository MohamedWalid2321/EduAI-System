namespace PresentationLayer.Controllers
{
	//[Authorize]
	public class DepartmentController(IServiceManager serviceManager):ApiControllerBase
	{
		[HttpGet]
		public async Task<IActionResult> GetDepartments()
		{
			var departments = await serviceManager.DepartmentService.GetAllDepartmentsAsync();
			return Ok(departments);
		}
		[HttpGet("{id}")]
		public async Task<IActionResult> GetDepartmentById(int id)
		{
			var department = await serviceManager.DepartmentService.GetDepartmentByIdAsync(id);
			if (department is null)
			{
				return NotFound();
			}
			return Ok(department);
		}
		[HttpPost]
		public async Task<IActionResult> CreateOrUpdateDepartment([FromBody] DepartmentDto departmentDto)
		{
			var createdOrUpdatedDepartment = await serviceManager.DepartmentService.CreateOrUpdateDepartmentAsync(departmentDto);
			return Ok(createdOrUpdatedDepartment);
		}
		[HttpDelete("{id}")]
		public async Task<IActionResult> DeleteDepartment(int id)
		{
			await serviceManager.DepartmentService.DeleteDepartmentAsync(id);
			return Ok();
		}
		[HttpGet("Course/{departmetnId}")]
		public async Task<IActionResult> GetAllCourseBydepartmentId(int departmetnId)
		{
			var courses = await serviceManager.DepartmentService.GetAllCourseBydepartmentIdAsync(departmetnId);
			return Ok(courses);
		}
		[HttpPost("{departmentId}/AssignCourse/{courseId}")]
		public async Task<IActionResult> AssignCourseToDepartment(int departmentId, int courseId)
		{
			var updatedDepartment = await serviceManager.DepartmentService.AssignCourseToDepartmentAsync(departmentId, courseId);
			return Ok(updatedDepartment);
		}
		[HttpPost("{departmentId}/RemoveCourse/{courseId}")]
		public async Task<IActionResult> RemoveCourseFromDepartment(int departmentId, int courseId)
		{
			var updatedDepartment = await serviceManager.DepartmentService.RemoveCourseFromDepartmentAsync(departmentId, courseId);
			return Ok(updatedDepartment);
		}

	}
}
