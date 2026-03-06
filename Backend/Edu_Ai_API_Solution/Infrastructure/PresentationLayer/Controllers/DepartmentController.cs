using Shared.Dtos.DepartmentDto.Request;

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
			return Ok(department);
		}
		[HttpPost]
		public async Task<IActionResult> AddDepartment([FromBody] DepartmentRequest request)
		{
			var createdDepartment = await serviceManager.DepartmentService.AddDepartmentAsync(request);
			return CreatedAtAction(nameof(GetDepartmentById), new { id = createdDepartment.Id }, createdDepartment);
		}
		[HttpPut("{id}")]
		public async Task<IActionResult> UpdateDepartment(int id, [FromBody] DepartmentRequest request)
		{
			await serviceManager.DepartmentService.UpdateDepartmentAsync(id, request);
			return Ok();
		}
		[HttpDelete("{id}")]
		public async Task<IActionResult> DeleteDepartment(int id)
		{
			await serviceManager.DepartmentService.DeleteDepartmentAsync(id);
			return Ok();
		}
		[HttpPost("{departmentId}/AssignCourse/{courseId}")]
		public async Task<IActionResult> AssignCourseToDepartment(int departmentId, int courseId)
		{
			 await serviceManager.DepartmentService.AssignCourseToDepartmentAsync(departmentId, courseId);
			return Ok();
		}
		[HttpPost("{departmentId}/RemoveCourse/{courseId}")]
		public async Task<IActionResult> RemoveCourseFromDepartment(int departmentId, int courseId)
		{
			await serviceManager.DepartmentService.RemoveCourseFromDepartmentAsync(departmentId, courseId);
			return Ok();
		}

	}
}
