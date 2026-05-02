using Shared.Dtos.DepartmentDto.Request;
using System.Threading;

namespace PresentationLayer.Controllers
{
    // [Authorize]
    public class DepartmentController(IServiceManager serviceManager) : ApiControllerBase
    {
        [HttpGet]
        public async Task<IActionResult> GetDepartments(CancellationToken cancellationToken)
        {
            var departments = await serviceManager.DepartmentService.GetAllDepartmentsAsync(cancellationToken);
            return Ok(departments);
        }

        [HttpGet("{id}")]
        public async Task<IActionResult> GetDepartmentById(int id, CancellationToken cancellationToken)
        {
            var department = await serviceManager.DepartmentService.GetDepartmentByIdAsync(id, cancellationToken);
            return Ok(department);
        }

        [HttpPost]
        public async Task<IActionResult> AddDepartment([FromBody] DepartmentRequest request, CancellationToken cancellationToken)
        {
            var createdDepartment = await serviceManager.DepartmentService.AddDepartmentAsync(request, cancellationToken);
            return CreatedAtAction(nameof(GetDepartmentById), new { id = createdDepartment.Id }, createdDepartment);
        }

        [HttpPut("{id}")]
        public async Task<IActionResult> UpdateDepartment(int id, [FromBody] DepartmentRequest request, CancellationToken cancellationToken)
        {
            await serviceManager.DepartmentService.UpdateDepartmentAsync(id, request, cancellationToken);
            return Ok();
        }

        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteDepartment(int id, CancellationToken cancellationToken)
        {
            await serviceManager.DepartmentService.DeleteDepartmentAsync(id, cancellationToken);
            return Ok();
        }

        [HttpPost("{departmentId}/AssignCourse/{courseId}")]
        public async Task<IActionResult> AssignCourseToDepartment(int departmentId, int courseId, CancellationToken cancellationToken)
        {
            await serviceManager.DepartmentService.AssignCourseToDepartmentAsync(departmentId, courseId, cancellationToken);
            return Ok();
        }

        [HttpPost("{departmentId}/RemoveCourse/{courseId}")]
        public async Task<IActionResult> RemoveCourseFromDepartment(int departmentId, int courseId, CancellationToken cancellationToken)
        {
            await serviceManager.DepartmentService.RemoveCourseFromDepartmentAsync(departmentId, courseId, cancellationToken);
            return Ok();
        }
    }
}
