


using Shared.Dtos.DepartmentDto.Request;
using Shared.Dtos.DepartmentDto.Response;

namespace ServiceAbstractionLayer
{
	public interface IDepartmentService
	{
		Task<IEnumerable<DepartmentResponse>> GetAllDepartmentsAsync();
		Task<DepartmentResponse> GetDepartmentByIdAsync(int departmentId);
		Task<DepartmentResponse> AddDepartmentAsync(DepartmentRequest request);
		Task UpdateDepartmentAsync(int departmentId, DepartmentRequest request);
		Task DeleteDepartmentAsync(int departmentId);
		Task AssignCourseToDepartmentAsync(int departmentId, int CourseId);
		Task RemoveCourseFromDepartmentAsync(int departmentId, int CourseId);

	}
}
