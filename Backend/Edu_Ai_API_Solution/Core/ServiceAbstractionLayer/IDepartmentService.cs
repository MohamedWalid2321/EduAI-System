


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
		Task<IEnumerable<CourseRequestDto>> GetAllCourseBydepartmentIdAsync(int departmetnId);
		Task<DepartmentRequest> AssignCourseToDepartmentAsync(int departmentId, int CourseId);
		Task<DepartmentRequest> RemoveCourseFromDepartmentAsync(int departmentId, int CourseId);

	}
}
