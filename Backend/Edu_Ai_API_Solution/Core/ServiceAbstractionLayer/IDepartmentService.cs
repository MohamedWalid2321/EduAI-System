using Shared.Dtos;
using Shared.Dtos.CourseDto.Request;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
	public interface IDepartmentService
	{
		Task<IEnumerable<DepartmentDto>> GetAllDepartmentsAsync();
		Task<DepartmentDto?> GetDepartmentByIdAsync(int departmentId);
		Task<DepartmentDto> CreateOrUpdateDepartmentAsync(DepartmentDto createDepartmentDto);
		Task DeleteDepartmentAsync(int departmentId);
		Task<IEnumerable<CourseRequestDto>> GetAllCourseBydepartmentIdAsync(int departmetnId);
		Task<DepartmentDto> AssignCourseToDepartmentAsync(int departmentId, int CourseId);
		Task<DepartmentDto> RemoveCourseFromDepartmentAsync(int departmentId, int CourseId);

	}
}
