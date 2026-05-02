using Shared.Dtos.DepartmentDto.Request;
using Shared.Dtos.DepartmentDto.Response;

namespace ServiceAbstractionLayer
{
    public interface IDepartmentService
    {
        Task<IEnumerable<DepartmentResponse>> GetAllDepartmentsAsync(CancellationToken cancellationToken = default);
        Task<DepartmentResponse> GetDepartmentByIdAsync(int departmentId, CancellationToken cancellationToken = default);
        Task<DepartmentResponse> AddDepartmentAsync(DepartmentRequest request, CancellationToken cancellationToken = default);
        Task UpdateDepartmentAsync(int departmentId, DepartmentRequest request, CancellationToken cancellationToken = default);
        Task DeleteDepartmentAsync(int departmentId, CancellationToken cancellationToken = default);
        Task AssignCourseToDepartmentAsync(int departmentId, int CourseId, CancellationToken cancellationToken = default);
        Task RemoveCourseFromDepartmentAsync(int departmentId, int CourseId, CancellationToken cancellationToken = default);
    }
}
