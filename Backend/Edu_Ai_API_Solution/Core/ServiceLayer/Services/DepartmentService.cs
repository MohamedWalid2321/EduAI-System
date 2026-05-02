using Shared.Dtos.DepartmentDto.Request;
using Shared.Dtos.DepartmentDto.Response;

namespace ServiceLayer.Services
{
	public class DepartmentService(IUnitOfWork unitOfWork) : IDepartmentService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		public async Task<IEnumerable<DepartmentResponse>> GetAllDepartmentsAsync(CancellationToken cancellationToken = default)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departments = await departmentRepository.GetAllAsync(cancellationToken);
			var departmentDtos = departments.Adapt<IEnumerable<DepartmentResponse>>();
			return departmentDtos;
		}
		public async Task<DepartmentResponse> GetDepartmentByIdAsync(int departmentId, CancellationToken cancellationToken = default)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentSpecificatioin = new DepartmentSpecification(departmentId);
			var DepartmentEntity = await departmentRepository.GetByIdAsync(departmentSpecificatioin, cancellationToken);
			if (DepartmentEntity is null)
			{
				throw new DepartmentNotFoundException(departmentId);
			}
			return DepartmentEntity.Adapt<DepartmentResponse>();
		}
		public async Task<DepartmentResponse> AddDepartmentAsync(DepartmentRequest request, CancellationToken cancellationToken = default)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var DuplicateDepartmentSpecification = new DepartmentByTitleSpecification(request.Title);
			if (await departmentRepository.GetByIdAsync(DuplicateDepartmentSpecification, cancellationToken) is not null)
			{
				throw new DuplicatedDepartmentException();
			}
			var DepartmentEntity = request.Adapt<Department>();
			await departmentRepository.AddAsync(DepartmentEntity, cancellationToken);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
			return DepartmentEntity.Adapt<DepartmentResponse>();
		}
		public async Task UpdateDepartmentAsync(int departmentId, DepartmentRequest request, CancellationToken cancellationToken = default)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentSpecificatioin = new DepartmentSpecification(departmentId);
			var DepartmentEntity = await departmentRepository.GetByIdAsync(departmentSpecificatioin, cancellationToken);
			if (DepartmentEntity is null)
			{
				throw new DepartmentNotFoundException(departmentId);
			}
			var DuplicateDepartmentSpecification = new DepartmentByTitleSpecification(request.Title);
			var duplicateDepartment = await departmentRepository.GetByIdAsync(DuplicateDepartmentSpecification, cancellationToken);
			if (duplicateDepartment is not null && duplicateDepartment.Id != departmentId)
			{
				throw new DuplicatedDepartmentException();
			}
			DepartmentEntity.Title = request.Title;
			departmentRepository.Update(DepartmentEntity);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}

		public async Task DeleteDepartmentAsync(int departmentId, CancellationToken cancellationToken = default)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var DepartmentEntity = await departmentRepository.GetByIdAsync(departmentId, cancellationToken);
			if (DepartmentEntity is null)
			{
				throw new DepartmentNotFoundException(departmentId);
			}
			departmentRepository.Delete(DepartmentEntity!);
			await _unitOfWork.SaveChangesAsync(cancellationToken);	
		}


		public async Task AssignCourseToDepartmentAsync(int departmentId, int CourseId, CancellationToken cancellationToken = default)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentSpecificatioin = new DepartmentSpecification(departmentId);
			var DepartmentEntity = await  departmentRepository.GetByIdAsync(departmentSpecificatioin, cancellationToken);
			if (DepartmentEntity is null)
			{
				throw new DepartmentNotFoundException(departmentId);
			}
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await CourseRepository.GetByIdAsync(CourseId, cancellationToken);
			if (courseEntity is null)
			{
				throw new CourseNotFoundException(CourseId);
			}
			if (DepartmentEntity.courses.Any(c => c.Id == CourseId))
			{
				throw new CourseDepartmentNotFoundException(CourseId, departmentId);
			}
			DepartmentEntity.courses.Add(courseEntity);
			departmentRepository.Update(DepartmentEntity);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}

		public async Task RemoveCourseFromDepartmentAsync(int departmentId, int CourseId, CancellationToken cancellationToken = default)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentSpecificatioin = new DepartmentSpecification(departmentId);
			var DepartmentEntity = await departmentRepository.GetByIdAsync(departmentSpecificatioin, cancellationToken);
			if (DepartmentEntity is null)
			{
				throw new DepartmentNotFoundException(departmentId);
			}
			if (DepartmentEntity.courses is null) {
				throw new CoursesInDepartmentNotFoundException(departmentId);
			}
			if (!DepartmentEntity.courses.Any(c => c.Id == CourseId))
			{ 
				throw new CourseDepartmentNotFoundException(CourseId, departmentId);
			}
			var courseEntity = DepartmentEntity.courses.FirstOrDefault(c => c.Id == CourseId);
			DepartmentEntity.courses.Remove(courseEntity!);
			departmentRepository.Update(DepartmentEntity);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}

		
	}
}
