using Shared.Dtos.DepartmentDto.Request;
using Shared.Dtos.DepartmentDto.Response;

namespace ServiceLayer.Services
{
	public class DepartmentService(IUnitOfWork unitOfWork) : IDepartmentService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		public async Task<IEnumerable<DepartmentResponse>> GetAllDepartmentsAsync()
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departments = await departmentRepository.GetAllAsync();
			var departmentDtos = departments.Adapt<IEnumerable<DepartmentResponse>>();
			return departmentDtos;
		}
		public async Task<DepartmentResponse> GetDepartmentByIdAsync(int departmentId)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentSpecificatioin = new DepartmentSpecification(departmentId);
			var DepartmentEntity = await departmentRepository.GetByIdAsync(departmentSpecificatioin);
			if (DepartmentEntity is null)
			{
				throw new DepartmentNotFoundException(departmentId);
			}
			return DepartmentEntity.Adapt<DepartmentResponse>();
		}
		public async Task<DepartmentResponse> AddDepartmentAsync(DepartmentRequest request)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var DuplicateDepartmentSpecification = new DepartmentByTitleSpecification(request.Title);
			if (await departmentRepository.GetByIdAsync(DuplicateDepartmentSpecification) is not null)
			{
				throw new DuplicatedDepartmentException();
			}
			var DepartmentEntity = request.Adapt<Department>();
			await departmentRepository.AddAsync(DepartmentEntity);
			await _unitOfWork.SaveChangesAsync();
			return DepartmentEntity.Adapt<DepartmentResponse>();
		}
		public async Task UpdateDepartmentAsync(int departmentId, DepartmentRequest request)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentSpecificatioin = new DepartmentSpecification(departmentId);
			var DepartmentEntity = await departmentRepository.GetByIdAsync(departmentSpecificatioin);
			if (DepartmentEntity is null)
			{
				throw new DepartmentNotFoundException(departmentId);
			}
			var DuplicateDepartmentSpecification = new DepartmentByTitleSpecification(request.Title);
			var duplicateDepartment = await departmentRepository.GetByIdAsync(DuplicateDepartmentSpecification);
			if (duplicateDepartment is not null && duplicateDepartment.Id != departmentId)
			{
				throw new DuplicatedDepartmentException();
			}
			DepartmentEntity.Title = request.Title;
			departmentRepository.Update(DepartmentEntity);
			await _unitOfWork.SaveChangesAsync();
		}

		public async Task DeleteDepartmentAsync(int departmentId)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var DepartmentEntity = await departmentRepository.GetByIdAsync(departmentId);
			if (DepartmentEntity is null)
			{
				throw new DepartmentNotFoundException(departmentId);
			}
			departmentRepository.Delete(DepartmentEntity!);
			await _unitOfWork.SaveChangesAsync();	
		}

		

		

		
		public async Task<IEnumerable<CourseRequestDto>> GetAllCourseBydepartmentIdAsync(int departmentId)
		{
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseSpecification = new CourseByDepartmentSpecification(departmentId);
			var courses = await CourseRepository.GetAllAsync(courseSpecification);
			if (courses is null || !courses.Any())
			{
				throw new CoursesInDepartmentNotFoundException(departmentId);
            }
            return courses.Adapt<IEnumerable<CourseRequestDto>>();
		}

		public async Task<DepartmentRequest> AssignCourseToDepartmentAsync(int departmentId, int CourseId)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentSpecificatioin = new DepartmentSpecification(departmentId);
			var DepartmentEntity = await  departmentRepository.GetByIdAsync(departmentSpecificatioin);
			if (DepartmentEntity is null)
			{
				throw new DepartmentNotFoundException(departmentId);
			}
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await CourseRepository.GetByIdAsync(CourseId);
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
			await _unitOfWork.SaveChangesAsync();
			return DepartmentEntity.Adapt<DepartmentRequest>();
		}

		public async Task<DepartmentRequest> RemoveCourseFromDepartmentAsync(int departmentId, int CourseId)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentSpecificatioin = new DepartmentSpecification(departmentId);
			var DepartmentEntity = await departmentRepository.GetByIdAsync(departmentSpecificatioin);
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
			await _unitOfWork.SaveChangesAsync();
			return DepartmentEntity.Adapt<DepartmentRequest>();
		}

		
	}
}
