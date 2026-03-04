using DomainLayer.Models;

namespace ServiceLayer.Services
{
	public class CourseService(IUnitOfWork unitOfWork,IFileStorageService fileStorageService) : ICourseService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		private readonly IFileStorageService _fileStorageService = fileStorageService;
		public async Task<IEnumerable<FullCourseResponse>> GetAllCourseAsync(int departmentId)
		{
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseSpecification = new CourseByDepartmentSpecification(departmentId);
			var courses = await CourseRepository.GetAllAsync(courseSpecification);
			if (courses is null || !courses.Any())
			{
				throw new CoursesInDepartmentNotFoundException(departmentId);
			}
			return courses.Adapt<IEnumerable<FullCourseResponse>>();
		}
		public async Task<CourseResponseDto> GetCourseByIdAsync(int departmentId, int courseId)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseExists = await courseRepository.GetByIdAsync(courseId);
			if (courseExists is null)
			{
				throw new CourseNotFoundException(courseId);
			}
			var courseSpecification = new CourseSpecification(departmentId, courseId);
			var course = await courseRepository.GetByIdAsync(courseSpecification);
			if (course is null)
			{
				throw new CourseDepartmentNotFoundException(courseId, departmentId);
			}
			
			return course.Adapt<CourseResponseDto>();
		}
		public async Task<CourseResponseDto> AddCourseAsync(int departmentId,CourseRequestDto request, IFormFile? ImageFile)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentExists = await departmentRepository.GetByIdAsync(departmentId);
			if (departmentExists is null)
			{
				throw new DepartmentNotFoundException(departmentId);
			}
			var courseEntity = request.Adapt<Course>();
			if (ImageFile is not null && ImageFile.Length > 0)
			{
				using var stream = ImageFile.OpenReadStream();
				var imagePath = await _fileStorageService.UploadFileAsync(
					stream,
					ImageFile.FileName,
					"Courses/images",
					ImageFile.ContentType);
				courseEntity.ImageUrl = imagePath;
			}
			courseEntity.Departments.Add(departmentExists);
			await courseRepository.AddAsync(courseEntity);
			await _unitOfWork.SaveChangesAsync();
			return courseEntity.Adapt<CourseResponseDto>();
		}
		public async Task UpdateCourseAsync(int departmentId, int courseId, CourseRequestDto request, IFormFile? ImageFile)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseSpecification = new CourseSpecification(departmentId, courseId);
			var courseEntity = await courseRepository.GetByIdAsync(courseSpecification);
			
			if (courseEntity is null)
			{
				// Check if course exists at all
				var courseExists = await courseRepository.GetByIdAsync(courseId);
				if (courseExists is null)
				{
					throw new CourseNotFoundException(courseId);
				}
				// Course exists but not in this department
				throw new CourseDepartmentNotFoundException(courseId, departmentId);
			}
			//if (!string.IsNullOrEmpty(courseEntity.ImageUrl))
			//{
			//	await _fileStorageService.DeleteFileAsync(courseEntity.ImageUrl);
			//}

			courseEntity = request.Adapt(courseEntity);

			if (ImageFile is not null && ImageFile.Length > 0)
			{
				using var stream = ImageFile.OpenReadStream();
				var imagePath = await _fileStorageService.UploadFileAsync(
					stream,
					ImageFile.FileName,
					"Courses/images",
					ImageFile.ContentType);
				courseEntity.ImageUrl = imagePath;
			}
			courseRepository.Update(courseEntity);
			await _unitOfWork.SaveChangesAsync();
		}
		public async Task<FullCourseResponse> AddAssesment(int CourseId, List<AssesmentDto> assesments)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var specification = new CourseSpecification(CourseId);
			var courseEntity = await courseRepository.GetByIdAsync(specification);
			if (courseEntity is null)
			{
				throw new CourseNotFoundException(CourseId);
			}
			var assesmentEntities = assesments.Adapt<List<Assessment>>();
			foreach (var assesment in assesmentEntities)
			{
				assesment.CourseId = CourseId;
				courseEntity.Assessments.Add(assesment);
			}
			courseRepository.Update(courseEntity);
			await _unitOfWork.SaveChangesAsync();
			return courseEntity.Adapt<FullCourseResponse>();
		}

		public async Task UpdateAssesment(int CourseId, List<AssesmentDto> assesments)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var specification = new CourseSpecification(CourseId);
			var courseEntity = await courseRepository.GetByIdAsync(specification);
			if (courseEntity is null)
			{
				throw new CourseNotFoundException(CourseId);
			}
			var assesmentEntities = assesments.Adapt<List<Assessment>>();
			courseEntity.Assessments.Clear();
			foreach (var assesment in assesmentEntities)
			{
				assesment.CourseId = CourseId;
				courseEntity.Assessments.Add(assesment);
			}
			courseRepository.Update(courseEntity);
			await _unitOfWork.SaveChangesAsync();
		}

		public async Task DeleteCourseAsync(int courseId)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await courseRepository.GetByIdAsync(courseId);
			if (courseEntity is null)
			{
				throw new CourseNotFoundException(courseId);
			}
			if (!string.IsNullOrEmpty(courseEntity.ImageUrl))
			{
				await _fileStorageService.DeleteFileAsync(courseEntity.ImageUrl);
			}
			courseRepository.Delete(courseEntity!);
			await _unitOfWork.SaveChangesAsync();
		}

		
	}
}
