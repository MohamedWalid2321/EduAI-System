using DomainLayer.Models;
using ServiceLayer.Specifications.CourseSpecifications;
using ServiceLayer.Specifications.InstructorCourseSpecifications;
using Shared.Constants;

namespace ServiceLayer.Services
{
	public class CourseService(
		IUnitOfWork unitOfWork,
		IFileStorageService fileStorageService,
		UserManager<ApplicationUser> userManager) : ICourseService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		private readonly IFileStorageService _fileStorageService = fileStorageService;
		private readonly UserManager<ApplicationUser> _userManager = userManager;

		public async Task<IEnumerable<CourseResponseDto>> GetAllCourseforDepartmentAsync(int departmentId)
		{
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courses = await CourseRepository.GetAllAsync(new CourseByDepartmentSpecification(departmentId));
			if (courses is null || !courses.Any())
			{
				throw new CoursesInDepartmentNotFoundException(departmentId);
			}
			return courses.Adapt<IEnumerable<CourseResponseDto>>();
		}
		public async Task<IEnumerable<CourseResponseDto>> GetAllStudentCourse(string UserId)
		{
			var user = await _userManager.FindByIdAsync(UserId);
			if (user is null)
			{
				throw new UserNotFound(UserId);
			}
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseSpecification = new StudentCourseSpecification(user.DepartmentId,user.AcademicYearEnum);
			var courses = await CourseRepository.GetAllAsync(courseSpecification);
			if (courses is null || !courses.Any())
			{
				throw  new CoursesInDepartmentNotFoundException(user.DepartmentId??0);
			}
			return courses.Adapt<IEnumerable<CourseResponseDto>>();
		}
		public async Task<IEnumerable<CourseResponseDto>> GetAllCourseAsync()
		{
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courses = await CourseRepository.GetAllAsync();
			return courses.Adapt<IEnumerable<CourseResponseDto>>();
		}
		public async Task<FullCourseResponse> GetCourseByIdAsync(int departmentId, int courseId)
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
			return course.Adapt<FullCourseResponse>();
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
					$"Courses/{request.Title}/images",
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
			var courseEntity = await courseRepository.GetByIdAsync(new CourseSpecification(departmentId, courseId));
			
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
			if (!string.IsNullOrEmpty(courseEntity.ImageUrl))
			{
				await _fileStorageService.DeleteFileAsync(courseEntity.ImageUrl);
			}

			courseEntity = request.Adapt(courseEntity);

			if (ImageFile is not null && ImageFile.Length > 0)
			{
				using var stream = ImageFile.OpenReadStream();
				var imagePath = await _fileStorageService.UploadFileAsync(
					stream,
					ImageFile.FileName,
					$"Courses/{request.Title}/images",
					ImageFile.ContentType);
				courseEntity.ImageUrl = imagePath;
			}
			courseRepository.Update(courseEntity);
			await _unitOfWork.SaveChangesAsync();
		}
		public async Task ToggleCouresStatus(int CourseId)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await courseRepository.GetByIdAsync(CourseId);
			if (courseEntity is null)
			{
				throw new CourseNotFoundException(CourseId);
			}
			courseEntity.IsPublished = !courseEntity.IsPublished;
			courseRepository.Update(courseEntity);
			await _unitOfWork.SaveChangesAsync();
		}
		public async Task<FullCourseResponse> AddAssesment(int CourseId, List<AssesmentDto> assesments)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await courseRepository.GetByIdAsync(new CourseSpecification(CourseId));
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
			var courseEntity = await courseRepository.GetByIdAsync(new CourseSpecification(CourseId));
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

		public async Task<InstructorCourseResponse> EnrollInstructorAsync(int courseId, string instructorId, string assignedBy)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var instructorCourseRepository = _unitOfWork.GetRepository<InstructorCourse, int>();
			
			// Verify course exists
			var course = await courseRepository.GetByIdAsync(courseId);
			if (course is null)
			{
				throw new CourseNotFoundException(courseId);
			}
			
			// Verify instructor exists and has Instructor role
			var instructor = await _userManager.FindByIdAsync(instructorId);
			if (instructor is null)
			{
				throw new UserNotFound(instructorId);
			}
			
			var isInstructor = await _userManager.IsInRoleAsync(instructor, DefaultRoles.Instructor);
			if (!isInstructor)
			{
				throw new IsNotInstructorException(instructorId);
			}
			
			// Check if already enrolled
			var existingEnrollment = await instructorCourseRepository
				.GetAllAsync(new InstructorCourseSpecification(courseId, instructorId));
			
			if (existingEnrollment.Any())
			{
				throw new DuplicatedInstructorEnrollmentException(instructorId, courseId);
			}
			
			var instructorCourse = new InstructorCourse
			{
				InstructorId = instructorId,
				CourseId = courseId,
				AssignedAt = DateTime.UtcNow,
				AssignedBy = assignedBy
			};
			
			await instructorCourseRepository.AddAsync(instructorCourse);
			await _unitOfWork.SaveChangesAsync();
			
			return new InstructorCourseResponse
			{
				Id = instructorCourse.Id,
				InstructorId = instructor.Id,
				InstructorName = $"{instructor.FirstName} {instructor.LastName}",
				InstructorEmail = instructor.Email!,
				CourseId = course.Id,
				CourseTitle = course.Title,
				AssignedAt = instructorCourse.AssignedAt
			};
		}

		public async Task UnenrollInstructorAsync(int courseId, string instructorId)
		{
			var instructorCourseRepository = _unitOfWork.GetRepository<InstructorCourse, int>();
			
			var enrollments = await instructorCourseRepository
				.GetAllAsync(new InstructorCourseSpecification(courseId, instructorId));
			
			var enrollment = enrollments.FirstOrDefault();
			if (enrollment is null)
			{
				throw new DuplicatedInstructorEnrollmentException(instructorId, courseId);
			}
			
			instructorCourseRepository.Delete(enrollment);
			await _unitOfWork.SaveChangesAsync();
		}

		public async Task<IEnumerable<InstructorCourseResponse>> GetCourseInstructorsAsync(int courseId)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var instructorCourseRepository = _unitOfWork.GetRepository<InstructorCourse, int>();
			
			var course = await courseRepository.GetByIdAsync(courseId);
			if (course is null)
			{
				throw new CourseNotFoundException(courseId);
			}
			
			var specification = new InstructorCourseSpecification(courseId);
			var enrollments = await instructorCourseRepository.GetAllAsync(specification);
			
			return enrollments.Select(e => new InstructorCourseResponse
				{
					Id = e.Id,
					InstructorId = e.InstructorId,
					InstructorName = $"{e.Instructor.FirstName} {e.Instructor.LastName}",
					InstructorEmail = e.Instructor.Email!,
					CourseId = e.CourseId,
					CourseTitle = e.Course.Title,
					AssignedAt = e.AssignedAt
				});
		}

		public async Task<IEnumerable<CourseResponseDto>> GetInstructorCoursesAsync(string instructorId)
		{
			var instructor = await _userManager.FindByIdAsync(instructorId);
			if (instructor is null)
			{
				throw new UserNotFound(instructorId);
			}
			
			var instructorCourseRepository = _unitOfWork.GetRepository<InstructorCourse, int>();
			var specification = new InstructorCourseByInstructorSpecification(instructorId);
			var enrollments = await instructorCourseRepository.GetAllAsync(specification);
			
			return enrollments.Select(e => e.Course).Adapt<IEnumerable<CourseResponseDto>>();
		}
	}
}
