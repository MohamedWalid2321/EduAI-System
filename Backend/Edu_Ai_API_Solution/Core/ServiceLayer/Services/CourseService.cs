using DomainLayer.Models;
using ServiceLayer.Specifications.CourseSpecifications;

namespace ServiceLayer.Services
{
	public class CourseService(
		IUnitOfWork unitOfWork,
		IFileStorageService fileStorageService,
		UserManager<ApplicationUser> userManager,
		RoleManager<ApplicationRole> roleManager) : ICourseService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		private readonly IFileStorageService _fileStorageService = fileStorageService;
		private readonly UserManager<ApplicationUser> _userManager = userManager;
		private readonly RoleManager<ApplicationRole> _roleManager = roleManager;

		public async Task<IEnumerable<CourseResponseDto>> GetAllCourseforDepartmentAsync(int departmentId)
		{
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courses = await CourseRepository.GetAllAsync(new CourseByDepartmentSpecification(departmentId));
			if (courses is null || !courses.Any())
				throw new CoursesInDepartmentNotFoundException(departmentId);
			return courses.Adapt<IEnumerable<CourseResponseDto>>();
		}

		public async Task<IEnumerable<CourseResponseDto>> GetAllCourseAsync()
		{
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courses = await CourseRepository.GetAllAsync();
			return courses.Adapt<IEnumerable<CourseResponseDto>>();
		}

		public async Task<IEnumerable<CourseResponseDto>> GetCoursesAsync(string userId, int? departmentId)
		{
			var user = await _userManager.FindByIdAsync(userId);
			if (user is null) throw new UserNotFound(userId);

			var userRoles = await _userManager.GetRolesAsync(user);
			var isEnrolledRole = false;
			foreach (var roleName in userRoles)
			{
				var role = await _roleManager.FindByNameAsync(roleName);
				if (role is not null && role.IsEnrollable)
				{
					isEnrolledRole = true;
					break;
				}
			}

			if (isEnrolledRole)
				return await GetUserEnrolledCoursesAsync(userId);

			if (departmentId.HasValue)
				return await GetAllCourseforDepartmentAsync(departmentId.Value);

			return await GetAllCourseAsync();

		}

		public async Task<IEnumerable<CourseResponseDto>> GetUserEnrolledCoursesAsync(string userId)
		{ 
			var user = await _userManager.FindByIdAsync(userId);
			if (user is null) throw new UserNotFound(userId);

			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseSpecification = new StudentCourseSpecification(user.DepartmentId,user.AcademicYearEnum);
			var courses = await CourseRepository.GetAllAsync(courseSpecification);
			if (courses is null || !courses.Any())
			{
				throw  new CoursesInDepartmentNotFoundException(user.DepartmentId??0);
			}
			return courses.Adapt<IEnumerable<CourseResponseDto>>();
		}

		public async Task<FullCourseResponse> GetCourseByIdAsync(int departmentId, int courseId)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseExists = await courseRepository.GetByIdAsync(courseId);
			if (courseExists is null)
				throw new CourseNotFoundException(courseId);

			var course = await courseRepository.GetByIdAsync(new CourseSpecification(departmentId, courseId));
			if (course is null)
				throw new CourseDepartmentNotFoundException(courseId, departmentId);

			return course.Adapt<FullCourseResponse>();
		}

		public async Task<CourseResponseDto> AddCourseAsync(int departmentId, CourseRequestDto request, IFormFile? ImageFile)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentExists = await departmentRepository.GetByIdAsync(departmentId);
			if (departmentExists is null)
				throw new DepartmentNotFoundException(departmentId);

			var courseEntity = request.Adapt<Course>();
			if (ImageFile is not null && ImageFile.Length > 0)
			{
				using var stream = ImageFile.OpenReadStream();
				var imagePath = await _fileStorageService.UploadFileAsync(
					stream, ImageFile.FileName,
					$"Courses/{request.Title}/images", ImageFile.ContentType);
				courseEntity.ImageUrl = imagePath;
			}
			courseEntity.Departments.Add(departmentExists);
			await courseRepository.AddAsync(courseEntity);
			await _unitOfWork.SaveChangesAsync();
			BackgroundJob.Enqueue<IEnrollmentService>(s => s.EnrollNewCourseAsync(courseEntity.Id));
			return courseEntity.Adapt<CourseResponseDto>();
		}

		public async Task UpdateCourseAsync(int departmentId, int courseId, CourseRequestDto request, IFormFile? ImageFile)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await courseRepository.GetByIdAsync(new CourseSpecification(departmentId, courseId));

			if (courseEntity is null)
			{
				var courseExists = await courseRepository.GetByIdAsync(courseId);
				if (courseExists is null)
					throw new CourseNotFoundException(courseId);
				throw new CourseDepartmentNotFoundException(courseId, departmentId);
			}
			if (!string.IsNullOrEmpty(courseEntity.ImageUrl))
				await _fileStorageService.DeleteFileAsync(courseEntity.ImageUrl);

			courseEntity = request.Adapt(courseEntity);

			if (ImageFile is not null && ImageFile.Length > 0)
			{
				using var stream = ImageFile.OpenReadStream();
				var imagePath = await _fileStorageService.UploadFileAsync(
					stream, ImageFile.FileName,
					$"Courses/{request.Title}/images", ImageFile.ContentType);
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
				throw new CourseNotFoundException(CourseId);

			courseEntity.IsPublished = !courseEntity.IsPublished;
			courseRepository.Update(courseEntity);
			await _unitOfWork.SaveChangesAsync();
			if (courseEntity.IsPublished)
				BackgroundJob.Enqueue<IEnrollmentService>(s => s.EnrollNewCourseAsync(CourseId));
		}

		public async Task<FullCourseResponse> AddAssesment(int CourseId, List<AssesmentDto> assesments)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await courseRepository.GetByIdAsync(new CourseSpecification(CourseId));
			if (courseEntity is null)
				throw new CourseNotFoundException(CourseId);

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
				throw new CourseNotFoundException(CourseId);

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
				throw new CourseNotFoundException(courseId);

			if (!string.IsNullOrEmpty(courseEntity.ImageUrl))
				await _fileStorageService.DeleteFileAsync(courseEntity.ImageUrl);

			courseRepository.Delete(courseEntity!);
			await _unitOfWork.SaveChangesAsync();
		}

		public async Task<UserCourseResponse> ManualEnrollUserAsync(int courseId, string userId, string enrolledBy)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var userCourseRepo = _unitOfWork.GetRepository<UserCourse, int>();

			var course = await courseRepository.GetByIdAsync(courseId);
			if (course is null) throw new CourseNotFoundException(courseId);

			var user = await _userManager.FindByIdAsync(userId);
			if (user is null) throw new UserNotFound(userId);

			// Verify user's role is an enrolled role
			var userRoles = await _userManager.GetRolesAsync(user);
			var isEnrolledRole = false;
			foreach (var roleName in userRoles)
			{
				var role = await _roleManager.FindByNameAsync(roleName);
				if (role is not null && role.IsEnrollable)
				{
					isEnrolledRole = true;
					break;
				}
			}
			if (!isEnrolledRole)
				throw new IsNotInstructorException(userId);

			// Check for duplicate enrollment
			var existing = await userCourseRepo.GetAllAsync(
				new UserCourseByUserAndCourseSpecification(userId, courseId));
			if (existing.Any())
				throw new DuplicatedInstructorEnrollmentException(userId, courseId);

			var userCourse = new UserCourse
			{
				UserId = userId,
				CourseId = courseId,
				EnrolledAt = DateTime.UtcNow,
				Status = EnrollmentStatus.Active,
				EnrolledBy = enrolledBy
			};

			await userCourseRepo.AddAsync(userCourse);
			await _unitOfWork.SaveChangesAsync();

			return new UserCourseResponse
			{
				Id = userCourse.Id,
				UserId = user.Id,
				UserName = $"{user.FirstName} {user.LastName}",
				UserEmail = user.Email!,
				CourseId = course.Id,
				CourseTitle = course.Title,
				EnrolledAt = userCourse.EnrolledAt,
				EnrolledBy = enrolledBy
			};
		}

		public async Task ManualUnenrollUserAsync(int courseId, string userId)
		{
			var userCourseRepo = _unitOfWork.GetRepository<UserCourse, int>();

			var enrollments = await userCourseRepo.GetAllAsync(
				new UserCourseByUserAndCourseSpecification(userId, courseId));

			var enrollment = enrollments.FirstOrDefault();
			if (enrollment is null)
				throw new DuplicatedInstructorEnrollmentException(userId, courseId);

			userCourseRepo.Delete(enrollment);
			await _unitOfWork.SaveChangesAsync();
		}

		public async Task<IEnumerable<UserCourseResponse>> GetCourseEnrolledUsersAsync(int courseId)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var userCourseRepo = _unitOfWork.GetRepository<UserCourse, int>();

			var course = await courseRepository.GetByIdAsync(courseId);
			if (course is null) throw new CourseNotFoundException(courseId);

			var enrollments = await userCourseRepo.GetAllAsync(new UserCoursesByCourseSpecification(courseId));

			return enrollments.Select(e => new UserCourseResponse
			{
				Id = e.Id,
				UserId = e.UserId,
				UserName = $"{e.User.FirstName} {e.User.LastName}",
				UserEmail = e.User.Email!,
				CourseId = e.CourseId,
				CourseTitle = e.Course.Title,
				EnrolledAt = e.EnrolledAt,
				EnrolledBy = e.EnrolledBy
			});
		}
	}
}
