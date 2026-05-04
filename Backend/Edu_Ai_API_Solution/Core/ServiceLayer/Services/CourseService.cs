using DomainLayer.Models;
using ServiceLayer.Specifications.CourseSpecifications;
using Shared.Dtos.AssesmentDto;

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

		public async Task<IEnumerable<CourseResponseDto>> GetAllCourseforDepartmentAsync(int departmentId, CancellationToken cancellationToken = default)
		{
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courses = await CourseRepository.GetAllAsync(new CourseByDepartmentSpecification(departmentId), cancellationToken);
			if (courses is null || !courses.Any())
				throw new CoursesInDepartmentNotFoundException(departmentId);
			return courses.Adapt<IEnumerable<CourseResponseDto>>();
		}

		public async Task<IEnumerable<CourseResponseDto>> GetAllCourseAsync(CancellationToken cancellationToken = default)
		{
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courses = await CourseRepository.GetAllAsync(cancellationToken);
			return courses.Adapt<IEnumerable<CourseResponseDto>>();
		}

		public async Task<IEnumerable<CourseResponseDto>> GetCoursesAsync(string userId, int? departmentId, CancellationToken cancellationToken = default)
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
				return await GetUserEnrolledCoursesAsync(userId, cancellationToken);

			if (departmentId.HasValue)
				return await GetAllCourseforDepartmentAsync(departmentId.Value, cancellationToken);

			return await GetAllCourseAsync(cancellationToken);
		}

		public async Task<IEnumerable<CourseResponseDto>> GetUserEnrolledCoursesAsync(string userId, CancellationToken cancellationToken = default)
		{
			var user = await _userManager.FindByIdAsync(userId);
			if (user is null) throw new UserNotFound(userId);

			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseSpecification = new StudentCourseSpecification(user.DepartmentId, user.AcademicYearEnum);
			var courses = await CourseRepository.GetAllAsync(courseSpecification, cancellationToken);
			if (courses is null || !courses.Any())
				throw new CoursesInDepartmentNotFoundException(user.DepartmentId ?? 0);
			return courses.Adapt<IEnumerable<CourseResponseDto>>();
		}

		public async Task<FullCourseResponse> GetCourseByIdAsync(int departmentId, int courseId, CancellationToken cancellationToken = default)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseExists = await courseRepository.GetByIdAsync(courseId, cancellationToken);
			if (courseExists is null)
				throw new CourseNotFoundException(courseId);

			var course = await courseRepository.GetByIdAsync(new CourseSpecification(departmentId, courseId), cancellationToken);
			if (course is null)
				throw new CourseDepartmentNotFoundException(courseId, departmentId);

			return course.Adapt<FullCourseResponse>();
		}

		public async Task<CourseResponseDto> AddCourseAsync(int departmentId, CourseRequestDto request, IFormFile? ImageFile, CancellationToken cancellationToken = default)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentExists = await departmentRepository.GetByIdAsync(departmentId, cancellationToken);
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
			await courseRepository.AddAsync(courseEntity, cancellationToken);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
			BackgroundJob.Enqueue<IEnrollmentService>(s => s.EnrollNewCourseAsync(courseEntity.Id, default));
			return courseEntity.Adapt<CourseResponseDto>();
		}

		public async Task UpdateCourseAsync(int courseId, CourseRequestDto request, IFormFile? ImageFile, CancellationToken cancellationToken = default)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await courseRepository.GetByIdAsync(new CourseSpecification(courseId), cancellationToken);
			if (courseEntity is null)
				throw new CourseNotFoundException(courseId);

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
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}

		public async Task ToggleCouresStatus(int CourseId, CancellationToken cancellationToken = default)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await courseRepository.GetByIdAsync(CourseId, cancellationToken);
			if (courseEntity is null)
				throw new CourseNotFoundException(CourseId);

			courseEntity.IsPublished = !courseEntity.IsPublished;
			courseRepository.Update(courseEntity);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
			if (courseEntity.IsPublished)
				BackgroundJob.Enqueue<IEnrollmentService>(s => s.EnrollNewCourseAsync(CourseId, default));
		}

		public async Task<FullCourseResponse> AddAssesment(int CourseId, List<AssesmentRequest> assesments, CancellationToken cancellationToken = default)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await courseRepository.GetByIdAsync(new CourseSpecification(CourseId), cancellationToken);
			if (courseEntity is null)
				throw new CourseNotFoundException(CourseId);

			var assesmentEntities = assesments.Adapt<List<Assessment>>();
			foreach (var assesment in assesmentEntities)
			{
				assesment.CourseId = CourseId;
				courseEntity.Assessments.Add(assesment);
			}
			courseRepository.Update(courseEntity);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
			return courseEntity.Adapt<FullCourseResponse>();
		}

		public async Task UpdateAssesment(int CourseId, List<AssesmentRequest> assesments, CancellationToken cancellationToken = default)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await courseRepository.GetByIdAsync(new CourseSpecification(CourseId), cancellationToken);
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
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}

		public async Task DeleteCourseAsync(int courseId, CancellationToken cancellationToken = default)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await courseRepository.GetByIdAsync(courseId, cancellationToken);
			if (courseEntity is null)
				throw new CourseNotFoundException(courseId);

			if (!string.IsNullOrEmpty(courseEntity.ImageUrl))
				await _fileStorageService.DeleteFileAsync(courseEntity.ImageUrl);

			courseRepository.Delete(courseEntity!);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}

		public async Task<UserCourseResponse> ManualEnrollUserAsync(int courseId, string userId, string enrolledBy, CancellationToken cancellationToken = default)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var userCourseRepo = _unitOfWork.GetRepository<UserCourse, int>();

			var course = await courseRepository.GetByIdAsync(courseId, cancellationToken);
			if (course is null) throw new CourseNotFoundException(courseId);

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
			if (!isEnrolledRole)
				throw new IsNotInstructorException(userId);

			var existing = await userCourseRepo.GetAllAsync(
				new UserCourseByUserAndCourseSpecification(userId, courseId), cancellationToken);
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

			await userCourseRepo.AddAsync(userCourse, cancellationToken);
			await _unitOfWork.SaveChangesAsync(cancellationToken);

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

		public async Task ManualUnenrollUserAsync(int courseId, string userId, CancellationToken cancellationToken = default)
		{
			var userCourseRepo = _unitOfWork.GetRepository<UserCourse, int>();
			var enrollments = await userCourseRepo.GetAllAsync(
				new UserCourseByUserAndCourseSpecification(userId, courseId), cancellationToken);

			var enrollment = enrollments.FirstOrDefault();
			if (enrollment is null)
				throw new DuplicatedInstructorEnrollmentException(userId, courseId);

			userCourseRepo.Delete(enrollment);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}

		public async Task<IEnumerable<UserCourseResponse>> GetCourseEnrolledUsersAsync(int courseId, CancellationToken cancellationToken = default)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var userCourseRepo = _unitOfWork.GetRepository<UserCourse, int>();

			var course = await courseRepository.GetByIdAsync(courseId, cancellationToken);
			if (course is null) throw new CourseNotFoundException(courseId);

			var enrollments = await userCourseRepo.GetAllAsync(new UserCoursesByCourseSpecification(courseId), cancellationToken);

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

		public async Task<IEnumerable<AssessmentResponseDto>> GetAssessmentsByCourseIdAsync(int courseId, CancellationToken cancellationToken = default)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var course = await courseRepository.GetByIdAsync(new CourseSpecification(courseId), cancellationToken);

			if (course is null)
				throw new CourseNotFoundException(courseId);

			return course.Assessments.Adapt<IEnumerable<AssessmentResponseDto>>();
		}
	}
}
