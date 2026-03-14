using DomainLayer.Models;
using Hangfire;
using ServiceLayer.Specifications.CourseSpecification;
using ServiceLayer.Specifications.CourseSpecifications;
using ServiceLayer.Specifications.UserCourseSpecifications;

namespace ServiceLayer.Services
{
	public class EnrollmentService(
		IUnitOfWork unitOfWork,
		UserManager<ApplicationUser> userManager) : IEnrollmentService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		private readonly UserManager<ApplicationUser> _userManager = userManager;

		public async Task AutoEnrollAsync(string userId)
		{
			var user = await _userManager.FindByIdAsync(userId);
			if (user is null || !user.DepartmentId.HasValue || !user.AcademicYear.HasValue)
				return;

			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var matchingCourses = await courseRepository.GetAllAsync(
				new StudentCourseSpecification(user.DepartmentId, user.AcademicYear));

			if (!matchingCourses.Any())
				return;

			var userCourseRepo = _unitOfWork.GetRepository<UserCourse, int>();
			var existing = await userCourseRepo.GetAllAsync(new UserCoursesByUserSpecification(userId));
			var alreadyEnrolledCourseIds = existing.Select(uc => uc.CourseId).ToHashSet();

			foreach (var course in matchingCourses.Where(c => !alreadyEnrolledCourseIds.Contains(c.Id)))
			{
				await userCourseRepo.AddAsync(new UserCourse
				{
					UserId = userId,
					CourseId = course.Id,
					EnrolledAt = DateTime.UtcNow,
					Status = EnrollmentStatus.Active
				});
			}

			await _unitOfWork.SaveChangesAsync();
		}

		public async Task ReEnrollAsync(string userId)
		{
			var userCourseRepo = _unitOfWork.GetRepository<UserCourse, int>();
			var existing = await userCourseRepo.GetAllAsync(new UserCoursesByUserSpecification(userId));

			foreach (var enrollment in existing)
				userCourseRepo.Delete(enrollment);

			await _unitOfWork.SaveChangesAsync();
			await AutoEnrollAsync(userId);
		}

		public async Task EnrollNewCourseAsync(int courseId)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var course = await courseRepository.GetByIdAsync(new CourseSpecification(courseId));
			if (course is null || !course.IsPublished)
				return;

			var departmentIds = course.Departments.Select(d => d.Id).ToList();
			if (!departmentIds.Any())
				return;

			var matchingUsers = await _userManager.Users
				.Where(u => u.DepartmentId.HasValue
						 && departmentIds.Contains(u.DepartmentId.Value)
						 && u.AcademicYear == course.AcademicLevel)
				.ToListAsync();

			if (!matchingUsers.Any())
				return;

			var userCourseRepo = _unitOfWork.GetRepository<UserCourse, int>();
			var existing = await userCourseRepo.GetAllAsync(new UserCoursesByCourseSpecification(courseId));
			var alreadyEnrolledUserIds = existing.Select(uc => uc.UserId).ToHashSet();
   			foreach (var user in matchingUsers.Where(u => !alreadyEnrolledUserIds.Contains(u.Id)))
			{
				await userCourseRepo.AddAsync(new UserCourse
				{
					UserId = user.Id,
					CourseId = courseId,
					EnrolledAt = DateTime.UtcNow,
					Status = EnrollmentStatus.Active
				});
			}

			await _unitOfWork.SaveChangesAsync();
		}
	}
}
