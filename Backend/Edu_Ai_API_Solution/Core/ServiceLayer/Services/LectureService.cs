using DomainLayer.Exceptions.Lecture;
using Microsoft.Extensions.Configuration;
using Shared.Dtos.LectureDto.Request;
using Shared.Dtos.LectureDto.Response;
using ServiceLayer.Specifications.LectureSpecifications;

namespace ServiceLayer.Services
{
	public class LectureService(IUnitOfWork unitOfWork, IConfiguration configuration,UserManager<ApplicationUser> userManager) : ILectureService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		private readonly UserManager<ApplicationUser> _userManager = userManager;
		private readonly string _jitsiDomain = configuration["Jitsi:Domain"] ?? "meet.jit.si";
		

		public async Task<IEnumerable<LectureResponse>> GetAllByCourseAsync(int courseId)
		{
			var repo = _unitOfWork.GetRepository<Lecture, int>();
			var lectures = await repo.GetAllAsync(new LecturesByCourseSpecification(courseId));
			return lectures.Adapt<IEnumerable<LectureResponse>>();
		}

		public async Task<LectureResponse> GetByIdAsync(int courseId, int lectureId)
		{
			var repo = _unitOfWork.GetRepository<Lecture, int>();
			var lecture = await repo.GetByIdAsync(new LecturesByCourseSpecification(courseId, lectureId));
			if (lecture is null)
				throw new LectureNotFoundException(lectureId);

			return lecture.Adapt<LectureResponse>();
		}

		public async Task<LectureResponse> CreateAsync(int courseId, string createdById, CreateLectureRequest request)
		{
			var courseRepo = _unitOfWork.GetRepository<Course, int>();
			var courseExists = await courseRepo.GetByIdAsync(courseId);
			if (courseExists is null)
				throw new CourseNotFoundException(courseId);

			var lecture = new Lecture
			{
				Title = request.Title,
				Description = request.Description,
				ScheduledAt = request.ScheduledAt,
				CourseId = courseId,
				CreatedById = createdById,
				RoomName = GenerateRoomName(courseId, request.Title),
				IsActive = false
			};

			var repo = _unitOfWork.GetRepository<Lecture, int>();
			await repo.AddAsync(lecture);
			await _unitOfWork.SaveChangesAsync();

			// Reload with CreatedBy navigation
			var created = await repo.GetByIdAsync(new LecturesByCourseSpecification(courseId, lecture.Id));
			return created!.Adapt<LectureResponse>();
		}

		public async Task UpdateAsync(int courseId, int lectureId, UpdateLectureRequest request)
		{
			var repo = _unitOfWork.GetRepository<Lecture, int>();
			var lecture = await repo.GetByIdAsync(new LecturesByCourseSpecification(courseId, lectureId));
			if (lecture is null)
				throw new LectureNotFoundException(lectureId);

			lecture.Title = request.Title;
			lecture.Description = request.Description;
			lecture.ScheduledAt = request.ScheduledAt;
			lecture.RoomName = GenerateRoomName(courseId, request.Title);

			repo.Update(lecture);
			await _unitOfWork.SaveChangesAsync();
		}

		public async Task DeleteAsync(int courseId, int lectureId)
		{
			var repo = _unitOfWork.GetRepository<Lecture, int>();
			var lecture = await repo.GetByIdAsync(new LecturesByCourseSpecification(courseId, lectureId));
			if (lecture is null)
				throw new LectureNotFoundException(lectureId);

			repo.Delete(lecture);
			await _unitOfWork.SaveChangesAsync();
		}

		public async Task ToggleActiveAsync(int courseId, int lectureId)
		{
			var repo = _unitOfWork.GetRepository<Lecture, int>();
			var lecture = await repo.GetByIdAsync(new LecturesByCourseSpecification(courseId, lectureId));
			if (lecture is null)
				throw new LectureNotFoundException(lectureId);

			lecture.IsActive = !lecture.IsActive;
			repo.Update(lecture);
			await _unitOfWork.SaveChangesAsync();
		}

		public async Task<LectureJoinResponse> JoinAsync(int lectureId, string userId)
		{
			var repo = _unitOfWork.GetRepository<Lecture, int>();
			var lecture = await repo.GetByIdAsync(lectureId);
			if (lecture is null)
				throw new LectureNotFoundException(lectureId);

			if (!lecture.IsActive)
				throw new LectureNotActiveException(lectureId);

			
			var user = await _userManager.FindByIdAsync(userId);

			var displayName = user is not null
				? $"{user.FirstName} {user.LastName}".Trim()
				: "Student";

			return new LectureJoinResponse
			{
				LectureId = lecture.Id,
				RoomName = lecture.RoomName,
				JitsiDomain = _jitsiDomain,
				DisplayName = displayName,
				JitsiUrl = $"https://{_jitsiDomain}/{lecture.RoomName}"
			};
		}

		// Generates a deterministic, URL-safe unique room name per course+lecture
		private static string GenerateRoomName(int courseId, string title)
		{
			var sanitized = new string(title
				.ToLowerInvariant()
				.Where(c => char.IsLetterOrDigit(c) || c == '-')
				.ToArray())
				.Replace(' ', '-');

			return $"lumino-course{courseId}-{sanitized}-{Guid.NewGuid():N}";
		}
	}
}