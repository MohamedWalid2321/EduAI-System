using Shared.Dtos.LectureDto.Request;
using Shared.Dtos.LectureDto.Response;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
	public interface ILectureService
	{
		Task<IEnumerable<LectureResponse>> GetAllByCourseAsync(int courseId);
		Task<LectureResponse> GetByIdAsync(int courseId, int lectureId);
		Task<LectureResponse> CreateAsync(int courseId, string createdById, CreateLectureRequest request);
		Task UpdateAsync(int courseId, int lectureId, UpdateLectureRequest request);
		Task DeleteAsync(int courseId, int lectureId);
		Task ToggleActiveAsync(int courseId, int lectureId);
		Task<LectureJoinResponse> JoinAsync(int lectureId, string userId);
	}
}