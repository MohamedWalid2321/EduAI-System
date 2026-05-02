using Shared.Dtos.LectureDto.Request;
using Shared.Dtos.LectureDto.Response;

namespace ServiceAbstractionLayer
{
    public interface ILectureService
    {
        Task<IEnumerable<LectureResponse>> GetAllByCourseAsync(int courseId, CancellationToken cancellationToken = default);
        Task<LectureResponse> GetByIdAsync(int courseId, int lectureId, CancellationToken cancellationToken = default);
        Task<LectureResponse> CreateAsync(int courseId, string createdById, CreateLectureRequest request, CancellationToken cancellationToken = default);
        Task UpdateAsync(int courseId, int lectureId, UpdateLectureRequest request, CancellationToken cancellationToken = default);
        Task DeleteAsync(int courseId, int lectureId, CancellationToken cancellationToken = default);
        Task ToggleActiveAsync(int courseId, int lectureId, CancellationToken cancellationToken = default);
        Task<LectureJoinResponse> JoinAsync(int lectureId, string userId, CancellationToken cancellationToken = default);
    }
}