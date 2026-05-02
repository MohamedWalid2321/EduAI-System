using Shared.Dtos.UserDto.Request;

namespace ServiceAbstractionLayer
{
    public interface IUserService
    {
        Task<IEnumerable<UserResponse>> GetAllAsync(bool? IncludeNotConfirmed = false, bool? includeDisabled = false, CancellationToken cancellationToken = default);
        Task<IEnumerable<InstructorsDetailsResponse>> GetAllInstructorByDepartmentIdAsync(int departmentId, CancellationToken cancellationToken = default);
        Task<UserResponse> GetAsync(string id, CancellationToken cancellationToken = default);
        Task<UserResponse> AddAsync(CreateUserRequest request, CancellationToken cancellationToken = default);
        Task UpdateAsync(string id, UpdateUserRequest request, CancellationToken cancellationToken = default);
        Task ToggleStatus(string id, CancellationToken cancellationToken = default);
        Task Unlock(string id, CancellationToken cancellationToken = default);
        Task LevelUp(string id, CancellationToken cancellationToken = default);
        Task<UserProfileResponse> GetUserProfileAsync(string userId, CancellationToken cancellationToken = default);
        Task UpdateUserProfileAsync(string userId, UpdateUserProfileRequest request, IFormFile? file, CancellationToken cancellationToken = default);
        Task ChangePasswordAsync(string userId, ChangePasswordRequest request, CancellationToken cancellationToken = default);
    }
}
