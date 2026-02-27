using Shared.Dtos.UserDto.Request;
namespace ServiceAbstractionLayer
{
	public interface IUserService
	{
		Task<IEnumerable<UserResponse>> GetAllAsync();
		Task<UserResponse> GetAsync(string id);
		Task<UserResponse> AddAsync(CreateUserRequest request);
		Task UpdateAsync(string id, UpdateUserRequest request);
		Task ToggleStatus(string id);
		Task Unlock(string id);
		Task<UserProfileResponse> GetUserProfileAsync(string userId);
		Task UpdateUserProfileAsync(string userId, UpdateUserProfileRequest request,IFormFile? file);
		Task ChangePasswordAsync(string userId, ChangePasswordRequest request);
	}
}
