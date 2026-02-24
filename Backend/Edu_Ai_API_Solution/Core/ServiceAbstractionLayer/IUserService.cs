using Shared.Dtos.UserDto.Request;
namespace ServiceAbstractionLayer
{
	public interface IUserService
	{
		Task<UserProfileResponse> GetUserProfileAsync(string userId);
		Task UpdateUserProfileAsync(string userId, UpdateUserRequest request);
		Task ChangePasswordAsync(string userId, ChangePasswordRequest request);
	}
}
