namespace ServiceLayer.Services
{
	public class UserService(UserManager<ApplicationUser> userManager):IUserService
	{
		private readonly UserManager<ApplicationUser>
			_userManager = userManager;
		public async Task<UserProfileResponse> GetUserProfileAsync(string userId)
		{
			var user = await _userManager.Users
				.Where(u => u.Id == userId)
				.ProjectToType<UserProfileResponse>()
				.SingleAsync();
			return user;
		}
		public async Task UpdateUserProfileAsync(string userId, UpdateUserRequest request)
		{
			var user = await _userManager.FindByIdAsync(userId);
			user = request.Adapt(user);
			await _userManager.UpdateAsync(user!);
		}
		public async Task ChangePasswordAsync(string userId, ChangePasswordRequest request)
		{
			var user = await _userManager.FindByIdAsync(userId);
			var result = await _userManager.ChangePasswordAsync(user!, request.CurrentPassword, request.NewPassword);
			if (!result.Succeeded)
			{
				throw new Exception(string.Join(", ", result.Errors.Select(e => e.Description)));
			}
		}

	

		
	}
}
