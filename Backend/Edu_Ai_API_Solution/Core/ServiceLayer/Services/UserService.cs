using Microsoft.AspNetCore.Mvc;
using ServiceAbstractionLayer;
using Shared.Constants;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace ServiceLayer.Services
{
	public class UserService(UserManager<ApplicationUser> userManager,IRoleService roleService,IFileStorageService fileStorageService):IUserService
	{
		private readonly UserManager<ApplicationUser>
			_userManager = userManager;
		private readonly IRoleService _roleService = roleService;
		private readonly IFileStorageService _fileStorageService = fileStorageService;

		public async Task<IEnumerable<UserResponse>> GetAllAsync() {
			var users = await _userManager.Users
			.Where(u => !u.IsDisabled)
			.ToListAsync();

			var userResponses = new List<UserResponse>();

			foreach (var user in users)
			{
				var roles = await _userManager.GetRolesAsync(user);
					userResponses.Add(new UserResponse
					{
						Id = user.Id,
						FirstName = user.FirstName!,
						LastName = user.LastName!,
						Email = user.Email!,
						IsDisabled = user.IsDisabled,
						Roles = roles
					});
				
			}

			return userResponses;
		}

		public async Task<UserResponse> GetAsync(string id)
		{
			if (await _userManager.FindByIdAsync(id) is not { } user)
				throw new UserNotFound(id);

			var userRoles = await _userManager.GetRolesAsync(user);

			var response = (user, userRoles).Adapt<UserResponse>();

			return response;
		}

		public async Task<UserResponse> AddAsync(CreateUserRequest request)
		{
			var emailIsExists = await _userManager.Users.AnyAsync(x => x.Email == request.Email);

			if (emailIsExists)
				throw new DuplicatedEmail(request.Email);

			var allowedRoles = await _roleService.GetAllAsync();

			if (request.Roles.Except(allowedRoles.Select(x => x.Name)).Any())
				throw new InvalidRoles();

			var user = request.Adapt<ApplicationUser>();

			var result = await _userManager.CreateAsync(user, request.Password);

			if (result.Succeeded)
			{
				await _userManager.AddToRolesAsync(user, request.Roles);

				var response = (user, request.Roles).Adapt<UserResponse>();

				return response;
			}

			var error = result.Errors.First();

			throw new IdentityResultError(error.Description);
		}

		public async Task UpdateAsync(string id, UpdateUserRequest request)
		{
			var emailIsExists = await _userManager.Users.AnyAsync(x => x.Email == request.Email && x.Id != id);

			if (emailIsExists)
				throw new DuplicatedEmail(request.Email);

			var allowedRoles = await _roleService.GetAllAsync();

			if (request.Roles.Except(allowedRoles.Select(x => x.Name)).Any())
				throw new InvalidRoles();

			if (await _userManager.FindByIdAsync(id) is not { } user)
				throw new UserNotFound(id);

			user = request.Adapt(user);

			var result = await _userManager.UpdateAsync(user);

			if (result.Succeeded)
			{
				var CurrentRoles = await _userManager.GetRolesAsync(user);
				if (CurrentRoles.Any())
				{
					await _userManager.RemoveFromRolesAsync(user, CurrentRoles);
				}
				await _userManager.AddToRolesAsync(user, request.Roles);

				return ;
			}

			var error = result.Errors.First();

			throw new IdentityResultError(error.Description);
		}

		public async Task ToggleStatus(string id)
		{
			if (await _userManager.FindByIdAsync(id) is not { } user)
				throw new UserNotFound(id);

			user.IsDisabled = !user.IsDisabled;

			var result = await _userManager.UpdateAsync(user);

			if (result.Succeeded)
				return ;

			var error = result.Errors.First();

			throw new IdentityResultError(error.Description);
		}

		public async Task Unlock(string id)
		{
			if (await _userManager.FindByIdAsync(id) is not { } user)
				throw new UserNotFound(id);

			var result = await _userManager.SetLockoutEndDateAsync(user, null);

			if (result.Succeeded)
				return;

			var error = result.Errors.First();

			throw new IdentityResultError(error.Description);
		}
		public async Task<UserProfileResponse> GetUserProfileAsync(string userId)
		{
			var user = await _userManager.Users
				.Where(u => u.Id == userId)
				.ProjectToType<UserProfileResponse>()
				.SingleAsync();
			return user;
		}
		public async Task UpdateUserProfileAsync(string userId, UpdateUserProfileRequest request, IFormFile? file)
		{
			var user = await _userManager.FindByIdAsync(userId);
			user = request.Adapt(user);
			if (file is not null && file.Length > 0)
			{
				using var stream = file.OpenReadStream();
				var imagePath = await _fileStorageService.UploadFileAsync(
					stream,
					file.FileName,
					$"Users/{user!.Email}",
					file.ContentType);
				user.ProfilePictureUrl = imagePath;
				user.ProfilePictureBase64 = await ConvertFileToBase64Async(file);
			}
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


		private async Task<string> ConvertFileToBase64Async(IFormFile file)
		{
			if (file == null || file.Length == 0)
				return string.Empty;

			using var memoryStream = new MemoryStream();
			await file.CopyToAsync(memoryStream);
			return Convert.ToBase64String(memoryStream.ToArray());
		}

	}
}
