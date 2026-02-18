
using DomainLayer.Exceptions.User;
using Microsoft.AspNetCore.WebUtilities;
using Microsoft.EntityFrameworkCore;
using ServiceAbstractionLayer;
using Shared.Dtos.AuthDto.Request;
using Shared.Dtos.AuthDto.Response;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace ServiceLayer.Services
{
	public class AuthService(UserManager<ApplicationUser> userManager,
		SignInManager<ApplicationUser> signInManager,
		IJwtProvider jwtProvider,
		IFileStorageService fileStorageService) : IAuthunticationService
	{
		private readonly UserManager<ApplicationUser> _userManager = userManager;
		private readonly SignInManager<ApplicationUser> _signInManager = signInManager;
		private readonly IJwtProvider _jwtProvider = jwtProvider;
		private readonly IFileStorageService _fileStorageService = fileStorageService;
		private readonly int _RefreshTokenExpirationDays = 14;



		public async Task<AuthResponse> GetTokenAsync(string email, string password)
		{
			var user = await _userManager.FindByEmailAsync(email);
			if (user is null)
			{
				throw new InvalidCredentials();
			}
			var isPasswordValid = await _userManager.CheckPasswordAsync(user, password);
			if (!isPasswordValid)
			{
				throw new InvalidCredentials();
			}
			var result = await _signInManager.PasswordSignInAsync(user, password, false, false);
			if (result.Succeeded)
			{
				// Here you would typically generate a JWT token or similar
				var (token, ExpireIn) = _jwtProvider.GenerateToken(user);
				var refreshToken = GenerateRefreshToken();
				var refreshTokenExpiration = DateTime.UtcNow.AddDays(_RefreshTokenExpirationDays);
				user.RefreshTokens.Add(new RefreshToken
				{
					Token = refreshToken,
					ExpiresOn = refreshTokenExpiration
				});
				await _userManager.UpdateAsync(user);
				var response = new AuthResponse
				{
					FirstName = user.FirstName!,
					LastName = user.LastName!,
					ProfilePictureUrl = user.ProfilePictureUrl,
					Email = user.Email,
					token = token,
					ExpinresIn = ExpireIn,
					RefreshToken = refreshToken,
					RefreshTokenExpiration = refreshTokenExpiration
				};
				return response;

			}
			if(result.IsNotAllowed)
			{
				throw new EmailNotConfirmed(email);
			}
			else
			{
				throw new InvalidCredentials();
			}
		}
		public async Task RegisterAsync(RegisterRequest request, IFormFile? file)
		{
			var emailExists = await _userManager.Users.AnyAsync(u => u.Email == request.Email);
			if (emailExists)
			{
				throw new DuplicatedEmail(request.Email);
			}
			var user = new ApplicationUser
			{
				UserName = request.Email,
				Email = request.Email,
				FirstName = request.FirstName,
				LastName = request.LastName,
				DateOfBirth = request.DateOfBirth
			};
			if (file is not null && file.Length > 0)
			{
				using var stream = file.OpenReadStream();
				var imagePath = await _fileStorageService.UploadFileAsync(
					stream,
					file.FileName,
					$"Users/{user.Email}",
					file.ContentType);
				user.ProfilePictureUrl = imagePath;
				user.ProfilePictureBase64 = await ConvertFileToBase64Async(file);
			}
		
			var result = await _userManager.CreateAsync(user, request.Password);
			if (!result.Succeeded)
			{
				var error = result.Errors.FirstOrDefault();
				throw new Exception(error != null ? error.Description : "User registration failed");
			}
			//var code = await _userManager.GenerateEmailConfirmationTokenAsync(user);
			//code = WebEncoders.Base64UrlEncode(Encoding.UTF8.GetBytes(code));
			//_logger.LogInformation("User registered successfully. UserId: {UserId}, Email: {Email}, ConfirmationCode: {ConfirmationCode}", user.Id, user.Email, code);
			// TODO : Send confirmation email with the code
			//await SendConfirmationEmailAsync(user, code);
			

		}
		public async Task<AuthResponse> GetRefreshTokenAsync(string Token, string RefreshToken)
		{
			var userId = _jwtProvider.ValidateToken(Token);
			if (userId is null)
			{
				throw new InvalidJwtToken();
			}
			var user = await _userManager.FindByIdAsync(userId);
			if (user is null)
			{
				throw new InvalidJwtToken();
			}
			var storedRefreshToken = user.RefreshTokens.SingleOrDefault(u => u.Token == RefreshToken && u.IsActive);
			if (storedRefreshToken is null)
			{
				throw new InvalidJwtToken();
			}
			storedRefreshToken.RevokedOn = DateTime.UtcNow; // Invalidate the old refresh token

			var (newToken, expiresIn) = _jwtProvider.GenerateToken(user);
			var newRefreshToken = GenerateRefreshToken();
			var refreshTokenExpiration = DateTime.UtcNow.AddDays(_RefreshTokenExpirationDays);

			user.RefreshTokens.Add(new RefreshToken
			{
				Token = newRefreshToken,
				ExpiresOn = refreshTokenExpiration
			});

			await _userManager.UpdateAsync(user);

			var response = new AuthResponse {
				FirstName = user.FirstName!,
				LastName = user.LastName!,
				ProfilePictureUrl = user.ProfilePictureUrl,
				Email = user.Email,
				token = newToken,
				ExpinresIn = expiresIn,
				RefreshToken = newRefreshToken,
				RefreshTokenExpiration = refreshTokenExpiration
			};

			return response;
		}
		public async Task RevokeRefreshTokenAsync(string Token, string RefreshToken)
		{
			var userId = _jwtProvider.ValidateToken(Token);
			if (userId is null)
			{
				throw new InvalidRefreshToken();
			}
			var user = await _userManager.FindByIdAsync(userId);
			if (user is null)
			{
				throw new InvalidRefreshToken();

			}
			var storedRefreshToken = user.RefreshTokens.SingleOrDefault(u => u.Token == RefreshToken && u.IsActive);
			if (storedRefreshToken is null)
			{
				throw new InvalidRefreshToken();

			}
			storedRefreshToken.RevokedOn = DateTime.UtcNow; // Invalidate the old refresh token
			await _userManager.UpdateAsync(user);
		}

		private string GenerateRefreshToken()
		{
			return Convert.ToBase64String(RandomNumberGenerator.GetBytes(64));
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
