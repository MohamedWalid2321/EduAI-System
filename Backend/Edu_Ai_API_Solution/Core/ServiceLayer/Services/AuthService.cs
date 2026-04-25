namespace ServiceLayer.Services
{
	public class AuthService(UserManager<ApplicationUser> userManager,
		SignInManager<ApplicationUser> signInManager,
		IJwtProvider jwtProvider,
		IFileStorageService fileStorageService,
		IHttpContextAccessor httpContextAccessor,
		IEmailSender emailSender,
		IEmailBodyBuilder EmailBodyBuilder,
		RoleManager<ApplicationRole> roleManager,
		IUnitOfWork unitOfWork,
		ILogger<AuthService> logger) : IAuthunticationService
	{
		private readonly UserManager<ApplicationUser> _userManager = userManager;
		private readonly SignInManager<ApplicationUser> _signInManager = signInManager;
		private readonly IJwtProvider _jwtProvider = jwtProvider;
		private readonly IFileStorageService _fileStorageService = fileStorageService;
		private readonly IHttpContextAccessor _httpContextAccessor = httpContextAccessor;
		private readonly ILogger<AuthService> _logger = logger;
		private readonly IEmailSender _emailSender = emailSender;
		private readonly IEmailBodyBuilder _emailBodyBuilder = EmailBodyBuilder;
		private readonly RoleManager<ApplicationRole> _roleManager = roleManager;
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		private readonly int _RefreshTokenExpirationDays = 14;



		public async Task<AuthResponse> GetTokenAsync(string email, string password)
		{
			var user = await _userManager.FindByEmailAsync(email);
			if (user is null)
			{
				throw new InvalidCredentials();
			}
			if (user.IsDisabled)
				throw new DisabledUser(email);

			var result = await _signInManager.PasswordSignInAsync(user, password, false, true);
			if (result.Succeeded)
			{
				// Here you would typically generate a JWT token or similar
				var (UserRoles, UserPermissions) = await GetRolesAndPermissionAsync(user);
				var (token, ExpireIn) = _jwtProvider.GenerateToken(user, UserRoles, UserPermissions);
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
					id=user.Id,
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
			else if(result.IsLockedOut)
			{
				throw new UserLockedOut(email);
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
			if (!Enum.TryParse<AcademicYearEnum>(request.AcademicYear, true, out var academicYear))
			{
				throw new InvalidAcademicYear();
			}
			var DepartmentRepository = _unitOfWork.GetRepository<Department,int>();
			if (await DepartmentRepository.GetByIdAsync(request.DepartmentId) is not { }) {
				throw new DepartmentNotFoundException(request.DepartmentId);
			}

			var user = new ApplicationUser
			{
				UserName = request.Email,
				Email = request.Email,
				FirstName = request.FirstName,
				LastName = request.LastName,
				DateOfBirth = request.DateOfBirth,
				DepartmentId = request.DepartmentId,
                AcademicYearEnum = academicYear,
				IsEnrolled = true,
				EnrolledAt = DateTime.UtcNow
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
			}
		
			var result = await _userManager.CreateAsync(user, request.Password);
			if (!result.Succeeded)
			{
				var error = result.Errors.FirstOrDefault();
				throw new Exception(error != null ? error.Description : "User registration failed");
			}
			var code = await _userManager.GenerateEmailConfirmationTokenAsync(user);
			code = WebEncoders.Base64UrlEncode(Encoding.UTF8.GetBytes(code));
			_logger.LogInformation("User registered successfully. UserId: {UserId}, Email: {Email}, ConfirmationCode: {ConfirmationCode}", user.Id, user.Email, code);
			await SendConfirmationEmailAsync(user, code);


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
			if (user.IsDisabled)
				throw new DisabledUser(user.Email!);

			if (user.LockoutEnd > DateTime.UtcNow)
				throw new UserLockedOut(user.Email!);
			var storedRefreshToken = user.RefreshTokens.SingleOrDefault(u => u.Token == RefreshToken && u.IsActive);
			if (storedRefreshToken is null)
			{
				throw new InvalidJwtToken();
			}
			storedRefreshToken.RevokedOn = DateTime.UtcNow; // Invalidate the old refresh token
			var (UserRoles, UserPermissions) = await GetRolesAndPermissionAsync(user);
			var (newToken, expiresIn) = _jwtProvider.GenerateToken(user, UserRoles, UserPermissions);
			var newRefreshToken = GenerateRefreshToken();
			var refreshTokenExpiration = DateTime.UtcNow.AddDays(_RefreshTokenExpirationDays);

			user.RefreshTokens.Add(new RefreshToken
			{
				Token = newRefreshToken,
				ExpiresOn = refreshTokenExpiration
			});

			await _userManager.UpdateAsync(user);

			var response = new AuthResponse {
				id= user.Id,
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
		public async Task ConfirmEmailAsync(ConfirmEmailRequest request)
		{
			var user = await _userManager.FindByIdAsync(request.userId);
			if (user == null)
			{
				throw new InvalidCode();
			}
			if (user.EmailConfirmed)
			{
				throw new DuplicatedConfirmation(user.Email!);
			}
			var code = request.code;
			try
			{
				code = Encoding.UTF8.GetString(WebEncoders.Base64UrlDecode(request.code));
			}
			catch (FormatException)
			{

				throw new InvalidCode();
			}
			var result = await _userManager.ConfirmEmailAsync(user, code);
			if (!result.Succeeded)
			{
				var error = result.Errors.FirstOrDefault();

				//return Result.Failure(new Error(error.Code, error.Description, StatusCodes.Status400BadRequest));
				throw new Exception(error != null ? error.Description : "Email confirmation failed");
			}
			await _userManager.AddToRoleAsync(user, DefaultRoles.Student);
		BackgroundJob.Enqueue<IEnrollmentService>(s => s.AutoEnrollAsync(user.Id));
		}

		public async Task ResendConfirmEmailAsync(ResendConfirmEmailRequest request)
		{
			var user = await _userManager.FindByEmailAsync(request.Email);
			if (user is null)
			{
				// For security reasons, we don't reveal whether the email exists or not
				_logger.LogWarning("Resend confirmation requested for non-existent email: {Email}", request.Email);
				return;
			}
			if (user.EmailConfirmed)
			{
				throw new DuplicatedConfirmation(user.Email!);
			}
			var code = await _userManager.GenerateEmailConfirmationTokenAsync(user);
			code = WebEncoders.Base64UrlEncode(Encoding.UTF8.GetBytes(code));
			_logger.LogInformation("User registered successfully. UserId: {UserId}, Email: {Email}, ConfirmationCode: {ConfirmationCode}", user.Id, user.Email, code);
			// TODO : Send confirmation email with the code
			await SendConfirmationEmailAsync(user, code);
		}
		public async Task SendResetPasswordCodeAsync(string email)
		{
			if (await _userManager.FindByEmailAsync(email) is not { } user)
			{
				// For security reasons, we don't reveal whether the email exists or not
				_logger.LogWarning("Password reset requested for non-existent email: {Email}", email);
				return;
			}
			var code = await _userManager.GeneratePasswordResetTokenAsync(user);
			code = WebEncoders.Base64UrlEncode(Encoding.UTF8.GetBytes(code));
			_logger.LogInformation("Reset code: {code}", code);

			await SendResetPasswordEmail(user, code);

		}

		public async Task ResetPasswordAsync(ResetPasswordRequest request)
		{
			var user = await _userManager.FindByEmailAsync(request.Email);
			if (user is null || !user.EmailConfirmed)
			{
				throw new InvalidCode();
			}
			IdentityResult result;
			try
			{
				var code = Encoding.UTF8.GetString(WebEncoders.Base64UrlDecode(request.Code));
				result = await _userManager.ResetPasswordAsync(user, code, request.NewPassword);
			}
			catch (FormatException)
			{
				result = IdentityResult.Failed(_userManager.ErrorDescriber.InvalidToken());
			}
			if (!result.Succeeded)
			{
				var error = result.Errors.FirstOrDefault();
				throw new Exception(error != null ? error.Description : "Password reset failed");
			}
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
		private async Task SendConfirmationEmailAsync(ApplicationUser user, string code)
		{
			var origin = _httpContextAccessor.HttpContext!.Request.Headers.Origin;
			var body = _emailBodyBuilder.GenerateEmailBody("EmailConfirmation",
				new Dictionary<string, string> {
					{"{{name}}",user.FirstName!},
					{ "{{action_url}}", $"{origin}/Auth/emailConfrimation?userId={user.Id}&code={code}" }
				}
				);
			//await _emailSender.SendEmailAsync(user.Email!, "✅ Lumino: Confirm your email", body);
			BackgroundJob.Enqueue(() => _emailSender.SendEmailAsync(user.Email!, "✅ Lumino: Confirm your email", body));
			_logger.LogInformation("Confirmation email sent. UserId: {UserId}, Email: {Email}, ConfirmationCode: {ConfirmationCode}", user.Id, user.Email, code);
			await Task.CompletedTask;
		}
		private async Task SendResetPasswordEmail(ApplicationUser user, string code)
		{
			var origin = _httpContextAccessor.HttpContext?.Request.Headers.Origin;

			var emailBody = EmailBodyBuilder.GenerateEmailBody("ForgetPassword",
				 new Dictionary<string, string>
				{
				{ "{{name}}", user.FirstName! },
				{ "{{action_url}}", $"{origin}/auth/forgetPassword?email={user.Email}&code={code}" }
				}
			);

			//await _emailSender.SendEmailAsync(user.Email!, "✅ Lumino: Change Password", emailBody);
			BackgroundJob.Enqueue(() => _emailSender.SendEmailAsync(user.Email!, "✅ Lumino: Change Password", emailBody));
			await Task.CompletedTask;
		}

		private async Task<(IEnumerable<string> UserRoles, IEnumerable<string> UserPermissions)> GetRolesAndPermissionAsync(ApplicationUser user)
		{
			var userRoles = await _userManager.GetRolesAsync(user);
			var permissions = new List<string>();

			foreach (var roleName in userRoles)
			{
				var role = await _roleManager.FindByNameAsync(roleName);
				if (role != null)
				{
					var claims = await _roleManager.GetClaimsAsync(role);
					permissions.AddRange(claims.Select(c => c.Value!));
				}
			}

			return (userRoles, permissions.Distinct());
		}
	}
}
