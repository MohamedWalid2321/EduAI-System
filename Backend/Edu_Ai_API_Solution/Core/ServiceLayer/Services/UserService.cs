namespace ServiceLayer.Services
{
	public class UserService(UserManager<ApplicationUser> userManager,
		IRoleService roleService,
		IUnitOfWork unitOfWork,
		IFileStorageService fileStorageService) : IUserService
	{
		private readonly UserManager<ApplicationUser> _userManager = userManager;
		private readonly IRoleService _roleService = roleService;
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		private readonly IFileStorageService _fileStorageService = fileStorageService;

		public async Task<IEnumerable<UserResponse>> GetAllAsync(bool? IncludeNotConfirmed = false, bool? includeDisabled = false, CancellationToken cancellationToken = default)
		{
			var users = await _userManager.Users
				.Where(u => !(u.IsDisabled && !(includeDisabled.HasValue && includeDisabled.Value)) && (u.EmailConfirmed || (IncludeNotConfirmed.HasValue && IncludeNotConfirmed.Value)))
				.ToListAsync(cancellationToken);

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
					AcademicYear = user.AcademicYearEnum.ToString()!,
					DepartmentId = user.DepartmentId ?? 0,
					Roles = roles
				});
			}
			return userResponses;
		}

		public async Task<IEnumerable<InstructorsDetailsResponse>> GetAllInstructorByDepartmentIdAsync(int departmentId, CancellationToken cancellationToken = default)
		{
			var instructors = await _userManager.GetUsersInRoleAsync(DefaultRoles.Instructor);
			var filteredInstructors = instructors
				.Where(i => i.DepartmentId == departmentId)
				.ToList();
			return filteredInstructors.Adapt<IEnumerable<InstructorsDetailsResponse>>();
		}

		public async Task<UserResponse> GetAsync(string id, CancellationToken cancellationToken = default)
		{
			if (await _userManager.FindByIdAsync(id) is not { } user)
				throw new UserNotFound(id);

			var userRoles = await _userManager.GetRolesAsync(user);
			return (user, userRoles).Adapt<UserResponse>();
		}

		public async Task<UserResponse> AddAsync(CreateUserRequest request, CancellationToken cancellationToken = default)
		{
			var emailIsExists = await _userManager.Users.AnyAsync(x => x.Email == request.Email, cancellationToken);
			if (emailIsExists)
				throw new DuplicatedEmail(request.Email);

			var DepartmentRepository = _unitOfWork.GetRepository<Department, int>();
			if (request.DepartmentId.HasValue && await DepartmentRepository.GetByIdAsync(request.DepartmentId.Value, cancellationToken) is not { } department)
				throw new DepartmentNotFoundException(request.DepartmentId.Value);

			var allowedRoles = await _roleService.GetAllAsync(cancellationToken: cancellationToken);
			if (request.Roles.Except(allowedRoles.Select(x => x.Name)).Any())
				throw new InvalidRoles();

			var user = request.Adapt<ApplicationUser>();
			user.EnrolledAt = DateTime.UtcNow;

			var result = await _userManager.CreateAsync(user, request.Password);
			if (result.Succeeded)
			{
				await _userManager.AddToRolesAsync(user, request.Roles);
				return (user, request.Roles).Adapt<UserResponse>();
			}

			var error = result.Errors.First();
			throw new IdentityResultError(error.Description);
		}

		public async Task UpdateAsync(string id, UpdateUserRequest request, CancellationToken cancellationToken = default)
		{
			var emailIsExists = await _userManager.Users.AnyAsync(x => x.Email == request.Email && x.Id != id, cancellationToken);
			if (emailIsExists)
				throw new DuplicatedEmail(request.Email);

			if (!Enum.TryParse<AcademicYearEnum>(request.AcademicYear, true, out var academicYear))
				throw new InvalidAcademicYear();

			var DepartmentRepository = _unitOfWork.GetRepository<Department, int>();
			if (request.DepartmentId.HasValue && await DepartmentRepository.GetByIdAsync(request.DepartmentId.Value, cancellationToken) is not { } department)
				throw new DepartmentNotFoundException(request.DepartmentId.Value);

			var allowedRoles = await _roleService.GetAllAsync(cancellationToken: cancellationToken);
			if (request.Roles.Except(allowedRoles.Select(x => x.Name)).Any())
				throw new InvalidRoles();

			if (await _userManager.FindByIdAsync(id) is not { } user)
				throw new UserNotFound(id);

			var originalDepartmentId = user.DepartmentId;
			var originalAcademicYear = user.AcademicYearEnum;

			user = request.Adapt(user);
			user.AcademicYearEnum = academicYear;

			var result = await _userManager.UpdateAsync(user);
			if (result.Succeeded)
			{
				var CurrentRoles = await _userManager.GetRolesAsync(user);
				if (CurrentRoles.Any())
					await _userManager.RemoveFromRolesAsync(user, CurrentRoles);
				await _userManager.AddToRolesAsync(user, request.Roles);

				var enrollmentDataChanged = user.DepartmentId != originalDepartmentId || user.AcademicYearEnum != originalAcademicYear;
				if (enrollmentDataChanged && user.DepartmentId.HasValue && user.AcademicYearEnum.HasValue)
					BackgroundJob.Enqueue<IEnrollmentService>(s => s.ReEnrollAsync(id, default));
				return;
			}

			var error = result.Errors.First();
			throw new IdentityResultError(error.Description);
		}

		public async Task ToggleStatus(string id, CancellationToken cancellationToken = default)
		{
			if (await _userManager.FindByIdAsync(id) is not { } user)
				throw new UserNotFound(id);

			user.IsDisabled = !user.IsDisabled;
			var result = await _userManager.UpdateAsync(user);
			if (result.Succeeded)
				return;

			var error = result.Errors.First();
			throw new IdentityResultError(error.Description);
		}

		public async Task Unlock(string id, CancellationToken cancellationToken = default)
		{
			if (await _userManager.FindByIdAsync(id) is not { } user)
				throw new UserNotFound(id);

			var result = await _userManager.SetLockoutEndDateAsync(user, null);
			if (result.Succeeded)
				return;

			var error = result.Errors.First();
			throw new IdentityResultError(error.Description);
		}

		public async Task LevelUp(string id, CancellationToken cancellationToken = default)
		{
			if (await _userManager.FindByIdAsync(id) is not { } user)
				throw new UserNotFound(id);

			if (user.AcademicYearEnum == AcademicYearEnum.Fifth)
				throw new MaxAcademicYearReached();

			user.AcademicYearEnum += 1;
			var result = await _userManager.UpdateAsync(user);
			if (result.Succeeded)
			{
				BackgroundJob.Enqueue<IEnrollmentService>(s => s.ReEnrollAsync(id, default));
				return;
			}

			var error = result.Errors.First();
			throw new IdentityResultError(error.Description);
		}

		public async Task<UserProfileResponse> GetUserProfileAsync(string userId, CancellationToken cancellationToken = default)
		{
			var user = await _userManager.Users
				.Where(u => u.Id == userId)
				.ProjectToType<UserProfileResponse>()
				.SingleAsync(cancellationToken);
			return user;
		}

		public async Task UpdateUserProfileAsync(string userId, UpdateUserProfileRequest request, IFormFile? file, CancellationToken cancellationToken = default)
		{
			var user = await _userManager.FindByIdAsync(userId);
			if ((!string.IsNullOrWhiteSpace(request.AcademicYear)) && (!await _userManager.IsInRoleAsync(user!, DefaultRoles.Student)))
				throw new IsNotStudentException();

			if (!Enum.TryParse<AcademicYearEnum>(request.AcademicYear, true, out var academicYear))
				throw new InvalidAcademicYear();

			if (user!.IsDisabled)
				throw new DisabledUser(user.Email!);

			if (!string.IsNullOrEmpty(user!.ProfilePictureUrl))
				await _fileStorageService.DeleteFileAsync(user!.ProfilePictureUrl);

			var originalDepartmentId = user.DepartmentId;
			var originalAcademicYear = user.AcademicYearEnum;

			user = request.Adapt(user);
			user!.AcademicYearEnum = academicYear;

			if (file is not null && file.Length > 0)
			{
				using var stream = file.OpenReadStream();
				var imagePath = await _fileStorageService.UploadFileAsync(
					stream,
					file.FileName,
					$"Users/{user!.Email}",
					file.ContentType);
				user.ProfilePictureUrl = imagePath;
			}
			await _userManager.UpdateAsync(user!);

			var enrollmentDataChanged = user.DepartmentId != originalDepartmentId || user.AcademicYearEnum != originalAcademicYear;
			if (enrollmentDataChanged && (await _userManager.IsInRoleAsync(user!, DefaultRoles.Student)))
				BackgroundJob.Enqueue<IEnrollmentService>(s => s.ReEnrollAsync(userId, default));
		}

		public async Task ChangePasswordAsync(string userId, ChangePasswordRequest request, CancellationToken cancellationToken = default)
		{
			var user = await _userManager.FindByIdAsync(userId);
			var result = await _userManager.ChangePasswordAsync(user!, request.CurrentPassword, request.NewPassword);
			if (!result.Succeeded)
				throw new IdentityResultError(string.Join(", ", result.Errors.Select(e => e.Description)));
		}
	}
}
