namespace ServiceLayer.Services
{
	public class RoleService(RoleManager<ApplicationRole> roleManager ) : IRoleService
	{
		private readonly RoleManager<ApplicationRole> _roleManager = roleManager;

		public async Task<IEnumerable<RoleResponse>> GetAllAsync(bool? includeDisabled = false) =>
		await _roleManager.Roles
			.Where(x => !x.IsDefault && (!x.IsDeleted || (includeDisabled.HasValue && includeDisabled.Value)))
			.ProjectToType<RoleResponse>()
			.ToListAsync();

		public async Task<RoleDetailResponse> GetAsync(string id)
		{
			if (await _roleManager.FindByIdAsync(id) is not { } role)
				throw new RoleNotFound();

			var permissions = await _roleManager.GetClaimsAsync(role);
			var response = new RoleDetailResponse
			{
				Id = role.Id,
				Name = role.Name!,
				IsDeleted = role.IsDeleted,
				Permissions = permissions.Select(x => x.Value).ToList()
			};

			return response;
		}

		public async Task<RoleDetailResponse> AddAsync(RoleRequest request)
		{
			var roleIsExists = await _roleManager.RoleExistsAsync(request.Name);

			if (roleIsExists)
				//return Result.Failure<RoleDetailResponse>(RoleErrors.DuplicatedRole);
				throw new DuplicatedRole();

			var allowedPermissions = Permissions.GetAllPermissions();

			if (request.Permissions.Except(allowedPermissions).Any())
				throw new InvalidPermissions();

			var role = new ApplicationRole
			{
				Name = request.Name,
				ConcurrencyStamp = Guid.NewGuid().ToString()
			};

			var result = await _roleManager.CreateAsync(role);

			if (result.Succeeded)
			{
				foreach (var permission in request.Permissions.Distinct())
				{
					var claim = new Claim(Permissions.Type, permission);
					await _roleManager.AddClaimAsync(role, claim);
				}
				var response = new RoleDetailResponse
				{
					Id = role.Id,
					Name = role.Name!,
					IsDeleted = role.IsDeleted,
					Permissions = request.Permissions.Distinct().ToList()
				};

				return response;
			}

			var error = result.Errors.First();
			throw new IdentityResultError(error.Description);
		}

		public async Task UpdateAsync(string id, RoleRequest request)
		{
			var roleIsExists = await _roleManager.Roles.AnyAsync(x => x.Name == request.Name && x.Id != id);

			if (roleIsExists)
				throw new DuplicatedRole();

			if (await _roleManager.FindByIdAsync(id) is not { } role)
				throw new RoleNotFound();

			var allowedPermissions = Permissions.GetAllPermissions();

			if (request.Permissions.Except(allowedPermissions).Any())
				throw new InvalidPermissions();

			role.Name = request.Name;

			var result = await _roleManager.UpdateAsync(role);

			if (result.Succeeded)
			{
				var currentClaims = await _roleManager.GetClaimsAsync(role);
				var currentPermissions = currentClaims
					.Where(c => c.Type == Permissions.Type)
					.Select(c => c.Value!)
					.ToList();

				// Add new permissions
				var newPermissions = request.Permissions.Except(currentPermissions);
				foreach (var permission in newPermissions)
				{
					var claim = new Claim(Permissions.Type, permission);
					await _roleManager.AddClaimAsync(role, claim);
				}

				// Remove old permissions
				var removedPermissions = currentPermissions.Except(request.Permissions);
				foreach (var permission in removedPermissions)
				{
					var claim = new Claim(Permissions.Type, permission);
					await _roleManager.RemoveClaimAsync(role, claim);
				}
				return ;
			}

			var error = result.Errors.First();

			throw new IdentityResultError(error.Description);
		}

		public async Task ToggleStatusAsync(string id)
		{
			if (await _roleManager.FindByIdAsync(id) is not { } role)
				throw new RoleNotFound();

			role.IsDeleted = !role.IsDeleted;

			await _roleManager.UpdateAsync(role);

			return ;
		}
	}
}
