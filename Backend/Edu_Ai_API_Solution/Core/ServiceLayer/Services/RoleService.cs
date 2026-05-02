namespace ServiceLayer.Services
{
	public class RoleService(RoleManager<ApplicationRole> roleManager) : IRoleService
	{
		private readonly RoleManager<ApplicationRole> _roleManager = roleManager;

		public async Task<IEnumerable<RoleResponse>> GetAllAsync(bool? includeDisabled = false, CancellationToken cancellationToken = default) =>
			await _roleManager.Roles
				.Where(x => !x.IsDefault && (!x.IsDeleted || (includeDisabled.HasValue && includeDisabled.Value)))
				.ProjectToType<RoleResponse>()
				.ToListAsync(cancellationToken);

		public async Task<RoleDetailResponse> GetAsync(string id, CancellationToken cancellationToken = default)
		{
			if (await _roleManager.FindByIdAsync(id) is not { } role)
				throw new RoleNotFound();

			var permissions = await _roleManager.GetClaimsAsync(role);
			return new RoleDetailResponse
			{
				Id = role.Id,
				Name = role.Name!,
				IsDeleted = role.IsDeleted,
				IsEnrollable = role.IsEnrollable,
				Permissions = permissions.Select(x => x.Value).ToList()
			};
		}

		public async Task<RoleDetailResponse> AddAsync(RoleRequest request, CancellationToken cancellationToken = default)
		{
			var roleIsExists = await _roleManager.RoleExistsAsync(request.Name);
			if (roleIsExists)
				throw new DuplicatedRole();

			var allowedPermissions = Permissions.GetAllPermissions();
			if (request.Permissions.Except(allowedPermissions).Any())
				throw new InvalidPermissions();

			var role = new ApplicationRole
			{
				Name = request.Name,
				IsEnrollable = request.IsEnrollable,
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
				return new RoleDetailResponse
				{
					Id = role.Id,
					Name = role.Name!,
					IsDeleted = role.IsDeleted,
					IsEnrollable = role.IsEnrollable,
					Permissions = request.Permissions.Distinct().ToList()
				};
			}

			var error = result.Errors.First();
			throw new IdentityResultError(error.Description);
		}

		public async Task UpdateAsync(string id, RoleRequest request, CancellationToken cancellationToken = default)
		{
			var roleIsExists = await _roleManager.Roles.AnyAsync(x => x.Name == request.Name && x.Id != id, cancellationToken);
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

				foreach (var permission in request.Permissions.Except(currentPermissions))
					await _roleManager.AddClaimAsync(role, new Claim(Permissions.Type, permission));

				foreach (var permission in currentPermissions.Except(request.Permissions))
					await _roleManager.RemoveClaimAsync(role, new Claim(Permissions.Type, permission));

				return;
			}

			var error = result.Errors.First();
			throw new IdentityResultError(error.Description);
		}

		public async Task ToggleStatusAsync(string id, CancellationToken cancellationToken = default)
		{
			if (await _roleManager.FindByIdAsync(id) is not { } role)
				throw new RoleNotFound();

			role.IsDeleted = !role.IsDeleted;
			await _roleManager.UpdateAsync(role);
		}
	}
}
