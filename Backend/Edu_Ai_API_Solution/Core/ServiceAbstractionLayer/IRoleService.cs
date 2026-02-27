using Shared.Dtos.RolesDto.Request;
using Shared.Dtos.RolesDto.Response;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
	public interface IRoleService
	{
		Task<IEnumerable<RoleResponse>> GetAllAsync(bool? includeDisabled = false);
		Task<RoleDetailResponse> GetAsync(string id);
		Task<RoleDetailResponse> AddAsync(RoleRequest request);
		Task UpdateAsync(string id, RoleRequest request);
		Task ToggleStatusAsync(string id);
	}
}
