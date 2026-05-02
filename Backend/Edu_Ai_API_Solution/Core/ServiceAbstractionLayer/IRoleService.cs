using Shared.Dtos.RolesDto.Request;
using Shared.Dtos.RolesDto.Response;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace ServiceAbstractionLayer
{
    public interface IRoleService
    {
        Task<IEnumerable<RoleResponse>> GetAllAsync(bool? includeDisabled = false, CancellationToken cancellationToken = default);
        Task<RoleDetailResponse> GetAsync(string id, CancellationToken cancellationToken = default);
        Task<RoleDetailResponse> AddAsync(RoleRequest request, CancellationToken cancellationToken = default);
        Task UpdateAsync(string id, RoleRequest request, CancellationToken cancellationToken = default);
        Task ToggleStatusAsync(string id, CancellationToken cancellationToken = default);
    }
}
