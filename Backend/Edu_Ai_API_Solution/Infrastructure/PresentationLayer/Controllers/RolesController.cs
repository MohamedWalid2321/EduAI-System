using Shared.Dtos.RolesDto.Request;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PresentationLayer.Controllers
{

	[Route("api/[controller]")]
	[ApiController]
	public class RolesController(IServiceManager serviceManager) : ControllerBase
	{
		private readonly IServiceManager _serviceManager = serviceManager;

		[HttpGet("")]
		[HasPermission(Permissions.GetRoles)]
		public async Task<IActionResult> GetAll([FromQuery] bool includeDisabled)
		{
			var roles = await _serviceManager.RoleService.GetAllAsync(includeDisabled);

			return Ok(roles);
		}

		[HttpGet("{id}")]
		[HasPermission(Permissions.GetRoles)]
		public async Task<IActionResult> Get([FromRoute] string id)
		{
			var result = await _serviceManager.RoleService.GetAsync(id);

			return Ok(result);
		}

		[HttpPost("")]
		[HasPermission(Permissions.AddRoles)]
		public async Task<IActionResult> Add([FromBody] RoleRequest request)
		{
			var result = await _serviceManager.RoleService.AddAsync(request);

			return CreatedAtAction(nameof(Get), new { result.Id }, result);
		}

		[HttpPut("{id}")]
		[HasPermission(Permissions.UpdateRoles)]
		public async Task<IActionResult> Update([FromRoute] string id, [FromBody] RoleRequest request)
		{
			await _serviceManager.RoleService.UpdateAsync(id,request);

			return NoContent();
		}

		[HttpPut("{id}/toggle-status")]
		[HasPermission(Permissions.UpdateRoles)]
		public async Task<IActionResult> ToggleStatus([FromRoute] string id)
		{
			await _serviceManager.RoleService.ToggleStatusAsync(id);

			return NoContent();
		}
		[HttpGet("permissions/all")]
		public IActionResult GetAllPermissions()
		{
			var permissions = Permissions.GetAllPermissions()
				.Where(p => p != null)
				.OrderBy(p => p)
				.ToList();

			return Ok(new { count = permissions.Count, permissions });
		}
	}
}
