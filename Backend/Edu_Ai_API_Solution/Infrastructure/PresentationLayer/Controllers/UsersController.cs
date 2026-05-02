using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PresentationLayer.Controllers
{
	[Route("api/[controller]")]
	[ApiController]
	public class UsersController(IServiceManager serviceManager) : ControllerBase
	{
		private readonly IServiceManager _serviceManager = serviceManager;

		[HttpGet("")]
		[HasPermission(Permissions.GetUsers)]
		public async Task<IActionResult> GetAll([FromQuery] bool IncludeNotConfirmed, [FromQuery] bool includeDisabled, CancellationToken cancellationToken)
		{
			return Ok(await _serviceManager.UserService.GetAllAsync(IncludeNotConfirmed, includeDisabled, cancellationToken));
		}

		[HttpGet("{id}")]
		[HasPermission(Permissions.GetUsers)]
		public async Task<IActionResult> Get([FromRoute] string id, CancellationToken cancellationToken)
		{
			var result = await _serviceManager.UserService.GetAsync(id, cancellationToken);

			return Ok(result); 
		}
		[Authorize]
		[HttpGet("{DepartmentId}/Instructors")]
		public async Task<IActionResult> GetUsersByDepartmentId([FromRoute] int DepartmentId, CancellationToken cancellationToken)
		{
			var result = await _serviceManager.UserService.GetAllInstructorByDepartmentIdAsync(DepartmentId, cancellationToken);
			return Ok(result);
		}

		[HttpPost("")]
		[HasPermission(Permissions.AddUsers)]
		public async Task<IActionResult> Add([FromBody] CreateUserRequest request, CancellationToken cancellationToken)
		{
			var result = await _serviceManager.UserService.AddAsync(request, cancellationToken);
			return CreatedAtAction(nameof(Get), new { result.Id }, result);
		}

		[HttpPut("{id}")]
		[HasPermission(Permissions.UpdateUsers)]
		public async Task<IActionResult> Update([FromRoute] string id, [FromBody] UpdateUserRequest request, CancellationToken cancellationToken)
		{
			await _serviceManager.UserService.UpdateAsync(id, request, cancellationToken);

			return NoContent();
		}

		[HttpPut("{id}/toggle-status")]
		[HasPermission(Permissions.UpdateUsers)]
		public async Task<IActionResult> ToggleStatus([FromRoute] string id, CancellationToken cancellationToken)
		{
			await _serviceManager.UserService.ToggleStatus(id, cancellationToken);
			return NoContent();
		}

		[HttpPut("{id}/unlock")]
		[HasPermission(Permissions.UpdateUsers)]
		public async Task<IActionResult> Unlock([FromRoute] string id, CancellationToken cancellationToken)
		{
			 await _serviceManager.UserService.Unlock(id, cancellationToken);
			return  NoContent() ;
		}
	}
}
