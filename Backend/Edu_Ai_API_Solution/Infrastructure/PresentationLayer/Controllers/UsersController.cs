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
		public async Task<IActionResult> GetAll([FromQuery] bool IncludeNotConfirmed)
		{
			return Ok(await _serviceManager.UserService.GetAllAsync(IncludeNotConfirmed));
		}

		[HttpGet("{id}")]
		[HasPermission(Permissions.GetUsers)]
		public async Task<IActionResult> Get([FromRoute] string id)
		{
			var result = await _serviceManager.UserService.GetAsync(id);

			return Ok(result); 
		}
		[HttpGet("{DepartmentId}/Instructors")]
		public async Task<IActionResult> GetUsersByDepartmentId([FromRoute] int DepartmentId)
		{
			var result = await _serviceManager.UserService.GetAllInstructorByDepartmentIdAsync(DepartmentId);
			return Ok(result);
		}

		[HttpPost("")]
		[HasPermission(Permissions.AddUsers)]
		public async Task<IActionResult> Add([FromBody] CreateUserRequest request)
		{
			var result = await _serviceManager.UserService.AddAsync(request);
			return CreatedAtAction(nameof(Get), new { result.Id }, result);
		}

		[HttpPut("{id}")]
		[HasPermission(Permissions.UpdateUsers)]
		public async Task<IActionResult> Update([FromRoute] string id, [FromBody] UpdateUserRequest request)
		{
			await _serviceManager.UserService.UpdateAsync(id, request);

			return NoContent();
		}

		[HttpPut("{id}/toggle-status")]
		[HasPermission(Permissions.UpdateUsers)]
		public async Task<IActionResult> ToggleStatus([FromRoute] string id)
		{
			await _serviceManager.UserService.ToggleStatus(id);
			return NoContent();
		}

		[HttpPut("{id}/unlock")]
		[HasPermission(Permissions.UpdateUsers)]
		public async Task<IActionResult> Unlock([FromRoute] string id)
		{
			 await _serviceManager.UserService.Unlock(id);
			return  NoContent() ;
		}
	}
}
