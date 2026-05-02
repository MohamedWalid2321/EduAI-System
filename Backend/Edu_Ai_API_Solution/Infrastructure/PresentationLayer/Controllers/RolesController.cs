using Shared.Dtos.RolesDto.Request;
    
namespace PresentationLayer.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class RolesController(IServiceManager serviceManager) : ControllerBase
    {
        private readonly IServiceManager _serviceManager = serviceManager;

        [HttpGet("")]
        [HasPermission(Permissions.GetRoles)]
        public async Task<IActionResult> GetAll([FromQuery] bool includeDisabled, CancellationToken cancellationToken)
        {
            var roles = await _serviceManager.RoleService.GetAllAsync(includeDisabled, cancellationToken);
            return Ok(roles);
        }

        [HttpGet("{id}")]
        [HasPermission(Permissions.GetRoles)]
        public async Task<IActionResult> Get([FromRoute] string id, CancellationToken cancellationToken)
        {
            var result = await _serviceManager.RoleService.GetAsync(id, cancellationToken);
            return Ok(result);
        }

        [HttpPost("")]
        [HasPermission(Permissions.AddRoles)]
        public async Task<IActionResult> Add([FromBody] RoleRequest request, CancellationToken cancellationToken)
        {
            var result = await _serviceManager.RoleService.AddAsync(request, cancellationToken);
            return CreatedAtAction(nameof(Get), new { result.Id }, result);
        }

        [HttpPut("{id}")]
        [HasPermission(Permissions.UpdateRoles)]
        public async Task<IActionResult> Update([FromRoute] string id, [FromBody] RoleRequest request, CancellationToken cancellationToken)
        {
            await _serviceManager.RoleService.UpdateAsync(id, request, cancellationToken);
            return NoContent();
        }

        [HttpPut("{id}/toggle-status")]
        [HasPermission(Permissions.UpdateRoles)]
        public async Task<IActionResult> ToggleStatus([FromRoute] string id, CancellationToken cancellationToken)
        {
            await _serviceManager.RoleService.ToggleStatusAsync(id, cancellationToken);
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
