namespace PresentationLayer.Authorization
{
	public class PermissionRequirement(string permission) : IAuthorizationRequirement
	{
		public string Permission { get; } = permission;
	}
}
