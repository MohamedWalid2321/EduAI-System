namespace PresentationLayer.Attributes
{
	public class HasPermissionAttribute(string permission) : AuthorizeAttribute(permission)
	{
	}
}
