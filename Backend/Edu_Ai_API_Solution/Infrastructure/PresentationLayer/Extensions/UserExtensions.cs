namespace PresentationLayer.Extensions
{
	public static class UserExtensions
	{
		public static string? GetUserId(this ClaimsPrincipal user) =>
			user.FindFirstValue(ClaimTypes.NameIdentifier);
	}
}
