using DomainLayer.Exceptions.Department;

namespace PresentationLayer.Extensions
{
	public static class UserExtensions
	{
		public static string? GetUserId(this ClaimsPrincipal user) =>
			user.FindFirstValue(ClaimTypes.NameIdentifier);
		public static int? GetDepartmentId(this ClaimsPrincipal user)
		{
			var departmentIdClaim = user.FindFirstValue("departmentId")
									?? user.FindFirstValue("DepartmentId");

			return int.TryParse(departmentIdClaim, out var departmentId)
				? departmentId
				: null;
		}

		public static int GetDepartmentIdOrThrow(this ClaimsPrincipal user)
		{
			var departmentId = user.GetDepartmentId();

			if (departmentId is null)
				throw new DepartmentIdNotFoundInToken();

			return departmentId.Value;
		}
	}
}
