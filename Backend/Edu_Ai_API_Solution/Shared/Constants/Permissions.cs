using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Constants
{
	public static class Permissions
	{
		public static string Type { get; } = "Permissions";

		public const string GetAss = "Ass:read";
		public const string AddAss = "Ass:add";
		public const string UpdateAss = "Ass:update";
		public const string DeleteAss = "Ass:delete";
		public const string SolveAss = "Ass:solve";       // Student can submit assignment

		public const string GetContent = "Content:read";
		public const string AddContent = "Content:add";
		public const string UpdateContent = "Content:update";
		public const string DeleteContent = "Content:delete";

		public const string GetCourse = "Course:read";
		public const string AddCourse = "Course:add";
		public const string UpdateCourse = "Course:update";
		public const string DeleteCourse = "Course:delete";

		public const string GetDepartment = "Department:read";
		public const string AddDepartment = "Department:add";
		public const string UpdateDepartment = "Department:update";
		public const string DeleteDepartment = "Department:delete";

		// reminder to Also make permissions for Quiz or SomeThing Like Question Below 

		// Reminder To me (Mohannad) : Just For Now unImplement Permissions Until Communicate With Marawan

		public const string GetQuestions = "questions:read";
		public const string AddQuestions = "questions:add";
		public const string UpdateQuestions = "questions:update";
		public const string DeleteQuestions = "questions:delete";
		public const string SolveQuiz = "questions:solve"; // Student can solve quiz

		public const string GetUsers = "users:read";
		public const string AddUsers = "users:add";
		public const string UpdateUsers = "users:update";
		public const string DeleteUsers = "users:delete";

		public const string GetRoles = "roles:read";
		public const string AddRoles = "roles:add";
		public const string UpdateRoles = "roles:update";
		public const string DeleteRoles = "roles:delete";

		//public const string Results = "results:read";

		public static IList<string?> GetAllPermissions() =>
			typeof(Permissions).GetFields().Select(x => x.GetValue(x) as string).ToList();
	}
}
