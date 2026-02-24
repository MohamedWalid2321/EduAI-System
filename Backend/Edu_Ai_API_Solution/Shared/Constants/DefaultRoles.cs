using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Constants
{
	public static class DefaultRoles
	{
		public const string SuperAdmin = nameof(SuperAdmin);
		public const string SuperAdminRoleId = "71e40e16-7fe9-4f8b-807b-77c9da3f41a9";
		public const string SuperAdminRoleConcurrencyStamp = "5540e8da-f93d-4457-a355-f04bb15c4594";

		public const string Admin = nameof(Admin);
		public const string AdminRoleId = "92b75286-d8f8-4061-9995-e6e23ccdee94";
		public const string AdminRoleConcurrencyStamp = "f51e5a91-bced-49c2-8b86-c2e170c0846c";

		public const string Instructor = nameof(Instructor);
		public const string InstructorRoleId = "7e07bb31-26ad-47ac-880c-c5fdfa0516d3";
		public const string InstructorRoleConcurrencyStamp = "9bba17c3-ee48-423a-b2c0-a63245d0edf0";


		public const string Student = nameof(Student);
		public const string StudentRoleId = "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4";
		public const string StudentRoleConcurrencyStamp = "5ee6bc12-5cb0-4304-91e7-6a00744e042a";
	}
}
