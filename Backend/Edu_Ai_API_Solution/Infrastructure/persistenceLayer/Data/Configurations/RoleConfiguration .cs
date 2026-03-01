using Shared.Constants;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistenceLayer.Data.Configurations
{
	public class RoleConfiguration : IEntityTypeConfiguration<ApplicationRole>
	{
		public void Configure(EntityTypeBuilder<ApplicationRole> builder)
		{

			//Default Data

			var passwordHasher = new PasswordHasher<ApplicationUser>();

			builder.HasData([
			new ApplicationRole
			{
				Id = DefaultRoles.SuperAdminRoleId,
				Name = DefaultRoles.SuperAdmin,
				NormalizedName = DefaultRoles.SuperAdmin.ToUpper(),
				ConcurrencyStamp = DefaultRoles.SuperAdminRoleConcurrencyStamp
			},
		   new ApplicationRole
			{
				Id = DefaultRoles.AdminRoleId,
				Name = DefaultRoles.Admin,
				NormalizedName = DefaultRoles.Admin.ToUpper(),
				ConcurrencyStamp = DefaultRoles.AdminRoleConcurrencyStamp
			},
			new ApplicationRole
			{
				Id = DefaultRoles.InstructorRoleId,
				Name = DefaultRoles.Instructor,
				NormalizedName = DefaultRoles.Instructor.ToUpper(),
				ConcurrencyStamp = DefaultRoles.InstructorRoleConcurrencyStamp
			},
			new ApplicationRole
			{
				Id = DefaultRoles.StudentRoleId,
				Name = DefaultRoles.Student,
				NormalizedName = DefaultRoles.Student.ToUpper(),
				ConcurrencyStamp = DefaultRoles.StudentRoleConcurrencyStamp,
				IsDefault = true
			}
	   ]);
		}
	}
}
