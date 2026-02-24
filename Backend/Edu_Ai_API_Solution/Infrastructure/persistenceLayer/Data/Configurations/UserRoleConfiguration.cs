using Shared.Constants;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistenceLayer.Data.Configurations
{
	public class UserRoleConfiguration : IEntityTypeConfiguration<IdentityUserRole<string>>
	{
		public void Configure(EntityTypeBuilder<IdentityUserRole<string>> builder)
		{
			//Default Data
			builder.HasData(
				new IdentityUserRole<string>
				{
					UserId = DefaultUsers.SuperAdminId,
					RoleId = DefaultRoles.SuperAdminRoleId
				},
				new IdentityUserRole<string>
				{
					UserId = DefaultUsers.AdminId,
					RoleId = DefaultRoles.AdminRoleId
				}
			);
		}
	}
}
