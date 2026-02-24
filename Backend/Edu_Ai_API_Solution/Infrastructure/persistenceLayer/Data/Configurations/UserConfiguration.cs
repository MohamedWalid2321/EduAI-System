using Shared.Constants;

namespace persistenceLayer.Data.Configurations
{
	public class UserConfiguration : IEntityTypeConfiguration<ApplicationUser>
	{
		public void Configure(EntityTypeBuilder<ApplicationUser> builder)
		{
			builder.OwnsMany(u => u.RefreshTokens)
				.ToTable("RefreshTokens")
				.WithOwner()
				.HasForeignKey("UserId");
			builder.Property(u => u.FirstName)
				.HasMaxLength(100)
				.IsRequired(false);
			builder.Property(u => u.LastName)
				.HasMaxLength(100)
				.IsRequired(false);
			builder.Property(u => u.ProfilePictureUrl)
				.HasMaxLength(500)
				.IsRequired(false);
			// For very large base64 strings (SQL Server)
			builder.Property(u => u.ProfilePictureBase64)
				.HasColumnType("NVARCHAR(MAX)")
				.IsRequired(false);

			//Default Data

			var passwordHasher = new PasswordHasher<ApplicationUser>();

			builder.HasData([
				new ApplicationUser
			{
				Id = DefaultUsers.SuperAdminId,
				FirstName = "Lumino",
				LastName = "SuperAdmin",
				UserName = DefaultUsers.SuperAdminEmail,
				NormalizedUserName = DefaultUsers.SuperAdminEmail.ToUpper(),
				Email = DefaultUsers.SuperAdminEmail,
				NormalizedEmail = DefaultUsers.SuperAdminEmail.ToUpper(),
				SecurityStamp = DefaultUsers.SuperAdminSecurityStamp,
				ConcurrencyStamp = DefaultUsers.SuperAdminConcurrencyStamp,
				EmailConfirmed = true,
				PasswordHash = passwordHasher.HashPassword(null!, DefaultUsers.SuperAdminPassword)
			},
			new ApplicationUser
			{
				Id = DefaultUsers.AdminId,
				FirstName = "Lumino",
				LastName = "Admin",
				UserName = DefaultUsers.AdminEmail,
				NormalizedUserName = DefaultUsers.AdminEmail.ToUpper(),
				Email = DefaultUsers.AdminEmail,
				NormalizedEmail = DefaultUsers.AdminEmail.ToUpper(),
				SecurityStamp = DefaultUsers.AdminSecurityStamp,
				ConcurrencyStamp = DefaultUsers.AdminConcurrencyStamp,
				EmailConfirmed = true,
				PasswordHash = passwordHasher.HashPassword(null!, DefaultUsers.AdminPassword)
			}
			]);
		}
	}
}
