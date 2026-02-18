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
		}
	}
}
