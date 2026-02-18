namespace persistenceLayer.Data.Configurations
{
	public class ContentConfiguration : IEntityTypeConfiguration<Content>
	{
		public void Configure(EntityTypeBuilder<Content> builder)
		{
			builder.ToTable("Contents");
			
			builder.Property(c => c.Title).HasMaxLength(200).IsRequired();
			builder.Property(c => c.Body).HasMaxLength(5000).IsRequired();
			
			// Relationships
			builder.HasOne(c => c.Course)
				   .WithMany(co => co.Contents)
				   .HasForeignKey(c => c.CourseId)
				   .OnDelete(DeleteBehavior.Cascade);
			
			builder.HasMany(c => c.ContentAttachments)
				   .WithOne(ca => ca.Content)
				   .HasForeignKey(ca => ca.ContentId)
				   .OnDelete(DeleteBehavior.Cascade);
		}
	}
}