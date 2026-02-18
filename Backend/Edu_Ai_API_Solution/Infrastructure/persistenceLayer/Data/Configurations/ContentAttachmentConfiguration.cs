namespace persistenceLayer.Data.Configurations
{
	public class ContentAttachmentConfiguration : IEntityTypeConfiguration<ContentAttachment>
	{
		public void Configure(EntityTypeBuilder<ContentAttachment> builder)
		{
			builder.ToTable("ContentAttachments");
			
			builder.Property(ca => ca.FileName).HasMaxLength(255).IsRequired();
			builder.Property(ca => ca.FileUrl).HasMaxLength(500).IsRequired();
			builder.Property(ca => ca.ContentType).HasMaxLength(100).IsRequired();
			
			// Relationship
			builder.HasOne(ca => ca.Content)
				   .WithMany(c => c.ContentAttachments)
				   .HasForeignKey(ca => ca.ContentId)
				   .OnDelete(DeleteBehavior.Cascade);
		}
	}
}