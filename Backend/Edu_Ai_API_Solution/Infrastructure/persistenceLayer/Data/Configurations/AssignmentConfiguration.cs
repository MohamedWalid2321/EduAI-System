namespace persistenceLayer.Data.Configurations
{
	public class AssignmentConfiguration : IEntityTypeConfiguration<Assignment>
	{
		public void Configure(EntityTypeBuilder<Assignment> builder)
		{
			builder.ToTable("Assignments");
			
			builder.Property(a => a.Title).HasMaxLength(200).IsRequired();
			builder.Property(a => a.Description).HasMaxLength(1000).IsRequired();
			
			// Relationships
			builder.HasOne(a => a.Course)
				   .WithMany(c => c.Assignments)
				   .HasForeignKey(a => a.CourseId)
				   .OnDelete(DeleteBehavior.Cascade);
			
			builder.HasMany(a => a.AssignmentAttachments)
				   .WithOne(aa => aa.Assignment)
				   .HasForeignKey(aa => aa.AssignmentId)
				   .OnDelete(DeleteBehavior.Cascade);
		}
	}
}