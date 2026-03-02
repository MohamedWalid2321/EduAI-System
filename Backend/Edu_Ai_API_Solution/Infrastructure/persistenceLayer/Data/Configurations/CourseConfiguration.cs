namespace persistenceLayer.Data.Configurations
{
	public class CourseConfiguration : IEntityTypeConfiguration<Course>
	{
		public void Configure(EntityTypeBuilder<Course> builder)
		{
			builder.ToTable("Courses");
			builder.Property(c => c.Title).HasMaxLength(100).IsRequired();
			builder.Property(c => c.Description).HasMaxLength(500).IsRequired();
			builder.Property(c => c.ImageUrl).HasMaxLength(200).IsRequired();
			// Relationships
			builder.HasMany(c => c.Assignments)
				   .WithOne(a => a.Course)
				   .HasForeignKey(a => a.CourseId)
				   .OnDelete(DeleteBehavior.Cascade);
			builder.HasMany(c => c.Quizzes)
					.WithOne(q => q.Course)
				   .HasForeignKey(q => q.CourseId)
				   .OnDelete(DeleteBehavior.Cascade);
			builder.HasOne(c => c.PrerequisiteCourse)
				   .WithMany()
				   .HasForeignKey(c => c.PrerequisiteCourseId)
				   .OnDelete(DeleteBehavior.Restrict);
			builder.HasMany(c => c.Departments)
				   .WithMany(d => d.courses);
			builder.HasMany(c => c.Assessments)
				   .WithOne(a => a.Course)
				   .HasForeignKey(a => a.CourseId)
				   .OnDelete(DeleteBehavior.Cascade);
			builder.HasMany(c => c.Contents)
					.WithOne(ct => ct.Course)
				   .HasForeignKey(ct => ct.CourseId)
				   .OnDelete(DeleteBehavior.Cascade);
			builder.Property(c => c.LearningOutcomes).HasMaxLength(1000).IsRequired();
		}
	}
}
