using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace persistenceLayer.Data.Configurations
{
	public class QuizConfiguration : IEntityTypeConfiguration<Quiz>
	{
		public void Configure(EntityTypeBuilder<Quiz> builder)
		{
			builder.ToTable("Quizzes");
			
			builder.Property(q => q.Title).HasMaxLength(200).IsRequired();
			builder.Property(q => q.Description).HasMaxLength(1000).IsRequired();
			
			// Relationships
			builder.HasOne(q => q.Course)
				   .WithMany(c => c.Quizzes)
				   .HasForeignKey(q => q.CourseId)
				   .OnDelete(DeleteBehavior.Cascade);
			
			builder.HasMany(q => q.QuizQuestions)
				   .WithOne(qq => qq.Quiz)
				   .HasForeignKey(qq => qq.QuizId)
				   .OnDelete(DeleteBehavior.Cascade);
		}
	}
}