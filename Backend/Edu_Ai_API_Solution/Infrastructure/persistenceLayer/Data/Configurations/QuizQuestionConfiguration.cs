using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace persistenceLayer.Data.Configurations
{
	public class QuizQuestionConfiguration : IEntityTypeConfiguration<QuizQuestion>
	{
		public void Configure(EntityTypeBuilder<QuizQuestion> builder)
		{
			builder.ToTable("QuizQuestions");
			
			builder.Property(qq => qq.QuestionText).HasMaxLength(500).IsRequired();
			
			// Relationships
			builder.HasOne(qq => qq.Quiz)
				   .WithMany(q => q.QuizQuestions)
				   .HasForeignKey(qq => qq.QuizId)
				   .OnDelete(DeleteBehavior.Cascade);
			
			builder.HasMany(qq => qq.QuestionChoices)
				   .WithOne(qc => qc.QuizQuestion)
				   .HasForeignKey(qc => qc.QuizQuestionId)
				   .OnDelete(DeleteBehavior.Cascade);
		}
	}
}