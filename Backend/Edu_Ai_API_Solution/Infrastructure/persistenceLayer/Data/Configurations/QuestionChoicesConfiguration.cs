using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace persistenceLayer.Data.Configurations
{
	public class QuestionChoicesConfiguration : IEntityTypeConfiguration<QuestionChoices>
	{
		public void Configure(EntityTypeBuilder<QuestionChoices> builder)
		{
			builder.ToTable("QuestionChoices");
			
			builder.Property(qc => qc.ChoiceText).HasMaxLength(300).IsRequired();
			
			// Relationship
			builder.HasOne(qc => qc.QuizQuestion)
				   .WithMany(qq => qq.QuestionChoices)
				   .HasForeignKey(qc => qc.QuizQuestionId)
				   .OnDelete(DeleteBehavior.Cascade);
		}
	}
}