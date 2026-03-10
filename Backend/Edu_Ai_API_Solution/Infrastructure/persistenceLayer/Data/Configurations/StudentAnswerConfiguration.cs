using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistenceLayer.Data.Configurations
{
    public class StudentAnswerConfiguration : IEntityTypeConfiguration<StudentAnswer>
    {
        public void Configure(EntityTypeBuilder<StudentAnswer> builder)
        {
            builder.HasKey(a => a.Id);

            builder.Property(a => a.IsCorrect)
                .IsRequired();

            builder.HasOne(a => a.QuizQuestion)
                .WithMany()
                .HasForeignKey(a => a.QuizQuestionId)
                .OnDelete(DeleteBehavior.Restrict);

            builder.HasOne(a => a.QuestionChoice)
                .WithMany()
                .HasForeignKey(a => a.QuestionChoiceId)
                .OnDelete(DeleteBehavior.Restrict);
        }
    }
}
