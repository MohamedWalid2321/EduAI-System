using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistenceLayer.Data.Configurations
{
    public class QuizAttemptConfiguration : IEntityTypeConfiguration<QuizAttempt>
    {
        public void Configure(EntityTypeBuilder<QuizAttempt> builder)
        {
            builder.HasKey(a => a.Id);

            builder.Property(a => a.StudentId)
                .IsRequired();

            builder.Property(a => a.CreatedAt)
                .IsRequired();

            builder.Property(a => a.Score)
                .IsRequired();

            builder.HasOne(a => a.Quiz)
                .WithMany(q => q.QuizAttempts)
                .HasForeignKey(a => a.QuizId)
                .OnDelete(DeleteBehavior.Cascade);

            builder.HasMany(a => a.StudentAnswers)
                .WithOne(a => a.QuizAttempt)
                .HasForeignKey(a => a.QuizAttemptId)
                .OnDelete(DeleteBehavior.Cascade);

            
            builder.HasOne(a => a.User)
                .WithMany()
                .HasForeignKey(a => a.StudentId)
                .OnDelete(DeleteBehavior.Restrict);

            builder.HasIndex(a => new { a.StudentId, a.QuizId })
                .IsUnique();
        }
    }
}
