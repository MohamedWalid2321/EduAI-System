using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace persistenceLayer.Data.Configurations
{
    public class CheatingReportConfiguration : IEntityTypeConfiguration<CheatingReport>
    {
        public void Configure(EntityTypeBuilder<CheatingReport> builder)
        {
            builder.HasKey(r => r.Id);

            // One CheatingReport per QuizAttempt
            builder.HasIndex(r => r.QuizAttemptId).IsUnique();

            builder.HasOne(r => r.QuizAttempt)
                .WithOne()
                .HasForeignKey<CheatingReport>(r => r.QuizAttemptId)
                .OnDelete(DeleteBehavior.Cascade);

            builder.HasOne(r => r.Student)
                .WithMany()
                .HasForeignKey(r => r.StudentId)
                .OnDelete(DeleteBehavior.NoAction);

            builder.HasMany(r => r.Violations)
                .WithOne(v => v.CheatingReport)
                .HasForeignKey(v => v.CheatingReportId)
                .OnDelete(DeleteBehavior.Cascade);

            // RiskScore: 6 digits total, 4 after the decimal point (range 0–99.9999)
            builder.Property(r => r.RiskScore)
                .HasPrecision(6, 4)
                .IsRequired(false);
        }
    }
}