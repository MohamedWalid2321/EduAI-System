using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace persistenceLayer.Data.Configurations
{
    public class RiskAssessmentResultConfiguration : IEntityTypeConfiguration<RiskAssessmentResult>
    {
        public void Configure(EntityTypeBuilder<RiskAssessmentResult> builder)
        {
            builder.HasKey(r => r.Id);

            builder.Property(r => r.StudentId)
                .IsRequired()
                .HasMaxLength(450);

            builder.Property(r => r.SessionViolationRate)
                .IsRequired();

            // decimal(8,4): up to 9999.9999 — wide enough for any weighted sum
            builder.Property(r => r.OverallSessionRiskScore)
                .HasPrecision(8, 4)
                .IsRequired();

            // One-to-one with CheatingReport
            builder.HasOne(r => r.CheatingReport)
                .WithOne()
                .HasForeignKey<RiskAssessmentResult>(r => r.CheatingReportId)
                .OnDelete(DeleteBehavior.Cascade);

            // One-to-many with RiskQuestionResult
            builder.HasMany(r => r.Questions)
                .WithOne(q => q.RiskAssessmentResult)
                .HasForeignKey(q => q.RiskAssessmentResultId)
                .OnDelete(DeleteBehavior.Cascade);

            // Fast look-up by attempt
            builder.HasIndex(r => r.AttemptId);
        }
    }
}
