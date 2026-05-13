using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace persistenceLayer.Data.Configurations
{
    public class RiskQuestionResultConfiguration : IEntityTypeConfiguration<RiskQuestionResult>
    {
        public void Configure(EntityTypeBuilder<RiskQuestionResult> builder)
        {
            builder.HasKey(r => r.Id);

            // decimal(6,2): 9999.99 — covers any realistic weighted score rounded to 2 dp
            builder.Property(r => r.StudentRiskScore)
                .HasPrecision(6, 2)
                .IsRequired();

            builder.Property(r => r.CohortAvgRiskScore)
                .HasPrecision(6, 2)
                .IsRequired();

            builder.Property(r => r.QuestionId)
                .IsRequired();

            // Index for fast cohort queries by question
            builder.HasIndex(r => r.QuestionId);
        }
    }
}
