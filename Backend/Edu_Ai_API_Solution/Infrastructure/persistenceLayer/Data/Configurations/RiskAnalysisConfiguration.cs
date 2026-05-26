using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace persistenceLayer.Data.Configurations
{
    public class RiskAnalysisConfiguration : IEntityTypeConfiguration<RiskAnalysis>
    {
        public void Configure(EntityTypeBuilder<RiskAnalysis> builder)
        {
            builder.HasKey(r => r.Id);

            // Composite index: one row per (AttemptId, QuestionId); upserts depend on this.
            builder.HasIndex(r => new { r.AttemptId, r.QuestionId });

            builder.Property(r => r.StudentId)
                .IsRequired()
                .HasMaxLength(450);

            builder.Property(r => r.ViolationRate)
                .IsRequired();

            // Violation counts – default 0, not nullable
            builder.Property(r => r.FaceDetection).IsRequired();
            builder.Property(r => r.FaceRecognition).IsRequired();
            builder.Property(r => r.EyeGaze).IsRequired();
            builder.Property(r => r.SpeechDetection).IsRequired();
            builder.Property(r => r.ObjectDetection).IsRequired();

            // Weights – double precision, not nullable
            builder.Property(r => r.WeightFaceAbsenceMismatch).IsRequired();
            builder.Property(r => r.WeightSuspiciousMovement).IsRequired();
            builder.Property(r => r.WeightConversationNoise).IsRequired();
            builder.Property(r => r.WeightForbiddenObjects).IsRequired();
        }
    }
}
