using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace persistenceLayer.Data.Configurations
{
    public class CheatingViolationConfiguration : IEntityTypeConfiguration<CheatingViolation>
    {
        public void Configure(EntityTypeBuilder<CheatingViolation> builder)
        {
            builder.HasKey(v => v.Id);

            builder.Property(v => v.EvidenceUrl)
                .IsRequired()
                .HasMaxLength(1000);

            builder.Property(v => v.Description)
                .IsRequired()
                .HasMaxLength(1000);

            builder.Property(v => v.Timestamp)
                .IsRequired();
        }
    }
}