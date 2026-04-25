using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistenceLayer.Data.Configurations
{
    public class FeeConfiguration : IEntityTypeConfiguration<Fee>
    {
        public void Configure(EntityTypeBuilder<Fee> builder)
        {
            builder.ToTable(nameof(Fee));

            builder.HasKey(f => f.Id);

            builder.Property(f => f.Name)
                .IsRequired()
                .HasMaxLength(100);

            builder.Property(f => f.Amount)
                .HasColumnType("decimal(10,2)")
                .IsRequired();

            
            builder.HasOne(f => f.AcademicYear)
                .WithMany(a => a.Fees)
                .HasForeignKey(f => f.AcademicYearId);

            
            builder.HasIndex(f => new { f.AcademicYearId, f.Name })
                .IsUnique();
        }
    }
}
