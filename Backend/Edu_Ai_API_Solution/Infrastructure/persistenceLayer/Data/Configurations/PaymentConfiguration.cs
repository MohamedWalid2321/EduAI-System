using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistenceLayer.Data.Configurations
{
    public class PaymentConfiguration : IEntityTypeConfiguration<Payment>
    {
        public void Configure(EntityTypeBuilder<Payment> builder)
        {
            builder.ToTable("Payments");

            builder.HasKey(p => p.Id);


            builder.Property(p => p.Amount)
                .HasColumnType("decimal(18,2)")
                .IsRequired();

            builder.Property(p => p.TransactionId)
                .HasMaxLength(100);

            builder.Property(p => p.CreatedAt)
                .HasDefaultValueSql("GETUTCDATE()");

            builder.Property(p => p.PaymentDate)
                .HasDefaultValueSql("GETUTCDATE()");


           // builder.HasIndex(a => new { a.Student.Id, a.AcademicYearId })
              //  .IsUnique();

            

           /* builder.HasOne(p => p.AcademicYear)
                   .WithMany(a => a.Payments)
                   .HasForeignKey(p => p.AcademicYearId);
           */
            builder.HasIndex(p => p.TransactionId);
        }
    }
}
