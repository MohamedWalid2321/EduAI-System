using DomainLayer.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistenceLayer.Data.Configurations
{
    public class AcademicYearConfiguration : IEntityTypeConfiguration<AcademicYear>
    {
        public void Configure(EntityTypeBuilder<AcademicYear> builder)
        {
            builder.ToTable("AcademicYear");

            builder.HasMany(a => a.Fees)
                   .WithOne(f => f.AcademicYear)
                   .HasForeignKey(f => f.AcademicYearId);

            


        }
    }
}
