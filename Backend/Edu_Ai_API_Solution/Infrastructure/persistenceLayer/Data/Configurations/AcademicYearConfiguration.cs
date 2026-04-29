using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using Shared.Constants;

namespace persistenceLayer.Data.Configurations
{
	public class AcademicYearConfiguration : IEntityTypeConfiguration<AcademicYear>
	{
		public void Configure(EntityTypeBuilder<AcademicYear> builder)
		{
			builder.ToTable("AcademicYear");

			builder.HasKey(a => a.Id);

			builder.Property(a => a.Name)
				.IsRequired()
				.HasMaxLength(50);

			builder.HasMany(a => a.Fees)
				   .WithOne(f => f.AcademicYear)
				   .HasForeignKey(f => f.AcademicYearId)
				   .OnDelete(DeleteBehavior.Restrict);

			builder.HasData(
				new AcademicYear { Id = DefaultAcademicYear.FirstYearId,  Name = "First Year"  },
				new AcademicYear { Id = DefaultAcademicYear.SecondYearId, Name = "Second Year" },
				new AcademicYear { Id = DefaultAcademicYear.ThirdYearId,  Name = "Third Year"  },
				new AcademicYear { Id = DefaultAcademicYear.FourthYearId, Name = "Fourth Year" },
				new AcademicYear { Id = DefaultAcademicYear.FifthYearId,  Name = "Fifth Year"  }
			);
		}
	}
}
