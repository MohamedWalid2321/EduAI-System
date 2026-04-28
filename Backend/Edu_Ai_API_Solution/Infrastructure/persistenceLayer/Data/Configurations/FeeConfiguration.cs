using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DomainLayer.Enums;
using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using Shared.Constants;

namespace persistenceLayer.Data.Configurations
{
	public class FeeConfiguration : IEntityTypeConfiguration<Fee>
	{
		public void Configure(EntityTypeBuilder<Fee> builder)
		{
			builder.ToTable(nameof(Fee));

			builder.HasKey(f => f.Id);

			builder.Property(f => f.FeeType)
				.IsRequired()
				.HasConversion<string>()   // store as string ("Tuition","Books","Activities") instead of int
				.HasMaxLength(50);

			builder.Property(f => f.Amount)
				.HasColumnType("decimal(10,2)")
				.IsRequired();

			builder.HasOne(f => f.Department)
				.WithMany(d => d.Fees)
				.HasForeignKey(f => f.DepartmentId)
				.OnDelete(DeleteBehavior.Restrict);

			builder.HasIndex(f => new { f.AcademicYearId, f.DepartmentId, f.FeeType }).IsUnique();

			int id = 1;
			var departments = new[]
			{
				(Id: DefaultDepartment.ComputerEngineeringId,     Tuition: 6000m, Books: 800m, Activities: 300m),
				(Id: DefaultDepartment.ElectricalEngineeringId,   Tuition: 5500m, Books: 700m, Activities: 300m),
				(Id: DefaultDepartment.MechanicalEngineeringId,   Tuition: 5000m, Books: 600m, Activities: 300m),
				(Id: DefaultDepartment.CommunicationEngineeringId,Tuition: 5800m, Books: 750m, Activities: 300m),
				(Id: DefaultDepartment.BiomedicalEngineeringId,   Tuition: 6500m, Books: 900m, Activities: 300m),
			};

			var academicYears = new[]
			{
				DefaultAcademicYear.FirstYearId,
				DefaultAcademicYear.SecondYearId,
				DefaultAcademicYear.ThirdYearId,
				DefaultAcademicYear.FourthYearId,
				DefaultAcademicYear.FifthYearId,
			};

			var fees = new List<Fee>();
			foreach (var dept in departments)
			{
				for (int i = 0; i < academicYears.Length; i++)
				{
					decimal multiplier = 1 + (i * 0.10m);
					fees.Add(new Fee { Id = id++, DepartmentId = dept.Id, AcademicYearId = academicYears[i], FeeType = FeeType.Tuition,    Amount = Math.Round(dept.Tuition    * multiplier, 2) });
					fees.Add(new Fee { Id = id++, DepartmentId = dept.Id, AcademicYearId = academicYears[i], FeeType = FeeType.Books,       Amount = Math.Round(dept.Books      * multiplier, 2) });
					fees.Add(new Fee { Id = id++, DepartmentId = dept.Id, AcademicYearId = academicYears[i], FeeType = FeeType.Activities,  Amount = dept.Activities });
				}
			}

			builder.HasData(fees);
		}
	}
}
