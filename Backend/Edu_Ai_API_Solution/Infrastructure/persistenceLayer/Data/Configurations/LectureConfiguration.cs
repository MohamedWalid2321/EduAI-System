using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace persistenceLayer.Data.Configurations
{
	public class LectureConfiguration : IEntityTypeConfiguration<Lecture>
	{
		public void Configure(EntityTypeBuilder<Lecture> builder)
		{
			builder.HasKey(l => l.Id);

			builder.Property(l => l.Title)
				.IsRequired()
				.HasMaxLength(200);

			builder.Property(l => l.Description)
				.IsRequired()
				.HasMaxLength(1000);

			builder.Property(l => l.RoomName)
				.IsRequired()
				.HasMaxLength(300);

			builder.Property(l => l.CreatedById)
				.IsRequired();

			builder.HasOne(l => l.Course)
				.WithMany(c => c.Lectures)
				.HasForeignKey(l => l.CourseId)
				.OnDelete(DeleteBehavior.Cascade);

			builder.HasOne(l => l.CreatedBy)
				.WithMany()
				.HasForeignKey(l => l.CreatedById)
				.OnDelete(DeleteBehavior.Restrict);
		}
	}
}