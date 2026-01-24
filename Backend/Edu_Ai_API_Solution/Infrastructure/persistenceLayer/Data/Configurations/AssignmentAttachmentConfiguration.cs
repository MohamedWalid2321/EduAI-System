using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace persistenceLayer.Data.Configurations
{
	public class AssignmentAttachmentConfiguration : IEntityTypeConfiguration<AssignmentAttachment>
	{
		public void Configure(EntityTypeBuilder<AssignmentAttachment> builder)
		{
			builder.ToTable("AssignmentAttachments");
			
			builder.Property(aa => aa.FileName).HasMaxLength(255).IsRequired();
			builder.Property(aa => aa.FileUrl).HasMaxLength(500).IsRequired();
			builder.Property(aa => aa.Type).HasMaxLength(100).IsRequired();
			
			// Relationship
			builder.HasOne(aa => aa.Assignment)
				   .WithMany(a => a.AssignmentAttachments)
				   .HasForeignKey(aa => aa.AssignmentId)
				   .OnDelete(DeleteBehavior.Cascade);
		}
	}
}