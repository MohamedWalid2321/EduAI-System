using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistenceLayer.Data.Configurations
{
    public class AssignmentSubmissionConfiguration: IEntityTypeConfiguration<AssignmentSubmission>
    {
        public void Configure(EntityTypeBuilder<AssignmentSubmission> builder)
        {
            builder.ToTable("AssignmentSubmissions");

            builder.Property(a => a.StudentId).IsRequired();
            builder.Property(a => a.TextSubmission).HasMaxLength(1000);
            builder.Property(a => a.SubmittedAt).IsRequired();
            builder.Property(a => a.Feedback).HasMaxLength(1000);

            // Relationships
            builder.HasMany(aa => aa.AssignmentSubmissionAttachments)
                   .WithOne(a => a.AssignmentSubmission)
                   .HasForeignKey(aa => aa.AssignmentSubmissionId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.HasOne(a => a.Assignment)
                   .WithMany(aa=>aa.AssignmentSubmissions)
                   .HasForeignKey(a => a.AssignmentId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.HasIndex(a => new { a.AssignmentId, a.StudentId }).IsUnique();
        }  
    }
}
