namespace persistenceLayer.Data.Configurations
{
    public class AssignmentSubmissionAttachmentConfiguration: IEntityTypeConfiguration<AssignmentSubmissionAttachment>
    {
        public void Configure(EntityTypeBuilder<AssignmentSubmissionAttachment> builder)
        {
            builder.ToTable("AssignmentSubmissionAttachments");

            builder.Property(a => a.FileName).HasMaxLength(255).IsRequired();
            builder.Property(a => a.FileUrl).HasMaxLength(500).IsRequired();
            builder.Property(a => a.Type).HasMaxLength(100).IsRequired();
            // Relationships
            builder.HasOne(aa => aa.AssignmentSubmission)
                   .WithMany(a => a.AssignmentSubmissionAttachments)
                   .HasForeignKey(aa => aa.AssignmentSubmissionId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.HasIndex(a => new {a.AssignmentSubmissionId , a.FileName }).IsUnique();
        }  
    }
}
