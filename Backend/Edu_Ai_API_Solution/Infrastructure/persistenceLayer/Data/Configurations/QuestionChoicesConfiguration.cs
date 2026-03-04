namespace persistenceLayer.Data.Configurations
{
	public class QuestionChoicesConfiguration : IEntityTypeConfiguration<QuestionChoices>
	{
		public void Configure(EntityTypeBuilder<QuestionChoices> builder)
		{
			builder.ToTable("QuestionChoices");
			
			builder.Property(qc => qc.ChoiceText).HasMaxLength(300).IsRequired();

			builder.HasIndex(qc => new { qc.QuizQuestionId, qc.ChoiceText }).IsUnique();

            // Relationship
            builder.HasOne(qc => qc.QuizQuestion)
				   .WithMany(qq => qq.QuestionChoices)
				   .HasForeignKey(qc => qc.QuizQuestionId)
				   .OnDelete(DeleteBehavior.Cascade);
		}
	}
}