namespace persistenceLayer.Data
{
	public class LmsDBContext(DbContextOptions<LmsDBContext> options) : IdentityDbContext<ApplicationUser,ApplicationRole,string>(options)
	{
		public DbSet<Department> Departments { get; set; }
		public DbSet<Course> Courses { get; set; }
		public DbSet<Assessment> Assessments { get; set; }
		public DbSet<Content> Contents { get; set; }
		public DbSet<ContentAttachment> ContentAttachments { get; set; }
		public DbSet<Assignment> Assignments { get; set; }

		public DbSet<AssignmentAttachment> AssignmentAttachments { get; set; }
		public DbSet<Quiz> Quizzes { get; set; }

		public DbSet<QuizQuestion> QuizQuestions { get; set; }
		public DbSet<QuestionChoices> QuestionChoices { get; set; }
		public DbSet<QuizAttempt> QuizAttempts { get; set; }
		public DbSet<StudentAnswer> StudentAnswers { get; set; }
		

        override protected void OnModelCreating(ModelBuilder modelBuilder)
		{
			base.OnModelCreating(modelBuilder);
			// Apply all configurations from the current assembly
			modelBuilder.ApplyConfigurationsFromAssembly(typeof(LmsDBContext).Assembly);
		}
	}
}
