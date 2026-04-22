using DomainLayer.Models;
using Microsoft.AspNetCore.Http;

namespace persistenceLayer.Data
{
	public class LmsDBContext(DbContextOptions<LmsDBContext> options, IHttpContextAccessor httpContextAccessor) : IdentityDbContext<ApplicationUser, ApplicationRole, string>(options)
	{
		private readonly IHttpContextAccessor _httpContextAccessor = httpContextAccessor;

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
		public DbSet<UserCourse> UserCourses { get; set; }

		protected override void OnModelCreating(ModelBuilder modelBuilder)
		{
			base.OnModelCreating(modelBuilder);
			modelBuilder.ApplyConfigurationsFromAssembly(typeof(LmsDBContext).Assembly);
		}
		public override Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
		{
			var entries = ChangeTracker.Entries<BaseEntity>(); 
			var userId = _httpContextAccessor.HttpContext?.User.FindFirstValue(ClaimTypes.NameIdentifier)!;

			foreach (var entry in entries)
			{
				if (entry.State == EntityState.Added)
				{
					entry.Entity.CreatedBy = userId;
				}
				else if (entry.State == EntityState.Modified)
				{
					entry.Entity.LastUpdatedBy = userId;
					entry.Entity.LastUpdatedAt = DateTime.UtcNow;
				}
			}

			return base.SaveChangesAsync(cancellationToken);
		}

	}
}
