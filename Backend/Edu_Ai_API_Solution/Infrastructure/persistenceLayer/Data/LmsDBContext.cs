using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistenceLayer.Data
{
	public class LmsDBContext :DbContext
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


		public LmsDBContext(DbContextOptions<LmsDBContext> options):base(options)	
		{
			
		}
		override protected void OnModelCreating(ModelBuilder modelBuilder)
		{
			base.OnModelCreating(modelBuilder);
			// Apply all configurations from the current assembly
			modelBuilder.ApplyConfigurationsFromAssembly(typeof(LmsDBContext).Assembly);
		}
	}
}
