using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Constants
{
	public static class Permissions
	{
		public static string Type { get; } = "Permissions";
		public const string LevelUp = "Profile:levelUp";

		public const string GetAss = "Ass:read";
		public const string AddOrUpdateAss = "Ass:addOrUpdate";
		public const string DeleteAss = "Ass:delete";
		public const string GradeAss = "Ass:Grade";
		public const string SolveAss = "Ass:solve";

		// Assignment Submission Permissions
		public const string GetAssSubmission = "AssSubmission:read";         // View a single submission (student + instructor)
		public const string GetAllAssSubmissions = "AssSubmission:readAll";  // View all submissions for an assignment (instructor)
		public const string DeleteAssSubmission = "AssSubmission:delete";    // Delete a submission (instructor / admin)

		public const string GetContent = "Content:read";
		public const string AddContent = "Content:add";
		public const string UpdateContent = "Content:update";
		public const string DeleteContent = "Content:delete";

		public const string GetCourse = "Course:read";
		public const string GetAllCourses = "Course:readAll";
		public const string AddCourse = "Course:add";
		public const string UpdateCourse = "Course:update";
		public const string DeleteCourse = "Course:delete";
		public const string EnrollInstructor = "Course:enrollInstructor";
		public const string UnenrollInstructor = "Course:unenrollInstructor";
		public const string GetAssesment = "Course:readAssesment"; // Permission to read course assessments (for students)

		public const string GetDepartment = "Department:read";
		public const string AddDepartment = "Department:add";
		public const string UpdateDepartment = "Department:update";
		public const string DeleteDepartment = "Department:delete";

		public const string GetQuestions = "questions:read";
		public const string AddQuestions = "questions:add";
		public const string UpdateQuestions = "questions:update";
		public const string DeleteQuestions = "questions:delete";
		public const string SolveQuiz = "questions:solve";                   // Student can solve quiz

		public const string GetUsers = "users:read";
		public const string AddUsers = "users:add";
		public const string UpdateUsers = "users:update";
		public const string DeleteUsers = "users:delete";

		public const string GetRoles = "roles:read";
		public const string AddRoles = "roles:add";
		public const string UpdateRoles = "roles:update";
		public const string DeleteRoles = "roles:delete";

		// Lecture Permissions
		public const string CreateLecture = "Lecture:create";
		public const string UpdateLecture = "Lecture:update";
		public const string DeleteLecture = "Lecture:delete";
		public const string JoinLecture = "Lecture:join";

		// Cheating Report Permissions
		public const string GetCheatingReport = "CheatingReport:read";
		public const string AddCheatingReport = "CheatingReport:add";
		public const string DeleteCheatingReport = "CheatingReport:delete";
		// Quiz Permissions
		public const string GetQuizzes = "Quiz:read";
		public const string AddOrUpdateQuiz = "Quiz:addOrUpdate";
		public const string DeleteQuiz = "Quiz:delete";

		// Quiz Attempt Score Permissions
		/// <summary>Instructor-only: update a specific attempt score exactly once (then locked).</summary>
		public const string FinalizeAttemptScore = "AttemptScore:finalize";
		/// <summary>Admin / SuperAdmin: update a specific attempt score an unlimited number of times.</summary>
		public const string UpdateAttemptScore = "AttemptScore:update";

		public static IList<string?> GetAllPermissions() =>
			typeof(Permissions).GetFields().Select(x => x.GetValue(x) as string).ToList();
	}
}
