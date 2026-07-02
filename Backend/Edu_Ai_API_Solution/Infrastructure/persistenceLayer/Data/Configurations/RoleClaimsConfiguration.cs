using Microsoft.AspNetCore.Identity;
using Shared.Constants;

namespace persistenceLayer.Data.Configurations;

public class RoleClaimsConfiguration : IEntityTypeConfiguration<IdentityRoleClaim<string>>
{
    // ─── ID Blocks ──────────────────────────────────────────
    // SuperAdmin : 1001 – 1050
    // Admin      : 1051 – 1100
    // Instructor : 1101 – 1150
    // Student    : 1151 – 1200
    // Extended   : 1301 +   ← new permissions added after initial seeding
    // ────────────────────────────────────────────────────────

    public void Configure(EntityTypeBuilder<IdentityRoleClaim<string>> builder)
    {
        var claims = new List<IdentityRoleClaim<string>>();

        // ── SuperAdmin — all permissions except Course:read and GetStudentCourseGrades
        //    (GetStudentCourseGrades is seeded separately at ID 1301 to avoid collision)
        var superAdminPermissions = Permissions.GetAllPermissions()
            .Where(p => p != Permissions.GetCourse && p != Permissions.GetStudentCourseGrades)
            .ToArray();

        var superAdminStartId = 1001;
        for (var i = 0; i < superAdminPermissions.Length; i++)
        {
            claims.Add(new IdentityRoleClaim<string>
            {
                Id = superAdminStartId + i,
                RoleId = DefaultRoles.SuperAdminRoleId,
                ClaimType = Permissions.Type,
                ClaimValue = superAdminPermissions[i]!
            });
        }

        // ── Admin ────────────────────────────────────────────────────────────
        var adminPermissions = new[]
        {
            // Assignment
            Permissions.GetAss, Permissions.AddOrUpdateAss, Permissions.DeleteAss,
            Permissions.GradeAss,
            // Assignment Submission
            Permissions.GetAssSubmission, Permissions.GetAllAssSubmissions, Permissions.DeleteAssSubmission,
            // Content
            Permissions.GetContent, Permissions.AddContent, Permissions.UpdateContent, Permissions.DeleteContent,
            // Course
            Permissions.GetCourse, Permissions.AddCourse, Permissions.UpdateCourse, Permissions.DeleteCourse,
            Permissions.EnrollInstructor, Permissions.UnenrollInstructor, Permissions.GetAssesment,
            // Quiz 
            Permissions.GetQuizzes, Permissions.AddOrUpdateQuiz, Permissions.DeleteQuiz,
            // Questions
            Permissions.GetQuestions, Permissions.AddQuestions, Permissions.UpdateQuestions,
            // Lecture
            Permissions.CreateLecture, Permissions.UpdateLecture, Permissions.DeleteLecture, Permissions.JoinLecture,
            // Cheating Report — admin can read, add and delete
            Permissions.GetCheatingReport, Permissions.AddCheatingReport, Permissions.DeleteCheatingReport,
            // Quiz Attempt Score — admin can update score unlimited times
            Permissions.UpdateAttemptScore,
        };

        for (var i = 0; i < adminPermissions.Length; i++)
        {
            claims.Add(new IdentityRoleClaim<string>
            {
                Id = 1051 + i,
                RoleId = DefaultRoles.AdminRoleId,
                ClaimType = Permissions.Type,
                ClaimValue = adminPermissions[i]
            });
        }

        // ── Instructor ───────────────────────────────────────────────────────
        var instructorPermissions = new[]
        {
            // Assignment
            Permissions.GetAss, Permissions.AddOrUpdateAss, Permissions.DeleteAss,
            Permissions.GradeAss,
            // Assignment Submission (view only — no delete)
            Permissions.GetAssSubmission, Permissions.GetAllAssSubmissions,
            // Content
            Permissions.GetContent, Permissions.AddContent, Permissions.UpdateContent, Permissions.DeleteContent,
            // Course (read only)
            Permissions.GetCourse, Permissions.GetAssesment,
            // Quiz 
            Permissions.GetQuizzes, Permissions.AddOrUpdateQuiz, Permissions.DeleteQuiz,
            // Questions
            Permissions.GetQuestions, Permissions.AddQuestions, Permissions.UpdateQuestions,
            // Lecture
            Permissions.CreateLecture, Permissions.UpdateLecture, Permissions.DeleteLecture, Permissions.JoinLecture,
            // Cheating Report — instructor can read, add and delete
            Permissions.GetCheatingReport, Permissions.AddCheatingReport, Permissions.DeleteCheatingReport,
            // Quiz Attempt Score — instructor can finalize a score one time only
            Permissions.FinalizeAttemptScore,
        };

        for (var i = 0; i < instructorPermissions.Length; i++)
        {
            claims.Add(new IdentityRoleClaim<string>
            {
                Id = 1101 + i,
                RoleId = DefaultRoles.InstructorRoleId,
                ClaimType = Permissions.Type,
                ClaimValue = instructorPermissions[i]
            });
        }

        // ── Student ──────────────────────────────────────────────────────────
        var studentPermissions = new[]
        {
            // Profile
            Permissions.LevelUp,
            // Assignment
            Permissions.GetAss, Permissions.SolveAss,
            // Assignment Submission (view own only)
            Permissions.GetAssSubmission, Permissions.DeleteAssSubmission,
            // Content
            Permissions.GetContent,
            // Course
            Permissions.GetCourse, Permissions.GetAssesment,
            // Quiz
            Permissions.GetQuizzes,
            // Questions
            Permissions.GetQuestions, Permissions.SolveQuiz,
            // Lecture
            Permissions.JoinLecture,
            // Cheating Report — student can only add (submit evidence during exam)
            Permissions.AddCheatingReport,
        };

        for (var i = 0; i < studentPermissions.Length; i++)
        {
            claims.Add(new IdentityRoleClaim<string>
            {
                Id = 1151 + i,
                RoleId = DefaultRoles.StudentRoleId,
                ClaimType = Permissions.Type,
                ClaimValue = studentPermissions[i]
            });
        }

        // ── Extended block (1301+) — permissions added after initial seeding ─
        // SuperAdmin: GetStudentCourseGrades
        claims.Add(new IdentityRoleClaim<string>
        {
            Id = 1301,
            RoleId = DefaultRoles.SuperAdminRoleId,
            ClaimType = Permissions.Type,
            ClaimValue = Permissions.GetStudentCourseGrades
        });

        // Student: GetStudentCourseGrades
        claims.Add(new IdentityRoleClaim<string>
        {
            Id = 1302,
            RoleId = DefaultRoles.StudentRoleId,
            ClaimType = Permissions.Type,
            ClaimValue = Permissions.GetStudentCourseGrades
        });

        builder.HasData(claims);
    }
}