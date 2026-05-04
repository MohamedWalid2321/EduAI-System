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
    // ────────────────────────────────────────────────────────

    public void Configure(EntityTypeBuilder<IdentityRoleClaim<string>> builder)
    {
        var claims = new List<IdentityRoleClaim<string>>();

        // ── SuperAdmin — all permissions except Course:read ─────────────────
        var superAdminPermissions = Permissions.GetAllPermissions()
            .Where(p => p != Permissions.GetCourse)
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
            Permissions.EnrollInstructor, Permissions.UnenrollInstructor,Permissions.GetAssesment,
            // Questions
            Permissions.GetQuestions, Permissions.AddQuestions, Permissions.UpdateQuestions,
            // Lecture
            Permissions.CreateLecture, Permissions.UpdateLecture, Permissions.DeleteLecture, Permissions.JoinLecture,
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
            Permissions.GetCourse,Permissions.GetAssesment,
            // Questions
            Permissions.GetQuestions, Permissions.AddQuestions, Permissions.UpdateQuestions,
            // Lecture
            Permissions.CreateLecture, Permissions.UpdateLecture, Permissions.DeleteLecture, Permissions.JoinLecture,
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
            Permissions.GetAssSubmission,Permissions.DeleteAssSubmission,
            // Content
            Permissions.GetContent,
            // Course
            Permissions.GetCourse,Permissions.GetAssesment,
            // Questions/Quiz
            Permissions.GetQuestions, Permissions.SolveQuiz,
            // Lecture
            Permissions.JoinLecture,
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

        builder.HasData(claims);
    }
}