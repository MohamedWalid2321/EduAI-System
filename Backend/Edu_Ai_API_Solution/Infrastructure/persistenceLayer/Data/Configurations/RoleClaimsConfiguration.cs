using Microsoft.AspNetCore.Identity;
using Shared.Constants;

namespace persistenceLayer.Data.Configurations;

public class RoleClaimsConfiguration : IEntityTypeConfiguration<IdentityRoleClaim<string>>
{
    public void Configure(EntityTypeBuilder<IdentityRoleClaim<string>> builder)
    {
        var claims = new List<IdentityRoleClaim<string>>();
        var id = 1;

        // SuperAdmin - All Permissions
        foreach (var permission in Permissions.GetAllPermissions())
        {
            claims.Add(new IdentityRoleClaim<string>
            {
                Id = id++,
                RoleId = DefaultRoles.SuperAdminRoleId,
                ClaimType = Permissions.Type,
                ClaimValue = permission
            });
        }

        // Admin - All except Role Management
        var adminPermissions = new[]
        {
            // Assignment
            Permissions.GetAss, Permissions.AddAss, Permissions.UpdateAss, Permissions.DeleteAss,
            // Content
            Permissions.GetContent, Permissions.AddContent, Permissions.UpdateContent, Permissions.DeleteContent,
            // Course
            Permissions.GetCourse, Permissions.AddCourse, Permissions.UpdateCourse, Permissions.DeleteCourse,
            // Department
            Permissions.GetDepartment, Permissions.AddDepartment, Permissions.UpdateDepartment, Permissions.DeleteDepartment,
            // Questions
            Permissions.GetQuestions, Permissions.AddQuestions, Permissions.UpdateQuestions,
            // Users
            Permissions.GetUsers, Permissions.AddUsers, Permissions.UpdateUsers, Permissions.DeleteUsers
            // Note: No Role permissions (GetRoles, AddRoles, UpdateRoles, DeleteRoles)
        };

        foreach (var permission in adminPermissions)
        {
            claims.Add(new IdentityRoleClaim<string>
            {
                Id = id++,
                RoleId = DefaultRoles.AdminRoleId,
                ClaimType = Permissions.Type,
                ClaimValue = permission
            });
        }

        // Instructor - Course, Content, Assignment, Quiz management
        var instructorPermissions = new[]
        {
            // Assignment - Full management
            Permissions.GetAss, Permissions.AddAss, Permissions.UpdateAss, Permissions.DeleteAss,
            // Content - Full management
            Permissions.GetContent, Permissions.AddContent, Permissions.UpdateContent, Permissions.DeleteContent,
            // Course - Read only
            Permissions.GetCourse,
            // Questions/Quiz - Full management
            Permissions.GetQuestions, Permissions.AddQuestions, Permissions.UpdateQuestions
        };

        foreach (var permission in instructorPermissions)
        {
            claims.Add(new IdentityRoleClaim<string>
            {
                Id = id++,
                RoleId = DefaultRoles.InstructorRoleId,
                ClaimType = Permissions.Type,
                ClaimValue = permission
            });
        }

        // Student - Read-only access + solve quiz and assignment
        var studentPermissions = new[]
        {
            // Assignment - Read + Solve (submit)
            Permissions.GetAss,
            Permissions.SolveAss,
            // Content - Read only
            Permissions.GetContent,
            // Course - Read only
            Permissions.GetCourse,
            // Questions/Quiz - Read + Solve
            Permissions.GetQuestions,
            Permissions.SolveQuiz
        };

        foreach (var permission in studentPermissions)
        {
            claims.Add(new IdentityRoleClaim<string>
            {
                Id = id++,
                RoleId = DefaultRoles.StudentRoleId,
                ClaimType = Permissions.Type,
                ClaimValue = permission
            });
        }

        builder.HasData(claims);
    }
}