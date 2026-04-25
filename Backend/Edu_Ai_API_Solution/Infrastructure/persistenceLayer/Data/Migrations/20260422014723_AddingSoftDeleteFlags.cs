using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddingSoftDeleteFlags : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "UserCourses",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "UserCourses",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "UserCourses",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "StudentAnswers",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "StudentAnswers",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "StudentAnswers",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "Quizzes",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "Quizzes",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "Quizzes",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "QuizQuestions",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "QuizQuestions",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "QuizQuestions",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "QuizAttempts",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "QuizAttempts",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "QuizAttempts",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "QuestionChoices",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "QuestionChoices",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "QuestionChoices",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "Lecture",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "Lecture",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "Lecture",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "Departments",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "Departments",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "Departments",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "Courses",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "Courses",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "Courses",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "Contents",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "Contents",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "Contents",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "ContentAttachments",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "ContentAttachments",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "ContentAttachments",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "AssignmentSubmissions",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "AssignmentSubmissions",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "AssignmentSubmissions",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "AssignmentSubmissionAttachments",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "AssignmentSubmissionAttachments",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "AssignmentSubmissionAttachments",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "Assignments",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "Assignments",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "Assignments",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "AssignmentAttachments",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "AssignmentAttachments",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "AssignmentAttachments",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "Assessments",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "Assessments",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "Assessments",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAECabZUlAQHKtj7Dv9Q5bxDkMZroNP8XiFvnwjh9MHKEH5FzyQzEKgX0GskuJ65PHXA==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEKepBIO259Rbyb2FVs6w2oj16wTDb88YTh8XNVuTFZUBX37nH2VyL7PdvZyzpX89hg==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                columns: new[] { "CreatedAt", "DeletedAt", "DeletedBy", "IsDeleted" },
                values: new object[] { new DateTime(2026, 4, 22, 3, 47, 22, 757, DateTimeKind.Local).AddTicks(1475), null, null, false });

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                columns: new[] { "CreatedAt", "DeletedAt", "DeletedBy", "IsDeleted" },
                values: new object[] { new DateTime(2026, 4, 22, 3, 47, 22, 757, DateTimeKind.Local).AddTicks(1532), null, null, false });

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                columns: new[] { "CreatedAt", "DeletedAt", "DeletedBy", "IsDeleted" },
                values: new object[] { new DateTime(2026, 4, 22, 3, 47, 22, 757, DateTimeKind.Local).AddTicks(1537), null, null, false });

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                columns: new[] { "CreatedAt", "DeletedAt", "DeletedBy", "IsDeleted" },
                values: new object[] { new DateTime(2026, 4, 22, 3, 47, 22, 757, DateTimeKind.Local).AddTicks(1535), null, null, false });

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                columns: new[] { "CreatedAt", "DeletedAt", "DeletedBy", "IsDeleted" },
                values: new object[] { new DateTime(2026, 4, 22, 3, 47, 22, 757, DateTimeKind.Local).AddTicks(1536), null, null, false });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "UserCourses");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "UserCourses");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "UserCourses");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "StudentAnswers");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "StudentAnswers");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "StudentAnswers");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "Quizzes");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "Quizzes");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "Quizzes");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "QuizQuestions");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "QuizQuestions");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "QuizQuestions");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "QuizAttempts");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "QuizAttempts");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "QuizAttempts");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "QuestionChoices");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "QuestionChoices");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "QuestionChoices");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "Lecture");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "Lecture");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "Lecture");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "Departments");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "Departments");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "Departments");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "Courses");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "Courses");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "Courses");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "Contents");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "Contents");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "Contents");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "ContentAttachments");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "ContentAttachments");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "ContentAttachments");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "AssignmentSubmissions");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "AssignmentSubmissions");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "AssignmentSubmissions");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "AssignmentSubmissionAttachments");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "AssignmentSubmissionAttachments");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "AssignmentSubmissionAttachments");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "Assignments");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "Assignments");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "Assignments");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "AssignmentAttachments");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "AssignmentAttachments");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "AssignmentAttachments");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "Assessments");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "Assessments");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "Assessments");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEJ1lIaghDNNualwBRUIDehdAwM63E6kkMaV/MNCY57avT5iR2DkzjmqZCWrm22n1LQ==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEHhi/XxOdbVlzNMz77Rl3mBn133oJ7VNqit8hI9sZehJ0MYrbNa+vMHacNOAgwzTfw==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 22, 1, 47, 26, 587, DateTimeKind.Local).AddTicks(5722));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 22, 1, 47, 26, 587, DateTimeKind.Local).AddTicks(5767));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 22, 1, 47, 26, 587, DateTimeKind.Local).AddTicks(5772));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 22, 1, 47, 26, 587, DateTimeKind.Local).AddTicks(5769));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 22, 1, 47, 26, 587, DateTimeKind.Local).AddTicks(5770));
        }
    }
}
