using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddLecturesTable : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "Lecture",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Title = table.Column<string>(type: "nvarchar(200)", maxLength: 200, nullable: false),
                    Description = table.Column<string>(type: "nvarchar(1000)", maxLength: 1000, nullable: false),
                    RoomName = table.Column<string>(type: "nvarchar(300)", maxLength: 300, nullable: false),
                    ScheduledAt = table.Column<DateTime>(type: "datetime2", nullable: false),
                    IsActive = table.Column<bool>(type: "bit", nullable: false),
                    CourseId = table.Column<int>(type: "int", nullable: false),
                    CreatedById = table.Column<string>(type: "nvarchar(450)", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    LastUpdatedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    LastUpdatedBy = table.Column<string>(type: "nvarchar(max)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Lecture", x => x.Id);
                    table.ForeignKey(
                        name: "FK_Lecture_AspNetUsers_CreatedById",
                        column: x => x.CreatedById,
                        principalTable: "AspNetUsers",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_Lecture_Courses_CourseId",
                        column: x => x.CourseId,
                        principalTable: "Courses",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 35,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Lecture:create", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 36,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Lecture:update", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 37,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Lecture:delete", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 38,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Lecture:join", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 39,
                column: "ClaimValue",
                value: "Ass:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 40,
                column: "ClaimValue",
                value: "Ass:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 41,
                column: "ClaimValue",
                value: "Ass:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 42,
                column: "ClaimValue",
                value: "Ass:delete");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 43,
                column: "ClaimValue",
                value: "Content:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 44,
                column: "ClaimValue",
                value: "Content:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 45,
                column: "ClaimValue",
                value: "Content:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 46,
                column: "ClaimValue",
                value: "Content:delete");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 47,
                column: "ClaimValue",
                value: "Course:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 48,
                column: "ClaimValue",
                value: "Course:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 49,
                column: "ClaimValue",
                value: "Course:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 50,
                column: "ClaimValue",
                value: "Course:delete");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 51,
                column: "ClaimValue",
                value: "Course:enrollInstructor");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 52,
                column: "ClaimValue",
                value: "Course:unenrollInstructor");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 53,
                column: "ClaimValue",
                value: "Department:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 54,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "questions:read", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 55,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "questions:add", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 56,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "questions:update", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 57,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "users:read", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 58,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Lecture:create", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 59,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Lecture:update", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 60,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Lecture:delete", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 61,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Lecture:join", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 62,
                column: "ClaimValue",
                value: "Ass:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 63,
                column: "ClaimValue",
                value: "Ass:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 64,
                column: "ClaimValue",
                value: "Ass:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 65,
                column: "ClaimValue",
                value: "Ass:delete");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 66,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:read", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 67,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:add", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 68,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:update", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 69,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:delete", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 70,
                column: "RoleId",
                value: "7e07bb31-26ad-47ac-880c-c5fdfa0516d3");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 71,
                column: "RoleId",
                value: "7e07bb31-26ad-47ac-880c-c5fdfa0516d3");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 72,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "questions:add", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.InsertData(
                table: "AspNetRoleClaims",
                columns: new[] { "Id", "ClaimType", "ClaimValue", "RoleId" },
                values: new object[,]
                {
                    { 73, "Permissions", "questions:update", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 74, "Permissions", "Lecture:create", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 75, "Permissions", "Lecture:update", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 76, "Permissions", "Lecture:delete", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 77, "Permissions", "Lecture:join", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 78, "Permissions", "Profile:levelUp", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 79, "Permissions", "Ass:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 80, "Permissions", "Ass:solve", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 81, "Permissions", "Content:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 82, "Permissions", "Course:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 83, "Permissions", "questions:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 84, "Permissions", "questions:solve", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 85, "Permissions", "Lecture:join", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" }
                });

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEMt9ZJ4g5jJbKroRb86bqTCLOaTEhkS1KfhvqjtSlxf4t1/aAk0EbC/exQSh7sovaA==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEEfbGPcbdhufhwNBfhRLqbITBk24xYBI2z97IxGoaBAeskSwHOb0brqKK+FafMq1vw==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 10, 5, 4, 9, 744, DateTimeKind.Local).AddTicks(8063));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 10, 5, 4, 9, 744, DateTimeKind.Local).AddTicks(8132));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 10, 5, 4, 9, 744, DateTimeKind.Local).AddTicks(8141));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 10, 5, 4, 9, 744, DateTimeKind.Local).AddTicks(8136));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 10, 5, 4, 9, 744, DateTimeKind.Local).AddTicks(8139));

            migrationBuilder.CreateIndex(
                name: "IX_Lecture_CourseId",
                table: "Lecture",
                column: "CourseId");

            migrationBuilder.CreateIndex(
                name: "IX_Lecture_CreatedById",
                table: "Lecture",
                column: "CreatedById");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "Lecture");

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 73);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 74);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 75);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 76);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 77);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 78);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 79);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 80);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 81);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 82);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 83);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 84);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 85);

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 35,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:read", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 36,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:add", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 37,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:update", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 38,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:delete", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 39,
                column: "ClaimValue",
                value: "Content:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 40,
                column: "ClaimValue",
                value: "Content:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 41,
                column: "ClaimValue",
                value: "Content:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 42,
                column: "ClaimValue",
                value: "Content:delete");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 43,
                column: "ClaimValue",
                value: "Course:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 44,
                column: "ClaimValue",
                value: "Course:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 45,
                column: "ClaimValue",
                value: "Course:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 46,
                column: "ClaimValue",
                value: "Course:delete");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 47,
                column: "ClaimValue",
                value: "Course:enrollInstructor");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 48,
                column: "ClaimValue",
                value: "Course:unenrollInstructor");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 49,
                column: "ClaimValue",
                value: "Department:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 50,
                column: "ClaimValue",
                value: "questions:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 51,
                column: "ClaimValue",
                value: "questions:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 52,
                column: "ClaimValue",
                value: "questions:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 53,
                column: "ClaimValue",
                value: "users:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 54,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:read", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 55,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:add", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 56,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:update", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 57,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:delete", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 58,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:read", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 59,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:add", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 60,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:update", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 61,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:delete", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 62,
                column: "ClaimValue",
                value: "Course:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 63,
                column: "ClaimValue",
                value: "questions:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 64,
                column: "ClaimValue",
                value: "questions:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 65,
                column: "ClaimValue",
                value: "questions:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 66,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Profile:levelUp", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 67,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 68,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:solve", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 69,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 70,
                column: "RoleId",
                value: "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 71,
                column: "RoleId",
                value: "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 72,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "questions:solve", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" });

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEC45NhTu+4EXGkJ2zPFUcdMHYX6sR6vQ0gnlWvFjP4Hef++6GuqwWDTBGCSIFPR9vA==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEOYpha2BrH/tisDoF7Y69Qr6XUhOpKId0cyN3WCzpPeGhmVRB42dU5lwmc4K1BpSZQ==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 14, 14, 50, 29, 848, DateTimeKind.Local).AddTicks(1446));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 14, 14, 50, 29, 848, DateTimeKind.Local).AddTicks(1495));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 14, 14, 50, 29, 848, DateTimeKind.Local).AddTicks(1500));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 14, 14, 50, 29, 848, DateTimeKind.Local).AddTicks(1497));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 14, 14, 50, 29, 848, DateTimeKind.Local).AddTicks(1498));
        }
    }
}
