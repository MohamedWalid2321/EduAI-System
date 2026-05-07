using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddCheatingReportSystem : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "CheatingReports",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    QuizAttemptId = table.Column<int>(type: "int", nullable: false),
                    StudentId = table.Column<string>(type: "nvarchar(450)", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    LastUpdatedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    CreatedBy = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    LastUpdatedBy = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    IsDeleted = table.Column<bool>(type: "bit", nullable: false),
                    DeletedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    DeletedBy = table.Column<string>(type: "nvarchar(max)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_CheatingReports", x => x.Id);
                    table.ForeignKey(
                        name: "FK_CheatingReports_AspNetUsers_StudentId",
                        column: x => x.StudentId,
                        principalTable: "AspNetUsers",
                        principalColumn: "Id");
                    table.ForeignKey(
                        name: "FK_CheatingReports_QuizAttempts_QuizAttemptId",
                        column: x => x.QuizAttemptId,
                        principalTable: "QuizAttempts",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "CheatingViolations",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    EvidenceUrl = table.Column<string>(type: "nvarchar(1000)", maxLength: 1000, nullable: false),
                    Timestamp = table.Column<DateTime>(type: "datetime2", nullable: false),
                    Description = table.Column<string>(type: "nvarchar(1000)", maxLength: 1000, nullable: false),
                    CheatingReportId = table.Column<int>(type: "int", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    LastUpdatedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    CreatedBy = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    LastUpdatedBy = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    IsDeleted = table.Column<bool>(type: "bit", nullable: false),
                    DeletedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    DeletedBy = table.Column<string>(type: "nvarchar(max)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_CheatingViolations", x => x.Id);
                    table.ForeignKey(
                        name: "FK_CheatingViolations_CheatingReports_CheatingReportId",
                        column: x => x.CheatingReportId,
                        principalTable: "CheatingReports",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 284, DateTimeKind.Local).AddTicks(5960));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 284, DateTimeKind.Local).AddTicks(6023));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 284, DateTimeKind.Local).AddTicks(6026));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 284, DateTimeKind.Local).AddTicks(6029));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 284, DateTimeKind.Local).AddTicks(6030));

            migrationBuilder.InsertData(
                table: "AspNetRoleClaims",
                columns: new[] { "Id", "ClaimType", "ClaimValue", "RoleId" },
                values: new object[,]
                {
                    { 1042, "Permissions", "CheatingReport:read", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 1043, "Permissions", "CheatingReport:add", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 1044, "Permissions", "CheatingReport:delete", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" }
                });

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEGVubZdu+qAsd0IiHVazvmTYo5KZWI8iVQpFkRo9tTeZYMfjv5iq7hdGA0UJPxzpww==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEBNhefh4q5wZMmFfYXVjf/6jGeLMCYXxcvM1RuaOEYP0mefrvnssrieHjY6tWIAN4w==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 286, DateTimeKind.Local).AddTicks(8070));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 286, DateTimeKind.Local).AddTicks(8093));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 286, DateTimeKind.Local).AddTicks(8099));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 286, DateTimeKind.Local).AddTicks(8096));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 286, DateTimeKind.Local).AddTicks(8098));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(995));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1020));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1024));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1027));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1028));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 6,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1031));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 7,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1032));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 8,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1034));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 9,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1035));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 10,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1037));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 11,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1038));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 12,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1040));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 13,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1072));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 14,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1073));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 15,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1074));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 16,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1078));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 17,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1079));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 18,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1081));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 19,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1083));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 20,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1084));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 21,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1085));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 22,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1086));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 23,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1087));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 24,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1088));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 25,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1089));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 26,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1091));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 27,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1092));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 28,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1093));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 29,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1094));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 30,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1095));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 31,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1099));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 32,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1100));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 33,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1101));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 34,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1103));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 35,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1104));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 36,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1105));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 37,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1106));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 38,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1107));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 39,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1109));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 40,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1110));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 41,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1111));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 42,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1112));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 43,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1113));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 44,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1114));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 45,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1115));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 46,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1118));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 47,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1119));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 48,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1120));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 49,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1122));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 50,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1123));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 51,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1124));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 52,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1125));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 53,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1126));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 54,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1127));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 55,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1128));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 56,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1129));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 57,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1130));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 58,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1131));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 59,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1132));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 60,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1133));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 61,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1137));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 62,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1138));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 63,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1139));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 64,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1158));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 65,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1159));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 66,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1161));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 67,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1162));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 68,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1163));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 69,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1165));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 70,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1166));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 71,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1167));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 72,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1168));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 73,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1169));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 74,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1170));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 75,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 7, 23, 13, 14, 287, DateTimeKind.Local).AddTicks(1171));

            migrationBuilder.CreateIndex(
                name: "IX_CheatingReports_QuizAttemptId",
                table: "CheatingReports",
                column: "QuizAttemptId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_CheatingReports_StudentId",
                table: "CheatingReports",
                column: "StudentId");

            migrationBuilder.CreateIndex(
                name: "IX_CheatingViolations_CheatingReportId",
                table: "CheatingViolations",
                column: "CheatingReportId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "CheatingViolations");

            migrationBuilder.DropTable(
                name: "CheatingReports");

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1042);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1043);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1044);

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 986, DateTimeKind.Local).AddTicks(2371));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 986, DateTimeKind.Local).AddTicks(2421));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 986, DateTimeKind.Local).AddTicks(2423));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 986, DateTimeKind.Local).AddTicks(2425));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 986, DateTimeKind.Local).AddTicks(2426));

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAENwf1U1CLEaVQldm4HH2LP7/Y8P0O5zIk65tOXPH9y8Q31NjR5cRhl+McI8FGk2bFw==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEMz9ILvyG1G+dtPV3QVGvRm5Ej33e9EFyYr3uoy6CvUgLz77sZ7vjjJed8XlfOyBcQ==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(1758));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(1780));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(1784));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(1782));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(1783));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5876));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5901));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5905));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5907));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5908));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 6,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5911));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 7,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5913));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 8,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5914));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 9,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5915));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 10,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5917));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 11,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5918));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 12,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5919));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 13,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5920));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 14,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5921));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 15,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5922));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 16,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5926));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 17,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5927));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 18,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5929));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 19,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5930));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 20,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5931));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 21,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5932));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 22,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5934));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 23,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5961));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 24,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5963));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 25,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5964));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 26,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5965));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 27,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5966));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 28,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5967));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 29,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5969));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 30,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5970));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 31,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5973));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 32,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5974));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 33,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5975));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 34,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5978));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 35,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5979));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 36,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5980));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 37,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5981));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 38,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5982));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 39,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5983));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 40,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5984));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 41,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5985));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 42,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5986));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 43,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5987));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 44,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5988));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 45,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5989));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 46,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5992));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 47,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5993));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 48,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5994));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 49,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5996));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 50,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5997));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 51,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5998));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 52,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(5999));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 53,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6000));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 54,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6001));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 55,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6002));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 56,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6003));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 57,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6004));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 58,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6005));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 59,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6006));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 60,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6007));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 61,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6010));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 62,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6011));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 63,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6012));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 64,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6013));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 65,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6014));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 66,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6016));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 67,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6017));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 68,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6018));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 69,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6037));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 70,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6039));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 71,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6040));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 72,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6041));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 73,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6042));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 74,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6043));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 75,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 4, 22, 15, 20, 988, DateTimeKind.Local).AddTicks(6044));
        }
    }
}
