using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddRiskAnalysis : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<decimal>(
                name: "RiskScore",
                table: "CheatingReports",
                type: "decimal(6,4)",
                precision: 6,
                scale: 4,
                nullable: true);

            migrationBuilder.CreateTable(
                name: "RiskAnalyses",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    AttemptId = table.Column<int>(type: "int", nullable: false),
                    StudentId = table.Column<string>(type: "nvarchar(450)", maxLength: 450, nullable: false),
                    QuestionId = table.Column<int>(type: "int", nullable: false),
                    ViolationRate = table.Column<double>(type: "float", nullable: false),
                    FaceDetection = table.Column<int>(type: "int", nullable: false),
                    FaceRecognition = table.Column<int>(type: "int", nullable: false),
                    EyeGaze = table.Column<int>(type: "int", nullable: false),
                    SpeechDetection = table.Column<int>(type: "int", nullable: false),
                    ObjectDetection = table.Column<int>(type: "int", nullable: false),
                    WeightFaceAbsenceMismatch = table.Column<double>(type: "float", nullable: false),
                    WeightSuspiciousMovement = table.Column<double>(type: "float", nullable: false),
                    WeightConversationNoise = table.Column<double>(type: "float", nullable: false),
                    WeightForbiddenObjects = table.Column<double>(type: "float", nullable: false),
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
                    table.PrimaryKey("PK_RiskAnalyses", x => x.Id);
                });

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 995, DateTimeKind.Local).AddTicks(1313));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 995, DateTimeKind.Local).AddTicks(1368));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 995, DateTimeKind.Local).AddTicks(1371));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 995, DateTimeKind.Local).AddTicks(1373));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 995, DateTimeKind.Local).AddTicks(1375));

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEAHHmCYWtcnukIAavJvImwfFmERVV/OdHoohrMHzPO4qpqZnqLhGGlzQOOXqOVBFvg==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEGAuiWzcV0l8kaqTwErwsVg9M9b5GMgQxdY821T9etmsBIrBMLNBWt0LkGkiRvKmcQ==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(2528));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(2580));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(2586));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(2583));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(2584));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6469));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6502));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6505));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6508));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6560));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 6,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6564));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 7,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6566));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 8,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6568));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 9,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6569));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 10,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6572));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 11,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6573));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 12,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6574));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 13,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6576));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 14,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6577));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 15,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6578));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 16,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6583));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 17,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6584));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 18,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6586));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 19,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6587));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 20,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6589));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 21,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6590));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 22,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6591));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 23,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6592));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 24,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6594));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 25,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6595));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 26,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6596));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 27,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6597));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 28,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6599));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 29,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6600));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 30,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6601));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 31,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6605));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 32,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6606));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 33,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6607));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 34,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6609));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 35,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6610));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 36,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6611));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 37,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6613));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 38,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6614));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 39,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6615));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 40,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6616));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 41,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6617));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 42,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6619));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 43,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6620));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 44,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6621));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 45,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6622));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 46,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6626));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 47,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6627));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 48,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6628));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 49,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6629));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 50,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6631));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 51,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6632));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 52,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6633));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 53,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6634));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 54,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6668));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 55,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6669));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 56,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6670));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 57,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6672));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 58,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6673));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 59,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6674));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 60,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6675));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 61,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6680));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 62,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6681));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 63,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6682));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 64,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6683));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 65,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6685));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 66,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6687));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 67,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6688));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 68,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6689));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 69,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6690));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 70,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6692));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 71,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6693));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 72,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6694));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 73,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6695));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 74,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6696));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 75,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 13, 3, 15, 33, 998, DateTimeKind.Local).AddTicks(6698));

            migrationBuilder.CreateIndex(
                name: "IX_RiskAnalyses_AttemptId_QuestionId",
                table: "RiskAnalyses",
                columns: new[] { "AttemptId", "QuestionId" });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "RiskAnalyses");

            migrationBuilder.DropColumn(
                name: "RiskScore",
                table: "CheatingReports");

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 87, DateTimeKind.Local).AddTicks(7092));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 87, DateTimeKind.Local).AddTicks(7143));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 87, DateTimeKind.Local).AddTicks(7146));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 87, DateTimeKind.Local).AddTicks(7147));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 87, DateTimeKind.Local).AddTicks(7149));

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEO7rs/TEiozQSHLfzWtVEjO+LRSXzxzZWjSnollrzqNhBhTKB/5P2QoLWnhQwnMyaw==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEIEuQpzeqjDqiMlsPtnn2M8GZGzk1cEw9YDhVqyyyIAz2wJ/w/lJFkBaeabB8n+5Ow==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(212));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(244));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(250));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(247));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(248));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3710));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3742));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3747));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3751));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3753));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 6,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3758));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 7,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3760));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 8,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3762));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 9,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3764));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 10,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3768));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 11,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3770));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 12,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3772));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 13,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3774));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 14,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3776));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 15,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3778));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 16,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3784));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 17,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3785));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 18,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3788));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 19,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3791));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 20,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3792));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 21,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3794));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 22,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3796));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 23,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3798));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 24,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3800));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 25,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3802));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 26,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3804));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 27,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3806));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 28,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3808));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 29,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3809));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 30,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3811));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 31,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3816));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 32,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3818));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 33,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3820));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 34,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3823));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 35,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3825));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 36,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3827));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 37,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3829));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 38,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3831));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 39,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3833));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 40,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3835));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 41,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3836));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 42,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3838));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 43,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3840));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 44,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3842));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 45,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3844));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 46,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3850));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 47,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3887));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 48,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3890));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 49,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3892));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 50,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3894));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 51,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3896));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 52,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3898));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 53,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3900));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 54,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3902));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 55,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3905));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 56,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3906));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 57,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3908));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 58,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3910));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 59,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3912));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 60,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3913));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 61,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3919));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 62,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3921));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 63,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3922));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 64,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3924));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 65,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3926));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 66,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3929));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 67,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3931));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 68,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3933));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 69,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3935));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 70,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3937));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 71,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3938));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 72,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3940));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 73,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3941));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 74,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3942));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 75,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 8, 0, 5, 31, 91, DateTimeKind.Local).AddTicks(3947));
        }
    }
}
