using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class SomeChangeInFeesTable : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            // REMOVED: DropForeignKey FK_Courses_Courses_PrerequisiteCourseId
            // → Already dropped in migration 20260421234727_RemovePreRequistCouresRelation

            // REMOVED: DropTable InstructorCourse
            // → Already dropped in migration 20260313144217_MergeToUserCourse

            migrationBuilder.DropForeignKey(
                name: "FK_Fee_AcademicYear_AcademicYearId",
                table: "Fee");

            migrationBuilder.DropIndex(
                name: "IX_Fee_AcademicYearId_Name",
                table: "Fee");

            migrationBuilder.DropColumn(
                name: "Name",
                table: "Fee");

            // REMOVED: DropColumn PrerequisiteCourseId from Courses
            // → Already dropped in migration 20260421234727_RemovePreRequistCouresRelation

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "Payments",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "Payments",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "Payments",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "Fee",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "Fee",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "DepartmentId",
                table: "Fee",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<string>(
                name: "FeeType",
                table: "Fee",
                type: "nvarchar(50)",
                maxLength: 50,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "Fee",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AlterColumn<string>(
                name: "Name",
                table: "AcademicYear",
                type: "nvarchar(50)",
                maxLength: 50,
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(max)");

            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "AcademicYear",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeletedBy",
                table: "AcademicYear",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "AcademicYear",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.InsertData(
                table: "AcademicYear",
                columns: new[] { "Id", "CreatedAt", "CreatedBy", "DeletedAt", "DeletedBy", "IsDeleted", "LastUpdatedAt", "LastUpdatedBy", "Name" },
                values: new object[,]
                {
                    { 1, new DateTime(2026, 4, 27, 2, 12, 32, 357, DateTimeKind.Local).AddTicks(5336), null, null, null, false, null, null, "First Year" },
                    { 2, new DateTime(2026, 4, 27, 2, 12, 32, 357, DateTimeKind.Local).AddTicks(5386), null, null, null, false, null, null, "Second Year" },
                    { 3, new DateTime(2026, 4, 27, 2, 12, 32, 357, DateTimeKind.Local).AddTicks(5390), null, null, null, false, null, null, "Third Year" },
                    { 4, new DateTime(2026, 4, 27, 2, 12, 32, 357, DateTimeKind.Local).AddTicks(5432), null, null, null, false, null, null, "Fourth Year" },
                    { 5, new DateTime(2026, 4, 27, 2, 12, 32, 357, DateTimeKind.Local).AddTicks(5434), null, null, null, false, null, null, "Fifth Year" }
                });

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEERtknVH6hOWNtRiMw5W/+Kcc+TiLy3oj4pIy7twBJP1KTjYn3j2h3NSfZw8UfvJIA==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEG8nK904JtxhuSapyxrUc/8R5TO7ddjuyTIqPqeWA8jtFR0eY9V8AYtBmaocaRmyvg==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(3679));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(3716));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(3723));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(3719));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(3721));

            migrationBuilder.InsertData(
                table: "Fee",
                columns: new[] { "Id", "AcademicYearId", "Amount", "CreatedAt", "CreatedBy", "DeletedAt", "DeletedBy", "DepartmentId", "FeeType", "IsDeleted", "LastUpdatedAt", "LastUpdatedBy" },
                values: new object[,]
                {
                    { 1, 1, 6000.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8631), null, null, null, 1000, "Tuition", false, null, null },
                    { 2, 1, 800.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8665), null, null, null, 1000, "Books", false, null, null },
                    { 3, 1, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8669), null, null, null, 1000, "Activities", false, null, null },
                    { 4, 2, 6600.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8672), null, null, null, 1000, "Tuition", false, null, null },
                    { 5, 2, 880.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8674), null, null, null, 1000, "Books", false, null, null },
                    { 6, 2, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8679), null, null, null, 1000, "Activities", false, null, null },
                    { 7, 3, 7200.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8681), null, null, null, 1000, "Tuition", false, null, null },
                    { 8, 3, 960.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8683), null, null, null, 1000, "Books", false, null, null },
                    { 9, 3, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8685), null, null, null, 1000, "Activities", false, null, null },
                    { 10, 4, 7800.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8688), null, null, null, 1000, "Tuition", false, null, null },
                    { 11, 4, 1040.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8690), null, null, null, 1000, "Books", false, null, null },
                    { 12, 4, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8692), null, null, null, 1000, "Activities", false, null, null },
                    { 13, 5, 8400.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8694), null, null, null, 1000, "Tuition", false, null, null },
                    { 14, 5, 1120.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8696), null, null, null, 1000, "Books", false, null, null },
                    { 15, 5, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8698), null, null, null, 1000, "Activities", false, null, null },
                    { 16, 1, 5500.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8703), null, null, null, 1001, "Tuition", false, null, null },
                    { 17, 1, 700.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8705), null, null, null, 1001, "Books", false, null, null },
                    { 18, 1, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8708), null, null, null, 1001, "Activities", false, null, null },
                    { 19, 2, 6050.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8710), null, null, null, 1001, "Tuition", false, null, null },
                    { 20, 2, 770.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8712), null, null, null, 1001, "Books", false, null, null },
                    { 21, 2, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8713), null, null, null, 1001, "Activities", false, null, null },
                    { 22, 3, 6600.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8715), null, null, null, 1001, "Tuition", false, null, null },
                    { 23, 3, 840.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8717), null, null, null, 1001, "Books", false, null, null },
                    { 24, 3, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8718), null, null, null, 1001, "Activities", false, null, null },
                    { 25, 4, 7150.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8720), null, null, null, 1001, "Tuition", false, null, null },
                    { 26, 4, 910.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8721), null, null, null, 1001, "Books", false, null, null },
                    { 27, 4, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8791), null, null, null, 1001, "Activities", false, null, null },
                    { 28, 5, 7700.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8794), null, null, null, 1001, "Tuition", false, null, null },
                    { 29, 5, 980.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8795), null, null, null, 1001, "Books", false, null, null },
                    { 30, 5, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8797), null, null, null, 1001, "Activities", false, null, null },
                    { 31, 1, 5000.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8803), null, null, null, 1002, "Tuition", false, null, null },
                    { 32, 1, 600.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8805), null, null, null, 1002, "Books", false, null, null },
                    { 33, 1, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8806), null, null, null, 1002, "Activities", false, null, null },
                    { 34, 2, 5500.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8810), null, null, null, 1002, "Tuition", false, null, null },
                    { 35, 2, 660.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8812), null, null, null, 1002, "Books", false, null, null },
                    { 36, 2, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8813), null, null, null, 1002, "Activities", false, null, null },
                    { 37, 3, 6000.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8815), null, null, null, 1002, "Tuition", false, null, null },
                    { 38, 3, 720.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8817), null, null, null, 1002, "Books", false, null, null },
                    { 39, 3, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8818), null, null, null, 1002, "Activities", false, null, null },
                    { 40, 4, 6500.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8820), null, null, null, 1002, "Tuition", false, null, null },
                    { 41, 4, 780.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8822), null, null, null, 1002, "Books", false, null, null },
                    { 42, 4, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8823), null, null, null, 1002, "Activities", false, null, null },
                    { 43, 5, 7000.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8825), null, null, null, 1002, "Tuition", false, null, null },
                    { 44, 5, 840.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8827), null, null, null, 1002, "Books", false, null, null },
                    { 45, 5, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8828), null, null, null, 1002, "Activities", false, null, null },
                    { 46, 1, 5800.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8834), null, null, null, 1003, "Tuition", false, null, null },
                    { 47, 1, 750.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8835), null, null, null, 1003, "Books", false, null, null },
                    { 48, 1, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8837), null, null, null, 1003, "Activities", false, null, null },
                    { 49, 2, 6380.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8838), null, null, null, 1003, "Tuition", false, null, null },
                    { 50, 2, 825.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8840), null, null, null, 1003, "Books", false, null, null },
                    { 51, 2, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8842), null, null, null, 1003, "Activities", false, null, null },
                    { 52, 3, 6960.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8843), null, null, null, 1003, "Tuition", false, null, null },
                    { 53, 3, 900.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8845), null, null, null, 1003, "Books", false, null, null },
                    { 54, 3, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8846), null, null, null, 1003, "Activities", false, null, null },
                    { 55, 4, 7540.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8848), null, null, null, 1003, "Tuition", false, null, null },
                    { 56, 4, 975.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8850), null, null, null, 1003, "Books", false, null, null },
                    { 57, 4, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8851), null, null, null, 1003, "Activities", false, null, null },
                    { 58, 5, 8120.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8853), null, null, null, 1003, "Tuition", false, null, null },
                    { 59, 5, 1050.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8855), null, null, null, 1003, "Books", false, null, null },
                    { 60, 5, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8856), null, null, null, 1003, "Activities", false, null, null },
                    { 61, 1, 6500.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8861), null, null, null, 1004, "Tuition", false, null, null },
                    { 62, 1, 900.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8862), null, null, null, 1004, "Books", false, null, null },
                    { 63, 1, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8864), null, null, null, 1004, "Activities", false, null, null },
                    { 64, 2, 7150.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8866), null, null, null, 1004, "Tuition", false, null, null },
                    { 65, 2, 990.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8867), null, null, null, 1004, "Books", false, null, null },
                    { 66, 2, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8870), null, null, null, 1004, "Activities", false, null, null },
                    { 67, 3, 7800.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8872), null, null, null, 1004, "Tuition", false, null, null },
                    { 68, 3, 1080.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8873), null, null, null, 1004, "Books", false, null, null },
                    { 69, 3, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8875), null, null, null, 1004, "Activities", false, null, null },
                    { 70, 4, 8450.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8876), null, null, null, 1004, "Tuition", false, null, null },
                    { 71, 4, 1170.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8878), null, null, null, 1004, "Books", false, null, null },
                    { 72, 4, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8879), null, null, null, 1004, "Activities", false, null, null },
                    { 73, 5, 9100.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8929), null, null, null, 1004, "Tuition", false, null, null },
                    { 74, 5, 1260.00m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8931), null, null, null, 1004, "Books", false, null, null },
                    { 75, 5, 300m, new DateTime(2026, 4, 27, 2, 12, 32, 360, DateTimeKind.Local).AddTicks(8932), null, null, null, 1004, "Activities", false, null, null }
                });

            migrationBuilder.CreateIndex(
                name: "IX_Fee_AcademicYearId_DepartmentId_FeeType",
                table: "Fee",
                columns: new[] { "AcademicYearId", "DepartmentId", "FeeType" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_Fee_DepartmentId",
                table: "Fee",
                column: "DepartmentId");

            migrationBuilder.AddForeignKey(
                name: "FK_Fee_AcademicYear_AcademicYearId",
                table: "Fee",
                column: "AcademicYearId",
                principalTable: "AcademicYear",
                principalColumn: "Id",
                onDelete: ReferentialAction.Restrict);

            migrationBuilder.AddForeignKey(
                name: "FK_Fee_Departments_DepartmentId",
                table: "Fee",
                column: "DepartmentId",
                principalTable: "Departments",
                principalColumn: "Id",
                onDelete: ReferentialAction.Restrict);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Fee_AcademicYear_AcademicYearId",
                table: "Fee");

            migrationBuilder.DropForeignKey(
                name: "FK_Fee_Departments_DepartmentId",
                table: "Fee");

            migrationBuilder.DropIndex(
                name: "IX_Fee_AcademicYearId_DepartmentId_FeeType",
                table: "Fee");

            migrationBuilder.DropIndex(
                name: "IX_Fee_DepartmentId",
                table: "Fee");

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 1);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 2);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 3);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 4);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 5);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 6);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 7);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 8);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 9);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 10);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 11);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 12);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 13);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 14);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 15);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 16);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 17);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 18);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 19);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 20);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 21);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 22);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 23);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 24);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 25);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 26);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 27);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 28);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 29);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 30);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 31);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 32);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 33);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 34);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 35);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 36);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 37);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 38);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 39);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 40);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 41);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 42);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 43);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 44);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 45);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 46);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 47);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 48);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 49);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 50);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 51);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 52);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 53);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 54);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 55);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 56);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 57);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 58);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 59);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 60);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 61);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 62);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 63);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 64);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 65);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 66);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 67);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 68);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 69);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 70);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 71);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 72);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 73);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 74);

            migrationBuilder.DeleteData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 75);

            migrationBuilder.DeleteData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 1);

            migrationBuilder.DeleteData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 2);

            migrationBuilder.DeleteData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 3);

            migrationBuilder.DeleteData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 4);

            migrationBuilder.DeleteData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 5);

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "Payments");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "Payments");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "Payments");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "Fee");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "Fee");

            migrationBuilder.DropColumn(
                name: "DepartmentId",
                table: "Fee");

            migrationBuilder.DropColumn(
                name: "FeeType",
                table: "Fee");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "Fee");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "AcademicYear");

            migrationBuilder.DropColumn(
                name: "DeletedBy",
                table: "AcademicYear");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "AcademicYear");

            migrationBuilder.AddColumn<int>(
                name: "Name",
                table: "Fee",
                type: "int",
                maxLength: 100,
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "PrerequisiteCourseId",
                table: "Courses",
                type: "int",
                nullable: true);

            migrationBuilder.AlterColumn<string>(
                name: "Name",
                table: "AcademicYear",
                type: "nvarchar(max)",
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(50)",
                oldMaxLength: 50);

            migrationBuilder.CreateTable(
                name: "InstructorCourse",
                columns: table => new
                {
                    CourseId = table.Column<int>(type: "int", nullable: true),
                    CreatedById = table.Column<string>(type: "nvarchar(450)", nullable: true)
                },
                constraints: table =>
                {
                    table.ForeignKey(
                        name: "FK_InstructorCourse_AspNetUsers_CreatedById",
                        column: x => x.CreatedById,
                        principalTable: "AspNetUsers",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_InstructorCourse_Courses_CourseId",
                        column: x => x.CourseId,
                        principalTable: "Courses",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEOWZVo54O7t+j0m6vtcBTTF9ZYE0F8PDMvFfkTBcLVpPrcU8SyXaYxD1uMxwvVvtWg==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEMPNEtSHlXRLP5XwNknrW6yQcQlJG9m99gozXg3cangkZ8gfZN6m9AiXMAXETC2dGQ==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 24, 21, 55, 25, 307, DateTimeKind.Local).AddTicks(7366));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 24, 21, 55, 25, 307, DateTimeKind.Local).AddTicks(7390));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 24, 21, 55, 25, 307, DateTimeKind.Local).AddTicks(7392));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 24, 21, 55, 25, 307, DateTimeKind.Local).AddTicks(7391));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 24, 21, 55, 25, 307, DateTimeKind.Local).AddTicks(7392));

            migrationBuilder.CreateIndex(
                name: "IX_Fee_AcademicYearId_Name",
                table: "Fee",
                columns: new[] { "AcademicYearId", "Name" },
                unique: true);

            migrationBuilder.AddForeignKey(
                name: "FK_Courses_Courses_PrerequisiteCourseId",
                table: "Courses",
                column: "PrerequisiteCourseId",
                principalTable: "Courses",
                principalColumn: "Id",
                onDelete: ReferentialAction.Restrict);

            migrationBuilder.AddForeignKey(
                name: "FK_Fee_AcademicYear_AcademicYearId",
                table: "Fee",
                column: "AcademicYearId",
                principalTable: "AcademicYear",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);
        }
    }
}
