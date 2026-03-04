using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddingUserDepartmentRelationShipAndSeedingDepartment : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "AcademicYear",
                table: "AspNetUsers",
                type: "int",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "DepartmentId",
                table: "AspNetUsers",
                type: "int",
                nullable: true);

            migrationBuilder.AddColumn<DateTime>(
                name: "EnrolledAt",
                table: "AspNetUsers",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsEnrolled",
                table: "AspNetUsers",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                columns: new[] { "AcademicYear", "DepartmentId", "EnrolledAt", "IsEnrolled", "PasswordHash" },
                values: new object[] { null, null, null, false, "AQAAAAIAAYagAAAAEGe3aeN4lenqDfX1shgVnAEMcZqC4qnJ9caQkc5Mna9jcuQF7K9Q54rkCdNGDoDXQQ==" });

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                columns: new[] { "AcademicYear", "DepartmentId", "EnrolledAt", "IsEnrolled", "PasswordHash" },
                values: new object[] { null, null, null, false, "AQAAAAIAAYagAAAAEGmbmdjEHCLoPblShOhhHEPaFRq9OZBjWKqCb2xyKSo7ZU6gtdV9YFTY7wJ1lqdMuQ==" });

            migrationBuilder.InsertData(
                table: "Departments",
                columns: new[] { "Id", "CreatedAt", "CreatedBy", "LastUpdatedAt", "LastUpdatedBy", "Title" },
                values: new object[,]
                {
                    { 1000, new DateTime(2026, 3, 1, 12, 38, 39, 555, DateTimeKind.Local).AddTicks(1385), null, null, null, "CommunicationEngineering" },
                    { 1001, new DateTime(2026, 3, 1, 12, 38, 39, 555, DateTimeKind.Local).AddTicks(1449), null, null, null, "ElectricalEngineering" },
                    { 1002, new DateTime(2026, 3, 1, 12, 38, 39, 555, DateTimeKind.Local).AddTicks(1457), null, null, null, "MechanicalEngineering" },
                    { 1003, new DateTime(2026, 3, 1, 12, 38, 39, 555, DateTimeKind.Local).AddTicks(1452), null, null, null, "CommunicationEngineering" },
                    { 1004, new DateTime(2026, 3, 1, 12, 38, 39, 555, DateTimeKind.Local).AddTicks(1455), null, null, null, "BiomedicalEngineering" }
                });

            migrationBuilder.CreateIndex(
                name: "IX_AspNetUsers_DepartmentId",
                table: "AspNetUsers",
                column: "DepartmentId");

            migrationBuilder.AddForeignKey(
                name: "FK_AspNetUsers_Departments_DepartmentId",
                table: "AspNetUsers",
                column: "DepartmentId",
                principalTable: "Departments",
                principalColumn: "Id",
                onDelete: ReferentialAction.SetNull);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_AspNetUsers_Departments_DepartmentId",
                table: "AspNetUsers");

            migrationBuilder.DropIndex(
                name: "IX_AspNetUsers_DepartmentId",
                table: "AspNetUsers");

            migrationBuilder.DeleteData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000);

            migrationBuilder.DeleteData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001);

            migrationBuilder.DeleteData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002);

            migrationBuilder.DeleteData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003);

            migrationBuilder.DeleteData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004);

            migrationBuilder.DropColumn(
                name: "AcademicYear",
                table: "AspNetUsers");

            migrationBuilder.DropColumn(
                name: "DepartmentId",
                table: "AspNetUsers");

            migrationBuilder.DropColumn(
                name: "EnrolledAt",
                table: "AspNetUsers");

            migrationBuilder.DropColumn(
                name: "IsEnrolled",
                table: "AspNetUsers");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEOW7LUkBTomeYWgO3ngzfwf86of175TF2H9qWOzfw2HafEYEKNw6sKzsvcBNI8UF+w==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEE5AVwCBYWNI0qsQ9rFLnsEvzM3WDU/24AM1jdjzE+bWCJ3+LkhxaTpqWXchYT7iRw==");
        }
    }
}
