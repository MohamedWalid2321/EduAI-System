using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class RenameIsEnrolledToIsEnrollableInApplicationRole : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "IsEnrolled",
                table: "AspNetRoles",
                newName: "IsEnrollable");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEPmMjP34L9rh5inqeJbAG+5VidrwJ8r04kBRIc/emuMJsdIzwUbPzZf0xJeRDTJL8w==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAELHL30MJ1pzcz998YW+s8GI3sn3Jx65PauG0FwyTn9gLtHg/Eeyws/zT1QSpeWpR3Q==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 14, 11, 11, 16, 133, DateTimeKind.Local).AddTicks(4576));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 14, 11, 11, 16, 133, DateTimeKind.Local).AddTicks(4645));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 14, 11, 11, 16, 133, DateTimeKind.Local).AddTicks(4654));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 14, 11, 11, 16, 133, DateTimeKind.Local).AddTicks(4649));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 14, 11, 11, 16, 133, DateTimeKind.Local).AddTicks(4651));
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "IsEnrollable",
                table: "AspNetRoles",
                newName: "IsEnrolled");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEDShFc5DSR0jHLWcqlR1ipP+jG5hW/NJchS1Cl9kglletD/DlphOGf+PK6isWvl9Yg==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAELsmNGfy3zPxAAKPw+rD2Z5ZvBKY52BjAnXnivyal2qNXF+D8lJkmOa4PxihrRPHuQ==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 13, 16, 42, 16, 547, DateTimeKind.Local).AddTicks(2591));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 13, 16, 42, 16, 547, DateTimeKind.Local).AddTicks(2640));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 13, 16, 42, 16, 547, DateTimeKind.Local).AddTicks(2645));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 13, 16, 42, 16, 547, DateTimeKind.Local).AddTicks(2642));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 13, 16, 42, 16, 547, DateTimeKind.Local).AddTicks(2644));
        }
    }
}
