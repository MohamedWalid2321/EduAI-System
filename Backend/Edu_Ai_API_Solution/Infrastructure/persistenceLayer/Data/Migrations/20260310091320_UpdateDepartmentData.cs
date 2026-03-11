using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class UpdateDepartmentData : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEIiFB2MMRioICyCw9URGQjRe/VIWTH9jsdjA6r9ZE9b4fmizB4nwt0TeXjs/cF7ZsA==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEGGq+FLDC4gI/aE+akPn1bv6Ytf3FR0a7HQH0nwHdXnk9HkQMz66qy5Qo26EFgiI1w==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                columns: new[] { "CreatedAt", "Title" },
                values: new object[] { new DateTime(2026, 3, 10, 11, 13, 20, 426, DateTimeKind.Local).AddTicks(4903), "ComputerEngineering" });

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 11, 13, 20, 426, DateTimeKind.Local).AddTicks(4955));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 11, 13, 20, 426, DateTimeKind.Local).AddTicks(4959));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 11, 13, 20, 426, DateTimeKind.Local).AddTicks(4957));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 11, 13, 20, 426, DateTimeKind.Local).AddTicks(4958));
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEAlYxxw1o8vcGEof8LMiiY0aCC5rx76+XWHH3ku//qvAczklpMK2j+/Ly0FVSLuFmg==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAECf4gip53777EEfVWGMUjn+657/4yvjkUdYHyA7J3eE5khCakUSkJ9Eo3seX3Twvgw==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                columns: new[] { "CreatedAt", "Title" },
                values: new object[] { new DateTime(2026, 3, 10, 9, 53, 31, 783, DateTimeKind.Local).AddTicks(9364), "CommunicationEngineering" });

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 9, 53, 31, 783, DateTimeKind.Local).AddTicks(9408));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 9, 53, 31, 783, DateTimeKind.Local).AddTicks(9412));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 9, 53, 31, 783, DateTimeKind.Local).AddTicks(9410));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 9, 53, 31, 783, DateTimeKind.Local).AddTicks(9411));
        }
    }
}
