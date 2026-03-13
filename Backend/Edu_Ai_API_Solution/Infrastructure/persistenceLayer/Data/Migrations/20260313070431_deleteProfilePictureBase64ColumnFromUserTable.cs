using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class deleteProfilePictureBase64ColumnFromUserTable : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ProfilePictureBase64",
                table: "AspNetUsers");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEGLnoOA85tqx+Z2Mc19MX1RRUmD33IHwhU1v6KwS9d8aahv2Miw20VTTP9WN0wtZ6g==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEIm7s/BOS5SjCi0S08VAxTz/jOMFYswed3ARWAxvqA5r/47qK1gjFxKDPDtF94Wm+w==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 13, 9, 4, 31, 319, DateTimeKind.Local).AddTicks(4948));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 13, 9, 4, 31, 319, DateTimeKind.Local).AddTicks(4997));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 13, 9, 4, 31, 319, DateTimeKind.Local).AddTicks(5001));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 13, 9, 4, 31, 319, DateTimeKind.Local).AddTicks(4999));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 13, 9, 4, 31, 319, DateTimeKind.Local).AddTicks(5000));
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "ProfilePictureBase64",
                table: "AspNetUsers",
                type: "NVARCHAR(MAX)",
                nullable: true);

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                columns: new[] { "PasswordHash", "ProfilePictureBase64" },
                values: new object[] { "AQAAAAIAAYagAAAAEIiFB2MMRioICyCw9URGQjRe/VIWTH9jsdjA6r9ZE9b4fmizB4nwt0TeXjs/cF7ZsA==", "" });

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                columns: new[] { "PasswordHash", "ProfilePictureBase64" },
                values: new object[] { "AQAAAAIAAYagAAAAEGGq+FLDC4gI/aE+akPn1bv6Ytf3FR0a7HQH0nwHdXnk9HkQMz66qy5Qo26EFgiI1w==", "" });

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 11, 13, 20, 426, DateTimeKind.Local).AddTicks(4903));

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
    }
}
