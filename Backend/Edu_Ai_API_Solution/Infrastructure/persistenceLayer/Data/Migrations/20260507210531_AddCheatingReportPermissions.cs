using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddCheatingReportPermissions : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
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

            migrationBuilder.InsertData(
                table: "AspNetRoleClaims",
                columns: new[] { "Id", "ClaimType", "ClaimValue", "RoleId" },
                values: new object[,]
                {
                    { 1076, "Permissions", "CheatingReport:read", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 1077, "Permissions", "CheatingReport:add", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 1078, "Permissions", "CheatingReport:delete", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 1120, "Permissions", "CheatingReport:read", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 1121, "Permissions", "CheatingReport:add", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 1122, "Permissions", "CheatingReport:delete", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 1162, "Permissions", "CheatingReport:add", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" }
                });

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

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1076);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1077);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1078);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1120);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1121);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1122);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1162);

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
        }
    }
}
