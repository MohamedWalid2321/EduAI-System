using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddAttemptScorePermissions : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 188, DateTimeKind.Local).AddTicks(8229));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 188, DateTimeKind.Local).AddTicks(8274));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 188, DateTimeKind.Local).AddTicks(8277));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 188, DateTimeKind.Local).AddTicks(8278));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 188, DateTimeKind.Local).AddTicks(8280));

            migrationBuilder.InsertData(
                table: "AspNetRoleClaims",
                columns: new[] { "Id", "ClaimType", "ClaimValue", "RoleId" },
                values: new object[,]
                {
                    { 1048, "Permissions", "AttemptScore:finalize", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 1049, "Permissions", "AttemptScore:update", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 1082, "Permissions", "AttemptScore:update", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 1126, "Permissions", "AttemptScore:finalize", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" }
                });

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEPuYv/Mu6rIq7E2Tkpa1lrfpTRTJ6d9Cus7TKzNyNBpoLbFejwXndv9GTPv5a1Qb4Q==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEFF0hAI2XFu4zy65pzyJYMg25bHiFbVZzSp4HE5DnHP3mgCD8HQFjGCKtQTBKkxhTg==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(752));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(780));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(786));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(783));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(785));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5806));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5835));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5838));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5841));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5842));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 6,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5849));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 7,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5850));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 8,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5851));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 9,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5853));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 10,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5909));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 11,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5910));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 12,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5912));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 13,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5913));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 14,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5915));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 15,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5916));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 16,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5920));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 17,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5921));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 18,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5924));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 19,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5925));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 20,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5927));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 21,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5928));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 22,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5929));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 23,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5930));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 24,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5931));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 25,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5933));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 26,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5934));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 27,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5935));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 28,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5936));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 29,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5937));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 30,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5939));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 31,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5942));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 32,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5943));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 33,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5944));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 34,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5947));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 35,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5948));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 36,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5949));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 37,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5950));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 38,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5951));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 39,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5952));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 40,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5954));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 41,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5955));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 42,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5956));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 43,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5957));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 44,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5958));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 45,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5960));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 46,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5963));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 47,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5964));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 48,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5965));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 49,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5967));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 50,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5968));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 51,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5969));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 52,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5970));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 53,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5971));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 54,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5972));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 55,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5974));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 56,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5975));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 57,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5976));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 58,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5977));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 59,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5979));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 60,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5982));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 61,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5987));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 62,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5988));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 63,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5989));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 64,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5991));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 65,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5992));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 66,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5994));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 67,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5995));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 68,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5996));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 69,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5998));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 70,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(5999));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 71,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(6000));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 72,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(6001));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 73,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(6002));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 74,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(6004));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 75,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 32, 35, 191, DateTimeKind.Local).AddTicks(6005));
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1048);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1049);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1082);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1126);

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 552, DateTimeKind.Local).AddTicks(4595));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 552, DateTimeKind.Local).AddTicks(4645));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 552, DateTimeKind.Local).AddTicks(4648));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 552, DateTimeKind.Local).AddTicks(4649));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 552, DateTimeKind.Local).AddTicks(4651));

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEPjBLxN14N37HOD7NgdUCLwvNPAVc8w4Kg3zEIW3etreAr3j5LmVwIU/GRqm+w0HYA==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEKyb4s8uYKZuNpEDdOq0gqjAxdeAYq7WSsGffQ/QIra9PgedXamVkVX9kgvicn4TFg==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(2403));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(2446));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(2452));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(2449));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(2451));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6092));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6124));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6128));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6131));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6132));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 6,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6135));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 7,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6137));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 8,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6138));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 9,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6140));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 10,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6142));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 11,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6143));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 12,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6145));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 13,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6146));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 14,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6147));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 15,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6149));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 16,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6152));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 17,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6154));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 18,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6156));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 19,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6157));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 20,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6158));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 21,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6160));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 22,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6161));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 23,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6162));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 24,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6164));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 25,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6165));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 26,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6166));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 27,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6167));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 28,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6169));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 29,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6170));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 30,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6171));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 31,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6175));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 32,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6176));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 33,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6177));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 34,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6180));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 35,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6181));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 36,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6182));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 37,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6183));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 38,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6185));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 39,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6186));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 40,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6187));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 41,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6188));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 42,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6224));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 43,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6226));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 44,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6227));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 45,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6228));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 46,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6233));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 47,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6234));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 48,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6235));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 49,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6236));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 50,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6238));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 51,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6239));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 52,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6240));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 53,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6241));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 54,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6243));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 55,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6244));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 56,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6245));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 57,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6246));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 58,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6248));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 59,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6249));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 60,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6250));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 61,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6254));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 62,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6255));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 63,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6256));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 64,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6257));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 65,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6258));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 66,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6261));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 67,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6262));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 68,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6263));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 69,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6264));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 70,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6266));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 71,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6267));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 72,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6268));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 73,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6269));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 74,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6270));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 75,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 24, 5, 29, 21, 555, DateTimeKind.Local).AddTicks(6272));
        }
    }
}
