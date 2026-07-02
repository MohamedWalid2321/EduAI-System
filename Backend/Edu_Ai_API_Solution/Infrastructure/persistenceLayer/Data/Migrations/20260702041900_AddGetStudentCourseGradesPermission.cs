using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddGetStudentCourseGradesPermission : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 379, DateTimeKind.Local).AddTicks(2224));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 379, DateTimeKind.Local).AddTicks(2269));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 379, DateTimeKind.Local).AddTicks(2271));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 379, DateTimeKind.Local).AddTicks(2273));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 379, DateTimeKind.Local).AddTicks(2274));

            migrationBuilder.InsertData(
                table: "AspNetRoleClaims",
                columns: new[] { "Id", "ClaimType", "ClaimValue", "RoleId" },
                values: new object[,]
                {
                    { 1301, "Permissions", "AttemptScore:readByCourse", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 1302, "Permissions", "AttemptScore:readByCourse", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" }
                });

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEFAev6JPBq53Ym/UHvot2mvHcOSuc+fita08bvSr0flt//5tTvFBh67OzebmQ9xoeg==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEFkT6DRSEGG/RKHtls4e52KWKECNwmlFCKbkGF4VaDkCKjQqoHMarxdaeTGiMXxcAQ==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(4876));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(4900));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(4904));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(4902));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(4903));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8154));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8182));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8185));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8188));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8189));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 6,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8193));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 7,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8194));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 8,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8195));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 9,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8196));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 10,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8199));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 11,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8200));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 12,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8201));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 13,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8202));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 14,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8203));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 15,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8204));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 16,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8208));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 17,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8209));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 18,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8210));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 19,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8212));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 20,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8213));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 21,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8213));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 22,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8215));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 23,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8216));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 24,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8217));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 25,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8218));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 26,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8219));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 27,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8220));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 28,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8221));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 29,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8222));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 30,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8223));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 31,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8226));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 32,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8248));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 33,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8249));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 34,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8251));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 35,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8253));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 36,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8254));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 37,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8255));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 38,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8256));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 39,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8257));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 40,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8258));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 41,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8259));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 42,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8260));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 43,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8261));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 44,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8262));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 45,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8263));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 46,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8267));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 47,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8268));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 48,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8269));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 49,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8270));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 50,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8271));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 51,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8272));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 52,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8273));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 53,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8274));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 54,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8275));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 55,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8276));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 56,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8277));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 57,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8278));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 58,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8279));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 59,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8280));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 60,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8281));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 61,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8284));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 62,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8285));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 63,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8286));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 64,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8287));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 65,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8288));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 66,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8290));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 67,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8291));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 68,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8292));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 69,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8293));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 70,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8294));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 71,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8295));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 72,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8296));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 73,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8297));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 74,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8298));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 75,
                column: "CreatedAt",
                value: new DateTime(2026, 7, 2, 7, 18, 59, 381, DateTimeKind.Local).AddTicks(8299));
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1301);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1302);

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
    }
}
