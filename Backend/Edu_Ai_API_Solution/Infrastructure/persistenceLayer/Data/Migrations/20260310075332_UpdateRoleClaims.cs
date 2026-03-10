using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class UpdateRoleClaims : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 70);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 71);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 72);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 73);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 74);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 75);

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 48,
                column: "ClaimValue",
                value: "questions:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 49,
                column: "ClaimValue",
                value: "questions:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 50,
                column: "ClaimValue",
                value: "questions:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 51,
                column: "ClaimValue",
                value: "users:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 52,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:read", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 53,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:add", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 54,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:update", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 55,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:delete", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 56,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:read", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 57,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:add", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 58,
                column: "ClaimValue",
                value: "Content:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 59,
                column: "ClaimValue",
                value: "Content:delete");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 60,
                column: "ClaimValue",
                value: "Course:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 61,
                column: "ClaimValue",
                value: "questions:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 62,
                column: "ClaimValue",
                value: "questions:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 63,
                column: "ClaimValue",
                value: "questions:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 64,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 65,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Ass:solve", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 66,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 67,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Course:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 68,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "questions:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 69,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "questions:solve", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" });

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
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 9, 53, 31, 783, DateTimeKind.Local).AddTicks(9364));

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

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 48,
                column: "ClaimValue",
                value: "Department:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 49,
                column: "ClaimValue",
                value: "Department:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 50,
                column: "ClaimValue",
                value: "Department:delete");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 51,
                column: "ClaimValue",
                value: "questions:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 52,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "questions:add", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 53,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "questions:update", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 54,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "users:read", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 55,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "users:add", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 56,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "users:update", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 57,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "users:delete", "92b75286-d8f8-4061-9995-e6e23ccdee94" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 58,
                column: "ClaimValue",
                value: "Ass:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 59,
                column: "ClaimValue",
                value: "Ass:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 60,
                column: "ClaimValue",
                value: "Ass:update");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 61,
                column: "ClaimValue",
                value: "Ass:delete");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 62,
                column: "ClaimValue",
                value: "Content:read");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 63,
                column: "ClaimValue",
                value: "Content:add");

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 64,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:update", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 65,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Content:delete", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 66,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "Course:read", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 67,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "questions:read", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 68,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "questions:add", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.UpdateData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 69,
                columns: new[] { "ClaimValue", "RoleId" },
                values: new object[] { "questions:update", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" });

            migrationBuilder.InsertData(
                table: "AspNetRoleClaims",
                columns: new[] { "Id", "ClaimType", "ClaimValue", "RoleId" },
                values: new object[,]
                {
                    { 70, "Permissions", "Ass:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 71, "Permissions", "Ass:solve", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 72, "Permissions", "Content:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 73, "Permissions", "Course:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 74, "Permissions", "questions:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 75, "Permissions", "questions:solve", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" }
                });

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAECFQgUDn+fRni27of7x/C1/IRHb6CBOt2pne1i0t/SVyGrJPmDMDGZfKggEUsds66A==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEIAMkASPPncVRJoa1eN+dLeuuIhH3B7w0aydXbdDWwDyOaoiOuYEl4uSj99FgUtXyw==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 8, 55, 29, 69, DateTimeKind.Local).AddTicks(2644));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 8, 55, 29, 69, DateTimeKind.Local).AddTicks(2688));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 8, 55, 29, 69, DateTimeKind.Local).AddTicks(2693));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 8, 55, 29, 69, DateTimeKind.Local).AddTicks(2691));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 10, 8, 55, 29, 69, DateTimeKind.Local).AddTicks(2692));
        }
    }
}
