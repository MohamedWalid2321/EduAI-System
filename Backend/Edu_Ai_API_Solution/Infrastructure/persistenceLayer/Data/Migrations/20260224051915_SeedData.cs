using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class SeedData : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.InsertData(
                table: "AspNetRoles",
                columns: new[] { "Id", "ConcurrencyStamp", "IsDefault", "IsDeleted", "Name", "NormalizedName" },
                values: new object[,]
                {
                    { "71e40e16-7fe9-4f8b-807b-77c9da3f41a9", "5540e8da-f93d-4457-a355-f04bb15c4594", false, false, "SuperAdmin", "SUPERADMIN" },
                    { "7e07bb31-26ad-47ac-880c-c5fdfa0516d3", "9bba17c3-ee48-423a-b2c0-a63245d0edf0", false, false, "Instructor", "INSTRUCTOR" },
                    { "92b75286-d8f8-4061-9995-e6e23ccdee94", "f51e5a91-bced-49c2-8b86-c2e170c0846c", false, false, "Admin", "ADMIN" },
                    { "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4", "5ee6bc12-5cb0-4304-91e7-6a00744e042a", true, false, "Student", "STUDENT" }
                });

            migrationBuilder.InsertData(
                table: "AspNetUsers",
                columns: new[] { "Id", "AccessFailedCount", "ConcurrencyStamp", "DateOfBirth", "Email", "EmailConfirmed", "FirstName", "LastName", "LockoutEnabled", "LockoutEnd", "NormalizedEmail", "NormalizedUserName", "PasswordHash", "PhoneNumber", "PhoneNumberConfirmed", "ProfilePictureBase64", "ProfilePictureUrl", "SecurityStamp", "TwoFactorEnabled", "UserName" },
                values: new object[,]
                {
                    { "585c8473-10ce-4377-8407-1f64655876c1", 0, "7d47a4bf-ded7-4642-83fd-7b16df7ac368", new DateOnly(1, 1, 1), "superadmin@Lumino.com", true, "Lumino", "SuperAdmin", false, null, "SUPERADMIN@LUMINO.COM", "SUPERADMIN@LUMINO.COM", "AQAAAAIAAYagAAAAEMBc3qOHPL//BEZnCgyca/KLGEQ262Wo2wDKOzTgKKg2ouqaHQVLsF54hBWfO8XzDQ==", null, false, "", "", "911889FEF7B44646B1E278C5C4F7C893", false, "superadmin@Lumino.com" },
                    { "6dc6528a-b280-4770-9eae-82671ee81ef7", 0, "99d2bbc6-bc54-4248-a172-a77de3ae4430", new DateOnly(1, 1, 1), "admin@Lumino.com", true, "Lumino", "Admin", false, null, "ADMIN@LUMINO.COM", "ADMIN@LUMINO.COM", "AQAAAAIAAYagAAAAEFdGASuoR5l5i1HP59wnBPbdFSuPQqPX5qfgX6QA4R8b4BkcOtFCwNc2IiyNa5czpg==", null, false, "", "", "55BF92C9EF0249CDA210D85D1A851BC9", false, "admin@Lumino.com" }
                });

            migrationBuilder.InsertData(
                table: "AspNetRoleClaims",
                columns: new[] { "Id", "ClaimType", "ClaimValue", "RoleId" },
                values: new object[,]
                {
                    { 1, "Permissions", "Ass:read", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 2, "Permissions", "Ass:add", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 3, "Permissions", "Ass:update", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 4, "Permissions", "Ass:delete", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 5, "Permissions", "Ass:solve", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 6, "Permissions", "Content:read", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 7, "Permissions", "Content:add", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 8, "Permissions", "Content:update", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 9, "Permissions", "Content:delete", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 10, "Permissions", "Course:read", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 11, "Permissions", "Course:add", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 12, "Permissions", "Course:update", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 13, "Permissions", "Course:delete", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 14, "Permissions", "Department:read", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 15, "Permissions", "Department:add", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 16, "Permissions", "Department:update", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 17, "Permissions", "Department:delete", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 18, "Permissions", "questions:read", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 19, "Permissions", "questions:add", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 20, "Permissions", "questions:update", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 21, "Permissions", "questions:delete", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 22, "Permissions", "questions:solve", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 23, "Permissions", "users:read", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 24, "Permissions", "users:add", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 25, "Permissions", "users:update", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 26, "Permissions", "users:delete", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 27, "Permissions", "roles:read", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 28, "Permissions", "roles:add", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 29, "Permissions", "roles:update", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 30, "Permissions", "roles:delete", "71e40e16-7fe9-4f8b-807b-77c9da3f41a9" },
                    { 31, "Permissions", "Ass:read", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 32, "Permissions", "Ass:add", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 33, "Permissions", "Ass:update", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 34, "Permissions", "Ass:delete", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 35, "Permissions", "Content:read", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 36, "Permissions", "Content:add", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 37, "Permissions", "Content:update", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 38, "Permissions", "Content:delete", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 39, "Permissions", "Course:read", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 40, "Permissions", "Course:add", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 41, "Permissions", "Course:update", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 42, "Permissions", "Course:delete", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 43, "Permissions", "Department:read", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 44, "Permissions", "Department:add", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 45, "Permissions", "Department:update", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 46, "Permissions", "Department:delete", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 47, "Permissions", "questions:read", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 48, "Permissions", "questions:add", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 49, "Permissions", "questions:update", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 50, "Permissions", "users:read", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 51, "Permissions", "users:add", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 52, "Permissions", "users:update", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 53, "Permissions", "users:delete", "92b75286-d8f8-4061-9995-e6e23ccdee94" },
                    { 54, "Permissions", "Ass:read", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 55, "Permissions", "Ass:add", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 56, "Permissions", "Ass:update", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 57, "Permissions", "Ass:delete", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 58, "Permissions", "Content:read", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 59, "Permissions", "Content:add", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 60, "Permissions", "Content:update", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 61, "Permissions", "Content:delete", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 62, "Permissions", "Course:read", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 63, "Permissions", "questions:read", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 64, "Permissions", "questions:add", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 65, "Permissions", "questions:update", "7e07bb31-26ad-47ac-880c-c5fdfa0516d3" },
                    { 66, "Permissions", "Ass:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 67, "Permissions", "Ass:solve", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 68, "Permissions", "Content:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 69, "Permissions", "Course:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 70, "Permissions", "questions:read", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" },
                    { 71, "Permissions", "questions:solve", "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4" }
                });

            migrationBuilder.InsertData(
                table: "AspNetUserRoles",
                columns: new[] { "RoleId", "UserId" },
                values: new object[,]
                {
                    { "71e40e16-7fe9-4f8b-807b-77c9da3f41a9", "585c8473-10ce-4377-8407-1f64655876c1" },
                    { "92b75286-d8f8-4061-9995-e6e23ccdee94", "6dc6528a-b280-4770-9eae-82671ee81ef7" }
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 1);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 2);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 3);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 4);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 5);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 6);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 7);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 8);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 9);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 10);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 11);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 12);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 13);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 14);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 15);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 16);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 17);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 18);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 19);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 20);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 21);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 22);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 23);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 24);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 25);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 26);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 27);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 28);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 29);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 30);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 31);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 32);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 33);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 34);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 35);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 36);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 37);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 38);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 39);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 40);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 41);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 42);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 43);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 44);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 45);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 46);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 47);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 48);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 49);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 50);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 51);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 52);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 53);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 54);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 55);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 56);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 57);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 58);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 59);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 60);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 61);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 62);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 63);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 64);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 65);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 66);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 67);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 68);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 69);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 70);

            migrationBuilder.DeleteData(
                table: "AspNetRoleClaims",
                keyColumn: "Id",
                keyValue: 71);

            migrationBuilder.DeleteData(
                table: "AspNetUserRoles",
                keyColumns: new[] { "RoleId", "UserId" },
                keyValues: new object[] { "71e40e16-7fe9-4f8b-807b-77c9da3f41a9", "585c8473-10ce-4377-8407-1f64655876c1" });

            migrationBuilder.DeleteData(
                table: "AspNetUserRoles",
                keyColumns: new[] { "RoleId", "UserId" },
                keyValues: new object[] { "92b75286-d8f8-4061-9995-e6e23ccdee94", "6dc6528a-b280-4770-9eae-82671ee81ef7" });

            migrationBuilder.DeleteData(
                table: "AspNetRoles",
                keyColumn: "Id",
                keyValue: "71e40e16-7fe9-4f8b-807b-77c9da3f41a9");

            migrationBuilder.DeleteData(
                table: "AspNetRoles",
                keyColumn: "Id",
                keyValue: "7e07bb31-26ad-47ac-880c-c5fdfa0516d3");

            migrationBuilder.DeleteData(
                table: "AspNetRoles",
                keyColumn: "Id",
                keyValue: "92b75286-d8f8-4061-9995-e6e23ccdee94");

            migrationBuilder.DeleteData(
                table: "AspNetRoles",
                keyColumn: "Id",
                keyValue: "9eaa03df-8e4f-4161-85de-0f6e5e30bfd4");

            migrationBuilder.DeleteData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1");

            migrationBuilder.DeleteData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7");
        }
    }
}
