using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddIsScoreFinalizedToQuizAttempt : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "IsScoreFinalized",
                table: "QuizAttempts",
                type: "bit",
                nullable: false,
                defaultValue: false);

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

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "IsScoreFinalized",
                table: "QuizAttempts");

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 533, DateTimeKind.Local).AddTicks(428));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 533, DateTimeKind.Local).AddTicks(504));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 533, DateTimeKind.Local).AddTicks(507));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 533, DateTimeKind.Local).AddTicks(509));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 533, DateTimeKind.Local).AddTicks(511));

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAENTbGgjdfrpKZZ15A/jsQ2pPI+sT1KOnJyDVZ1R3rUBks7+4xznDOYXf/T1H0YtcBw==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEGC9tTyCVwR7QIlFbBgf4aWR9ww3yH7vXzchQu0oeCGn9KVvmhTqilj3yMLZQBU+OQ==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 536, DateTimeKind.Local).AddTicks(7370));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 536, DateTimeKind.Local).AddTicks(7424));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 536, DateTimeKind.Local).AddTicks(7432));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 536, DateTimeKind.Local).AddTicks(7428));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 536, DateTimeKind.Local).AddTicks(7430));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2589));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2632));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2636));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2640));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2642));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 6,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2645));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 7,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2647));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 8,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2649));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 9,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2650));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 10,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2653));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 11,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2655));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 12,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2656));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 13,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2658));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 14,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2660));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 15,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2662));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 16,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2668));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 17,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2692));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 18,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2695));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 19,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2697));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 20,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2699));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 21,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2701));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 22,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2702));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 23,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2704));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 24,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2705));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 25,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2707));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 26,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2709));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 27,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2710));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 28,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2712));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 29,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2714));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 30,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2716));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 31,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2721));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 32,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2723));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 33,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2724));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 34,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2727));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 35,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2729));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 36,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2730));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 37,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2732));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 38,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2733));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 39,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2735));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 40,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2736));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 41,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2738));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 42,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2739));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 43,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2741));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 44,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2743));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 45,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2744));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 46,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2749));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 47,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2750));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 48,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2751));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 49,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2753));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 50,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2755));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 51,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2756));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 52,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2758));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 53,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2759));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 54,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2761));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 55,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2763));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 56,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2764));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 57,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2766));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 58,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2767));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 59,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2769));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 60,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2770));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 61,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2775));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 62,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2776));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 63,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2778));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 64,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2779));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 65,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2781));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 66,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2851));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 67,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2853));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 68,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2854));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 69,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2856));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 70,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2858));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 71,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2860));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 72,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2861));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 73,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2863));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 74,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2865));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 75,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2866));
        }
    }
}
